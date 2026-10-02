import pyarrow as pa
import pyarrow.compute as pc
import arcpy
from arcpy import SpatialReference
from dataclasses import dataclass
from pathlib import Path


from arcpy_arrow.helpers import ap_workspace, ap_fields

_UINT_TO_SINT = {
                pa.uint8(): pa.int16(),
                pa.uint16(): pa.int32(),
                pa.uint32(): pa.int64(),
                pa.uint64(): pa.int64(),
            }


# File geodatabase drops rows with BigInteger fields>+-2**53
_FGDB_MAX_INT = 2**53

### Template validation ####################################
@dataclass
class DestField():
    name : str
    field_arcpy: arcpy.Field
    field_pyarrow: pa.Field




def _field_metadata(field: pa.Field) -> dict[str, str]:

    metadata = field.metadata or {}

    def _decode_mtdta_entry(e):
        return e.decode() if isinstance(e, bytes) else e

    return {_decode_mtdta_entry(k): _decode_mtdta_entry(v) for k, v in metadata.items()}

def _arr_fieldtype_category(field: pa.Field) -> str:
    """pyarrow fieldtype category, used for _COMPATIBLE"""
    t = field.type

    if _field_metadata(field).get("esri.interop.type") == "esri.timestamp_offset":
        return "timestamp_offset"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_date(t):
        return "date"
    if pa.types.is_time(t):
        return "time"
    return "other"

# Compatible fieldtype category combinations
_ALLOWED_CASTS = {
    ("int", "int"), 
    ("int", "float"), 
    ("float", "float"), 
    ("float", "int"),
    ("str", "str"),
    ("timestamp", "timestamp"), 
    ("timestamp", "date"),
    ("date", "date"), 
    ("date", "timestamp"),
    ("time", "time"),
    ("timestamp_offset", "timestamp_offset"),
    ("other", "other"),
}


_BAD_PA_TYPES = [
        (pa.types.is_list, "List types are not supported"),
        (pa.types.is_duration, "Duration types are not supported"),
        (pa.types.is_null, "All-null columns are not supported"),
        (pa.types.is_decimal, "Decimal types are not supported")
]


_PARSE = "Parse it first, e.g. pd.to_numeric(df[col])."
_FORMAT = "Format it first, e.g. df[col].astype(str)."
_FIXES = {
    ("str", "int"): _PARSE,
    ("str", "float"): _PARSE,
    ("str", "timestamp"): "Parse it first, e.g. pd.to_datetime(df[col]).",
    ("str", "date"): "Parse it first, e.g. pd.to_datetime(df[col]).dt.date.",
    ("str", "timestamp_offset"): "Parse it first, e.g. pd.to_datetime(df[col], utc=True).",
    ("int", "str"): _FORMAT,
    ("float", "str"): _FORMAT,
    ("timestamp_offset", "timestamp"): (
        "The column is timezone-aware but a Date field is not. Convert it explicitly: "
        "df[col].dt.tz_convert('UTC').dt.tz_localize(None) stores UTC, "
        "df[col].dt.tz_localize(None) keeps local time."
    ),
    ("timestamp_offset", "date"): "The column is timezone-aware. Convert it explicitly with dt.tz_localize(None).",
    ("timestamp", "timestamp_offset"): "The column has no timezone but the field stores an offset. Localize it first, e.g. df[col].dt.tz_localize('UTC').",
}

def is_pa_safecast(column: pa.ChunkedArray, target_type: pa.DataType) -> bool:
    """
    Can pyarrow safely cast all values in :attr:`column`  to :attr:`target_type` 
    """
    try:
        column.cast(target_type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return False
    return True

def _all_midnight(column: pa.ChunkedArray) -> bool:
    """
    Check if all timestamps in a pyarrow arrow are midnight (lack useful time info)
    """
    midnight = pc.equal(pc.floor_temporal(column, unit="day"), column)
    return pc.all(midnight, min_count=0).as_py()      # min_count=0: True when empty or all null

def _longest_str(column: pa.ChunkedArray) -> int:
    return pc.max(pc.utf8_length(column)).as_py() or 0

def arr_compare_fields(column: pa.ChunkedArray,
                       orig_field: pa.Field,
                       dest_field: DestField,
                       input_is_fgdb: bool = False
                    ) -> list[str]:
    """
    Validate problems loading :attr:`column` into :attr:`dest_field`. 
    """

    cast_label = f"'{orig_field.name}' ({orig_field.type} -> {dest_field.field_arcpy.type})"
    orig_kind = _arr_fieldtype_category(orig_field)
    dest_kind = _arr_fieldtype_category(dest_field.field_pyarrow)
    problems = []

    # Is casting these type categories allowed?
    if (orig_kind, dest_kind) not in _ALLOWED_CASTS:
        problems.append(f"{cast_label}: can not be converted. {_FIXES.get((orig_kind, dest_kind), '')}".rstrip())

    # Is this numeric cast lossy?
    elif orig_kind in ("int", "float") and dest_kind in ("int", "float"):
        if not is_pa_safecast(column, dest_field.field_pyarrow.type):
            problems.append(f"{cast_label}:  pyarrow can not cast these types without overflow or rounding")

        elif input_is_fgdb and orig_kind == "int" and dest_field.field_pyarrow.type == pa.int64():
            limits = pc.min_max(column).as_py()
            if limits["min"] is not None and (limits["min"] < -_FGDB_MAX_INT or limits["max"] > _FGDB_MAX_INT):
                problems.append(f"{cast_label}: a file geodatabase drops rows with integers beyond +-2**53 "
                                f"(got {limits['min']} to {limits['max']})")

    # Would this cast drop time info in a timestamp?
    elif (orig_kind, dest_kind) == ("timestamp", "date") and not _all_midnight(column):
        problems.append(f"{cast_label}: values have a time of day that a date-only field would drop")

    # Would this cast truncate strings longer than the arcpy field length?
    elif (orig_kind, dest_kind) == ("str", "str"):
        longest = _longest_str(column)
        if longest > dest_field.field_arcpy.length:
            problems.append(f"{cast_label}: longest value is {longest} characters, field length is {dest_field.field_arcpy.length}")

    # Would this cast insert nulls from the orig into a non-nullable dest?
    if not dest_field.field_pyarrow.nullable and column.null_count:
        problems.append(f"{cast_label}: {column.null_count} null value(s) but the dest field does not allow nulls")

    return problems


### TimestampOffset ######################################

_TIMESTAMP_OFFSET_METADATA = {"esri.interop.type": "esri.timestamp_offset"}


def _tz_aware_to_offset_strings(column: pa.ChunkedArray) -> pa.Array:
    """ISO 8601 strings with a UTC offset, e.g. ``2024-07-01T12:00:00.123-07:00``.
    """
    s = column.to_pandas()
    base = s.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]    # microseconds -> milliseconds
    offset = s.dt.strftime("%z")                              # "-0700"
    return pa.array(base + offset.str[:3] + ":" + offset.str[3:], type=pa.string(), from_pandas=True)


def _spatial_reference(field: pa.Field) -> arcpy.SpatialReference:
    sr = arcpy.SpatialReference()
    sr.loadFromString(_field_metadata(field)["esri.sr_wkt"])
    return sr

def _geometry_field(schema: pa.Schema) -> pa.Field | None:
    return next((f for f in schema if "esri.sr_wkt" in _field_metadata(f)), None)


def _spatial_reference_mismatches(orig_geom: pa.Field | None, 
                                  dest_geom: pa.Field | None, 
                                  template: str) -> list[str]:
    
    if orig_geom is None and dest_geom is None:
        return []

    if orig_geom is None or dest_geom is None:
        what = "a table into a feature class" if orig_geom is None else "a feature class into a table"
        return [f"dataType mismatch: writing {what} ({template})"]

    orig_sr = _spatial_reference(orig_geom)
    dest_sr = _spatial_reference(dest_geom)

    factory_codes_eq = (orig_sr.factoryCode == dest_sr.factoryCode)
    wkts_eq =  orig_sr.exportToString() == dest_sr.exportToString()


    if (factory_codes_eq or wkts_eq) and orig_sr.factoryCode != 0:
        return []

    return [f"spatial reference is not the same: '{orig_sr.name}' ({orig_sr.factoryCode}) in the data, "
            f"'{dest_sr.name}' ({dest_sr.factoryCode}) in {template}. Reproject first, e.g. gdf.to_crs({dest_sr.factoryCode})."]

def get_system_fields(desc):

    o = set()

    editor_fields = ["created_user", "created_date", "last_edited_user", "last_edited_date"]

    for system_field in ["lengthFieldName", "areaFieldName", "shapeFieldName"]:

        field_name = getattr(desc, system_field, "")

        if field_name:
            o.add(field_name.lower())

    o.update(editor_fields)

    return o



def get_dest_fields(template_p: str):

    dest_schema = arcpy.da.TableToArrowTable(template_p, where_clause = "1=2").schema
    desc = arcpy.Describe(template_p)
    arcpy_fields = {f.name.lower(): f for f in ap_fields(template_p)}

    system_names = get_system_fields(desc)

    dest_fields = {}
    for dest_field in dest_schema:

        field_arcpy = arcpy_fields[dest_field.name.lower()]
        if field_arcpy.type in ("OID", "Geometry") or dest_field.name.lower() in system_names:
            continue
        dest_fields[dest_field.name.lower()] = DestField(name = dest_field.name,
                                                            field_arcpy = field_arcpy,
                                                            field_pyarrow = dest_field)

    return dest_fields, dest_schema


def _validate_against_template(table: pa.Table,
                               template: str|Path):
    """Raise a ValueError listing everything that stops :attr:`table` being appended to :attr:`template`.

    Columns with no matching field in the template are not a problem here: they are skipped
    (and warned about when appending). OBJECTID and arcpy's Shape_Length / Shape_Area are never compared.
    """

    template_p = str(template)

    workspace = str(ap_workspace(template_p))
    input_is_fgdb = workspace.lower().endswith(".gdb")

    
    dest_fields, dest_schema = get_dest_fields(template_p)

    orig_geom = _geometry_field(table.schema)
    dest_geom = _geometry_field(dest_schema)

    problems = []

    spatial_reference_mismatches = _spatial_reference_mismatches(orig_geom, dest_geom, template_p)

    problems.extend(spatial_reference_mismatches)

    for idx, orig_field in enumerate(table.schema):

        if orig_geom is not None and orig_field.name == orig_geom.name:
            continue

        name_validated = arcpy.ValidateFieldName(name = orig_field.name, workspace = workspace)
        dest_field = dest_fields.get(name_validated.lower())

        if dest_field is not None:
            problems += arr_compare_fields(column = table.column(idx), 
                                           orig_field=orig_field, 
                                           dest_field=dest_field, 
                                           input_is_fgdb = input_is_fgdb)

    if problems:
        raise ValueError(f"The data can not be written to {template_p}:\n  - " + "\n  - ".join(problems))

def _pa_fieldtype_supported(field: pa.Field):

    for is_func, message in _BAD_PA_TYPES:
        if is_func(field):
            raise ValueError(f"Problem with field '{field.name}': {message}",
                             "Drop or cast this column before calling.")

def _pa_fieldname_isascii(field: pa.Field):
    if not field.name.isascii():
        raise ValueError(f"Problem with field '{field.name}': Field names must contain ASCII characters only.")

    
def _sanitize_arr_tbl(table: pa.Table,
                      shape_col:str | None  = None,
                      sr:SpatialReference | None = None,
                      template:Path|str|None = None
                      ) -> pa.Table:
    """
    Sanitize an Arrow table for use with ``arcpy.management.CopyRows()``, ``CopyFeatures()``, and ``Append()``.

    Normalises field types that arcpy cannot handle: upcasts unsigned integers,
    collapses large strings and dictionary-encoded columns, normalises timestamps
    to milliseconds, and attaches ESRI geometry metadata to the shape column.
    Resets metadata on OBJECTID and drops unsupported binary columns.

    Timezone-aware timestamp columns become ISO 8601 strings with a UTC offset,
    tagged so arcpy writes a TimestampOffset field. Each row keeps its offset.
    Naive timestamp columns stay timestamps and are written as Date fields.

    Args:
        table: The Arrow table to sanitize.
        shape_col: Name of the geometry column. Must be binary (WKB). If provided,
            :attr:`sr` must also be supplied.
        sr: Spatial reference used to build the ``esri.sr_wkt`` field metadata.

    Raises:
        TypeError: :attr:`table` is not a ``pa.Table``.
        ValueError: :attr:`shape_col` is provided but :attr:`sr` is ``None``.
        TypeError: :attr:`sr` is not an ``arcpy.SpatialReference``.
        TypeError: :attr:`shape_col` column is not binary-like.
        TypeError: A non-shape column has a binary-like type (unsupported).
        RuntimeError: :attr:`shape_col` was specified but not found in the schema.

    Returns:
        A sanitized ``pa.Table`` ready to pass to arcpy write functions.
    """


    if not isinstance(table, pa.Table):
        raise TypeError("The passed table must be a pyarrow table")
    if sr is None and shape_col is not None: 
        raise ValueError(f"Geometry column '{shape_col}' passed but no spatial reference provided")
    if sr is not None and not isinstance(sr, SpatialReference):
        raise TypeError("Invalid spatial reference argument")
    
    fields = []
    columns = []
    parsed_shape = shape_col is None


    for i, field in enumerate(table.schema):


        _pa_fieldname_isascii(table)
        _pa_fieldtype_supported(field)

        column = table.column(i)

        if field.name == "OBJECTID":
            new_field = pa.field(
                field.name,
                pa.int64(),
                metadata={'esri.oid': 'esri.int64'}
            )

        elif field.name == "GlobalID":
            new_field = pa.field(
                field.name,
                field.type,
                metadata = {'esri.interop.type': 'esri.global_id'}
            )

        elif (field.name == shape_col):
            if not (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)):
                raise TypeError(f"Shape field {shape_col} is wrong type: should be binary-like, got {field.type}")

            new_field = pa.field(
                name = field.name,
                type = pa.binary(),
                nullable = True,
                metadata={b'esri.encoding': "WKB", b'esri.sr_wkt': sr.exportToString()}
            )

            parsed_shape = True

        elif (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)):
            raise TypeError("Binary-like fields for non-shape columns are not supported.")

        # Convert large strings to short strings
        elif pa.types.is_large_string(field.type):
            new_field = pa.field(
                name=field.name,
                type=pa.string(),
                nullable=field.nullable,
                metadata=field.metadata,
            )

        # ESRI tables to not have bools
        elif pa.types.is_boolean(field.type):
            new_field = pa.field(
                name=field.name,
                type=pa.int32(),
                nullable=field.nullable,
                metadata=field.metadata,
            )

        elif pa.types.is_timestamp(field.type):

            # Naive: milliseconds, arcpy's resolution. Finer precision is truncated, not rejected.
            if field.type.tz is None:
                new_field = pa.field(
                    name=field.name,
                    type=pa.timestamp("ms"),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                column = column.cast(new_field.type, safe=False)

            # Timezone-aware: arcpy Date fields are naive, so write a TimestampOffset field instead.
            else:
                new_field = pa.field(
                    name=field.name,
                    type=pa.string(),
                    nullable=field.nullable,
                    metadata=_TIMESTAMP_OFFSET_METADATA,
                )
                column = _tz_aware_to_offset_strings(column)

        # Flatten dictionary-encoded columns
        elif pa.types.is_dictionary(field.type):
            value_type = field.type.value_type
            if pa.types.is_large_string(value_type):
                value_type = pa.string()
            new_field = pa.field(
                name=field.name,
                type=value_type,
                nullable=field.nullable,
                metadata=field.metadata,
            )

        # Upcast unsigned integers to the smallest signed type that fits.
        elif pa.types.is_unsigned_integer(field.type):
            new_field = pa.field(
                name=field.name,
                type=_UINT_TO_SINT[field.type],
                nullable=field.nullable,
                metadata=field.metadata,
            )

        else:
            new_field = pa.field(
                name=field.name,
                type=field.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )

        fields.append(new_field)
        try:
            columns.append(column.cast(new_field.type))
        except pa.ArrowInvalid as e:
            raise pa.ArrowInvalid(f"Column '{field.name}': {e}") from e

    if not parsed_shape:
        raise RuntimeError(f"shape_col={shape_col} but no shape column was detected")

    tbl = pa.Table.from_arrays(columns, schema=pa.schema(fields))

    if template is not None:
        _validate_against_template(tbl, template)

    return tbl
