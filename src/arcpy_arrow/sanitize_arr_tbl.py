import pyarrow as pa
from arcpy import SpatialReference

_UINT_TO_SINT = {
                pa.uint8(): pa.int16(),
                pa.uint16(): pa.int32(),
                pa.uint32(): pa.int64(),
                pa.uint64(): pa.int64(),
            }


def _sanitize_arr_tbl(table: pa.Table, 
                      shape_col:str | None  = None, 
                      sr:SpatialReference | None = None
                      ) -> pa.Table:
    """Sanitize an Arrow table for use with ``arcpy.management.CopyRows()``, ``CopyFeatures()``, and ``Append()``.

    Normalises field types that arcpy cannot handle: upcasts unsigned integers,
    collapses large strings and dictionary-encoded columns, normalises timestamps
    to milliseconds, and attaches ESRI geometry metadata to the shape column.
    Resets metadata on OBJECTID and drops unsupported binary columns.

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
    parsed_shape = shape_col is None

    for field in table.schema:
        if field.name == "OBJECTID":
            fields.append(pa.field(
                                    field.name,
                                    pa.int64(),
                                    metadata={'esri.oid': 'esri.int64'}
                                    ))

        elif (field.name == shape_col):
            if not (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)):
                raise TypeError(f"Shape field {shape_col} is wrong type: should be binary-like, got {field.type}")
        
            fields.append(pa.field(
                name = field.name,
                type = pa.binary(),
                nullable = True,
                metadata={b'esri.encoding': "WKB", b'esri.sr_wkt': sr.exportToString()}
            ))

            parsed_shape = True

        elif (pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type)):
            raise TypeError("Binary-like fields for non-shape columns are not supported.")
        
        # Convert large strings to short strings
        elif pa.types.is_large_string(field.type):
            fields.append(pa.field(
                name=field.name,
                type=pa.string(),
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # ESRI tables to not have bools
        elif pa.types.is_boolean(field.type):
            fields.append(pa.field(
                name=field.name,
                type=pa.int32(),
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # Normalise timestamp units to milliseconds
        # Timezone info is stripped: arcpy Date fields are timezone-naive.
        elif pa.types.is_timestamp(field.type):
            fields.append(pa.field(
                name=field.name,
                type=pa.timestamp("ms"),
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # Flatten dictionary-encoded columns
        elif pa.types.is_dictionary(field.type):
            value_type = field.type.value_type
            if pa.types.is_large_string(value_type):
                value_type = pa.string()
            fields.append(pa.field(
                name=field.name,
                type=value_type,
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # Upcast unsigned integers to the smallest signed type that fits.
        elif pa.types.is_unsigned_integer(field.type):
            
            fields.append(pa.field(
                name=field.name,
                type=_UINT_TO_SINT[field.type],
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        else:
            fields.append(pa.field(
                name=field.name,
                type=field.type,
                nullable=field.nullable,
                metadata=field.metadata,
            ))

    if not parsed_shape:
        raise RuntimeError(f"shape_col={shape_col} but no shape column was detected")

    schema = pa.schema(fields)

    return table.cast(schema)
