from __future__ import annotations

from uuid import uuid4
import warnings
import pyarrow as pa
from pathlib import Path
from shapely import from_wkb
from typing import Literal, overload
import pandas as pd
from dataclasses import dataclass
from functools import cached_property
import arcpy
import geopandas as gpd

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

_UINT_TO_SINT = {
                pa.uint8(): pa.int16(),
                pa.uint16(): pa.int32(),
                pa.uint32(): pa.int64(),
                pa.uint64(): pa.int64(),
            }

@dataclass
class ESRITable:
    """A lightweight reference to an ArcGIS feature class or table.

    Stores the components needed to construct a catalog path and provides
    convenience accessors for common ``arcpy.Describe`` properties and
    Arrow-based I/O. ``arcpy`` calls are deferred until first access and
    then cached, so constructing an ``ESRITable`` is cheap.

    Args:
        name: The feature class or table name (without workspace or dataset prefix).
        workspace: Path to the geodatabase or folder workspace (e.g. a ``.gdb``
            or ``.sde`` file).
        dataset_name: Name of the containing feature dataset, if any.
        schema_prefix: Prefix applied to both ``dataset_name`` and ``name`` when
            building the catalog path — typically the ``<owner>.`` prefix used
            in enterprise geodatabases (e.g. ``"GIS."``).

    Examples:
        ```
        hydrants_fc = ESRITable(
            name="hydrants",
            workspace=Path("C:/connections/prod.sde"),
            dataset_name="Water",
            schema_prefix="sde.GIS.",
        )
        hydrants_fc.path  
        # >> Path("C:/connections/prod.sde/sde.GIS.Water/sde.GIS.hydrants")
        ```
    """

    name: str
    workspace: Path
    dataset_name: str | None = None
    schema_prefix : str = ""

    @property
    def path(self) -> Path:
        if self.dataset_name:
            return (
                self.workspace
                / f"{self.schema_prefix}{self.dataset_name}"
                / f"{self.schema_prefix}{self.name}"
            )
        else:
            return self.workspace / f"{self.schema_prefix}{self.name}"

    ### Cache properties that make arcpy calls

    @cached_property
    def describe(self):
        return arcpy.Describe(str(self.path))

    @cached_property
    def fields(self) -> list:
        return arcpy.ListFields(str(self.path))

    @cached_property
    def exists(self):
        """Whether the feature class or table exists at :attr:`path`.

        Cached on first access — will not reflect creation or deletion after
        the first call. Invalidate by deleting the attribute: ``del tbl.exists``.
        """
        return arcpy.Exists(str(self.path))

    ### Don't cache simple lookups

    @property
    def data_type(self) -> str:
        return self.describe.dataType

    @property
    def workspace_catalog_path(self) -> Path:
        return Path(self.describe.workspace.catalogPath)

    @property
    def oid_field_name(self) -> str:
        return self.describe.OIDFieldName

    @property
    def is_versioned(self) -> bool:
        return self.describe.isVersioned

    @property
    def spatial_reference(self):
        return self.describe.spatialReference
    
    @property
    def shape_col_name(self) -> str:
        return self.describe.shapeFieldName

    @property
    def fields_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def field_types(self) -> dict[str, str]:
        return {f.name: f.type for f in self.fields}


    def row_count(self):
        return int(arcpy.GetCount_management(str(self.path)).getOutput(0))

    def as_data_frame(self, 
                      field_names: list[str] | None = None, 
                      where_clause: str | None = None, 
                      df_type: Literal["pandas", "polars", "pyarrow"] = "pandas"
                      ) -> "pd.DataFrame|pl.DataFrame|pa.Table":
        
        """Fetch this table's attributes as a DataFrame (non-spatial).

        Args:
            field_names: Subset of fields to fetch. Fetches all fields if ``None``.
            where_clause: SQL where clause to filter rows. No filter if ``None``.
            df_type: Output type — ``"pandas"``, ``"polars"``, or ``"pyarrow"``.

        Raises:
            ValueError: One or more :attr:`field_names` are not present in the table.
            ImportError: :attr:`df_type` is ``"polars"`` but polars is not installed.
            TypeError: :attr:`df_type` is not one of the allowed values.

        Returns:
            A DataFrame of the requested type containing the table's attributes.
        """

        return fc_to_df(fc = self,
                        field_names=field_names,
                        where_clause=where_clause, 
                        df_type=df_type)

        
    def as_geodataframe(self, 
                    where_clause: str | None = None,                     
                    field_names:list[str] | None = None, 
                    fallback_crs : int | None = None,
                    allow_null_geometries:bool = False,
                    to_crs:bool = False,
                    ) -> gpd.GeoDataFrame:
        
        """Fetch this feature class as a GeoDataFrame with shapely geometries.

        Args:
            field_names: Subset of fields to fetch. Fetches all fields if ``None``.
                The geometry column is always included.
            where_clause: SQL where clause to filter rows. No filter if ``None``.
            allow_null_geometries: If ``True``, warn on null/invalid geometries instead
                of raising. Covers missing values and shapes unparseable by shapely.
            fallback_crs: EPSG code to use when the feature class spatial reference
                has no resolvable factory code (e.g. an improperly defined CRS).
            to_crs: If provided, reproject the result to this EPSG code.

        Raises:
            ValueError: One or more :attr:`field_names` are not present in the feature class.

        Returns:
            A GeoDataFrame with the feature class attributes and shapely geometries.
        """

        return fc_to_gdf(fc = self,
                         field_names=field_names,
                         where_clause=where_clause,
                         allow_null_geometries=allow_null_geometries,
                         shape_col=self.shape_col_name,
                         fallback_crs=fallback_crs,
                         to_crs=to_crs
                         )
        


    def __repr__(self):
        cls = self.__class__.__name__
        parts = [f"name='{self.name}'"]

        if self.dataset_name:
            parts.append(f"dataset='{self.dataset_name}'")

        parts.append(f"conn='{self.workspace.name}'")
        parts.append(f"prefix='{self.schema_prefix}'")

        return f"{cls}({', '.join(parts)})"


    def __str__(self):
        return str(self.path)


def memory_str() -> str:
    """Return a unique ArcGIS in-memory path, e.g. ``"memory/fc_4af874bd5b654bb7914e4165e2667a19"``."""
    return f"memory/fc_{uuid4().hex}"


def _sanitize_arr_tbl(table: pa.Table, 
                      shape_col:str | None  = None, 
                      sr:arcpy.SpatialReference | None = None
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
    if sr is not None and not isinstance(sr, arcpy.SpatialReference):
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

        # arcpy has no boolean field type — cast to int32 (matches ESRI's own convention).
        elif pa.types.is_boolean(field.type):
            fields.append(pa.field(
                name=field.name,
                type=pa.int32(),
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # Normalise timestamp units to milliseconds — arcpy's TableToArrowTable
        # produces timestamp[ms] and the write path expects the same unit.
        # Polars defaults to us, newer pandas to ns; both need normalising.
        # Timezone info is stripped: arcpy Date fields are timezone-naive.
        elif pa.types.is_timestamp(field.type):
            fields.append(pa.field(
                name=field.name,
                type=pa.timestamp("ms"),
                nullable=field.nullable,
                metadata=field.metadata,
            ))

        # Flatten dictionary-encoded columns to their value type.
        # pandas Categorical → dictionary<int32, utf8>
        # polars Categorical/Enum → dictionary<int32, large_utf8>
        # large_utf8 value types are also downcasted to utf8.
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
        # uint64 → int64 is the only potentially lossy cast (values > 2^63-1).
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


### df -> fc
def _sr_from_gdf(gdf: gpd.GeoDataFrame) -> arcpy.SpatialReference:
    """Derive an ``arcpy.SpatialReference`` from a GeoDataFrame's CRS.

    Prefers EPSG factory code; falls back to the projection's authority string
    for CRS that have no clean EPSG code.

    Args:
        gdf: A GeoDataFrame with a defined CRS.

    Raises:
        ValueError: The GeoDataFrame has no CRS set.

    Returns:
        An ``arcpy.SpatialReference`` matching the GeoDataFrame's CRS.
    """

    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS")
    
    epsg = gdf.crs.to_epsg()
    if epsg is not None:
        return arcpy.SpatialReference(epsg)
    
    # Fall back to WKT for CRS without a clean EPSG code
    sr = arcpy.SpatialReference()
    sr.loadFromString(gdf.crs.to_wkt())
    return sr


@overload
def df_to_fc(df: "pl.DataFrame|pd.DataFrame|gpd.GeoDataFrame", fc: ESRITable, mode: Literal["new", "overwrite", "append"] = ...) -> ESRITable: ...
@overload
def df_to_fc(df: "pl.DataFrame|pd.DataFrame|gpd.GeoDataFrame", fc: str|Path|None, mode: Literal["new", "overwrite", "append"] = ...) -> str: ...
def df_to_fc(df: "pl.DataFrame|pd.DataFrame|gpd.GeoDataFrame",
             fc: str|Path|ESRITable|None,
             mode: Literal["new", "overwrite", "append"] = "new",
             ) -> str|ESRITable:
    
    """Write a DataFrame to an ArcGIS feature class or table.

    Args:
        df: A pandas DataFrame, GeoPandas GeoDataFrame, or Polars DataFrame.
        fc: Destination path. Accepts a string, ``Path``, ``ESRITable``, or
            ``None`` to write to a new in-memory workspace path.
        mode: ``"new"`` creates the feature class (errors if it exists),
            ``"overwrite"`` replaces it, ``"append"`` adds rows to an existing one.

    Raises:
        TypeError: :attr:`df` is not a supported DataFrame type.
        FileNotFoundError: :attr:`mode` is ``"append"`` but :attr:`fc` does not exist.

    Returns:
        The path to the written feature class. Returns the original ``ESRITable``
        when one is passed as :attr:`fc`, otherwise returns the path as a string.
    """

    sr = None
    is_spatial = False
    shape_col = None

    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):  # type: ignore[union-attr]
        df_arrow = df.to_arrow()
    # GeoDataFrame is a subclass of DataFrame and must be checked first
    elif isinstance(df, gpd.GeoDataFrame):
        shape_col = str(df.geometry.name)
        _raw = df.to_arrow(geometry_encoding="WKB")
        # Newer geopandas returns a wrapper, not a pa.Table.  Any Arrow-compatible
        # object supports __arrow_c_stream__, so we round-trip through that.
        if not isinstance(_raw, pa.Table):
            _raw = pa.RecordBatchReader.from_stream(_raw).read_all()
        df_arrow = _raw
        sr = _sr_from_gdf(df)
        is_spatial = True
    elif isinstance(df, pd.DataFrame):
        df_arrow = pa.Table.from_pandas(df, preserve_index=False)
    else:
        raise TypeError("df must be a valid polars, geopandas, or pandas dataframe")

    df_san = _sanitize_arr_tbl(table=df_arrow, shape_col=shape_col, sr=sr)

    if fc is None:
        fc_p = memory_str()
    elif isinstance(fc, ESRITable):
        fc_p = str(fc.path)
    else:
        fc_p = str(fc)

    if mode == "append":
        if not arcpy.Exists(fc_p):
            raise FileNotFoundError(f"Cannot append: target does not exist: {fc_p}")
        temp_fc = memory_str()
        try:
            if is_spatial:
                arcpy.management.CopyFeatures(in_features=df_san, out_feature_class=temp_fc)
            else:
                arcpy.management.CopyRows(in_rows=df_san, out_table=temp_fc)
            arcpy.management.Append(
                inputs=temp_fc,
                target=fc_p,
                schema_type="NO_TEST",
            )
            arcpy_warnings = arcpy.GetMessages(1)
            if arcpy_warnings:
                warnings.warn(arcpy_warnings, stacklevel=2)
        finally:
            if arcpy.Exists(temp_fc):
                arcpy.management.Delete(temp_fc)
    else:
        with arcpy.EnvManager(overwriteOutput=(mode == "overwrite")):
            if is_spatial:
                arcpy.management.CopyFeatures(in_features=df_san, out_feature_class=fc_p)
            else:
                arcpy.management.CopyRows(in_rows=df_san, out_table=fc_p)

    if isinstance(fc, ESRITable):
        return fc
    return fc_p


def _resolve_fc(fc: str|Path|ESRITable) -> tuple[bool, list[str], str]:
    """
    Get info needed to validate a fc. Helper for `fc_to_df()`
    """

    if isinstance(fc, ESRITable):
        exists = fc.exists
        available_fields = fc.fields_names
        fc_p = str(fc.path)
    else:
        fc_p = str(fc)
        exists = arcpy.Exists(fc_p)
        available_fields = [f.name for f in arcpy.ListFields(fc_p)] if exists else []
        
    return exists, available_fields, fc_p


def fc_to_df(fc: str|Path|ESRITable,
             field_names: list[str] | None = None,
             where_clause: str | None = None,
             df_type: Literal["pandas", "polars", "pyarrow"] = "pandas",
            ) -> "pd.DataFrame|pl.DataFrame|pa.Table":
    
    """Fetch a table or feature class as a DataFrame (attributes only).

    Wraps ``arcpy.da.TableToArrowTable()`` with field validation and existence
    checks. For spatial data with usable geometries use ``fc_to_gdf()`` instead.

    Args:
        fc: The table or feature class to read.
        field_names: Subset of fields to fetch. Fetches all fields if ``None``.
        where_clause: SQL where clause to filter rows. No filter if ``None``.
        df_type: Output type — ``"pandas"``, ``"polars"``, or ``"pyarrow"``.

    Raises:
        FileNotFoundError: :attr:`fc` does not exist.
        ValueError: One or more :attr:`field_names` are not present in the table.
        ImportError: :attr:`df_type` is ``"polars"`` but polars is not installed.
        TypeError: :attr:`df_type` is not one of the allowed values.

    Returns:
        A DataFrame of the requested type containing the table's attributes.
    """


    exists, available_fields, fc_p = _resolve_fc(fc)

    if not exists:
        raise FileNotFoundError(f"Feature Class does not exist: {fc}")

    if field_names is not None:
        bad_fields = [f for f in field_names if f not in available_fields]
        if bad_fields:  
            raise ValueError(f"The requested fields do not exist: {', '.join(bad_fields)}")
        
    arr = arcpy.da.TableToArrowTable(in_table = fc_p, 
                                     field_names=field_names, 
                                     where_clause=where_clause
                                     )
    if df_type == "pandas":
        return arr.to_pandas()
    elif df_type == "polars":
        if not _POLARS_AVAILABLE:
            raise ImportError("polars is not installed. Install it with: pip install arcpy_arrow[polars]")
        return pl.from_arrow(arr)  # type: ignore[union-attr]
    elif df_type == "pyarrow":
        return arr
    else:
        raise TypeError(f"df_type '{df_type}' is invalid: Allowed options are ['pandas', 'polars', 'pyarrow']")
    

def _load_gdf_from_arr(arr: pa.Table,
                       sr: arcpy.SpatialReference,
                       allow_null_geometries: bool = False,
                       shape_col: str = "Shape",
                       fallback_crs: int | None = None,
                       to_crs: int | None = None
                       ):
    
    """Build a GeoDataFrame from an Arrow table produced by ``arcpy.da.TableToArrowTable()``.

    Args:
        arr: Arrow table with a WKB geometry column.
        sr: Spatial reference of the source feature class, used to set the CRS.
        allow_null_geometries: If ``True``, warn on null/invalid geometries instead
            of raising. Applies to both missing values and shapely parse failures.
        shape_col: Name of the WKB geometry column in :attr:`arr`.
        fallback_crs: EPSG code to use when :attr:`sr`.factoryCode is 0 (unresolvable).
        to_crs: If provided, reproject the GeoDataFrame to this EPSG code after
            construction.

    Raises:
        ValueError: Null geometries are present and :attr:`allow_null_geometries` is ``False``.
        AttributeError: :attr:`sr`.factoryCode is 0 and no :attr:`fallback_crs` was given.

    Returns:
        A GeoDataFrame with shapely geometries and CRS set from :attr:`sr`.
    """
    
    df = arr.to_pandas()
    
    handle_invalid = "warn" if allow_null_geometries else "raise"
    geometries = from_wkb(df[shape_col].values, on_invalid= handle_invalid)

    null_mask = df[shape_col].isna()
    if null_mask.any():
        msg = f"{null_mask.sum()} null geometries found in column '{shape_col}'"
        if allow_null_geometries:
            warnings.warn(msg)
        else:
            raise ValueError(msg)


    factory_code_proj = sr.factoryCode

    if factory_code_proj == 0:
        if fallback_crs is None:
            raise AttributeError(f"You must provide a source_crs when the source feature class hass factoryCode=0.")
        else:
            factory_code_proj = fallback_crs


    # Drop the raw WKB bytes column — geometry is now held as shapely objects.
    # Keeping it would leave a stale binary column that breaks a subsequent
    # df_to_fc call (two binary columns, only one being the active geometry).
    gdf = gpd.GeoDataFrame(df.drop(columns=[shape_col]),
                            geometry=geometries,
                            crs=factory_code_proj
                            )
            
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    
    return gdf


def fc_to_gdf(fc: str|Path|ESRITable,
             field_names: list[str] | None = None, 
             where_clause: str | None = None, 
             allow_null_geometries : bool = False,
             shape_col: str | None= None,
             fallback_crs: int| None = None,
             to_crs: int | None = None
             ):
    """Fetch a feature class as a GeoDataFrame with shapely geometries.

    Args:
        fc: The feature class to read.
        field_names: Subset of fields to fetch. Fetches all fields if ``None``.
            The geometry column is always included.
        where_clause: SQL where clause to filter rows. No filter if ``None``.
        allow_null_geometries: If ``True``, warn on null/invalid geometries instead
            of raising. Covers missing values and shapes unparseable by shapely.
        shape_col: Name of the geometry column. Auto-detected from :attr:`fc` when
            an ``ESRITable`` is passed; defaults to ``"Shape"`` otherwise.
        fallback_crs: EPSG code to use when the feature class spatial reference
            has no resolvable factory code (e.g. an improperly defined CRS).
        to_crs: If provided, reproject the result to this EPSG code.

    Raises:
        FileNotFoundError: :attr:`fc` does not exist.
        ValueError: One or more :attr:`field_names` are not present in the feature class.

    Returns:
        A GeoDataFrame with the feature class attributes and shapely geometries.
    """

    exists, available_fields, fc_p = _resolve_fc(fc) 
    if not exists:
        raise FileNotFoundError(f"Feature Class does not exist: {fc}")

    if field_names is not None:
        bad_fields = [f for f in field_names if f not in available_fields]
        if bad_fields:  
            raise ValueError(f"The requested fields do not exist: {', '.join(bad_fields)}")
    
    shape_col = shape_col or (fc.shape_col_name if isinstance(fc, ESRITable) else "Shape")
    sr = fc.spatial_reference if isinstance(fc, ESRITable) else arcpy.Describe(fc_p).spatialReference

    # Always include the geometry column even when field_names is a subset.
    if field_names is not None and shape_col not in field_names:
        field_names = [shape_col, *field_names]

    arr = arcpy.da.TableToArrowTable(in_table = fc_p,
                                     field_names=field_names,
                                     where_clause=where_clause,
                                     geometry_encoding="WKB"
                                     )
    
    return _load_gdf_from_arr(arr = arr,
                                  sr = sr,
                                  allow_null_geometries = allow_null_geometries,
                                  shape_col = shape_col,
                                  fallback_crs=fallback_crs,
                                  to_crs=to_crs)
