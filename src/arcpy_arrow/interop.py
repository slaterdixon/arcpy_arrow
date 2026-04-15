from uuid import uuid4
import warnings
import pyarrow as pa
from pathlib import Path
from shapely import from_wkb
from typing import Literal
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
                    ):

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
    """
    Get a random in-memory string to write to using a universally unique identifier

    :return: A string like "memory/fc_4af874bd5b654bb7914e4165e2667a19"
    :rtype: str
    """
    return f"memory/fc_{uuid4().hex}"

def _sanitize_arr_tbl(table: pa.Table, 
                      shape_col:str | None  = None, 
                      sr:arcpy.SpatialReference | None = None
                      ) -> pa.Table:
    """
    Sanitize types in an Arrow table so it can be passed into arcpy functions
    like arcpy.management.CopyRows(), arcpy.management.CopyFeatures(), arcpy.management.Append(), etc.

    You must explicitly pass a `shape_col` and `sr` for geometry encoding.

    This function drops existing metadata in OBJECTID, Binary columns
    
    :param table: The Arrow Table to be converted
    :param sr: A spatial reference used to construct ESRI geometry metadata
    :return: An arrow table with sanitized columns 
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
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS")
    
    epsg = gdf.crs.to_epsg()
    if epsg is not None:
        return arcpy.SpatialReference(epsg)
    
    # Fall back to WKT for CRS without a clean EPSG code
    sr = arcpy.SpatialReference()
    sr.loadFromString(gdf.crs.to_wkt())
    return sr


def df_to_fc(df: "pl.DataFrame|pd.DataFrame|gpd.GeoDataFrame",
             fc: str|Path|ESRITable|None,
             mode: Literal["new", "overwrite", "append"] = "new",
             ) -> str:

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

    return fc_p


def _resolve_fc (fc: str|Path|ESRITable):
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
