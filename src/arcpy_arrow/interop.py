from __future__ import annotations

from uuid import uuid4
import warnings
import pyarrow as pa
from pathlib import Path
from shapely import from_wkb
from typing import Callable, Literal, TypeAlias, TYPE_CHECKING
import pandas as pd
from dataclasses import dataclass
import arcpy
import geopandas as gpd
from geopandas.array import (
        LINE_GEOM_TYPES,
        POINT_GEOM_TYPES,
        POLYGON_GEOM_TYPES,
    )

from .sanitize_arr_tbl import _sanitize_arr_tbl
from .helpers import ap_exists, ap_has_zm, ap_row_count, ap_shapetype, ap_workspace

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from pyproj import CRS

# geopandas accepts multiple types as CRS: EPSG code, "EPSG:2227", WKT, or a pyproj CRS
CRSLike: TypeAlias = "int | str | CRS"

def memory_str() -> str:
    """Return a unique ArcGIS in-memory path, e.g. ``"memory/fc_4af874bd5b654bb7914e4165e2667a19"``."""
    return f"memory/fc_{uuid4().hex}"

def _qualify_db_path(
    name: str,
    *,
    db_conn: Path,
    db_prefix: str,
    dataset_name: str | None = None
) -> Path:
    """Fully-qualified SDE or FGDB catalog path, 
    like ``C:/connections/database.sde/sde.GIS.Address/sde.GIS.AddressPoints``
    """


    if dataset_name:
        base_p = db_conn / f"{db_prefix}{dataset_name}" 
    else:
        base_p = db_conn

    if name and name is not None:
        o =  base_p  / f"{db_prefix}{name}"
    else:
        o = base_p

    return o
    

def _qualify_memory_path(name: str) -> str:
    """In-memory workspace path"""
    return name if name.startswith(("memory/", "memory\\")) else f"memory/{name}"

@dataclass
class CatalogPath:
    """
    arcpy represents catalog paths as strings. For fully-qualified enterprise geodatabase
    paths, this means joining 
        1. The .sde or .gdb file
        2. The db prefix
        3. The dataset name, 
        4. The table name.

    To simplify this process CatalogPath parametizes each of these inputs, and resolves them as
    a unified pathlib object, ``CatalogPath.path``, or string, ``str(CatalogPath)``.
     
    It also includes convenience functions for fetching metadata about the feature class
    referenced by the CatalogPath, if it exists. 

    Memory and File Geodatabase workspaces are supported by accepting the default ``db_prefix = ''``
    and passing either 1. The FGDB (.fgdb) directory path or 2. The literal string "memory" 
    """

    name: str
    db_conn: Path | str 
    db_prefix: str = ""
    dataset_name: str | None = None


    ### Possible whether table exists or not
    @property
    def full_name(self) -> str:
        return f"{self.db_prefix}{self.name}"


    @property
    def path(self) -> Path:
        if self.db_conn == "memory":
            return Path(_qualify_memory_path(self.name))

        return _qualify_db_path(
            name=self.name,
            db_conn=Path(self.db_conn),
            db_prefix=self.db_prefix,
            dataset_name=self.dataset_name,
        )
    
    @property
    def dataset_path(self) -> Path|None:
        if self.dataset_name is not None:
            return _qualify_db_path(
                name=self.dataset_name,
                db_conn=Path(self.db_conn),
                db_prefix=self.db_prefix,
            )
        else:
            return None

    @property                
    def pretty_path(self, depth: int|None = None) -> str:
        """
        A short path like `./database.sde/LandRecords/Parcels` for logging
        """

        parts = self.path.parts
        depth = depth or (3 if self.dataset_name is not None else 2)
        return './' + '/'.join(parts[-depth:])

    def exists(self) -> bool:
        return ap_exists(str(self))

    def assert_exists(self) -> bool:
        e = self.exists()

        if not e:
            raise FileNotFoundError(f"CatalogPath does not exist: {self.path}")
        
        return e

    @property
    def parent_path(self) -> Path|None:
        """The dataset or workspace this table lives in. ``None`` for the memory workspace."""
        if self.db_conn == "memory":
            return None
        return self.dataset_path or Path(self.db_conn)

    def assert_parent_exists(self) -> bool:
        parent = self.parent_path

        if parent is not None and not arcpy.Exists(str(parent)):
            raise FileNotFoundError(f"CatalogPath parent workspace or dataset does not exist: {parent}")

        return True

    ### Only possible on table that exist
    # Describe properties
    def describe(self):
        self.assert_exists()
        return arcpy.Describe(str(self.path))

    def data_type(self) -> str:
        return self.describe().dataType

    def workspace(self) -> Path|str:
        return ap_workspace(self.path)

    def oid_field_name(self) -> str|None:
        d =  self.describe()

        if hasattr(d, "OIDFieldName"):
            return d.OIDFieldName
        else: 
            return None

    def shape_field_name(self) -> str|None:
        d =  self.describe()

        # Describe objects doesn't implement 
        # getattr(d, "shapeFieldName", None) correctly
        if hasattr(d, "shapeFieldName"):
            return d.shapeFieldName
        else: 
            return None

    def is_versioned(self) -> bool:
        return self.describe().isVersioned

    
    def assert_versioning_aligns(self) -> bool:

        if self.dataset_name:
            fc_isversioned = self.is_versioned()

            dataset_isversioned = arcpy.Describe(str(self.dataset_path)).isVersioned

            if not fc_isversioned == dataset_isversioned:
                raise EnvironmentError("Versioning misalignment in FC / Dataset : " \
                                        f"isVersioned({self.name})={fc_isversioned} != isVersioned({self.dataset_name})={dataset_isversioned}"
                                       )
            
        return True
        
    def spatial_reference(self) -> arcpy.SpatialReference|None:
        d =  self.describe()

        if hasattr(d, "spatialReference"):
            return d.spatialReference
        else: 
            return None


    def crs(self) -> int|None:

        sr = self.spatial_reference()

        if sr is not None:
            sr = int(sr.factoryCode)

        return sr

    ### Fields 
    def fields(self) -> list:
        self.assert_exists()
        return arcpy.ListFields(str(self.path))

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields()]

    def field_types(self) -> dict[str, str]:
        return {f.name: f.type for f in self.fields()}

    def row_count(self):
        self.assert_exists()
        return ap_row_count(self.path)

    def as_dataframe(self,
                     *,
                     field_names: list[str] | None = None,
                     where_clause: str | None = None,
                     df_type: Literal["pandas", "polars", "pyarrow"] = "pandas"
                     ) -> "pd.DataFrame|pl.DataFrame|pa.Table":
        
        """Fetch this table's attributes as a DataFrame (non-spatial).

        Args:
            field_names: Subset of fields to fetch. Fetches all fields if ``None``.
            where_clause: SQL where clause to filter rows. No filter if ``None``.
            df_type: df type to return, one of ``"pandas"``, ``"polars"``, or ``"pyarrow"``.

        Raises:
            ValueError: One or more :attr:`field_names` are not present in the table.
            ImportError: :attr:`df_type` is ``"polars"`` but polars is not installed.
            TypeError: :attr:`df_type` is not one of the allowed values.

        Returns:
            A DataFrame of the requested type containing the table's attributes.
        """

        self.assert_exists()

        return tbl_to_df(tbl = self,
                        field_names=field_names,
                        where_clause=where_clause, 
                        df_type=df_type)

        
    def as_geodataframe(self,
                    *,
                    field_names: list[str] | None = None,
                    where_clause: str | None = None,
                    allow_null_geometries: bool = False,
                    fallback_crs: CRSLike | None = None,
                    to_crs: CRSLike | None = None,
                    ) -> gpd.GeoDataFrame:
        
        """Fetch this feature class as a GeoDataFrame with shapely geometries.

        Args:
            field_names: Subset of fields to fetch. Fetches all fields if ``None``.
                The geometry column is always included.
            where_clause: SQL where clause to filter rows. No filter if ``None``.
            allow_null_geometries: If ``True``, warn on null/invalid geometries instead
                of raising.
            fallback_crs: CRS to use when the feature class spatial reference
                has no resolvable factory code (e.g. an improperly defined CRS).
                Accepts anything geopandas does: EPSG code, ``"EPSG:2227"``, WKT, or ``pyproj.CRS``.
            to_crs: If provided, reproject the result to this CRS.

        Raises:
            ValueError: One or more :attr:`field_names` are not present in the feature class.

        Returns:
            A GeoDataFrame with the feature class attributes and shapely geometries.
        """

        self.assert_exists()

        return fc_to_gdf(fc = self,
                         field_names=field_names,
                         where_clause=where_clause,
                         allow_null_geometries=allow_null_geometries,
                         shape_col=self.shape_field_name(),
                         fallback_crs=fallback_crs,
                         to_crs=to_crs
                         )
        
    def __repr__(self):
        parts = [f"name='{self.name}'"]

        if self.dataset_name:
            parts.append(f"dataset='{self.dataset_name}'")

        parts.append(f"conn='{self.db_conn}'")
        parts.append(f"prefix='{self.db_prefix}'")

        return f"{self.__class__.__name__}({', '.join(parts)})"


    def __str__(self):
        return str(self.path)
    

    def __fspath__ (self):
        return str(self.path)

    def __post_init__(self):
        if self.db_prefix:
            if not self.db_prefix.endswith("."):
                self.db_prefix += "."
        if self.db_conn == Path("memory"):
            self.db_conn = "memory"


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
    else:
        raise ValueError(f"Could not build SpatialReference from CRS '{gdf.crs}' with no EPSG. Non-EPSG references are currently not supported.")



WriteMode = Literal["new", "overwrite", "append"]
_WRITE_MODES = ("new", "overwrite", "append")


def _check_mode(mode: str) -> None:
    if mode not in _WRITE_MODES:
        raise ValueError(f"'mode' must be one of {list(_WRITE_MODES)} (got '{mode}')")


def _dropped_fields(source_p: str, target_p: str) -> list[str]:
    """Writable fields in ``source_p`` with no same-named field in ``target_p``.

    ``Append(schema_type="NO_TEST")``` skips these.
    """
    target_names = {f.name.lower() for f in arcpy.ListFields(target_p)}
    return [f.name for f in arcpy.ListFields(source_p)
            if f.type not in ("OID", "Geometry")
            and f.editable
            and f.name.lower() not in target_names]

def _shapetypes_align(temp_p: str, target_p: str):

    src_shapetype =  ap_shapetype(temp_p) 
    dest_shapetype = ap_shapetype(target_p)

    if src_shapetype != dest_shapetype:
        raise ValueError(f"Can not append features with shapeType '{src_shapetype}' into "
                         f"Feature Class with shapeType '{dest_shapetype}'")

    src_z, src_m = ap_has_zm(temp_p)
    dest_z, dest_m = ap_has_zm(target_p)

    if src_z != dest_z:
        raise ValueError(f"Can not append features with hasZ={src_z} into Feature Class with hasZ={dest_z}")
    if src_m != dest_m:
        raise ValueError(f"Can not append features with hasM={src_m} into Feature Class with hasM={dest_m}")


def _append_via_memory(df_san: pa.Table,
                       target_p: str,
                       copy_tool: Callable[[pa.Table, str], object]) -> None:
    """Stage :attr:`df_san` in memory, then Append it to the existing :attr:`target_p`.

    :attr:`df_san` must already have been validated against the target with
    ``_sanitize_arr_tbl(template=target_p)``. Nothing is deleted from the target.
    """
    temp_p = memory_str()
    try:
        copy_tool(df_san, temp_p)

        staged_count = ap_row_count(temp_p)
        if staged_count != df_san.num_rows:
            raise ValueError(f"Staging dropped rows: {df_san.num_rows} in the table, {staged_count} staged")

        dropped = _dropped_fields(temp_p, target_p)
        if dropped:
            warnings.warn(f"{len(dropped)} column(s) are not in {target_p} and will not be written: "
                          f"{', '.join(dropped)}", stacklevel=3)

        _shapetypes_align(temp_p, target_p)

        arcpy.management.Append(
            inputs=temp_p,
            target=target_p,
            schema_type="NO_TEST",
        )
        arcpy_warnings = arcpy.GetMessages(1)
        if arcpy_warnings:
            warnings.warn(arcpy_warnings, stacklevel=3)
    finally:
        if arcpy.Exists(temp_p):
            arcpy.management.Delete(temp_p)


def _resolve_target(dest: str|Path|CatalogPath|None, mode: WriteMode) -> str:
    """Path string to write to, after checking that :attr:`dest` suits :attr:`mode`.

    ``None`` is a new in-memory path. ``"append"`` needs an existing target and ``"new"`` needs
    a free one. A ``CatalogPath`` that doesn't exist yet needs an existing dataset or workspace.
    """
    if dest is None:
        if mode == "append":
            raise FileNotFoundError("Cannot use mode='append' without a target to append to")
        return memory_str()

    path = str(dest.path) if isinstance(dest, CatalogPath) else str(dest)
    exists = arcpy.Exists(path)

    if mode == "append" and not exists:
        raise FileNotFoundError(f"Cannot use mode='append': target does not exist: {path}")
    if mode == "new" and exists:
        raise FileExistsError(f"Cannot use mode='new': target already exists: {path}")
    if not exists and isinstance(dest, CatalogPath):
        dest.assert_parent_exists()

    return path


def df_to_tbl(df: "pl.DataFrame|pd.DataFrame",
              tbl: str|Path|CatalogPath|None = None,
              *,
              mode: WriteMode = "new",
             ) -> str|Path|CatalogPath:

    """Write a DataFrame to an ArcGIS table.

    Args:
        df: A pandas or Polars DataFrame.
        tbl: Destination path. Accepts a string, ``Path``, ``CatalogPath``, or
            ``None`` to write to a new in-memory workspace path.
        mode: ``"new"`` creates the table (errors if it exists), ``"overwrite"``
            drops and recreates it, ``"append"`` adds rows to an existing one.
            Rows are never deleted from an existing table: replacing them atomically is up to the caller.

    Raises:
        TypeError: :attr:`df` is not a supported DataFrame type.
        ValueError: :attr:`mode` is not one of the allowed values, or, for ``"append"``,
            the data would be lost or changed by the existing table's schema.
        FileExistsError: :attr:`mode` is ``"new"`` but :attr:`tbl` already exists.
        FileNotFoundError: :attr:`mode` is ``"append"`` but :attr:`tbl` does not exist, or
            :attr:`tbl` is a new ``CatalogPath`` whose dataset or workspace does not exist.

    Warns:
        UserWarning: Columns in :attr:`df` with no matching field in an existing
            :attr:`tbl` are not written (``"append"``).

    Returns:
        :attr:`tbl` exactly as passed, or the new in-memory path string when
        :attr:`tbl` is ``None``.
    """
    _check_mode(mode)

    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):  # type: ignore[union-attr]
        df_arrow = df.to_arrow()
    # Don't allow passing a geodataframe
    elif not isinstance(df, gpd.GeoDataFrame) and isinstance(df, pd.DataFrame):
        df_arrow = pa.Table.from_pandas(df, preserve_index=False)
    else:
        if isinstance(df, gpd.GeoDataFrame):
            raise TypeError("Passing a GeoDataFrame is not supported. Drop the geometry first (e.g. df.drop(columns=df.geometry.name)), or use"
                            "gdf_to_fc() to write to a Feature Class.")
        else:
            raise TypeError("df must be a valid non-spatial pandas or polars dataframe.")

    tbl_p = _resolve_target(tbl, mode)

    # Appending validates the data against the existing table before anything is staged or written
    dest_type = "FGDB" if ".gdb" in str(tbl_p) else "OTHER"
    df_san = _sanitize_arr_tbl(table=df_arrow,
                               template=tbl_p if mode == "append" else None,
                               dest_type=dest_type)

    if mode == "append":
        _append_via_memory(df_san, tbl_p,
                           copy_tool=lambda src, dst: arcpy.management.CopyRows(in_rows=src, out_table=dst))
    else:
        with arcpy.EnvManager(overwriteOutput=(mode == "overwrite")):
            arcpy.management.CopyRows(in_rows=df_san, out_table=tbl_p)

    return tbl if tbl is not None else tbl_p

def gdf_geomtypes_consistent(gdf: gpd.GeoDataFrame):
    """Ensure all geometries are the same type (borrowed from gpd.overlay())
    """

    poly_check = gdf.geom_type.isin(POLYGON_GEOM_TYPES).any()
    lines_check = gdf.geom_type.isin(LINE_GEOM_TYPES).any()
    points_check = gdf.geom_type.isin(POINT_GEOM_TYPES).any()
    if sum([poly_check, lines_check, points_check]) > 1:
        raise ValueError(f"GeoDataFrame has mixed geometry types, arcpy will drop rows with one or the other")

def gdf_geoms_nonnull(gdf: gpd.GeoDataFrame):
    """Ensure all geometries are nonnull
    """

    if gdf.geometry.isna().any():
        raise ValueError(f"GeoDataFrame has rows with Null geometries, arcpy will drop these rows")


def gdf_to_fc(gdf: "gpd.GeoDataFrame",
             fc: str|Path|CatalogPath|None = None,
             *,
             mode: WriteMode = "new",
             ) -> str|Path|CatalogPath:

    """Write a GeoDataFrame to an ArcGIS feature class.

    Args:
        gdf: A GeoPandas GeoDataFrame
        fc: Destination path. Accepts a string, ``Path``, ``CatalogPath``, or
            ``None`` to write to a new in-memory workspace path.
        mode: ``"new"`` creates the feature class (errors if it exists), ``"overwrite"``
            drops and recreates it, ``"append"`` adds rows to an existing one.
            Rows are never deleted from an existing feature class: replacing them atomically is up to the caller.

    Raises:
        TypeError: :attr:`gdf` is not a GeoDataFrame.
        ValueError: :attr:`mode` is not one of the allowed values, or, for ``"append"``,
            the data would be lost or changed by the existing feature class.
        FileExistsError: :attr:`mode` is ``"new"`` but :attr:`fc` already exists.
        FileNotFoundError: :attr:`mode` is ``"append"`` but :attr:`fc` does not exist, or
            :attr:`fc` is a new ``CatalogPath`` whose dataset or workspace does not exist.

    Warns:
        UserWarning: Columns in :attr:`gdf` with no matching field in an existing
            :attr:`fc` are not written (``"append"``).

    Returns:
        :attr:`fc` exactly as passed, or the new in-memory path string when
        :attr:`fc` is ``None``.
    """
    _check_mode(mode)

    sr = None
    shape_col = None

    if isinstance(gdf, gpd.GeoDataFrame):
        gdf_geomtypes_consistent(gdf)
        gdf_geoms_nonnull(gdf)
        shape_col = str(gdf.geometry.name)
        _raw = gdf.to_arrow(geometry_encoding="WKB", index = False)
        # Newer geopandas returns a wrapper, not a pa.Table.  
        if not isinstance(_raw, pa.Table):
            _raw = pa.RecordBatchReader.from_stream(_raw).read_all()
        df_arrow = _raw
        sr = _sr_from_gdf(gdf)
    else:
        raise TypeError("gdf must be a geopandas GeoDataFrame. Use df_to_tbl() for non-spatial dataframes.")

    fc_p = _resolve_target(fc, mode)

    dest_type = "FGDB" if ".gdb" in str(fc_p) else "OTHER"

    # Appending validates the data against the existing feature class before anything is staged or written
    df_san = _sanitize_arr_tbl(table=df_arrow, shape_col=shape_col, sr=sr,
                               template = (fc_p if mode == "append" else None),
                               dest_type = dest_type
                               
                               )

    if mode == "append":
        _append_via_memory(df_san, fc_p,
                           copy_tool=lambda src, dst: arcpy.management.CopyFeatures(in_features=src, out_feature_class=dst))
    else:
        with arcpy.EnvManager(overwriteOutput=(mode == "overwrite")):
            arcpy.management.CopyFeatures(in_features=df_san, out_feature_class=fc_p)

    return fc if fc is not None else fc_p

def _resolve_tbl(tbl: str|Path|CatalogPath, 
                 check_dtype: str = "FeatureClass") -> tuple[list[str], str]:
    """
    Get info needed to validate a tbl. Helper for `tbl_to_df()` and `fc_to_df()`
    """

    def _checkdtype(data_type):
        if (data_type != check_dtype) and check_dtype != "Table":
            raise ValueError(f"The table is of the wrong dataType (expected '{check_dtype}', got '{data_type}'). Use tbl_to_df() to fetch non-spatial tables.")

    if isinstance(tbl, CatalogPath):

        tbl.assert_exists()
        data_type = tbl.describe().dataType
        _checkdtype(data_type)
        available_fields = tbl.field_names()
        tbl_p = str(tbl)
        
    else:
        tbl_p = str(tbl)
        exists = arcpy.Exists(tbl_p)
        if not exists:
            raise FileNotFoundError(f"{check_dtype} does not exist: {tbl}")
        data_type = arcpy.Describe(str(tbl_p)).dataType
        _checkdtype(data_type)
        available_fields = [f.name for f in arcpy.ListFields(tbl_p)]
        
    return available_fields, tbl_p


def tbl_to_df(tbl: str|Path|CatalogPath,
             *,
             field_names: list[str] | None = None,
             where_clause: str | None = None,
             df_type: Literal["pandas", "polars", "pyarrow"] = "pandas",
            ) -> "pd.DataFrame|pl.DataFrame|pa.Table":
    
    """Fetch a table or the non-spatial attributes of a feature class as a DataFrame.

    Wraps ``arcpy.da.TableToArrowTable()`` with field validation and existence
    checks. For spatial data with usable geometries use ``fc_to_gdf()`` instead.

    Args:
        tbl: The table or feature class to read.
        field_names: Subset of fields to fetch. Fetches all fields if ``None``.
        where_clause: SQL where clause to filter rows. No filter if ``None``.
        df_type: Output type — ``"pandas"``, ``"polars"``, or ``"pyarrow"``.

    Raises:
        FileNotFoundError: :attr:`tbl` does not exist.
        ValueError: One or more :attr:`field_names` are not present in the table.
        ImportError: :attr:`df_type` is ``"polars"`` but polars is not installed.
        TypeError: :attr:`df_type` is not one of the allowed values.

    Returns:
        A DataFrame of the requested type containing the table's attributes.
    """


    available_fields, tbl_p = _resolve_tbl(tbl, check_dtype = "Table")


    if field_names is not None:
        bad_fields = [f for f in field_names if f not in available_fields]
        if bad_fields:  
            raise ValueError(f"The requested fields do not exist: {', '.join(bad_fields)}")
        

    arr = arcpy.da.TableToArrowTable(in_table = tbl_p, 
                                     field_names=field_names, 
                                     where_clause=where_clause
                                     )
    if df_type == "pandas":
        return arr.to_pandas(types_mapper=pd.ArrowDtype)
    elif df_type == "polars":
        if not _POLARS_AVAILABLE:
            raise ImportError("polars is not installed.")
        return pl.from_arrow(arr)  # type: ignore[union-attr]
    elif df_type == "pyarrow":
        return arr
    else:
        raise TypeError(f"df_type '{df_type}' is invalid: Allowed options are ['pandas', 'polars', 'pyarrow']")
    

def _load_gdf_from_arr(arr: pa.Table,
                       sr: arcpy.SpatialReference,
                       allow_null_geometries: bool = False,
                       shape_col: str = "Shape",
                       fallback_crs: CRSLike | None = None,
                       to_crs: CRSLike | None = None
                       ):
    
    """Build a GeoDataFrame from an Arrow table produced by ``arcpy.da.TableToArrowTable()``.

    Args:
        arr: Arrow table with a WKB geometry column.
        sr: Spatial reference of the source feature class, used to set the CRS.
        allow_null_geometries: If ``True``, warn on null/invalid geometries instead
            of raising. Applies to both missing values and shapely parse failures.
        shape_col: Name of the WKB geometry column in :attr:`arr`.
        fallback_crs: CRS to use when :attr:`sr`.factoryCode is 0 (unresolvable).
        to_crs: If provided, reproject the GeoDataFrame to this CRS after
            construction.

    Raises:
        ValueError: Null geometries are present and :attr:`allow_null_geometries` is ``False``.
        AttributeError: :attr:`sr`.factoryCode is 0 and no :attr:`fallback_crs` was given.

    Returns:
        A GeoDataFrame with shapely geometries and CRS set from :attr:`sr`.
    """
    
    wkb = arr[shape_col]

    handle_invalid = "warn" if allow_null_geometries else "raise"
    geometries = from_wkb(wkb, on_invalid= handle_invalid)

    if wkb.null_count:
        msg = f"{wkb.null_count} null geometries found in column '{shape_col}'"
        if allow_null_geometries:
            warnings.warn(msg)
        else:
            raise ValueError(msg)

    factory_code_proj = getattr(sr, "factoryCode", 0)

    if factory_code_proj == 0:
        if fallback_crs is None:
            raise AttributeError(f"You must provide a fallback_crs when the source feature class has factoryCode=0.")
        else:
            factory_code_proj = fallback_crs


    # Attributes keep their Arrow types, consistent with tbl_to_df(); the raw WKB column is dropped
    df = arr.drop_columns([shape_col]).to_pandas(types_mapper=pd.ArrowDtype)
    gdf = gpd.GeoDataFrame(df,
                            geometry=geometries,
                            crs=factory_code_proj
                            )
            
    if to_crs is not None:
        gdf = gdf.to_crs(to_crs)
    
    return gdf


def fc_to_gdf(fc: str|Path|CatalogPath,
             *,
             field_names: list[str] | None = None, 
             where_clause: str | None = None, 
             allow_null_geometries : bool = False,
             shape_col: str | None= None,
             fallback_crs: CRSLike | None = None,
             to_crs: CRSLike | None = None
             ):
    """Fetch a feature class as a GeoDataFrame with shapely geometries.

    Args:
        fc: The feature class to read.
        field_names: Subset of fields to fetch. Fetches all fields if ``None``.
            The geometry column is always included.
        where_clause: SQL where clause to filter rows. No filter if ``None``.
        allow_null_geometries: If ``True``, warn on null/invalid geometries instead of raising.
        shape_col: Name of the geometry column. Auto-detected from :attr:`fc` when
            an ``CatalogPath`` is passed; defaults to ``"Shape"`` otherwise.
        fallback_crs: CRS to use when the feature class spatial reference
            has no resolvable factory code (e.g. an improperly defined CRS).
        to_crs: Optionally return a gdf reprojected to this crs

    Raises:
        FileNotFoundError: :attr:`fc` does not exist.
        ValueError: One or more :attr:`field_names` are not present in the feature class.

    Returns:
        A GeoDataFrame with the feature class attributes and shapely geometries.
    """

    available_fields, fc_p  = _resolve_tbl(fc, check_dtype="FeatureClass") 

    if field_names is not None:
        bad_fields = [f for f in field_names if f not in available_fields]
        if bad_fields:  
            raise ValueError(f"The requested fields do not exist: {', '.join(bad_fields)}")

    shape_col = shape_col or (fc.shape_field_name() if isinstance(fc, CatalogPath) else "Shape")


    if isinstance(fc, CatalogPath):
        sr = fc.spatial_reference()
    else: 
        d = arcpy.Describe(fc_p)
        sr = getattr(d, "spatialReference")

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
