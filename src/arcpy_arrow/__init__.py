"""Lightweight translation between ESRI tables/feature classes and Python dataframes via Apache Arrow."""


_ESRI_DEPS = ["arcpy", "pyarrow", "pandas", "geopandas", "shapely"]
_missing = [pkg for pkg in _ESRI_DEPS if __import__("importlib.util", fromlist=["find_spec"]).find_spec(pkg) is None]
if _missing:
    raise ImportError(
        f"arcpy_arrow requires the following packages that were not found: {_missing}. "
        "These are expected to be present in the ArcGIS Pro conda environment."
    )

from arcpy_arrow.interop import (
    CatalogPath,
    df_to_tbl,
    tbl_to_df,
    gdf_to_fc,
    fc_to_gdf,
    memory_str,
)

__all__ = [
    "CatalogPath",
    "df_to_tbl",
    "tbl_to_df",
    "gdf_to_fc",
    "fc_to_gdf",
    "memory_str",
]
