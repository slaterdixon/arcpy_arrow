"""
.. include:: ./README.md

**arcpy_arrow** is a Python package that provides lightweight translation between ESRI tables/feature classes and standard Python dataframes, powered by the Apache Arrow format. 

**Example workflows:**
- Join and clean Excel exports in `pandas` and write the results straight to a EGDB table, 
- Bulk update feature attributes in seconds, without touching a `UpdateCursor()` or Field Calc
- Buffer, dissolve, and spatially-join data in `geopandas` without dealing with workspaces or intermediate results — then visualize the final feature class in Pro
- Easily audit data and dump archives to disk as Excel, parquet, or feather files

---
"""


_ESRI_DEPS = ["arcpy", "pyarrow", "pandas", "geopandas", "shapely"]
_missing = [pkg for pkg in _ESRI_DEPS if __import__("importlib.util", fromlist=["find_spec"]).find_spec(pkg) is None]
if _missing:
    raise ImportError(
        f"arcpy_arrow requires the following packages that were not found: {_missing}. "
        "These are expected to be present in the ArcGIS Pro conda environment."
    )

from arcpy_arrow.interop import (
    ESRITable,
    df_to_fc,
    fc_to_df,
    fc_to_gdf,
    memory_str,
)

__all__ = [
    "ESRITable",
    "df_to_fc",
    "fc_to_df",
    "fc_to_gdf",
    "memory_str",
]
