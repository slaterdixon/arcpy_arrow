# arcpy_arrow

**arcpy_arrow** is a lightweight library for roundtripping ArcGIS data and Python DataFrames, backed by Esri's native Apache Arrow support. 

Leave the world of Field Mappings, Join/Calc, and UpdateCursors behind, without sacrificing compatiblity with your geodatabases.

*Use cases include*:
- Interactively query, join, and transform your data as `pandas`, `geopandas`, and `polars` dataframes
- Quickly dump a Feature Class or Table to DuckDB, parquet, or Excel
- Write robust, Pythonic scripts using `CatalogPath` objects


|         | DB → DF         | DF → DB       |
| ------- | --------------- | ------------- |
| *Tabular* | `tbl_to_df()` | `df_to_tbl()` |
| *Spatial* | `fc_to_gdf()`   | `gdf_to_fc()` |

## Requirements 
* ArcGIS Pro 3.5+ (Python ≥ 3.11)
* A conda environment [cloned](https://doc.esri.com/en/arcgis-pro/latest/arcpy/get-started/clone-an-environment.html) from `arcgispro-py3`

## Installation
```{cmd}
conda activate <your env> 
conda install shapely geopandas
conda install -c conda-forge arcpy_arrow
```

To work with `polars` DataFrames, also install:
```{cmd}
conda install polars
```
## Quick Start
### Interop
To get started, fetch a Feature Class as a GeoDataFrame:

```{python}
from arcpy_arrow.interop import fc_to_gdf

states_url = ("https://services.arcgis.com/ue9rwulIoeLEI9bj/" 
            "arcgis/rest/services/US_StateBoundaries/FeatureServer/0")

TARGET_CRS = 4269
WHERE_C = "STATE_ABBR NOT IN ('HI', 'AK', 'PR', 'GU')"

gdf = fc_to_gdf(fc = states_url, 
                where_clause=WHERE_C,
                to_crs = TARGET_CRS,
            )

gdf.plot(column = "POP")
```
And write results to a Enterprise Geodatabase table:

```{python}
from arcpy_arrow import gdf_to_fc

gdf_to_fc(
    gdf = gdf,
    dest = Path(r"C:\connections\db.sde\Continental_States"),
    mode = "new"
)
```

`fc_to_gdf` allows a number of modes, `"new", "overwrite", "append", and "delete_append"` depending on whether the destination already exists. When appending to an existing table, schema violations in your DataFrame will raise loud errors to prevent silent data loss/truncation. 

### CatalogPath
`CatalogPath` is an alternative to the standard string or `pathlib` objects used to resolve Feature Classes and Tables via .sde connection files. It makes path construction easier and exposes several helpful attributes and methods that return objects as native Python types, not fiddly `arcpy` objects. 

For example, `CatalogPath.field_names()` returns a native list of string names. 

```{python}
from arcpy_arrow import CatalogPath
from pathlib import Path

tbl = CatalogPath(
    name = "wMain",
    db_conn = r"C:\connections\db.sde",
    dataset_name="WaterDistribution",
    db_prefix="sde.gis."
)

main _names = tbl.field_names()

main _names
>>> ['OBJECTID',
>>>  'MAINID',
>>>  'DIAMETER',
>>>  'MATERIAL',
>>>  'TRANSMISSION',
>>>  'SUBDIVISION',
>>>    ...]
```

```{python}
main _names == [f.name for f in arcpy.ListFields(str(tbl))]
>>> True
```

The `CatalogPath` helpers are available individually in the `helpers` module.

```{python}
from arcpy_arrow.helpers import ap_field_names

ap_field_names(Path(r"C:\connections\db.sde\sde.gis.WaterDistribution\wMain"))

>>> ['OBJECTID',
>>>  'MAINID',
>>>  'DIAMETER',
>>>  'MATERIAL',
>>>  'TRANSMISSION',
>>>  'SUBDIVISION',
>>>    ...]
```

## Examples
For more help, see the example notebooks in [examples/](examples/):

* [Basic usage](examples/01_BasicUsage.ipynb)
* [Using CatalogPaths](examples/02_UsingCatalogPaths.ipynb)
* [Benchmark vs. the Spatially Enabled DataFrame](examples/03_BenchmarkVsSEDF.ipynb)

## Development
From a cloned `arcgispro-py3` environment:
```{cmd}
conda install shapely geopandas polars pytest
pip install --no-deps -e .
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) file for details.


