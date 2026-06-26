# arcpy_arrow

**arcpy_arrow** is a package for round-tripping ArcGIS Feature Classes/Tables and Python DataFrames. It gives ArcGIS Pro users access to the speed and ergonomics of modern data analysis workflows. 

using the Apache Arrow format as a seamless translation layer. For more information, see the project [documentation]().

## Quick Start

Install into a [cloned](https://pro.arcgis.com/en/pro-app/3.4/arcpy/get-started/clone-an-environment.htm) ArcGIS Pro conda environment with the following command:

```{python}
conda activate <your env> && conda install -c ... arcpy_arrow
```

Then, fetch a feature class as a GeoDataFrame:

```{python}
from arcpy_arrow.interop import fc_to_gdf

states_url = ("https://services.arcgis.com/ue9rwulIoeLEI9bj/" 
            "arcgis/rest/services/US_StateBoundaries/FeatureServer/0")

where_c = "STATE_ABBR NOT IN ('HI', 'AK', 'PR', 'GU')"

fc_to_gdf(fc = states_url, 
            where_clause=where_c).plot(column = "POP")
```
![alt text](image-2.png)

Full installation instructions are available [here]().

## License & Disclaimer




