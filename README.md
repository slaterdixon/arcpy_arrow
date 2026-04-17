**arcpy_arrow** is a Python package that provides lightweight translation between ESRI tables/feature classes and standard Python dataframes, powered by the Apache Arrow format. 

**Example workflows:**
- Join and clean Excel exports in `pandas` and write the results straight to a EGDB table, 
- Bulk update feature attributes in seconds, without touching a `UpdateCursor()` or Field Calc
- Buffer, dissolve, and spatially-join data in `geopandas` without dealing with workspaces or intermediate results — then visualize the final feature class in Pro
- Easily audit data and dump archives to disk as Excel, parquet, or feather files

## Installation
1. Clone your ArcGIS Pro python environment in conda
2. Install the package from conda using `conda install -c ... arcpy_arrow`

## Usage

### Fetch a table as a DataFrame

```

from arcpy_arrow import fc_to_df

EXAMPLE_TABLE = "https://sampleserver6.arcgisonline.com/arcgis/rest/services/NYTimes_Covid19Cases_USCounties/MapServer/5"

covid_cases_sd = fc_to_df(EXAMPLE_TABLE,
         where_clause="state = 'South Dakota'",
         field_names=["date", "cases", "deaths"]
         )


print(covid_cases_sd.head())

>>>                 date  cases  deaths
>>> 0 2020-03-10 04:00:00      5       1
>>> 1 2020-03-11 04:00:00      8       1
>>> 2 2020-03-12 04:00:00      8       1
>>> 3 2020-03-13 04:00:00      9       1
>>> 4 2020-03-14 04:00:00      9       1
```


### Fetch a feature class as a GeoDataFrame


```{python}

from arcpy_arrow import fc_to_gdf

ESRI_CITIES = "https://sampleserver6.arcgisonline.com/arcgis/rest/services/USA/MapServer/0"

sd_cities_gdf = fc_to_gdf(ESRI_CITIES, 
          where_clause = "st = 'SD'",
          shape_col="shape")



print(sd_cities_gdf.head())
>>>     objectid   areaname class  st capital  pop2000  geometry
>>> 0      2963   Aberdeen  city  SD       N    24658   POINT (-98.48129 45.46371)
>>> 1      2964  Brookings  city  SD       N    18504   POINT (-96.78811 44.30625)
>>> 2      2965      Huron  city  SD       N    11893   POINT (-98.21815 44.35914)
>>> 3      2966   Mitchell  city  SD       N    14558   POINT (-98.02628 43.7139)
>>> 4      2967     Pierre  city  SD       Y    13876   POINT (-100.33638 44.36797)
```

### Write a DataFrame to a feature class





### Append rows from a DataFrame into a feature class


## Contributing

1. Clone the repository using git clone
2. Install as a editable pip package using `pip install --no-deps -e .`
3. Create a pull request for your code

## License

[MIT](https://choosealicense.com/licenses/mit/)