import arcpy
from pathlib import Path

def ap_workspace(p: str|Path) -> Path|str:
    cp = arcpy.Describe(str(p)).workspace.catalogPath
    return cp if cp.startswith("memory") else Path(cp) 

def ap_exists(p: str|Path) -> bool:
    return arcpy.Exists(str(p))

def ap_fields(p: str|Path) -> list[arcpy.Field]:
    return [f for f in arcpy.ListFields(str(p))]

def ap_field_names(p: str|Path) -> list[str]:
    return [f.name for f in ap_fields(str(p))]

def ap_row_count(p: str|Path) -> int:
    return int(arcpy.GetCount_management(str(p)).getOutput(0))