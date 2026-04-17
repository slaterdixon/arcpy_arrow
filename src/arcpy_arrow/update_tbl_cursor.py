from __future__ import annotations

"""
Delete and append rows/features
"""


import pathlib
import arcpy
import logging
from shared.esri_interop import arr_from_df, rand_in_memory_string, EGDBTable
import polars as pl
import pandas as pd
import numpy as np
import geopandas as gpd


    
def update_tbl_cursor(
        orig: pd.DataFrame|pl.DataFrame|EGDBTable,
        dest: EGDBTable,
        orig_key_col: str,
        dest_key_col: str,
        cols_to_update: list|dict[str, str],
        check_unique_keys: bool = True,
        logger: logging.Logger | None = None
):
    
    logger = logger or logging.getLogger(__file__)

    if isinstance(orig, EGDBTable):
        orig_df: pd.DataFrame = orig.as_data_frame(df_type = "pandas")  # type: ignore[assignment]
    elif isinstance(orig, pd.DataFrame):
        orig_df: pd.DataFrame = orig
    elif isinstance(orig, pl.DataFrame):
        orig_df: pd.DataFrame = orig.to_pandas()
    else:
        raise ValueError(f"orig must be a DataFrame or EGDBTable, got {type(orig)}")

    if isinstance(cols_to_update, list):
        missing_orig = [c for c in cols_to_update if c not in orig_df.columns]
        missing_dest = [c for c in cols_to_update if c not in dest.fields_names]
    
    elif isinstance(cols_to_update, dict):
        missing_orig = [c for c in [orig_key_col] + list(cols_to_update.keys()) if c not in orig_df.columns]
        missing_dest = [c for c in [dest_key_col] + list(cols_to_update.values()) if c not in dest.fields_names]
    else:
        raise ValueError("cols_to_update must be a list or dict with column names")
    
    if missing_orig:
        raise ValueError(f"The following cols_to_update were not found in orig: {', '.join(missing_orig)}")
    if missing_dest:
        raise ValueError(f"The following cols_to_update were not found in dest: {', '.join(missing_dest)}")
    
    if isinstance(cols_to_update, dict):
        orig_cols = [k for k in cols_to_update.keys() if k != orig_key_col]
        dest_cols = [cols_to_update[k] for k in cols_to_update.keys() if k != orig_key_col]
    else:
        orig_cols = [c for c in cols_to_update if c != orig_key_col]
        dest_cols = orig_cols

    orig_df = (orig_df.set_index(orig_key_col)
                        .replace(r'^\s*$', None, regex=True) # All whitespace strings -> None
                        .replace({np.nan: None}) # Missing entries -> None
                )
    if not check_unique_keys:
        orig_df = orig_df.drop_duplicates()

    logger.info(f"STARTED - update_fc_cursor on {dest.path}")
    logger.info(f"  Source rows: {len(orig_df)}, Dest fields: {dest_cols}")
    logger.info(f"  Key mapping: {orig_key_col} -> {dest_key_col}")

    updated_rows = 0
    skipped_rows = 0
    cursor_rows = dest.row_count()
    log_interval = max(cursor_rows // 4, 1)
    with arcpy.da.Editor(workspace = str(dest.workspace), multiuser_mode = dest.is_versioned):
        with arcpy.da.UpdateCursor(in_table = str(dest.path), field_names= [dest_key_col] + dest_cols) as fc_cursor:
            for row in fc_cursor:
                try:
                    dest_key = row[0]

                    try:
                        orig_row = orig_df.loc[dest_key]
                    except KeyError:
                        skipped_rows += 1
                        continue

                    if isinstance(orig_row, pd.DataFrame):
                        if check_unique_keys:
                            raise KeyError(f"{dest_key_col}={dest_key} is duplicated in the source df")
                        else:
                            skipped_rows += 1
                            continue

                    new_row_ordered = [dest_key] + [getattr(orig_row, c) for c in orig_cols]

                    fc_cursor.updateRow(new_row_ordered)
                    updated_rows += 1

                    if updated_rows % log_interval == 0:
                        logger.info(f"  Updated {updated_rows} / {cursor_rows} rows...")
                except Exception:
                    logger.warning(f"  Failed to update row: {row}", exc_info=True)
                    skipped_rows += 1
                    continue


    logger.info(f"FINISHED - update_fc_cursor: {updated_rows} updated, {skipped_rows} skipped")

    return {"n_total_rows": cursor_rows, "n_updated_rows":  updated_rows, "n_skipped_rows": skipped_rows}
