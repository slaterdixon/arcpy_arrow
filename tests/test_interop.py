"""
Tests for src/arcpy_arrow/interop.py.

Requires an ArcGIS Pro Python environment with arcpy available.
Integration tests use arcpy's in-memory workspace so no on-disk
geodatabase is needed.
"""

import re
from pathlib import Path

import arcpy
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import geopandas as gpd
from shapely.geometry import Point

from arcpy_arrow.interop import (
    CatalogPath,
    _load_gdf_from_arr,
    _sanitize_arr_tbl,
    df_to_tbl,
    fc_to_gdf,
    fc_to_gdf,
    memory_str,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(**columns) -> pa.Table:
    """Convenience wrapper for pa.table()."""
    return pa.table(columns)


def _wkb_point(x: float = 0.0, y: float = 0.0) -> bytes:
    from shapely import to_wkb
    from shapely.geometry import Point

    return to_wkb(Point(x, y))


def _sr(epsg: int = 4326) -> arcpy.SpatialReference:
    return arcpy.SpatialReference(epsg)


# ===========================================================================
# memory_str
# ===========================================================================


class TestMemoryStr:
    def test_has_memory_prefix(self):
        assert memory_str().startswith("memory/")

    def test_suffix_is_fc_plus_32_char_hex(self):
        suffix = memory_str().split("/", 1)[1]
        assert re.fullmatch(r"fc_[0-9a-f]{32}", suffix), f"Unexpected format: {suffix!r}"

    def test_each_call_is_unique(self):
        for _ in range(10):
            assert memory_str() != memory_str()


# ===========================================================================
# _sanitize_arr_tbl
# ===========================================================================


class TestSanitizeArrTbl:
    # --- input validation ---------------------------------------------------

    def test_raises_type_error_for_non_table(self):
        with pytest.raises(TypeError, match="pyarrow table"):
            _sanitize_arr_tbl("not a table")

    def test_raises_value_error_shape_col_without_sr(self):
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.binary()))
        with pytest.raises(ValueError, match="no spatial reference"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=None)

    def test_raises_type_error_for_invalid_sr(self):
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.binary()))
        with pytest.raises(TypeError, match="Invalid spatial reference"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr="not_a_spatial_reference")

    def test_raises_type_error_shape_col_non_binary(self):
        tbl = _table(Shape=pa.array(["wkt_data"], type=pa.string()))
        with pytest.raises(TypeError, match="wrong type"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())

    def test_raises_runtime_error_shape_col_name_not_in_table(self):
        """shape_col is specified but no column by that name exists."""
        tbl = _table(value=[1, 2])
        with pytest.raises(RuntimeError):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())

    # --- OBJECTID field handling --------------------------------------------

    def test_objectid_cast_to_int64(self):
        tbl = _table(
            OBJECTID=pa.array([1, 2], type=pa.int32()),
            Shape=pa.array([b"\x01", b"\x01"], type=pa.binary()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("OBJECTID").type == pa.int64()

    def test_objectid_gets_esri_metadata(self):
        tbl = _table(
            OBJECTID=pa.array([1], type=pa.int32()),
            Shape=pa.array([b"\x01"], type=pa.binary()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        meta = result.schema.field("OBJECTID").metadata
        assert meta.get(b"esri.oid") == b"esri.int64"

    # --- shape column handling ----------------------------------------------

    def test_shape_col_type_is_binary(self):
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.binary()))
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("Shape").type == pa.binary()

    def test_shape_col_gets_encoding_metadata(self):
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.binary()))
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("Shape").metadata.get(b"esri.encoding") == b"WKB"

    def test_shape_col_gets_sr_wkt_metadata(self):
        sr = _sr(4326)
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.binary()))
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=sr)
        meta = result.schema.field("Shape").metadata
        assert b"esri.sr_wkt" in meta
        assert len(meta[b"esri.sr_wkt"]) > 0

    def test_large_binary_shape_col_accepted(self):
        tbl = _table(Shape=pa.array([b"\x01"], type=pa.large_binary()))
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("Shape").type == pa.binary()

    # --- string column handling ---------------------------------------------

    def test_large_string_downcast_to_string(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            label=pa.array(["foo"], type=pa.large_utf8()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("label").type == pa.string()

    def test_regular_string_unchanged(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            label=pa.array(["foo"], type=pa.utf8()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("label").type == pa.string()

    # --- bool → int32 -------------------------------------------------------

    def test_bool_cast_to_int32(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            flag=pa.array([True], type=pa.bool_()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("flag").type == pa.int32()

    def test_bool_values_preserved_after_cast(self):
        tbl = _table(
            Shape=pa.array([b"\x01", b"\x01"], type=pa.binary()),
            flag=pa.array([True, False], type=pa.bool_()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.column("flag").to_pylist() == [1, 0]

    # --- timestamp normalisation to ms --------------------------------------

    @pytest.mark.parametrize("ts_type", [
        pa.timestamp("ns"),
        pa.timestamp("us"),
        pa.timestamp("ms"),
        pa.timestamp("us", tz="UTC"),
        pa.timestamp("ns", tz="America/New_York"),
    ])
    def test_timestamp_normalised_to_ms(self, ts_type):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            dt=pa.array([0], type=ts_type),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("dt").type == pa.timestamp("ms")

    def test_timestamp_tz_stripped(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            dt=pa.array([0], type=pa.timestamp("us", tz="UTC")),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("dt").type.tz is None

    # --- dictionary flattening ----------------------------------------------

    @pytest.mark.parametrize("dict_type,expected", [
        (pa.dictionary(pa.int32(), pa.utf8()),       pa.string()),
        (pa.dictionary(pa.int8(),  pa.utf8()),       pa.string()),
        (pa.dictionary(pa.int32(), pa.large_utf8()), pa.string()),
    ])
    def test_dictionary_flattened_to_value_type(self, dict_type, expected):
        # pa.array() has no converter for large_utf8 dictionary types, so
        # build all variants via DictionaryArray.from_arrays for consistency.
        cat = pa.DictionaryArray.from_arrays(
            pa.array([0], type=dict_type.index_type),
            pa.array(["a"], type=dict_type.value_type),
        )
        tbl = pa.table({"Shape": pa.array([b"\x01"], type=pa.binary()), "cat": cat})
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("cat").type == expected

    def test_dictionary_values_preserved_after_flattening(self):
        tbl = _table(
            Shape=pa.array([b"\x01", b"\x01", b"\x01"], type=pa.binary()),
            cat=pa.array(["x", "y", "x"], type=pa.dictionary(pa.int32(), pa.utf8())),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.column("cat").to_pylist() == ["x", "y", "x"]

    # --- unsigned integer upcasting -----------------------------------------

    @pytest.mark.parametrize("uint_type,expected", [
        (pa.uint8(),  pa.int16()),
        (pa.uint16(), pa.int32()),
        (pa.uint32(), pa.int64()),
        (pa.uint64(), pa.int64()),
    ])
    def test_unsigned_int_upcast_to_signed(self, uint_type, expected):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            val=pa.array([1], type=uint_type),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("val").type == expected

    def test_unsigned_int_data_preserved_after_cast(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            val=pa.array([255], type=pa.uint8()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.column("val")[0].as_py() == 255

    # --- other columns pass through -----------------------------------------

    def test_integer_column_type_preserved(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            count=pa.array([42], type=pa.int32()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("count").type == pa.int32()

    def test_float_column_type_preserved(self):
        tbl = _table(
            Shape=pa.array([b"\x01"], type=pa.binary()),
            value=pa.array([3.14], type=pa.float64()),
        )
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.schema.field("value").type == pa.float64()


# ===========================================================================
# _load_gdf_from_arr
# ===========================================================================


class TestLoadGdfFromArr:
    """Tests for _load_gdf_from_arr using real WKB geometries."""

    @pytest.fixture()
    def single_point_table(self):
        wkb = _wkb_point(10.0, 50.0)
        return pa.table({"Shape": pa.array([wkb], type=pa.binary()), "id": [1]})

    @pytest.fixture()
    def table_with_null(self):
        wkb = _wkb_point(10.0, 50.0)
        return pa.table(
            {"Shape": pa.array([wkb, None], type=pa.binary()), "id": [1, 2]}
        )

    # --- fallback_crs -------------------------------------------------------

    def test_raises_when_factory_code_zero_no_fallback(self, single_point_table):
        sr = arcpy.SpatialReference()  # empty — factoryCode is 0
        with pytest.raises(AttributeError, match="source_crs"):
            _load_gdf_from_arr(single_point_table, sr, shape_col="Shape")

    def test_fallback_crs_used_when_factory_code_zero(self, single_point_table):
        sr = arcpy.SpatialReference()
        gdf = _load_gdf_from_arr(
            single_point_table, sr, shape_col="Shape", fallback_crs=4326
        )
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    # --- to_crs reprojection ------------------------------------------------

    def test_to_crs_reprojects(self, single_point_table):
        sr = _sr(4326)
        gdf = _load_gdf_from_arr(
            single_point_table, sr, shape_col="Shape", to_crs=3857
        )
        assert gdf.crs is not None and gdf.crs.to_epsg() == 3857

    def test_no_to_crs_keeps_source_crs(self, single_point_table):
        sr = _sr(4326)
        gdf = _load_gdf_from_arr(single_point_table, sr, shape_col="Shape")
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    # --- null geometry handling ---------------------------------------------

    def test_allow_null_geometries_warns(self, table_with_null):
        sr = _sr(4326)
        with pytest.warns(UserWarning):
            gdf = _load_gdf_from_arr(
                table_with_null, sr, shape_col="Shape", allow_null_geometries=True
            )
        assert len(gdf) == 2

    def test_null_geometry_raises_by_default(self, table_with_null):
        sr = _sr(4326)
        with pytest.raises(ValueError):
            _load_gdf_from_arr(table_with_null, sr, shape_col="Shape")

    # --- output structure ---------------------------------------------------

    def test_result_is_geodataframe(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, _sr(4326), shape_col="Shape")
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_non_shape_columns_preserved(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, _sr(4326), shape_col="Shape")
        assert "id" in gdf.columns


# ===========================================================================
# CatalogPath — pure property tests (no arcpy calls)
# ===========================================================================


class TestCatalogPathPath:
    def test_path_without_dataset(self):
        t = CatalogPath(name="MyTable", workspace=Path("/gdb/mydb.gdb"))
        assert t.path == Path("/gdb/mydb.gdb/MyTable")

    def test_path_with_dataset(self):
        t = CatalogPath(name="MyFC", workspace=Path("/gdb/mydb.gdb"), dataset_name="MyDS")
        assert t.path == Path("/gdb/mydb.gdb/MyDS/MyFC")

    def test_path_with_prefix_and_dataset(self):
        t = CatalogPath(
            name="Roads",
            workspace=Path("/gdb/mydb.gdb"),
            schema_prefix="sde.",
            dataset_name="Transport",
        )
        assert t.path == Path("/gdb/mydb.gdb/sde.Transport/sde.Roads")

    def test_path_with_prefix_no_dataset(self):
        t = CatalogPath(name="Roads", workspace=Path("/gdb/mydb.gdb"), schema_prefix="sde.")
        assert t.path == Path("/gdb/mydb.gdb/sde.Roads")


class TestCatalogPathDunderMethods:
    def test_str_returns_str_of_path(self):
        t = CatalogPath(name="Parcels", workspace=Path("/gdb/mydb.gdb"))
        assert str(t) == str(Path("/gdb/mydb.gdb/Parcels"))

    def test_repr_starts_with_class_name(self):
        t = CatalogPath(name="X", workspace=Path("/gdb/x.gdb"))
        assert repr(t).startswith("CatalogPath(")

    def test_repr_contains_name(self):
        t = CatalogPath(name="Parcels", workspace=Path("/gdb/mydb.gdb"))
        assert "name='Parcels'" in repr(t)

    def test_repr_contains_conn_basename(self):
        t = CatalogPath(name="Parcels", workspace=Path("/gdb/mydb.gdb"))
        assert "conn='mydb.gdb'" in repr(t)

    def test_repr_omits_dataset_when_absent(self):
        t = CatalogPath(name="Parcels", workspace=Path("/gdb/mydb.gdb"))
        assert "dataset" not in repr(t)

    def test_repr_includes_dataset_when_present(self):
        t = CatalogPath(name="Roads", workspace=Path("/gdb/db.gdb"), dataset_name="Transport")
        assert "dataset='Transport'" in repr(t)


# ===========================================================================
# CatalogPath — integration tests (require arcpy + in-memory workspace)
# ===========================================================================


@pytest.fixture(scope="module")
def in_memory_point_fc():
    """Create a simple point feature class in arcpy's in-memory workspace."""
    fc_path = "memory/test_points"
    sr = arcpy.SpatialReference(4326)

    arcpy.management.CreateFeatureclass(
        out_path="memory",
        out_name="test_points",
        geometry_type="POINT",
        spatial_reference=sr,
    )
    arcpy.management.AddField(fc_path, "label", "TEXT", field_length=50)

    with arcpy.da.InsertCursor(fc_path, ["SHAPE@WKT", "label"]) as cur:
        cur.insertRow(["POINT (10 50)", "alpha"])
        cur.insertRow(["POINT (20 60)", "beta"])

    yield fc_path

    arcpy.management.Delete(fc_path)


class TestCatalogPathExists:
    def test_returns_false_for_missing_table(self):
        t = CatalogPath(name="DoesNotExist", workspace=Path("memory"))
        assert not t.exists

    def test_returns_true_for_existing_table(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert t.exists


class TestCatalogPathFields:
    def test_fields_returns_list(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert isinstance(t.fields, list)
        assert len(t.fields) > 0

    def test_field_names_contains_label(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert "label" in t.fields_names

    def test_field_types_is_dict(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert isinstance(t.field_types, dict)

    def test_field_types_label_is_string(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert t.field_types["label"] == "String"


class TestCatalogPathRowCount:
    def test_row_count_matches_inserted_rows(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        assert t.row_count() == 2


# ===========================================================================
# fc_to_gdf
# ===========================================================================


class TestFcToGdf:
    # --- input validation ---------------------------------------------------

    def test_raises_file_not_found_for_missing_fc(self):
        with pytest.raises(FileNotFoundError):
            fc_to_gdf("memory/does_not_exist")

    def test_raises_value_error_for_bad_field_names(self, in_memory_point_fc):
        with pytest.raises(ValueError, match="do not exist"):
            fc_to_gdf(in_memory_point_fc, field_names=["no_such_field"])

    # --- output type --------------------------------------------------------

    def test_returns_geodataframe_from_string_path(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_returns_geodataframe_from_esri_table(self, in_memory_point_fc):
        t = CatalogPath(name="test_points", workspace=Path("memory"))
        gdf = fc_to_gdf(t)
        assert isinstance(gdf, gpd.GeoDataFrame)

    # --- row and column selection -------------------------------------------

    def test_row_count_matches_source(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc)
        assert len(gdf) == 2

    def test_field_names_restricts_columns(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc, field_names=["label"])
        assert "label" in gdf.columns

    def test_where_clause_filters_rows(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc, where_clause="label = 'alpha'")
        assert len(gdf) == 1
        assert gdf.iloc[0]["label"] == "alpha"

    # --- CRS handling -------------------------------------------------------

    def test_crs_assigned_from_source_sr(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc)
        assert gdf.crs is not None and gdf.crs.to_epsg() == 4326

    def test_to_crs_reprojects(self, in_memory_point_fc):
        gdf = fc_to_gdf(in_memory_point_fc, to_crs=3857)
        assert gdf.crs is not None and gdf.crs.to_epsg() == 3857

    # --- string path vs CatalogPath give same result --------------------------

    def test_string_and_esri_table_return_same_rows(self, in_memory_point_fc):
        gdf_str = fc_to_gdf(in_memory_point_fc)
        gdf_tbl = fc_to_gdf(CatalogPath(name="test_points", workspace=Path("memory")))
        assert len(gdf_str) == len(gdf_tbl)
        assert set(gdf_str["label"]) == set(gdf_tbl["label"])


# ===========================================================================
# Round-trip tests — write GeoDataFrame → FC → GeoDataFrame, N times
# ===========================================================================


@pytest.fixture(scope="module")
def source_gdf():
    """GeoDataFrame with known geometries and attributes, EPSG:4326."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "label": ["alpha", "beta", "gamma"]},
        geometry=[Point(10.0, 50.0), Point(20.0, 60.0), Point(-5.0, 45.0)],
        crs="EPSG:4326",
    )


def _geoms_equal(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> None:
    """Assert all geometries are exactly equal after aligning on 'id'."""
    a = gdf_a.sort_values("id").geometry.reset_index(drop=True)
    b = gdf_b.sort_values("id").geometry.reset_index(drop=True)
    assert len(a) == len(b), f"Row count mismatch: {len(a)} vs {len(b)}"
    for i, (ga, gb) in enumerate(zip(a, b)):
        assert ga.equals_exact(gb, tolerance=1e-9), f"Geometry mismatch at id={i+1}: {ga} != {gb}"


def _write_and_read(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Write gdf to an in-memory FC and read it back as a GeoDataFrame."""
    fc_path = memory_str()
    df_to_tbl(gdf, fc_path)
    result = fc_to_gdf(fc_path)
    arcpy.management.Delete(fc_path)
    return result


# ===========================================================================
# df_to_tbl — non-spatial writes (pandas / polars)
# ===========================================================================


class TestDfToFcTabular:
    def test_pandas_write_row_count(self):
        df = pd.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "c"]})
        fc_path = memory_str()
        try:
            df_to_tbl(df, fc_path)
            result = fc_to_gdf(fc_path)
            assert len(result) == 3
        finally:
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)

    def test_pandas_write_values_preserved(self):
        df = pd.DataFrame({"id": [10, 20], "label": ["x", "y"]})
        fc_path = memory_str()
        try:
            df_to_tbl(df, fc_path)
            result = fc_to_gdf(fc_path)
            assert set(result["label"]) == {"x", "y"}
        finally:
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)

    def test_polars_write_row_count(self):
        df = pl.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "c"]})
        fc_path = memory_str()
        try:
            df_to_tbl(df, fc_path)
            result = fc_to_gdf(fc_path)
            assert len(result) == 3
        finally:
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)

    def test_polars_write_values_preserved(self):
        df = pl.DataFrame({"id": [10, 20], "label": ["x", "y"]})
        fc_path = memory_str()
        try:
            df_to_tbl(df, fc_path)
            result = fc_to_gdf(fc_path)
            assert set(result["label"]) == {"x", "y"}
        finally:
            if arcpy.Exists(fc_path):
                arcpy.management.Delete(fc_path)

    def test_raises_for_invalid_df_type(self):
        with pytest.raises(TypeError):
            df_to_tbl("not a dataframe", memory_str())  # type: ignore[arg-type]

    def test_fc_to_gdf_polars_raises_import_error_when_unavailable(self, in_memory_point_fc):
        import arcpy_arrow.interop as interop
        original = interop._POLARS_AVAILABLE
        try:
            interop._POLARS_AVAILABLE = False
            with pytest.raises(ImportError, match="polars"):
                fc_to_gdf(in_memory_point_fc, df_type="polars")
        finally:
            interop._POLARS_AVAILABLE = original

    def test_df_to_tbl_polars_raises_type_error_when_unavailable(self):
        import arcpy_arrow.interop as interop
        original = interop._POLARS_AVAILABLE
        try:
            interop._POLARS_AVAILABLE = False
            df = pl.DataFrame({"id": [1, 2], "label": ["a", "b"]})
            with pytest.raises(TypeError, match="polars, geopandas, or pandas"):
                df_to_tbl(df, memory_str())
        finally:
            interop._POLARS_AVAILABLE = original


class TestRoundTrip:
    def test_single_round_trip_preserves_geometry(self, source_gdf):
        result = _write_and_read(source_gdf)
        _geoms_equal(source_gdf, result)

    def test_single_round_trip_preserves_row_count(self, source_gdf):
        result = _write_and_read(source_gdf)
        assert len(result) == len(source_gdf)

    def test_repeated_round_trips_preserve_geometry(self, source_gdf):
        """Geometry must survive three consecutive write→read cycles."""
        current = source_gdf
        for _ in range(3):
            current = _write_and_read(current)
        _geoms_equal(source_gdf, current)
