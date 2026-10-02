"""
Tests for arcpy_arrow.

Requires an ArcGIS Pro Python environment with arcpy available.

Most write tests run twice: once against arcpy's in-memory workspace and once
against a real file geodatabase. The two are not interchangeable. Several
failure modes (rows dropped for out-of-range values, null geometries
discarded) only happen in a file geodatabase.

Written data is checked with arcpy directly (ListFields, SearchCursor,
Describe, CheckGeometry), never with fc_to_gdf/tbl_to_df, so a bug in the
readers cannot hide a bug in the writers.

The contract the write tests enforce: a write either produces exactly the
rows, columns, values, geometries and spatial reference it was given, or it
raises. A silent change is always a failure.
"""

import datetime as dt
import os
import re
import subprocess
import sys
import textwrap
import warnings
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

arcpy = pytest.importorskip("arcpy")

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import shapely
from pyproj import CRS
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

import arcpy_arrow.interop as interop
from arcpy_arrow import (
    CatalogPath,
    df_to_tbl,
    fc_to_gdf,
    gdf_to_fc,
    memory_str,
    tbl_to_df,
)
from arcpy_arrow.interop import _load_gdf_from_arr
from arcpy_arrow.sanitize_arr_tbl import _sanitize_arr_tbl

pl = pytest.importorskip("polars")


# ===========================================================================
# Fixtures and arcpy-side verification helpers
# ===========================================================================


@pytest.fixture(scope="session")
def fgdb(tmp_path_factory) -> str:
    folder = tmp_path_factory.mktemp("fgdb")
    return arcpy.management.CreateFileGDB(str(folder), "test.gdb").getOutput(0)


@pytest.fixture(params=["memory", "fgdb"])
def new_path(request, fgdb):
    """Factory for unique, not-yet-existing destination paths.

    Parametrized so every test using it runs against both workspaces.
    ``new_path.workspace`` names the current one.
    """
    created = []

    def make() -> str:
        if request.param == "memory":
            p = memory_str()
        else:
            p = f"{fgdb}/t_{uuid4().hex[:16]}"
        created.append(p)
        return p

    make.workspace = request.param
    yield make

    for p in created:
        if arcpy.Exists(p):
            arcpy.management.Delete(p)


@pytest.fixture(params=["gdf_to_fc", "df_to_tbl"])
def write_columns(request, new_path):
    """Write a dict of equal-length columns through one of the public writers.

    For ``gdf_to_fc`` the columns get a point geometry per row (EPSG:4326).
    Returns the written path.
    """

    def write(columns: dict, *, index=None, **kwargs) -> str:
        n = len(next(iter(columns.values())))
        p = new_path()
        if request.param == "gdf_to_fc":
            frame = gpd.GeoDataFrame(
                columns,
                geometry=[Point(i, i) for i in range(n)],
                crs=4326,
                index=index,
            )
            return gdf_to_fc(frame, p, **kwargs)
        return df_to_tbl(pd.DataFrame(columns, index=index), p, **kwargs)

    write.writer = request.param
    return write


def _user_fields(p: str) -> dict[str, str]:
    """``{name: arcpy type}`` for fields other than the OID, geometry, and
    system-maintained Shape_Length / Shape_Area fields."""
    d = arcpy.Describe(p)
    system = {getattr(d, "lengthFieldName", ""), getattr(d, "areaFieldName", "")} - {""}
    return {
        f.name: f.type
        for f in arcpy.ListFields(p)
        if f.type not in ("OID", "Geometry") and f.name not in system
    }


def _count(p: str) -> int:
    return int(arcpy.management.GetCount(p).getOutput(0))


def _rows(p: str, cols: list[str]) -> list[tuple]:
    """Rows in OID order, read with a SearchCursor."""
    with arcpy.da.SearchCursor(str(p), ["OID@", *cols]) as cur:
        return [r[1:] for r in sorted(cur, key=lambda r: r[0])]


def _column(p: str, col: str) -> list:
    return [r[0] for r in _rows(p, [col])]


def _geoms(p: str) -> list:
    """Shapely geometries in OID order, read with a SearchCursor (``None`` for null shapes).

    Uses WKT because arcpy's ``SHAPE@WKB`` token drops Z values.
    """
    return [None if wkt is None else shapely.from_wkt(wkt) for (wkt,) in _rows(p, ["SHAPE@WKT"])]


def _xy_tolerance(p: str) -> float:
    return arcpy.Describe(str(p)).spatialReference.XYTolerance


def _geometry_problems(p: str) -> list[tuple]:
    """Problems reported by ``arcpy.management.CheckGeometry``."""
    out = memory_str()
    try:
        arcpy.management.CheckGeometry(str(p), out)
        if not arcpy.Exists(out):
            return []
        with arcpy.da.SearchCursor(out, ["FEATURE_ID", "PROBLEM"]) as cur:
            return list(cur)
    finally:
        if arcpy.Exists(out):
            arcpy.management.Delete(out)


def _assert_geoms_match(expected, actual, tol: float) -> None:
    """Every written geometry is within ``tol`` of its source.

    arcpy snaps coordinates to the spatial reference's XY resolution grid, so
    exact equality is not expected. Polygons come back as MultiPolygons and
    lines as MultiLineStrings, so geometry types are not compared here.
    """
    assert len(expected) == len(actual), f"row count {len(actual)} != {len(expected)}"
    for i, (e, a) in enumerate(zip(expected, actual)):
        assert a is not None, f"row {i}: geometry was written as null"
        dist = shapely.hausdorff_distance(e, a)
        assert dist <= tol, f"row {i}: geometry moved {dist} (tolerance {tol}): {e.wkt} -> {a.wkt}"
        assert shapely.get_coordinate_dimension(a) == shapely.get_coordinate_dimension(e), f"row {i}: Z changed"
        if e.area:
            # Each vertex can move by up to tol, so area can shift by about perimeter * tol.
            assert a.area == pytest.approx(e.area, abs=e.length * tol), f"row {i}: area changed"


def _rows_survive_or_raise(write, path_of, n_expected: int) -> None:
    """Loud failure is acceptable. Losing rows silently is not."""
    try:
        written = write()
    except Exception:
        return
    got = _count(path_of(written))
    assert got == n_expected, f"write succeeded but silently dropped {n_expected - got} of {n_expected} rows"


def _column_survives_or_raise(write, col: str) -> None:
    try:
        written = write()
    except Exception:
        return
    assert col in _user_fields(written), f"write succeeded but silently dropped column {col!r}: {_user_fields(written)}"


# ===========================================================================
# memory_str
# ===========================================================================


class TestMemoryStr:
    def test_format(self):
        assert re.fullmatch(r"memory/fc_[0-9a-f]{32}", memory_str())

    def test_each_call_is_unique(self):
        assert len({memory_str() for _ in range(100)}) == 100


# ===========================================================================
# _sanitize_arr_tbl
# ===========================================================================


def _shape(n: int = 1) -> pa.Array:
    return pa.array([b"\x01"] * n, type=pa.binary())


def _sr(epsg: int = 4326):
    return arcpy.SpatialReference(epsg)


class TestSanitizeArrTbl:
    # --- input validation ---------------------------------------------------

    def test_raises_type_error_for_non_table(self):
        with pytest.raises(TypeError, match="pyarrow table"):
            _sanitize_arr_tbl("not a table")

    def test_raises_value_error_shape_col_without_sr(self):
        with pytest.raises(ValueError, match="no spatial reference"):
            _sanitize_arr_tbl(pa.table({"Shape": _shape()}), shape_col="Shape", sr=None)

    def test_raises_type_error_for_invalid_sr(self):
        with pytest.raises(TypeError, match="Invalid spatial reference"):
            _sanitize_arr_tbl(pa.table({"Shape": _shape()}), shape_col="Shape", sr="4326")

    def test_raises_type_error_shape_col_non_binary(self):
        tbl = pa.table({"Shape": pa.array(["POINT (0 0)"])})
        with pytest.raises(TypeError, match="wrong type"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())

    def test_raises_when_shape_col_missing(self):
        with pytest.raises(RuntimeError):
            _sanitize_arr_tbl(pa.table({"value": [1, 2]}), shape_col="Shape", sr=_sr())

    def test_raises_for_non_shape_binary_column(self):
        tbl = pa.table({"Shape": _shape(), "blob": pa.array([b"x"], type=pa.binary())})
        with pytest.raises(TypeError, match="Binary-like"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())

    # --- OBJECTID -----------------------------------------------------------

    def test_objectid_cast_to_int64_with_esri_metadata(self):
        tbl = pa.table({"OBJECTID": pa.array([1, 2], type=pa.int32()), "Shape": _shape(2)})
        field = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr()).schema.field("OBJECTID")
        assert field.type == pa.int64()
        assert field.metadata.get(b"esri.oid") == b"esri.int64"

    # --- shape column -------------------------------------------------------

    @pytest.mark.parametrize("binary_type", [pa.binary(), pa.large_binary()])
    def test_shape_col_becomes_binary_with_esri_metadata(self, binary_type):
        tbl = pa.table({"Shape": pa.array([b"\x01"], type=binary_type)})
        field = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr(4326)).schema.field("Shape")
        assert field.type == pa.binary()
        assert field.metadata[b"esri.encoding"] == b"WKB"
        assert CRS.from_wkt(field.metadata[b"esri.sr_wkt"].decode()).to_epsg() == 4326

    # --- type normalisation -------------------------------------------------

    @pytest.mark.parametrize("arr,expected", [
        (pa.array(["a"], type=pa.large_utf8()), pa.string()),
        (pa.array(["a"], type=pa.utf8()), pa.string()),
        (pa.array([True]), pa.int32()),
        (pa.array([1], type=pa.uint8()), pa.int16()),
        (pa.array([1], type=pa.uint16()), pa.int32()),
        (pa.array([1], type=pa.uint32()), pa.int64()),
        (pa.array([1], type=pa.uint64()), pa.int64()),
        (pa.array([1], type=pa.int32()), pa.int32()),
        (pa.array([1.5], type=pa.float64()), pa.float64()),
        (pa.array([0], type=pa.timestamp("ns")), pa.timestamp("ms")),
    ], ids=lambda x: str(x.type) if isinstance(x, pa.Array) else str(x))
    def test_type_normalisation(self, arr, expected):
        tbl = pa.table({"Shape": _shape(), "col": arr})
        assert _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr()).schema.field("col").type == expected

    @pytest.mark.parametrize("index_type,value_type", [
        (pa.int32(), pa.utf8()),
        (pa.int8(), pa.utf8()),
        (pa.int32(), pa.large_utf8()),
    ])
    def test_dictionary_flattened_to_string(self, index_type, value_type):
        cat = pa.DictionaryArray.from_arrays(
            pa.array([0, 1, 0], type=index_type), pa.array(["x", "y"], type=value_type)
        )
        result = _sanitize_arr_tbl(pa.table({"Shape": _shape(3), "cat": cat}), shape_col="Shape", sr=_sr())
        assert result.schema.field("cat").type == pa.string()
        assert result.column("cat").to_pylist() == ["x", "y", "x"]

    def test_values_preserved(self):
        tbl = pa.table({
            "Shape": _shape(2),
            "flag": [True, False],
            "u8": pa.array([0, 255], type=pa.uint8()),
            "u32": pa.array([0, 2**32 - 1], type=pa.uint32()),
        })
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.column("flag").to_pylist() == [1, 0]
        assert result.column("u8").to_pylist() == [0, 255]
        assert result.column("u32").to_pylist() == [0, 2**32 - 1]

    def test_cast_failure_names_the_column(self):
        tbl = pa.table({"Shape": _shape(), "too_big": pa.array([2**63 + 5], type=pa.uint64())})
        with pytest.raises(pa.ArrowInvalid, match="Column 'too_big'"):
            _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())

    def test_column_order_and_field_metadata_preserved(self):
        tbl = pa.table({"b": [1], "Shape": _shape(), "a": ["x"]})
        result = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr())
        assert result.column_names == ["b", "Shape", "a"]
        assert result.schema.field("Shape").metadata[b"esri.encoding"] == b"WKB"

    # --- timezone-aware timestamps -> TimestampOffset -----------------------

    @staticmethod
    def _tz_column(values, tz, unit="us"):
        return pa.array(pd.to_datetime(values, format="ISO8601").tz_localize(tz), type=pa.timestamp(unit, tz=tz))

    def test_tz_aware_becomes_tagged_iso_string(self):
        tbl = pa.table({"Shape": _shape(), "t": self._tz_column(["2024-07-01 12:00:00"], "UTC")})
        field = _sanitize_arr_tbl(tbl, shape_col="Shape", sr=_sr()).schema.field("t")
        assert field.type == pa.string()
        assert field.metadata == {b"esri.interop.type": b"esri.timestamp_offset"}

    def test_utc_offset_formatting(self):
        tbl = pa.table({"t": self._tz_column(["2024-07-01 12:00:00"], "UTC")})
        assert _sanitize_arr_tbl(tbl).column("t").to_pylist() == ["2024-07-01T12:00:00.000+00:00"]

    def test_zoned_column_gets_per_row_dst_offsets(self):
        """Los Angeles is UTC-7 in July and UTC-8 in January; each row keeps its own offset."""
        tbl = pa.table({"t": self._tz_column(["2024-07-01 12:00:00", "2024-01-01 08:30:00"], "America/Los_Angeles")})
        assert _sanitize_arr_tbl(tbl).column("t").to_pylist() == [
            "2024-07-01T12:00:00.000-07:00",
            "2024-01-01T08:30:00.000-08:00",
        ]

    def test_half_hour_offset(self):
        tbl = pa.table({"t": self._tz_column(["2024-07-01 12:00:00"], "Asia/Kolkata")})
        assert _sanitize_arr_tbl(tbl).column("t").to_pylist() == ["2024-07-01T12:00:00.000+05:30"]

    def test_tz_aware_nulls_preserved(self):
        arr = pa.array([dt.datetime(2024, 7, 1, tzinfo=dt.timezone.utc), None], type=pa.timestamp("us", tz="UTC"))
        result = _sanitize_arr_tbl(pa.table({"t": arr}))
        assert result.column("t").to_pylist() == ["2024-07-01T00:00:00.000+00:00", None]

    def test_tz_aware_sub_millisecond_is_truncated_not_rejected(self):
        tbl = pa.table({"t": self._tz_column(["2024-07-01 12:00:00.123456789"], "UTC", unit="ns")})
        assert _sanitize_arr_tbl(tbl).column("t").to_pylist() == ["2024-07-01T12:00:00.123+00:00"]

    def test_naive_timestamp_is_not_tagged(self):
        field = _sanitize_arr_tbl(pa.table({"t": pa.array([0], type=pa.timestamp("us"))})).schema.field("t")
        assert field.type == pa.timestamp("ms")
        assert not field.metadata

    def test_only_tz_aware_columns_are_converted(self):
        tbl = pa.table({
            "aware": self._tz_column(["2024-07-01 12:00:00"], "UTC"),
            "naive": pa.array([0], type=pa.timestamp("us")),
            "n": [1],
        })
        schema = _sanitize_arr_tbl(tbl).schema
        assert [f.type for f in schema] == [pa.string(), pa.timestamp("ms"), pa.int64()]

    def test_sub_millisecond_timestamp_truncated_not_rejected(self):
        """Timestamps from ``pd.Timestamp.now()`` or polars carry µs/ns precision.
        arcpy Date fields hold milliseconds, so the extra precision must be
        truncated, not turned into an ArrowInvalid."""
        tbl = pa.table({"t": pa.array([1_704_067_200_123_456], type=pa.timestamp("us"))})
        result = _sanitize_arr_tbl(tbl)
        assert result.column("t").to_pylist() == [dt.datetime(2024, 1, 1, 0, 0, 0, 123000)]


# ===========================================================================
# _load_gdf_from_arr
# ===========================================================================


class TestLoadGdfFromArr:
    @pytest.fixture()
    def single_point_table(self):
        return pa.table({"Shape": pa.array([shapely.to_wkb(Point(10, 50))], type=pa.binary()), "id": [1]})

    @pytest.fixture()
    def table_with_null(self):
        wkb = shapely.to_wkb(Point(10, 50))
        return pa.table({"Shape": pa.array([wkb, None], type=pa.binary()), "id": [1, 2]})

    def test_raises_when_factory_code_zero_no_fallback(self, single_point_table):
        with pytest.raises(AttributeError, match="fallback_crs"):
            _load_gdf_from_arr(single_point_table, arcpy.SpatialReference(), shape_col="Shape")

    def test_fallback_crs_used_when_factory_code_zero(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, arcpy.SpatialReference(), shape_col="Shape", fallback_crs=4326)
        assert gdf.crs.to_epsg() == 4326

    def test_to_crs_reprojects(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, _sr(4326), shape_col="Shape", to_crs=3857)
        assert gdf.crs.to_epsg() == 3857
        assert gdf.geometry.iloc[0].x == pytest.approx(1113194.9, abs=1)

    def test_no_to_crs_keeps_source_crs(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, _sr(4326), shape_col="Shape")
        assert gdf.crs.to_epsg() == 4326

    def test_allow_null_geometries_warns(self, table_with_null):
        with pytest.warns(UserWarning, match="null geometries"):
            gdf = _load_gdf_from_arr(table_with_null, _sr(4326), shape_col="Shape", allow_null_geometries=True)
        assert len(gdf) == 2

    def test_null_geometry_raises_by_default(self, table_with_null):
        with pytest.raises(ValueError, match="null geometries"):
            _load_gdf_from_arr(table_with_null, _sr(4326), shape_col="Shape")

    def test_output_structure(self, single_point_table):
        gdf = _load_gdf_from_arr(single_point_table, _sr(4326), shape_col="Shape")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert list(gdf.columns) == ["id", "geometry"]
        assert gdf.geometry.iloc[0].equals(Point(10, 50))


# ===========================================================================
# CatalogPath: pure path logic (no arcpy calls)
# ===========================================================================


class TestCatalogPathPath:
    def test_path_without_dataset(self):
        t = CatalogPath(name="MyTable", db_conn=Path("/gdb/mydb.gdb"))
        assert t.path == Path("/gdb/mydb.gdb/MyTable")

    def test_path_with_dataset(self):
        t = CatalogPath(name="MyFC", db_conn=Path("/gdb/mydb.gdb"), dataset_name="MyDS")
        assert t.path == Path("/gdb/mydb.gdb/MyDS/MyFC")

    def test_prefix_applied_to_dataset_and_name(self):
        t = CatalogPath(name="Roads", db_conn="/c/db.sde", db_prefix="sde.gis.", dataset_name="Transport")
        assert t.path == Path("/c/db.sde/sde.gis.Transport/sde.gis.Roads")

    def test_prefix_without_dataset(self):
        t = CatalogPath(name="Roads", db_conn="/c/db.sde", db_prefix="sde.gis.")
        assert t.path == Path("/c/db.sde/sde.gis.Roads")

    def test_prefix_gets_trailing_dot(self):
        t = CatalogPath(name="Roads", db_conn="/c/db.sde", db_prefix="sde.gis")
        assert t.db_prefix == "sde.gis."
        assert t.full_name == "sde.gis.Roads"

    def test_dataset_path(self):
        t = CatalogPath(name="Roads", db_conn="/c/db.sde", db_prefix="sde.", dataset_name="Transport")
        assert t.dataset_path == Path("/c/db.sde/sde.Transport")
        assert CatalogPath(name="Roads", db_conn="/c/db.sde").dataset_path is None

    def test_parent_path(self):
        assert CatalogPath(name="R", db_conn="/c/x.gdb").parent_path == Path("/c/x.gdb")
        assert CatalogPath(name="R", db_conn="/c/x.gdb", dataset_name="D").parent_path == Path("/c/x.gdb/D")

    @pytest.mark.parametrize("db_conn", ["memory", Path("memory")])
    def test_memory_workspace(self, db_conn):
        """``Path("memory")`` must behave like the string ``"memory"``."""
        t = CatalogPath(name="scratch", db_conn=db_conn)
        assert t.path == Path("memory/scratch")
        assert t.parent_path is None

    def test_pretty_path(self):
        t = CatalogPath(name="Parcels", db_conn="/c/db.sde", dataset_name="LandRecords")
        assert t.pretty_path == "./db.sde/LandRecords/Parcels"


class TestCatalogPathDunderMethods:
    def test_str_and_fspath_match_path(self):
        t = CatalogPath(name="Parcels", db_conn=Path("/gdb/mydb.gdb"))
        assert str(t) == str(Path("/gdb/mydb.gdb/Parcels"))
        assert Path(t) == t.path

    def test_repr(self):
        t = CatalogPath(name="Roads", db_conn="/gdb/db.gdb", dataset_name="Transport")
        r = repr(t)
        assert r.startswith("CatalogPath(")
        assert "name='Roads'" in r
        assert "dataset='Transport'" in r
        assert "conn='/gdb/db.gdb'" in r

    def test_repr_omits_dataset_when_absent(self):
        assert "dataset" not in repr(CatalogPath(name="Parcels", db_conn="/gdb/mydb.gdb"))


# ===========================================================================
# CatalogPath: against real data
# ===========================================================================


@pytest.fixture(scope="module")
def point_fc():
    """A two-row point feature class in the memory workspace, built with arcpy only."""
    name = f"pts_{uuid4().hex[:8]}"
    fc_path = f"memory/{name}"
    arcpy.management.CreateFeatureclass("memory", name, "POINT", spatial_reference=arcpy.SpatialReference(4326))
    arcpy.management.AddField(fc_path, "label", "TEXT", field_length=50)
    arcpy.management.AddField(fc_path, "value", "DOUBLE")
    with arcpy.da.InsertCursor(fc_path, ["SHAPE@XY", "label", "value"]) as cur:
        cur.insertRow([(10, 50), "alpha", 1.5])
        cur.insertRow([(20, 60), "beta", None])
    yield fc_path
    arcpy.management.Delete(fc_path)


@pytest.fixture(scope="module")
def plain_table():
    name = f"tbl_{uuid4().hex[:8]}"
    p = f"memory/{name}"
    arcpy.management.CreateTable("memory", name)
    arcpy.management.AddField(p, "label", "TEXT", field_length=50)
    with arcpy.da.InsertCursor(p, ["label"]) as cur:
        cur.insertRow(["alpha"])
        cur.insertRow(["beta"])
    yield p
    arcpy.management.Delete(p)


def _catalog_path_for(p: str) -> CatalogPath:
    return CatalogPath(name=p.split("/", 1)[1], db_conn="memory")


class TestCatalogPathOnData:
    def test_exists(self, point_fc):
        assert _catalog_path_for(point_fc).exists() is True
        assert CatalogPath(name="does_not_exist", db_conn="memory").exists() is False

    def test_assert_exists_raises_for_missing(self):
        with pytest.raises(FileNotFoundError):
            CatalogPath(name="does_not_exist", db_conn="memory").assert_exists()

    def test_field_accessors(self, point_fc):
        t = _catalog_path_for(point_fc)
        assert {"label", "value"} <= set(t.field_names())
        assert t.field_types()["label"] == "String"
        assert t.field_types()["value"] == "Double"
        assert all(hasattr(f, "name") for f in t.fields())

    def test_describe_accessors(self, point_fc):
        t = _catalog_path_for(point_fc)
        assert t.data_type() == "FeatureClass"
        assert t.oid_field_name() == "OBJECTID"
        assert t.shape_field_name() == "Shape"
        assert t.crs() == 4326

    def test_table_has_no_shape_or_crs(self, plain_table):
        t = _catalog_path_for(plain_table)
        assert t.shape_field_name() is None
        assert t.crs() is None

    def test_row_count(self, point_fc):
        assert _catalog_path_for(point_fc).row_count() == 2

    def test_as_dataframe_and_geodataframe(self, point_fc):
        t = _catalog_path_for(point_fc)
        assert list(t.as_dataframe(field_names=["label"])["label"]) == ["alpha", "beta"]
        assert len(t.as_geodataframe()) == 2


# ===========================================================================
# Readers: fc_to_gdf and tbl_to_df
# ===========================================================================


class TestFcToGdf:
    def test_raises_file_not_found_for_missing_fc(self):
        with pytest.raises(FileNotFoundError):
            fc_to_gdf("memory/does_not_exist")

    def test_raises_value_error_for_bad_field_names(self, point_fc):
        with pytest.raises(ValueError, match="do not exist"):
            fc_to_gdf(point_fc, field_names=["no_such_field"])

    def test_raises_for_non_spatial_table(self, plain_table):
        with pytest.raises(ValueError, match="tbl_to_df"):
            fc_to_gdf(plain_table)

    @pytest.mark.parametrize("as_type", [str, Path, _catalog_path_for])
    def test_accepts_str_path_and_catalog_path(self, point_fc, as_type):
        gdf = fc_to_gdf(as_type(point_fc))
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert list(gdf["label"]) == ["alpha", "beta"]

    def test_values_and_geometry(self, point_fc):
        gdf = fc_to_gdf(point_fc)
        assert gdf.crs.to_epsg() == 4326
        assert [(g.x, g.y) for g in gdf.geometry] == [pytest.approx((10, 50)), pytest.approx((20, 60))]
        assert gdf["value"].iloc[0] == 1.5
        assert pd.isna(gdf["value"].iloc[1])

    def test_field_names_restricts_columns(self, point_fc):
        gdf = fc_to_gdf(point_fc, field_names=["label"])
        assert set(gdf.columns) == {"label", "geometry"}

    def test_where_clause_filters_rows(self, point_fc):
        gdf = fc_to_gdf(point_fc, where_clause="label = 'alpha'")
        assert list(gdf["label"]) == ["alpha"]

    def test_to_crs_reprojects(self, point_fc):
        gdf = fc_to_gdf(point_fc, to_crs=3857)
        assert gdf.crs.to_epsg() == 3857
        assert gdf.geometry.iloc[0].x == pytest.approx(1113194.9, abs=1)


@pytest.fixture()
def nullable_bigint_fc():
    """Feature class whose BigInteger column has a null and a value above 2**53.

    Only the memory workspace can store the large value; a file geodatabase drops it.
    """
    name = f"big_{uuid4().hex[:8]}"
    p = f"memory/{name}"
    arcpy.management.CreateFeatureclass("memory", name, "POINT", spatial_reference=arcpy.SpatialReference(4326))
    arcpy.management.AddField(p, "big", "BIGINTEGER")
    arcpy.management.AddField(p, "small", "LONG")
    with arcpy.da.InsertCursor(p, ["SHAPE@XY", "big", "small"]) as cur:
        cur.insertRow([(0, 0), 2**53 + 1, 1])
        cur.insertRow([(1, 1), None, None])
    yield p
    arcpy.management.Delete(p)


class TestArrowBackedColumns:
    """Both readers keep Arrow dtypes, so integer columns with nulls stay integers.

    NumPy-backed pandas turns them into float64, which silently rounds values above 2**53.
    """

    @pytest.mark.parametrize("reader", [tbl_to_df, fc_to_gdf])
    def test_integer_with_nulls_keeps_exact_values(self, nullable_bigint_fc, reader):
        df = reader(nullable_bigint_fc)
        assert isinstance(df["big"].dtype, pd.ArrowDtype)
        assert df["big"].iloc[0] == 2**53 + 1
        assert pd.isna(df["big"].iloc[1])
        assert pd.api.types.is_integer_dtype(df["small"].dtype)

    def test_round_trip_keeps_integer_field_types(self, nullable_bigint_fc, new_path):
        p = gdf_to_fc(fc_to_gdf(nullable_bigint_fc), new_path())
        assert _user_fields(p)["small"] in ("Integer", "BigInteger")
        assert _user_fields(p)["big"] == "BigInteger"


class TestTblToDf:
    def test_reads_table(self, plain_table):
        assert list(tbl_to_df(plain_table)["label"]) == ["alpha", "beta"]

    def test_reads_feature_class_attributes(self, point_fc):
        assert list(tbl_to_df(point_fc, field_names=["label"]).columns) == ["label"]

    @pytest.mark.parametrize("df_type,cls", [
        ("pandas", pd.DataFrame),
        ("polars", pl.DataFrame),
        ("pyarrow", pa.Table),
    ])
    def test_df_type(self, plain_table, df_type, cls):
        assert isinstance(tbl_to_df(plain_table, df_type=df_type), cls)

    def test_invalid_df_type(self, plain_table):
        with pytest.raises(TypeError, match="df_type"):
            tbl_to_df(plain_table, df_type="spark")

    def test_polars_unavailable(self, plain_table, monkeypatch):
        monkeypatch.setattr(interop, "_POLARS_AVAILABLE", False)
        with pytest.raises(ImportError, match="polars"):
            tbl_to_df(plain_table, df_type="polars")

    def test_bad_field_names(self, plain_table):
        with pytest.raises(ValueError, match="do not exist"):
            tbl_to_df(plain_table, field_names=["nope"])

    def test_missing_table(self):
        with pytest.raises(FileNotFoundError):
            tbl_to_df("memory/does_not_exist")

    def test_where_clause(self, plain_table):
        assert list(tbl_to_df(plain_table, where_clause="label = 'beta'")["label"]) == ["beta"]


# ===========================================================================
# Writers: schema integrity (index columns, field names)
# ===========================================================================


def _index_cases():
    return {
        "range": None,
        "filtered_range": pd.Index([0, 2, 5]),
        "unnamed_int": pd.Index([10, 20, 30]),
        "named_int": pd.Index([10, 20, 30], name="key"),
        "unnamed_str": pd.Index(["p", "q", "r"]),
        "named_str": pd.Index(["p", "q", "r"], name="key"),
        "datetime": pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "multi": pd.MultiIndex.from_tuples([(1, "x"), (2, "y"), (3, "z")], names=["k1", "k2"]),
    }


class TestNoIndexColumns:
    """The DataFrame index is never written as a field, whatever its type.

    ``df_to_tbl`` uses ``preserve_index=False``; ``gdf_to_fc`` has to match it.
    """

    @pytest.mark.parametrize("index_name", list(_index_cases()))
    def test_only_user_columns_written(self, write_columns, index_name):
        p = write_columns({"a": [1, 2, 3]}, index=_index_cases()[index_name])
        assert _user_fields(p) == {"a": "BigInteger"}
        assert _column(p, "a") == [1, 2, 3]


class TestFieldNames:
    @pytest.mark.parametrize("name", ["a", "Label", "snake_case_name", "x" * 60])
    def test_valid_names_written_verbatim(self, write_columns, name):
        assert list(_user_fields(write_columns({name: [1, 2, 3]}))) == [name]

    def test_non_ascii_name_not_mangled(self, write_columns):
        """A non-ASCII name must be written as-is or rejected, never garbled
        (currently ``día`` becomes ``dÃ_a``)."""
        _column_survives_or_raise(lambda: write_columns({"día": [1, 2, 3]}), "día")

    @pytest.mark.parametrize("name,written_as", [
        ("my col", "my_col"),
        ("a-b", "a_b"),
        ("1col", "F1col"),
        ("SELECT", "SELECT_"),
    ])
    def test_invalid_names_are_validated_by_arcpy(self, write_columns, name, written_as):
        """Documents current behavior: arcpy renames invalid names. A reader
        of the output sees a different column name than was written."""
        p = write_columns({name: [1, 2, 3]})
        assert list(_user_fields(p)) == [written_as]
        assert _column(p, written_as) == [1, 2, 3]

    def test_case_insensitive_duplicates_both_kept(self, write_columns):
        p = write_columns({"abc": [1, 2, 3], "ABC": [4, 5, 6]})
        fields = _user_fields(p)
        assert len(fields) == 2
        assert sorted(v for f in fields for v in _column(p, f)) == [1, 2, 3, 4, 5, 6]

    def test_objectid_column_becomes_oid_not_a_duplicate(self, write_columns):
        p = write_columns({"OBJECTID": [7, 8, 9], "a": [1, 2, 3]})
        assert [f.name for f in arcpy.ListFields(p) if f.type == "OID"] == ["OBJECTID"]
        assert _user_fields(p) == {"a": "BigInteger"}


# ===========================================================================
# Writers: dtypes and values
# ===========================================================================

_TS = [dt.datetime(2024, 1, 1, 12), None, dt.datetime(2024, 3, 1)]

# id -> (column values, expected arcpy field type, expected values via SearchCursor)
PANDAS_DTYPE_CASES = {
    "int64": (pd.Series([1, -2, 2**40], dtype="int64"), "BigInteger", [1, -2, 2**40]),
    "int32": (pd.Series([1, -2, 3], dtype="int32"), "Integer", [1, -2, 3]),
    "int16": (pd.Series([1, -2, 3], dtype="int16"), "SmallInteger", [1, -2, 3]),
    "int8": (pd.Series([1, -2, 3], dtype="int8"), "SmallInteger", [1, -2, 3]),
    "uint8": (pd.Series([0, 1, 255], dtype="uint8"), "SmallInteger", [0, 1, 255]),
    "uint16": (pd.Series([0, 1, 65535], dtype="uint16"), "Integer", [0, 1, 65535]),
    "uint32": (pd.Series([0, 1, 2**32 - 1], dtype="uint32"), "BigInteger", [0, 1, 2**32 - 1]),
    "Int64_nullable": (pd.array([1, None, 3], dtype="Int64"), "BigInteger", [1, None, 3]),
    "float64": (pd.Series([1.5, np.nan, -3.25]), "Double", [1.5, None, -3.25]),
    "float32": (pd.Series([1.5, 2.0, -3.25], dtype="float32"), "Single", [1.5, 2.0, -3.25]),
    "int_with_nan": (pd.Series([1, np.nan, 3]), "Double", [1.0, None, 3.0]),
    "bool": (pd.Series([True, False, True]), "Integer", [1, 0, 1]),
    "boolean_nullable": (pd.array([True, None, False], dtype="boolean"), "Integer", [1, None, 0]),
    "object_str": (pd.Series(["a", None, "c"], dtype=object), "String", ["a", None, "c"]),
    "string": (pd.array(["a", None, "c"], dtype="string"), "String", ["a", None, "c"]),
    "string_pyarrow": (pd.array(["a", None, "c"], dtype="string[pyarrow]"), "String", ["a", None, "c"]),
    "unicode_values": (pd.Series(["día", "北京", "naïve"]), "String", ["día", "北京", "naïve"]),
    "empty_string": (pd.Series(["", "a", ""]), "String", ["", "a", ""]),
    "category": (pd.Categorical(["x", "y", "x"]), "String", ["x", "y", "x"]),
    "datetime": (pd.to_datetime(pd.Series(_TS)), "Date", _TS),
    "datetime_ms": (pd.to_datetime(pd.Series(["2024-01-01 00:00:00.123"] * 3)), "Date",
                    [dt.datetime(2024, 1, 1, 0, 0, 0, 123000)] * 3),
    "datetime_old_and_far": (pd.to_datetime(pd.Series(["1600-01-01", "1970-01-01", "2262-01-01"])), "Date",
                             [dt.datetime(1600, 1, 1), dt.datetime(1970, 1, 1), dt.datetime(2262, 1, 1)]),
    "date": (pd.Series([dt.date(2024, 1, 1), None, dt.date(2024, 3, 1)]), "DateOnly",
             [dt.date(2024, 1, 1), None, dt.date(2024, 3, 1)]),
    "time": (pd.Series([dt.time(1, 2, 3), None, dt.time(23, 59)]), "TimeOnly",
             [dt.time(1, 2, 3), None, dt.time(23, 59)]),
}


class TestDtypes:
    @pytest.mark.parametrize("case", list(PANDAS_DTYPE_CASES))
    def test_field_type_and_values(self, write_columns, case):
        values, field_type, expected = PANDAS_DTYPE_CASES[case]
        p = write_columns({"c": values})
        assert _user_fields(p) == {"c": field_type}
        assert _column(p, "c") == expected

    @staticmethod
    def _instants_and_offsets(p: str) -> list:
        """``(instant, utc offset in hours)`` per row, as arcpy reports them (``None`` for nulls)."""
        return [None if v is None else (v, v.utcoffset().total_seconds() / 3600) for v in _column(p, "c")]

    def test_tz_aware_datetime_written_as_timestamp_offset(self, write_columns):
        """A tz-aware column becomes a TimestampOffset field and keeps its offset,
        not a UTC Date field with the zone thrown away."""
        ts = pd.Series(pd.to_datetime(["2024-07-01 12:00"] * 3)).dt.tz_localize("America/Los_Angeles")
        p = write_columns({"c": ts})
        assert _user_fields(p) == {"c": "TimestampOffset"}
        expected = dt.datetime(2024, 7, 1, 12, tzinfo=dt.timezone(dt.timedelta(hours=-7)))
        assert self._instants_and_offsets(p) == [(expected, -7.0)] * 3

    def test_tz_aware_dst_offsets_kept_per_row(self, write_columns):
        ts = pd.Series(pd.to_datetime(["2024-07-01 12:00", "2024-01-01 12:00"])).dt.tz_localize("America/Los_Angeles")
        p = write_columns({"c": ts})
        assert [off for _, off in self._instants_and_offsets(p)] == [-7.0, -8.0]

    def test_tz_aware_utc_and_fixed_offset(self, write_columns):
        utc = pd.Series(pd.to_datetime(["2024-07-01 12:00"])).dt.tz_localize("UTC")
        fixed = pd.Series(pd.to_datetime(["2024-07-01 12:00"])).dt.tz_localize(dt.timezone(dt.timedelta(hours=5, minutes=30)))
        assert [off for _, off in self._instants_and_offsets(write_columns({"c": utc}))] == [0.0]
        assert [off for _, off in self._instants_and_offsets(write_columns({"c": fixed}))] == [5.5]

    def test_tz_aware_nulls_and_milliseconds(self, write_columns):
        ts = pd.Series(pd.to_datetime(["2024-07-01 12:00:00.123456", None, "2024-07-01 13:00"], format="ISO8601")).dt.tz_localize("UTC")
        values = self._instants_and_offsets(write_columns({"c": ts}))
        assert values[0][0] == dt.datetime(2024, 7, 1, 12, 0, 0, 123000, tzinfo=dt.timezone.utc)
        assert values[1] is None
        assert values[2][0] == dt.datetime(2024, 7, 1, 13, tzinfo=dt.timezone.utc)

    def test_naive_datetime_still_written_as_date(self, write_columns):
        p = write_columns({"c": pd.to_datetime(pd.Series(["2024-07-01 12:00"] * 2))})
        assert _user_fields(p) == {"c": "Date"}

    @pytest.mark.parametrize("precision", ["us", "ns"])
    def test_sub_millisecond_datetime_is_written(self, write_columns, precision):
        """``pd.Timestamp.now()`` and polars datetimes carry sub-ms precision.
        Writing them must work (truncated to ms), not raise ArrowInvalid."""
        ts = pd.Series(pd.to_datetime(["2024-01-01 00:00:00.123456789"] * 3)).astype(f"datetime64[{precision}]")
        assert _column(write_columns({"c": ts}), "c") == [dt.datetime(2024, 1, 1, 0, 0, 0, 123000)] * 3

    def test_infinity_not_silently_nulled(self, write_columns):
        """arcpy stores ±inf as null. That must raise, not silently change the value."""
        try:
            p = write_columns({"c": [1.0, np.inf, -np.inf]})
        except Exception:
            return
        assert _column(p, "c") == [1.0, np.inf, -np.inf]

    def test_mixed_object_column_raises(self, write_columns):
        with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError)):
            write_columns({"c": [1, "a", 2.0]})

    def test_uint64_beyond_int64_raises(self, write_columns):
        with pytest.raises((pa.ArrowInvalid, OverflowError, ValueError)):
            write_columns({"c": pd.Series([2**63 + 5, 1, 2], dtype="uint64")})

    @pytest.mark.parametrize("case,values", [
        ("all_null", [None, None, None]),
        ("timedelta", pd.to_timedelta([1, 2, 3], unit="D")),
        ("list", [[1], [2, 3], []]),
        ("dict", [{"k": 1}, {"k": 2}, {"k": 3}]),
        ("decimal", [Decimal("1.10"), Decimal("2.20"), Decimal("3.30")]),
    ])
    def test_unsupported_column_not_silently_dropped(self, write_columns, case, values):
        """arcpy drops columns whose Arrow type it can't map. The writer
        must raise (or write them), never return successfully without them."""
        _column_survives_or_raise(lambda: write_columns({"c": values, "keep": [1, 2, 3]}), "c")


class TestNoSilentRowLoss:
    """Values arcpy can't store make it skip the row without an error."""

    def test_string_longer_than_5000_chars(self, write_columns):
        _rows_survive_or_raise(lambda: write_columns({"s": ["x" * 5001, "a", "b"]}), str, 3)

    def test_string_of_exactly_5000_chars_is_written(self, write_columns):
        p = write_columns({"s": ["x" * 5000, "a", "b"]})
        assert _column(p, "s") == ["x" * 5000, "a", "b"]

    def test_integer_beyond_2_pow_53(self, write_columns):
        """File geodatabase BigInteger fields hold ±2**53; larger values drop the row."""
        _rows_survive_or_raise(lambda: write_columns({"i": [2**53 + 1, 1, 2]}), str, 3)

    def test_integer_at_2_pow_53_is_written(self, write_columns):
        p = write_columns({"i": [2**53, -(2**53), 0]})
        assert _column(p, "i") == [2**53, -(2**53), 0]

    def test_empty_frame_writes_empty_output(self, write_columns):
        p = write_columns({"i": pd.Series([], dtype="int64")})
        assert _count(p) == 0
        assert _user_fields(p) == {"i": "BigInteger"}


# ===========================================================================
# Writers: polars input
# ===========================================================================


class TestPolarsWrite:
    def test_dtypes_and_values(self, new_path):
        df = pl.DataFrame({
            "s": ["a", None, "c"],
            "cat": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
            "enum": pl.Series(["x", "y", "x"], dtype=pl.Enum(["x", "y"])),
            "b": [True, None, False],
            "u32": pl.Series([1, 2, 2**32 - 1], dtype=pl.UInt32),
            "i8": pl.Series([1, -2, 3], dtype=pl.Int8),
            "d": [dt.date(2024, 1, 1), None, dt.date(2024, 3, 1)],
            "f": [1.5, None, 2.5],
        })
        p = df_to_tbl(df, new_path())
        assert _user_fields(p) == {
            "s": "String", "cat": "String", "enum": "String", "b": "Integer",
            "u32": "BigInteger", "i8": "SmallInteger", "d": "DateOnly", "f": "Double",
        }
        assert _rows(p, list(df.columns)) == [
            ("a", "x", "x", 1, 1, 1, dt.date(2024, 1, 1), 1.5),
            (None, "y", "y", None, 2, -2, None, None),
            ("c", "x", "x", 0, 2**32 - 1, 3, dt.date(2024, 3, 1), 2.5),
        ]

    def test_tz_aware_datetime_written_as_timestamp_offset(self, new_path):
        df = pl.DataFrame({"t": [dt.datetime(2024, 7, 1, 12)]}).with_columns(
            pl.col("t").dt.replace_time_zone("America/Los_Angeles")
        )
        p = df_to_tbl(df, new_path())
        assert _user_fields(p) == {"t": "TimestampOffset"}
        (value,) = _column(p, "t")
        assert value == dt.datetime(2024, 7, 1, 12, tzinfo=dt.timezone(dt.timedelta(hours=-7)))

    def test_default_datetime_is_written(self, new_path):
        """Polars datetimes default to µs precision."""
        df = pl.DataFrame({"t": [dt.datetime(2024, 1, 1, 0, 0, 0, 123456)]})
        assert _column(df_to_tbl(df, new_path()), "t") == [dt.datetime(2024, 1, 1, 0, 0, 0, 123000)]

    @pytest.mark.parametrize("col", [
        pl.Series("c", [None, None], dtype=pl.Null),
        pl.Series("c", [[1], [2, 3]]),
        pl.Series("c", [{"k": 1}, {"k": 2}]),
    ], ids=["null", "list", "struct"])
    def test_unsupported_column_not_silently_dropped(self, new_path, col):
        df = pl.DataFrame([col, pl.Series("keep", [1, 2])])
        _column_survives_or_raise(lambda: df_to_tbl(df, new_path()), "c")

    def test_polars_rejected_when_unavailable(self, new_path, monkeypatch):
        monkeypatch.setattr(interop, "_POLARS_AVAILABLE", False)
        with pytest.raises(TypeError, match="pandas or polars"):
            df_to_tbl(pl.DataFrame({"a": [1]}), new_path())


# ===========================================================================
# Writers: geometry and spatial reference
# ===========================================================================

_SQ = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])  # counter-clockwise exterior
_SQ_CW = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])  # clockwise exterior
_DONUT = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)], [[(10, 10), (20, 10), (20, 20), (10, 20)]])

# id -> (geometries, expected arcpy shapeType, expected hasZ)
GEOMETRY_CASES = {
    "point": ([Point(1, 2), Point(-3, 4)], "Point", False),
    "point_z": ([Point(1, 2, 3), Point(4, 5, -6)], "Point", True),
    "multipoint": ([MultiPoint([(0, 0), (1, 1)]), MultiPoint([(5, 5)])], "Multipoint", False),
    "linestring": ([LineString([(0, 0), (1, 1), (2, 0)])], "Polyline", False),
    "multilinestring": ([MultiLineString([[(0, 0), (1, 1)], [(5, 5), (6, 6)]])], "Polyline", False),
    "linestring_z": ([LineString([(0, 0, 1), (1, 1, 2)])], "Polyline", True),
    "polygon_ccw": ([_SQ], "Polygon", False),
    "polygon_cw": ([_SQ_CW], "Polygon", False),
    "polygon_with_hole": ([_DONUT], "Polygon", False),
    "multipolygon": ([MultiPolygon([_SQ, Polygon([(50, 50), (60, 50), (60, 60)])])], "Polygon", False),
    "polygon_and_multipolygon": ([_SQ, MultiPolygon([Polygon([(50, 50), (60, 50), (60, 60)])])], "Polygon", False),
    "polygon_z": ([Polygon([(0, 0, 1), (10, 0, 2), (10, 10, 3), (0, 10, 4)])], "Polygon", True),
}


def _write_geoms(new_path, geoms, crs=2227, **cols) -> tuple[gpd.GeoDataFrame, str]:
    gdf = gpd.GeoDataFrame({"id": range(len(geoms)), **cols}, geometry=list(geoms), crs=crs)
    return gdf, gdf_to_fc(gdf, new_path())


class TestGeometry:
    @pytest.mark.parametrize("case", list(GEOMETRY_CASES))
    def test_geometry_written_intact(self, new_path, case):
        geoms, shape_type, has_z = GEOMETRY_CASES[case]
        gdf, p = _write_geoms(new_path, geoms)
        d = arcpy.Describe(p)
        assert d.shapeType == shape_type
        assert bool(d.hasZ) == has_z
        assert _count(p) == len(geoms)
        _assert_geoms_match(geoms, _geoms(p), _xy_tolerance(p))
        assert _geometry_problems(p) == []

    @pytest.mark.parametrize("poly", [_SQ, _SQ_CW, _DONUT], ids=["ccw", "cw", "hole"])
    def test_polygon_area_positive_regardless_of_winding(self, new_path, poly):
        """shapely and Esri use opposite ring orientations. A wrong winding
        order shows up as a negative or zero area."""
        _, p = _write_geoms(new_path, [poly])
        (area,) = _column(p, "SHAPE@AREA")
        assert area == pytest.approx(poly.area, abs=poly.length * _xy_tolerance(p))

    def test_attributes_stay_with_their_geometry(self, new_path):
        geoms = [Point(i, i * 2) for i in range(50)]
        gdf, p = _write_geoms(new_path, geoms, label=[f"row{i}" for i in range(50)])
        rows = _rows(p, ["id", "label", "SHAPE@XY"])
        for i, (id_, label, (x, y)) in enumerate(rows):
            assert (id_, label) == (i, f"row{i}")
            assert (x, y) == pytest.approx((i, i * 2), abs=_xy_tolerance(p))

    def test_renamed_geometry_column(self, new_path):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(1, 1)], crs=4326).rename_geometry("geom")
        p = gdf_to_fc(gdf, new_path())
        assert _user_fields(p) == {"a": "BigInteger"}
        _assert_geoms_match([Point(1, 1)], _geoms(p), _xy_tolerance(p))

    def test_extra_geometry_column_raises(self, new_path):
        gdf = gpd.GeoDataFrame({"a": [1], "other": gpd.GeoSeries([Point(9, 9)])}, geometry=[Point(0, 0)], crs=4326)
        with pytest.raises(TypeError):
            gdf_to_fc(gdf, new_path())

    def test_empty_geometry_written_as_null(self, new_path):
        _, p = _write_geoms(new_path, [Point(0, 0), Point()])
        assert _count(p) == 2
        assert _geoms(p)[1] is None

    def test_null_geometry_row_not_silently_dropped(self, new_path):
        """A file geodatabase drops rows whose geometry is None."""
        _rows_survive_or_raise(lambda: _write_geoms(new_path, [Point(0, 0), None]), lambda r: r[1], 2)

    @pytest.mark.parametrize("geoms", [
        [Point(0, 0), LineString([(0, 0), (1, 1)])],
        [LineString([(0, 0), (1, 1)]), _SQ],
    ], ids=["point+line", "line+polygon"])
    def test_mixed_geometry_types_raise(self, new_path, geoms):
        """A feature class has one geometry type. Rows of another type are
        written with null geometry (memory) or dropped (file gdb) unless
        the writer refuses."""
        with pytest.raises((TypeError, ValueError)):
            _write_geoms(new_path, geoms)

    def test_self_intersecting_polygon_does_not_crash_python(self):
        """Writing a bowtie polygon currently kills the Python process.

        Run in a subprocess so a crash fails this test instead of the session.
        Raising a Python exception or writing the feature are both acceptable.
        """
        code = textwrap.dedent("""
            import geopandas as gpd
            from shapely.geometry import Polygon
            from arcpy_arrow import gdf_to_fc
            bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
            try:
                gdf_to_fc(gpd.GeoDataFrame({"a": [1]}, geometry=[bowtie], crs=2227))
            except Exception as e:
                print("raised", type(e).__name__)
        """)
        src = str(Path(interop.__file__).parents[1])
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONPATH": src},
        )
        assert result.returncode == 0, f"process crashed (exit {result.returncode}): {result.stderr[-500:]}"


class TestSpatialReference:
    @pytest.mark.parametrize("epsg", [4326, 3857, 2227, 26910])
    def test_epsg_crs_preserved(self, new_path, epsg):
        _, p = _write_geoms(new_path, [Point(1, 1)], crs=epsg)
        assert arcpy.Describe(p).spatialReference.factoryCode == epsg

    def test_crs_without_epsg_code_preserved(self, new_path):
        """A CRS with no EPSG code falls back to WKT. It must not come out as
        an 'Unknown' spatial reference."""
        crs = CRS.from_proj4(
            "+proj=tmerc +lat_0=0 +lon_0=-120.5 +k=0.9996 +x_0=500000 +y_0=0 +datum=WGS84 +units=m"
        )
        assert crs.to_epsg() is None
        _, p = _write_geoms(new_path, [Point(1, 1)], crs=crs)
        sr = arcpy.Describe(p).spatialReference
        assert sr.name.lower() != "unknown"
        assert CRS.from_wkt(sr.exportToString()).equals(crs, ignore_axis_order=True)

    def test_missing_crs_raises(self, new_path):
        gdf = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(1, 1)])
        with pytest.raises(ValueError, match="no CRS"):
            gdf_to_fc(gdf, new_path())


# ===========================================================================
# Writers: modes
# ===========================================================================


def _gdf(crs=4326, geoms=None, **cols) -> gpd.GeoDataFrame:
    n = len(next(iter(cols.values())))
    return gpd.GeoDataFrame(cols, geometry=geoms or [Point(i, i) for i in range(n)], crs=crs)


@pytest.fixture(params=["gdf_to_fc", "df_to_tbl"])
def writer(request):
    """``(write, make_frame)`` for one public writer, so mode tests cover both."""
    if request.param == "gdf_to_fc":
        return gdf_to_fc, _gdf
    return df_to_tbl, lambda **cols: pd.DataFrame(cols)


class TestModes:
    def test_invalid_mode_raises_before_writing(self, writer, new_path):
        write, frame = writer
        p = new_path()
        with pytest.raises(ValueError, match="mode"):
            write(frame(a=[1]), p, mode="upsert")
        assert not arcpy.Exists(p)

    def test_new_raises_if_target_exists_and_leaves_it_intact(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2]), new_path())
        with pytest.raises(FileExistsError):
            write(frame(a=[9]), p, mode="new")
        assert _column(p, "a") == [1, 2]

    def test_overwrite_replaces_schema_and_rows(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2, 3]), new_path())
        write(frame(b=["x"]), p, mode="overwrite")
        assert _user_fields(p) == {"b": "String"}
        assert _column(p, "b") == ["x"]

    def test_overwrite_creates_missing_target(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1]), new_path(), mode="overwrite")
        assert _column(p, "a") == [1]

    @pytest.mark.parametrize("mode", ["append", "delete_append"])
    def test_existing_target_modes_require_target(self, writer, new_path, mode):
        write, frame = writer
        with pytest.raises(FileNotFoundError):
            write(frame(a=[1]), new_path(), mode=mode)

    def test_append_adds_rows_and_keeps_existing(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2], s=["x", "y"]), new_path())
        write(frame(a=[3], s=["z"]), p, mode="append")
        assert _rows(p, ["a", "s"]) == [(1, "x"), (2, "y"), (3, "z")]

    def test_append_warns_about_and_skips_unknown_columns(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1], s=["x"]), new_path())
        with pytest.warns(UserWarning, match="extra"):
            write(frame(a=[2], extra=[5]), p, mode="append")
        assert set(_user_fields(p)) == {"a", "s"}
        assert _rows(p, ["a", "s"]) == [(1, "x"), (2, None)]

    def test_append_string_into_integer_field_not_silently_zeroed(self, writer, new_path):
        """``Append(schema_type="NO_TEST")`` writes 0 for strings it can't
        convert. The mismatch must raise."""
        write, frame = writer
        p = write(frame(a=[1, 2]), new_path())
        try:
            write(frame(a=["q", "r"]), p, mode="append")
        except Exception:
            assert _column(p, "a") == [1, 2]
            return
        pytest.fail(f"append of strings into an integer field succeeded: {_column(p, 'a')}")

    def test_append_float_into_integer_field_not_silently_rounded(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2]), new_path())
        try:
            write(frame(a=[1.7, 2.2]), p, mode="append")
        except Exception:
            return
        assert _column(p, "a")[2:] == [1.7, 2.2], "fractional values were silently rounded"

    def test_append_long_string_into_short_field(self, writer, new_path):
        """Target field length is 3; a 6-character value must not be silently
        truncated or dropped."""
        write, frame = writer
        short = new_path()
        ws, name = short.rsplit("/", 1)
        if write is gdf_to_fc:
            arcpy.management.CreateFeatureclass(ws, name, "POINT", spatial_reference=arcpy.SpatialReference(4326))
        else:
            arcpy.management.CreateTable(ws, name)
        arcpy.management.AddField(short, "s", "TEXT", field_length=3)
        try:
            write(frame(s=["abcdef", "ab"]), short, mode="append")
        except Exception:
            return
        assert _column(short, "s") == ["abcdef", "ab"]

    def test_delete_append_replaces_rows_and_keeps_schema(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2, 3], keep=["x", "y", "z"]), new_path())
        arcpy.management.AddField(p, "added_later", "TEXT", field_length=10)
        write(frame(a=[9]), p, mode="delete_append")
        assert set(_user_fields(p)) == {"a", "keep", "added_later"}
        assert _rows(p, ["a", "keep"]) == [(9, None)]

    def test_delete_append_keeps_rows_when_input_is_invalid(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2, 3]), new_path())
        with pytest.raises(Exception):
            write(frame(a=[1, "a", 2.0]), p, mode="delete_append")
        assert _column(p, "a") == [1, 2, 3]

    def test_delete_append_keeps_rows_when_append_fails(self, writer, new_path, monkeypatch):
        """If Append fails after DeleteRows, the target must not be left empty.

        Needs the delete and the append to share an edit session/transaction.
        """
        write, frame = writer
        p = write(frame(a=[1, 2, 3]), new_path())

        def failing_append(*args, **kwargs):
            raise arcpy.ExecuteError("simulated Append failure")

        monkeypatch.setattr(arcpy.management, "Append", failing_append)
        with pytest.raises(arcpy.ExecuteError):
            write(frame(a=[9]), p, mode="delete_append")
        monkeypatch.undo()
        assert _column(p, "a") == [1, 2, 3], "target rows were deleted and not restored"

    def test_delete_append_keeps_rows_when_new_rows_would_be_dropped(self, writer, new_path):
        write, frame = writer
        p = write(frame(a=[1, 2, 3], s=["a", "b", "c"]), new_path())
        try:
            write(frame(a=[4, 5, 6], s=["x" * 6000, "y", "z"]), p, mode="delete_append")
        except Exception:
            assert _column(p, "a") == [1, 2, 3]
            return
        assert _column(p, "a") == [4, 5, 6], "delete_append succeeded but silently dropped rows"

    def test_append_reprojects_or_rejects_other_crs(self, new_path):
        """A point in EPSG:3857 appended to an EPSG:4326 feature class must end
        up at the right location, not have its metres read as degrees."""
        p = gdf_to_fc(_gdf(a=[1]), new_path())
        lon1_lat0_in_3857 = Point(111319.49, 0)
        try:
            gdf_to_fc(_gdf(crs=3857, geoms=[lon1_lat0_in_3857], a=[2]), p, mode="append")
        except Exception:
            return
        (x, y) = _rows(p, ["SHAPE@XY"])[1][0]
        assert (x, y) == pytest.approx((1, 0), abs=1e-6)

    def test_append_wrong_geometry_type_raises(self, new_path):
        p = gdf_to_fc(_gdf(a=[1]), new_path())
        with pytest.raises(Exception):
            gdf_to_fc(_gdf(geoms=[_SQ], a=[2]), p, mode="append")
        assert _count(p) == 1

    @pytest.mark.parametrize("dest_kind", ["str", "Path", "CatalogPath", "None"])
    def test_return_value(self, writer, new_path, fgdb, dest_kind):
        write, frame = writer
        p = new_path()
        if dest_kind == "str":
            dest = p
        elif dest_kind == "Path":
            dest = Path(p)
        elif dest_kind == "CatalogPath":
            ws, name = p.rsplit("/", 1)
            dest = CatalogPath(name=name, db_conn=ws)
        else:
            dest = None
        result = write(frame(a=[1]), dest)
        if dest is None:
            assert isinstance(result, str) and result.startswith("memory/")
            arcpy.management.Delete(result)
        else:
            assert result is dest
            assert _column(str(dest), "a") == [1]

    def test_catalog_path_with_missing_parent_raises(self, writer):
        write, frame = writer
        dest = CatalogPath(name="t", db_conn=r"C:\does\not\exist.gdb")
        with pytest.raises(FileNotFoundError):
            write(frame(a=[1]), dest)


class TestDfToTblInput:
    def test_rejects_non_dataframe(self, new_path):
        with pytest.raises(TypeError):
            df_to_tbl("not a dataframe", new_path())

    def test_rejects_geodataframe_with_pointer_to_gdf_to_fc(self, new_path):
        """A GeoDataFrame is a pandas DataFrame, so it passes the type check
        and then fails deep inside pyarrow. It should be rejected up front."""
        with pytest.raises(TypeError, match="gdf_to_fc"):
            df_to_tbl(_gdf(a=[1]), new_path())

    def test_gdf_to_fc_rejects_plain_dataframe(self, new_path):
        with pytest.raises(TypeError, match="df_to_tbl"):
            gdf_to_fc(pd.DataFrame({"a": [1]}), new_path())


# ===========================================================================
# Round trips: read -> write -> read must be stable
# ===========================================================================


ROUND_TRIP_GEOMS = {
    "point": [Point(1, 2), Point(3, 4), Point(5, 6)],
    "line": [LineString([(0, 0), (1, 1)]), LineString([(2, 2), (3, 5)]), LineString([(9, 9), (8, 7)])],
    "polygon": [_SQ, _DONUT, MultiPolygon([_SQ, Polygon([(50, 50), (60, 50), (60, 60)])])],
}


class TestRoundTrip:
    @pytest.mark.parametrize("kind", list(ROUND_TRIP_GEOMS))
    def test_feature_class_round_trip_is_stable(self, new_path, kind):
        """Three read/write cycles: no accumulating OBJECTID_1 / Shape_Length_1
        fields, no dtype drift, no value drift, no geometry drift."""
        geoms = ROUND_TRIP_GEOMS[kind]
        source = gpd.GeoDataFrame(
            {
                "i": [1, 2, 3],
                "f": [1.5, np.nan, 3.0],
                "s": ["a", None, "c"],
                "t": pd.to_datetime(["2024-01-01 00:00", None, "2024-03-01 12:30"]),
            },
            geometry=geoms,
            crs=2227,
        )
        p = gdf_to_fc(source, new_path())
        first_fields = _user_fields(p)
        assert first_fields == {"i": "BigInteger", "f": "Double", "s": "String", "t": "Date"}
        first = fc_to_gdf(p)

        current = first
        for _ in range(3):
            p = gdf_to_fc(current, new_path())
            assert _user_fields(p) == first_fields
            assert [f.name for f in arcpy.ListFields(p) if f.type == "OID"] == ["OBJECTID"]
            current = fc_to_gdf(p)

        assert list(current.columns) == list(first.columns)
        assert current.dtypes.to_dict() == first.dtypes.to_dict()
        attrs = ["i", "f", "s", "t"]
        pd.testing.assert_frame_equal(current[attrs], first[attrs])
        _assert_geoms_match(list(first.geometry), list(current.geometry), 1e-9)
        _assert_geoms_match(geoms, list(current.geometry), _xy_tolerance(p))

    def test_table_round_trip_is_stable(self, new_path):
        source = pd.DataFrame({"i": [1, 2], "s": ["a", None], "b": [True, False]})
        first = tbl_to_df(df_to_tbl(source, new_path()))
        current = first
        for _ in range(3):
            current = tbl_to_df(df_to_tbl(current, new_path()))
        pd.testing.assert_frame_equal(current.drop(columns="OBJECTID"), first.drop(columns="OBJECTID"))
        assert list(current.columns) == list(first.columns)

    def test_append_read_back_into_source(self, new_path):
        """Reading a feature class and appending the result back doubles the
        rows without disturbing the originals or adding fields."""
        p = gdf_to_fc(gpd.GeoDataFrame({"a": [1, 2]}, geometry=[_SQ, _DONUT], crs=2227), new_path())
        fields_before = _user_fields(p)
        gdf_to_fc(fc_to_gdf(p), p, mode="append")
        assert _user_fields(p) == fields_before
        assert _column(p, "a") == [1, 2, 1, 2]
        areas = _column(p, "SHAPE@AREA")
        assert areas[:2] == pytest.approx(areas[2:])
