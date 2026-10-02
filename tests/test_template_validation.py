"""
Tests for the template validation in ``_sanitize_arr_tbl(table, template=...)``.

Appending to an existing table validates the data against that table before anything is
staged or deleted, and raises ``ValueError`` listing every way the data would be lost or
changed. These tests build real arcpy tables for both sides. ``validate_append(orig, dest)``
below feeds the data of the staged table ``orig`` through the same validation that
``df_to_tbl`` / ``gdf_to_fc`` run against ``dest``.

What counts as a problem is decided by the *data*, not just the field definitions:

* The staged table comes from Arrow, so its string fields are always 5000
  characters wide and its integers are BigInteger. Comparing field definitions
  would reject every append. A string is a problem only if a value is longer
  than the target field; an integer only if a value doesn't fit the target type.
* Alias, default value, domain and editable never matter for an append.
* A conversion the library would have to guess at (string into number, naive datetime
  into an offset field, ...) raises, and the message says how to convert explicitly.
"""

import datetime as dt
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

arcpy = pytest.importorskip("arcpy")

from arcpy_arrow import CatalogPath, df_to_tbl, memory_str
from arcpy_arrow.sanitize_arr_tbl import _sanitize_arr_tbl


def validate_append(orig, dest) -> None:
    """Validate the data in the staged table ``orig`` against the existing table ``dest``.

    Reads ``orig`` back as Arrow and runs the same ``_sanitize_arr_tbl(..., template=dest)``
    that ``df_to_tbl`` / ``gdf_to_fc`` run when appending. Raises ValueError if it can't be written.
    """
    orig_p = str(orig)
    desc = arcpy.Describe(orig_p)

    if desc.dataType == "FeatureClass":
        table = arcpy.da.TableToArrowTable(orig_p, geometry_encoding="WKB")
        sr = arcpy.SpatialReference()       # Describe() returns a different class than arcpy.SpatialReference
        sr.loadFromString(desc.spatialReference.exportToString())
        _sanitize_arr_tbl(table, shape_col=desc.shapeFieldName, sr=sr, template=str(dest))
    else:
        _sanitize_arr_tbl(arcpy.da.TableToArrowTable(orig_p), template=str(dest))


# ===========================================================================
# Fixtures
# ===========================================================================


def F(name: str, field_type: str, **kwargs) -> tuple:
    """Field spec for ``make_table``: ``F("label", "TEXT", field_length=10)``."""
    return (name, field_type, kwargs)


@pytest.fixture(scope="module")
def fgdb(tmp_path_factory) -> str:
    folder = tmp_path_factory.mktemp("validate_append_fgdb")
    return arcpy.management.CreateFileGDB(str(folder), "test.gdb").getOutput(0)


@pytest.fixture(scope="module")
def coded_domain(fgdb) -> str:
    arcpy.management.CreateDomain(fgdb, "status_dom", "Status", "SHORT", "CODED")
    arcpy.management.AddCodedValueToDomain(fgdb, "status_dom", 1, "Open")
    arcpy.management.AddCodedValueToDomain(fgdb, "status_dom", 2, "Closed")
    return "status_dom"


@pytest.fixture()
def make_table(fgdb):
    """Build a table or feature class, optionally with rows, and delete it afterwards.

    ``rows`` are tuples matching ``fields``. Feature classes are created with
    null shapes, since these tests only care about attributes and schema.
    """
    created: list[str] = []

    def make(fields=(), rows=(), *, geom=None, epsg=4326, has_z=False, has_m=False, ws="memory") -> str:
        name = f"cs_{uuid4().hex[:12]}"
        out_ws = "memory" if ws == "memory" else fgdb
        path = f"memory/{name}" if ws == "memory" else f"{fgdb}/{name}"

        if geom:
            arcpy.management.CreateFeatureclass(
                out_ws, name, geom,
                spatial_reference=arcpy.SpatialReference(epsg),
                has_z="ENABLED" if has_z else "DISABLED",
                has_m="ENABLED" if has_m else "DISABLED",
            )
        else:
            arcpy.management.CreateTable(out_ws, name)
        created.append(path)

        for field_name, field_type, kwargs in fields:
            arcpy.management.AddField(path, field_name, field_type, **kwargs)

        if rows:
            with arcpy.da.InsertCursor(path, [f[0] for f in fields]) as cur:
                for row in rows:
                    cur.insertRow(row)
        return path

    yield make

    for p in created:
        if arcpy.Exists(p):
            arcpy.management.Delete(p)


def _count(p: str) -> int:
    return int(arcpy.management.GetCount(p).getOutput(0))


def _column(p: str, col: str) -> list:
    with arcpy.da.SearchCursor(p, ["OID@", col]) as cur:
        return [r[1] for r in sorted(cur)]


# ===========================================================================
# Schemas that are safe to append
# ===========================================================================


class TestCompatibleSchemas:
    def test_identical_schema(self, make_table):
        fields = [F("label", "TEXT", field_length=20), F("n", "LONG"), F("x", "DOUBLE")]
        validate_append(make_table(fields, [("a", 1, 1.5)]), make_table(fields))

    def test_staged_table_with_extra_columns(self, make_table):
        """Columns missing from the target are reported by the dropped-column warning, not here."""
        staged = make_table([F("n", "LONG"), F("extra", "LONG")], [(1, 2)])
        validate_append(staged, make_table([F("n", "LONG")]))

    def test_target_with_extra_columns(self, make_table):
        staged = make_table([F("n", "LONG")], [(1,)])
        validate_append(staged, make_table([F("n", "LONG"), F("only_in_target", "TEXT", field_length=5)]))

    def test_no_columns_in_common(self, make_table):
        validate_append(make_table([F("a", "LONG")], [(1,)]), make_table([F("b", "LONG")]))

    def test_field_names_match_case_insensitively(self, make_table):
        """arcpy matches fields by name without regard to case, so a case-only
        difference must compare the two fields, not skip them."""
        validate_append(
            make_table([F("Label", "TEXT", field_length=20)], [("abc",)]),
            make_table([F("label", "TEXT", field_length=20)]),
        )

    def test_case_insensitive_match_still_catches_mismatch(self, make_table):
        with pytest.raises(ValueError, match="(?i)label"):
            validate_append(
                make_table([F("Label", "TEXT", field_length=20)], [("abc",)]),
                make_table([F("label", "LONG")]),
            )

    def test_system_fields_are_ignored(self, make_table):
        """A file geodatabase feature class has Shape_Length / Shape_Area; a staged memory table doesn't."""
        staged = make_table([F("n", "LONG")], [(1,)], geom="POLYGON", epsg=2227)
        target = make_table([F("n", "LONG")], geom="POLYGON", epsg=2227, ws="fgdb")
        validate_append(staged, target)

    @pytest.mark.parametrize("source,target,value", [
        ("SHORT", "LONG", 5),
        ("SHORT", "BIGINTEGER", 5),
        ("LONG", "BIGINTEGER", 5),
        ("LONG", "DOUBLE", 5),
        ("SHORT", "DOUBLE", 5),
        ("FLOAT", "DOUBLE", 1.5),
        ("DATEONLY", "DATE", dt.date(2024, 1, 2)),
    ])
    def test_widening_is_allowed(self, make_table, source, target, value):
        validate_append(make_table([F("c", source)], [(value,)]), make_table([F("c", target)]))

    @pytest.mark.parametrize("source,target,value", [
        ("BIGINTEGER", "LONG", 5),
        ("BIGINTEGER", "SHORT", 5),
        ("LONG", "SHORT", 5),
        ("DOUBLE", "LONG", 5.0),
        ("DOUBLE", "SHORT", -3.0),
        ("DATE", "DATEONLY", dt.datetime(2024, 1, 2, 0, 0, 0)),
    ])
    def test_narrowing_is_allowed_when_every_value_fits(self, make_table, source, target, value):
        """The Arrow writer produces BigInteger / Double / Date; the target
        is often a narrower type. What matters is whether the values fit."""
        validate_append(make_table([F("c", source)], [(value,)]), make_table([F("c", target)]))

    def test_narrowing_with_no_rows(self, make_table):
        validate_append(make_table([F("c", "BIGINTEGER")]), make_table([F("c", "SHORT")]))

    def test_narrowing_at_the_type_boundaries(self, make_table):
        validate_append(
            make_table([F("c", "LONG")], [(-32768,), (32767,)]),
            make_table([F("c", "SHORT")]),
        )

    def test_nulls_do_not_block_narrowing(self, make_table):
        validate_append(
            make_table([F("c", "BIGINTEGER")], [(None,), (7,)]),
            make_table([F("c", "SHORT")]),
        )

    def test_long_string_field_with_short_values(self, make_table):
        """Staged string fields are 5000 wide. The target's 10 is fine if the values fit."""
        validate_append(
            make_table([F("s", "TEXT", field_length=5000)], [("abc",), (None,)]),
            make_table([F("s", "TEXT", field_length=10)]),
        )

    def test_string_exactly_as_long_as_the_target_field(self, make_table):
        validate_append(
            make_table([F("s", "TEXT", field_length=50)], [("12345",)]),
            make_table([F("s", "TEXT", field_length=5)]),
        )

    def test_unicode_length_is_counted_in_characters(self, make_table):
        """``北京`` is 2 characters (6 UTF-8 bytes) and fits a length-2 field."""
        validate_append(
            make_table([F("s", "TEXT", field_length=50)], [("北京",)]),
            make_table([F("s", "TEXT", field_length=2)]),
        )

    def test_nullable_staged_column_without_nulls_into_non_nullable_target(self, make_table):
        validate_append(
            make_table([F("n", "LONG")], [(1,), (2,)]),
            make_table([F("n", "LONG", field_is_nullable="NON_NULLABLE")]),
        )

    def test_integer_at_2_pow_53_into_file_geodatabase_biginteger(self, make_table):
        validate_append(
            make_table([F("n", "BIGINTEGER")], [(2**53,), (-(2**53),)]),
            make_table([F("n", "BIGINTEGER")], ws="fgdb"),
        )

    def test_large_integer_into_memory_biginteger(self, make_table):
        """Only a file geodatabase has the 2**53 limit."""
        validate_append(
            make_table([F("n", "BIGINTEGER")], [(2**53 + 1,)]),
            make_table([F("n", "BIGINTEGER")]),
        )

    def test_alias_and_domain_on_target_do_not_block(self, make_table, coded_domain):
        staged = make_table([F("status", "SHORT"), F("label", "TEXT", field_length=5000)], [(1, "x")])
        target = make_table(
            [
                F("status", "SHORT", field_alias="Case Status", field_domain=coded_domain),
                F("label", "TEXT", field_length=10, field_alias="Pretty Label"),
            ],
            ws="fgdb",
        )
        validate_append(staged, target)

    def test_realistic_staged_table_from_arrow(self, make_table):
        """The staged table ``df_to_tbl`` really produces (5000-wide strings,
        BigInteger, Double) against a typical hand-built target."""
        staged = df_to_tbl(pd.DataFrame({"label": ["a", "bb"], "n": [1, 2], "x": [1.5, 2.5]}), memory_str())
        try:
            validate_append(
                staged,
                make_table([F("label", "TEXT", field_length=10), F("n", "LONG"), F("x", "DOUBLE")], ws="fgdb"),
            )
        finally:
            arcpy.management.Delete(staged)

    @pytest.mark.parametrize("wrap", [str, Path, lambda p: CatalogPath(name=p.split("/", 1)[1], db_conn="memory")],
                             ids=["str", "Path", "CatalogPath"])
    def test_accepts_str_path_and_catalog_path(self, make_table, wrap):
        fields = [F("n", "LONG")]
        validate_append(wrap(make_table(fields, [(1,)])), wrap(make_table(fields)))

    def test_does_not_modify_either_table(self, make_table):
        staged = make_table([F("n", "LONG")], [(1,), (2,)])
        target = make_table([F("n", "LONG")], [(9,)])
        validate_append(staged, target)
        assert _column(staged, "n") == [1, 2]
        assert _column(target, "n") == [9]


# ===========================================================================
# Schemas that would lose or change data
# ===========================================================================


class TestIncompatibleFieldTypes:
    @pytest.mark.parametrize("source,target,value", [
        ("TEXT", "LONG", "1"),
        ("TEXT", "DOUBLE", "1.5"),
        ("TEXT", "DATE", "2024-01-01"),
        ("LONG", "TEXT", 1),
        ("DOUBLE", "TEXT", 1.5),
        ("DATE", "TEXT", dt.datetime(2024, 1, 1)),
        ("DATE", "LONG", dt.datetime(2024, 1, 1)),
        ("LONG", "DATE", 1),
        ("DOUBLE", "DATE", 1.5),
        ("DATE", "TIMEONLY", dt.datetime(2024, 1, 1, 1, 2)),
    ])
    def test_incompatible_types_raise(self, make_table, source, target, value):
        with pytest.raises(ValueError, match="c"):
            validate_append(make_table([F("c", source)], [(value,)]), make_table([F("c", target)]))

    def test_type_mismatch_raises_even_without_rows(self, make_table):
        """A type mismatch is a schema problem, regardless of how much data is staged."""
        with pytest.raises(ValueError):
            validate_append(make_table([F("c", "TEXT")]), make_table([F("c", "LONG")]))

    def test_message_names_the_field_and_both_types(self, make_table):
        with pytest.raises(ValueError) as exc:
            validate_append(make_table([F("score", "TEXT")], [("x",)]), make_table([F("score", "LONG")]))
        message = str(exc.value)
        assert "score" in message
        assert "String" in message or "string" in message
        assert "Integer" in message or "int" in message


class TestValuesThatDoNotFit:
    def test_fractional_double_into_integer(self, make_table):
        with pytest.raises(ValueError, match="frac"):
            validate_append(make_table([F("frac", "DOUBLE")], [(1.0,), (2.5,)]), make_table([F("frac", "LONG")]))

    @pytest.mark.parametrize("target,value", [
        ("SHORT", 32768),
        ("SHORT", -32769),
        ("LONG", 2**31),
        ("LONG", -(2**31) - 1),
    ])
    def test_integer_outside_target_range(self, make_table, target, value):
        with pytest.raises(ValueError, match="big"):
            validate_append(make_table([F("big", "BIGINTEGER")], [(1,), (value,)]), make_table([F("big", target)]))

    def test_double_outside_integer_range(self, make_table):
        with pytest.raises(ValueError, match="big"):
            validate_append(make_table([F("big", "DOUBLE")], [(3e9,)]), make_table([F("big", "LONG")]))

    @pytest.mark.parametrize("value", [2**53 + 1, -(2**53) - 1])
    def test_integer_beyond_2_pow_53_into_file_geodatabase(self, make_table, value):
        """A file geodatabase drops rows holding such values. No error comes from arcpy."""
        with pytest.raises(ValueError, match="big"):
            validate_append(make_table([F("big", "BIGINTEGER")], [(value,)]), make_table([F("big", "BIGINTEGER")], ws="fgdb"))

    def test_timestamp_with_time_of_day_into_date_only(self, make_table):
        with pytest.raises(ValueError, match="when"):
            validate_append(
                make_table([F("when", "DATE")], [(dt.datetime(2024, 1, 2, 0, 0),), (dt.datetime(2024, 1, 2, 13, 30),)]),
                make_table([F("when", "DATEONLY")]),
            )

    def test_string_longer_than_target_field(self, make_table):
        with pytest.raises(ValueError, match="label"):
            validate_append(
                make_table([F("label", "TEXT", field_length=50)], [("ok",), ("abcdef",)]),
                make_table([F("label", "TEXT", field_length=5)]),
            )

    def test_string_length_message_includes_both_lengths(self, make_table):
        with pytest.raises(ValueError) as exc:
            validate_append(
                make_table([F("label", "TEXT", field_length=50)], [("abcdefgh",)]),
                make_table([F("label", "TEXT", field_length=5)]),
            )
        assert "8" in str(exc.value)
        assert "5" in str(exc.value)

    def test_unicode_string_longer_than_target_in_characters(self, make_table):
        with pytest.raises(ValueError):
            validate_append(
                make_table([F("s", "TEXT", field_length=50)], [("北京市",)]),
                make_table([F("s", "TEXT", field_length=2)]),
            )

    def test_null_into_non_nullable_target(self, make_table):
        with pytest.raises(ValueError, match="n"):
            validate_append(
                make_table([F("n", "LONG")], [(1,), (None,)]),
                make_table([F("n", "LONG", field_is_nullable="NON_NULLABLE")]),
            )

    def test_null_string_into_non_nullable_target(self, make_table):
        with pytest.raises(ValueError, match="s"):
            validate_append(
                make_table([F("s", "TEXT", field_length=50)], [("a",), (None,)]),
                make_table([F("s", "TEXT", field_length=50, field_is_nullable="NON_NULLABLE")]),
            )

    def test_every_problem_is_reported_not_just_the_first(self, make_table):
        """One failed append shouldn't need three runs to find three problems."""
        staged = make_table(
            [F("bad_type", "TEXT"), F("bad_len", "TEXT", field_length=50), F("bad_range", "LONG")],
            [("x", "abcdefgh", 100000)],
        )
        target = make_table([F("bad_type", "LONG"), F("bad_len", "TEXT", field_length=3), F("bad_range", "SHORT")])
        with pytest.raises(ValueError) as exc:
            validate_append(staged, target)
        for name in ("bad_type", "bad_len", "bad_range"):
            assert name in str(exc.value)


# ===========================================================================
# Geometry: type, Z/M, spatial reference
# ===========================================================================


class TestGeometryCompatibility:
    @pytest.mark.parametrize("geom", ["POINT", "POLYLINE", "POLYGON", "MULTIPOINT"])
    def test_same_geometry_type(self, make_table, geom):
        validate_append(make_table([F("n", "LONG")], geom=geom), make_table([F("n", "LONG")], geom=geom))

    @pytest.mark.xfail(
        strict=True,
        reason="An Arrow table carries no geometry type, Z or M. Needs a check against the GeoDataFrame in gdf_to_fc.",
    )
    @pytest.mark.parametrize("staged_geom,target_geom", [
        ("POINT", "POLYGON"),
        ("POLYGON", "POINT"),
        ("POLYLINE", "POLYGON"),
        ("POINT", "MULTIPOINT"),
    ])
    def test_different_geometry_type_raises(self, make_table, staged_geom, target_geom):
        """Rows of the wrong type are written as null shapes (memory) or dropped (file geodatabase)."""
        with pytest.raises(ValueError, match="(?i)shapeType|geometry"):
            validate_append(make_table(geom=staged_geom), make_table(geom=target_geom))

    @pytest.mark.xfail(
        strict=True,
        reason="An Arrow table carries no geometry type, Z or M. Needs a check against the GeoDataFrame in gdf_to_fc.",
    )
    def test_z_into_non_z_raises(self, make_table):
        with pytest.raises(ValueError, match="(?i)hasZ|Z"):
            validate_append(make_table(geom="POINT", has_z=True), make_table(geom="POINT", has_z=False))

    @pytest.mark.xfail(
        strict=True,
        reason="An Arrow table carries no geometry type, Z or M. Needs a check against the GeoDataFrame in gdf_to_fc.",
    )
    def test_non_z_into_z_raises(self, make_table):
        with pytest.raises(ValueError, match="(?i)hasZ|Z"):
            validate_append(make_table(geom="POINT", has_z=False), make_table(geom="POINT", has_z=True))

    def test_same_z(self, make_table):
        validate_append(make_table(geom="POINT", has_z=True), make_table(geom="POINT", has_z=True))

    @pytest.mark.xfail(
        strict=True,
        reason="An Arrow table carries no geometry type, Z or M. Needs a check against the GeoDataFrame in gdf_to_fc.",
    )
    def test_m_mismatch_raises(self, make_table):
        with pytest.raises(ValueError, match="(?i)hasM|M"):
            validate_append(make_table(geom="POINT", has_m=True), make_table(geom="POINT", has_m=False))

    def test_same_crs(self, make_table):
        validate_append(make_table(geom="POINT", epsg=2227), make_table(geom="POINT", epsg=2227))

    def test_different_crs_raises(self, make_table):
        """Append would reproject, so the data is safe. Silent reprojection
        still hides a mistake (wrong source CRS), so the caller must reproject
        explicitly with ``to_crs``."""
        with pytest.raises(ValueError, match="(?i)spatial|crs|factoryCode"):
            validate_append(make_table(geom="POINT", epsg=4326), make_table(geom="POINT", epsg=3857))

    def test_feature_class_into_table_raises(self, make_table):
        with pytest.raises(ValueError, match="(?i)dataType"):
            validate_append(make_table(geom="POINT"), make_table())

    def test_table_into_feature_class_raises(self, make_table):
        with pytest.raises(ValueError, match="(?i)dataType"):
            validate_append(make_table(), make_table(geom="POINT"))

    def test_plain_tables_have_no_geometry_checks(self, make_table):
        validate_append(make_table([F("n", "LONG")], [(1,)]), make_table([F("n", "LONG")]))


# ===========================================================================
# Wiring: the validation must run before anything in the target changes
# ===========================================================================


def _append(df: pd.DataFrame, target: str, *, delete_first: bool) -> None:
    df_to_tbl(df, target, mode="delete_append" if delete_first else "append")


class TestAppendValidatesFirst:
    def test_mismatched_append_raises_value_error_and_leaves_target_untouched(self, make_table):
        target = make_table([F("n", "LONG")], [(1,), (2,)])
        with pytest.raises(ValueError, match="n"):
            _append(pd.DataFrame({"n": ["a", "b"]}), target, delete_first=False)
        assert _column(target, "n") == [1, 2]

    def test_delete_append_never_deletes_when_schema_is_bad(self, make_table, monkeypatch):
        target = make_table([F("n", "LONG")], [(1,), (2,)])

        def fail_if_called(*args, **kwargs):
            raise AssertionError("DeleteRows ran before the data was validated")

        monkeypatch.setattr(arcpy.management, "DeleteRows", fail_if_called)
        with pytest.raises(ValueError):
            _append(pd.DataFrame({"n": ["a"]}), target, delete_first=True)
        monkeypatch.undo()
        assert _column(target, "n") == [1, 2]

    def test_value_that_does_not_fit_stops_delete_append(self, make_table):
        target = make_table([F("n", "SHORT")], [(1,), (2,)])
        with pytest.raises(ValueError):
            _append(pd.DataFrame({"n": [1, 100000]}), target, delete_first=True)
        assert _column(target, "n") == [1, 2]

    def test_string_longer_than_field_stops_delete_append(self, make_table):
        target = make_table([F("s", "TEXT", field_length=5)], [("old",)])
        with pytest.raises(ValueError, match="s"):
            _append(pd.DataFrame({"s": ["far too long"]}), target, delete_first=True)
        assert _column(target, "s") == ["old"]

    def test_every_problem_is_listed_in_one_error(self, make_table):
        target = make_table([F("a", "LONG"), F("b", "TEXT", field_length=3)])
        with pytest.raises(ValueError) as exc:
            _append(pd.DataFrame({"a": ["x"], "b": ["too long"]}), target, delete_first=False)
        assert "'a'" in str(exc.value) and "'b'" in str(exc.value)

    def test_compatible_append_still_works(self, make_table):
        target = make_table([F("label", "TEXT", field_length=10), F("n", "LONG")], [("old", 1)], ws="fgdb")
        _append(pd.DataFrame({"label": ["a", "b"], "n": [2, 3]}), target, delete_first=False)
        assert sorted(_column(target, "n")) == [1, 2, 3]

    def test_compatible_delete_append_still_works(self, make_table):
        target = make_table([F("label", "TEXT", field_length=10), F("n", "LONG")], [("old", 1)], ws="fgdb")
        _append(pd.DataFrame({"label": ["a", "b"], "n": [2, 3]}), target, delete_first=True)
        assert sorted(_column(target, "n")) == [2, 3]

    def test_unmatched_columns_are_skipped_not_rejected(self, make_table):
        target = make_table([F("n", "LONG")], [(1,)])
        with pytest.warns(UserWarning, match="extra"):
            _append(pd.DataFrame({"n": [2], "extra": ["x"]}), target, delete_first=False)
        assert sorted(_column(target, "n")) == [1, 2]
