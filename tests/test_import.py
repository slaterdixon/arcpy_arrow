"""
Check for circular imports
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("arcpy")


@pytest.mark.parametrize("module", [
    "arcpy_arrow",
    "arcpy_arrow.interop",
    "arcpy_arrow.sanitize_arr_tbl",
])
def test_module_imports_first(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr[-800:]
