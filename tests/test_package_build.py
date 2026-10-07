import os
import shutil
import subprocess
import sys
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.mark.slow
def test_build_requirements_repair_old_packaging(tmp_path):
    tomllib = pytest.importorskip("tomllib")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for the build integration test")
    repository = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE.txt"):
        shutil.copy2(repository / filename, project / filename)
    shutil.copytree(repository / "src" / "mllooper", project / "src" / "mllooper")
    requirements = tomllib.loads((project / "pyproject.toml").read_text())["build-system"]["requires"]
    environment = tmp_path / "build-env"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def run(*args):
        result = subprocess.run(args, cwd=project, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr

    run(uv, "venv", "--python", sys.executable, str(environment))
    run(
        uv,
        "pip",
        "install",
        "--python",
        str(python),
        "setuptools==77.0.3",
        "wheel==0.45.1",
        "packaging==24.1",
    )
    # A build frontend must be able to repair this environment using the declared requirements.
    run(uv, "pip", "install", "--python", str(python), *requirements)
    (project / "dist").mkdir()
    run(
        str(python),
        "-c",
        "from setuptools.build_meta import get_requires_for_build_wheel, build_wheel; "
        "get_requires_for_build_wheel(); build_wheel('dist')",
    )

    wheel = next((project / "dist").glob("*.whl"))
    with ZipFile(wheel) as archive:
        metadata_file = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_file).decode())
    assert metadata["License-Expression"] == "AGPL-3.0-only"
