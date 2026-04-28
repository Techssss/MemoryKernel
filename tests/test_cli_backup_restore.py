import json
from pathlib import Path

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from memk.cli.main import app
from memk.workspace.schema import WorkspaceManifest


def test_backup_restore_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memk_dir = tmp_path / ".memk"
    state_dir = memk_dir / "state"
    state_dir.mkdir(parents=True)

    manifest = WorkspaceManifest(workspace_root=str(tmp_path))
    (memk_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump()),
        encoding="utf-8",
    )
    db_path = state_dir / "state.db"
    db_path.write_bytes(b"original-db")

    runner = CliRunner()
    archive = tmp_path / "backup.zip"

    backup = runner.invoke(app, ["backup", "--output", str(archive)])
    assert backup.exit_code == 0
    assert archive.exists()

    db_path.write_bytes(b"changed-db")

    restore = runner.invoke(app, ["restore", str(archive), "--force"])
    assert restore.exit_code == 0
    assert db_path.read_bytes() == b"original-db"


def test_restore_requires_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"not-a-real-zip")

    runner = CliRunner()
    result = runner.invoke(app, ["restore", str(archive)])

    assert result.exit_code == 1
    assert "--force" in result.output
