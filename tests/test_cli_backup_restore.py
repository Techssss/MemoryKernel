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


def test_init_is_lightweight_and_prints_next_steps(tmp_path, monkeypatch):
    import memk.cli.main as cli_main

    monkeypatch.chdir(tmp_path)

    def fail_get_service():
        raise AssertionError("init should not load the service runtime")

    monkeypatch.setattr(cli_main, "get_service", fail_get_service)

    runner = CliRunner()
    result = runner.invoke(app, ["init"])

    assert result.exit_code == 0
    assert "Next Steps" in result.output
    assert (tmp_path / ".memk" / "manifest.json").exists()
    assert (tmp_path / ".memk" / "state" / "state.db").exists()


def test_guide_prints_first_run_flow():
    runner = CliRunner()
    result = runner.invoke(app, ["guide"])

    assert result.exit_code == 0
    assert "memk remember" in result.output
    assert "memk setup claude" in result.output
