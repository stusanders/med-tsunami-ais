"""Tests for T21 HTML report generator (scripts/make_report.py)."""
import importlib.util

from tsunami.config import load_config, repo_path


def _load_report_module():
    path = repo_path("scripts/make_report.py")
    spec = importlib.util.spec_from_file_location("report_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_report_empty_dir_handles_pending_cleanly(tmp_path):
    mod = _load_report_module()
    out = mod.build_report(load_config(), out_dir=tmp_path, embed=False)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Ships as a Tsunami Network" in text
    assert "Criterion 1" in text
    assert "Criterion 2" in text
    assert "Criterion 3" in text
    assert "Criterion 4" in text
    assert "Pending:" in text
