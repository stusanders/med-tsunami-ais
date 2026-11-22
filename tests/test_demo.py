"""T27: demo framework checks (scene table, cut lengths, placeholder rendering)."""
import importlib.util
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[1] / "scripts" / "demo"
sys.path.insert(0, str(DEMO))
cm = pytest.importorskip("common")


def test_scene_table_matches_script():
    assert sorted(cm.SCENES) == list(range(1, 19))
    tags = [cm.SCENES[n].tag for n in cm.ORDER]
    assert tags == ["01", "02", "03", "04", "05", "06", "07", "12", "10", "14", "08", "09", "11", "13", "09b", "15", "end"]
    assert sum(s.full_s for s in cm.SCENES.values()) == 305          # script lengths, seven stretched to fit narration
    in_3min = [cm.SCENES[n].tag for n in cm.ORDER if cm.SCENES[n].cut3_s is not None]
    assert in_3min == ["01", "02", "04", "06", "07", "12", "10", "08", "09", "11", "13", "09b", "15", "end"]
    assert sum(cm.SCENES[n].cut3_s or 0 for n in cm.ORDER) == 223


def test_scene_id_accepts_labels():
    assert cm.scene_id("9b") == cm.scene_id("09b") == cm.scene_id("scene_09b") == 16
    assert cm.scene_id("8") == cm.scene_id(8) == 8


def test_out_paths_only_split_when_lengths_differ():
    assert cm.out_paths(8, "3min")[0] == cm.out_paths(8, "full")[0]
    assert cm.out_paths(2, "3min")[0].name == "scene_02_3min.mp4"


def test_ramp_and_fade():
    assert cm.ramp(0, 1, 2) == 0 and cm.ramp(3, 1, 2) == 1
    assert 0 < cm.ramp(1.5, 1, 2) < 1
    assert cm.fade(5, 1, t_out=3) == 0


def test_placeholder_renders(tmp_path):
    fig, chrome = cm.new_figure(10, caption="test")
    arts = cm.placeholder(fig, "something", needs="outputs/x.csv", rect=(0.1, 0.1, 0.8, 0.7))
    assert any(getattr(a, "get_text", lambda: "")() == "RESULT PENDING" for a in arts)
    fig.savefig(tmp_path / "p.png", dpi=20)


@pytest.mark.parametrize("num", sorted(cm.SCENES))
def test_scene_module_contract(num):
    path = DEMO / f"{cm.SCENES[num].stem}.py"
    if not path.exists():
        pytest.skip(f"{path.name} not written yet")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SCENE == num and callable(mod.build)
