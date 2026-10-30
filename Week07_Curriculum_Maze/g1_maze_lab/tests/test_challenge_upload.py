"""Upload checks of scripts/challenge_server.py (no simulator, no server: the check functions only)."""

import importlib.util
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    os.environ["CHALLENGE_DIR"] = str(tmp_path_factory.mktemp("subs"))
    spec = importlib.util.spec_from_file_location("challenge_server", os.path.join(ROOT, "scripts", "challenge_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_policy_g1_rough_is_accepted(server):
    with open(os.path.join(ROOT, "pretrained", "g1_rough", "model_pretrained.pt"), "rb") as f:
        assert server.check_policy(f.read()) is None


def test_policy_other_network_is_refused(server):
    with open(os.path.join(ROOT, "pretrained", "g1_flat", "model_pretrained.pt"), "rb") as f:
        assert "network" in server.check_policy(f.read())
    assert server.check_policy(b"x" * 5000) is not None
    assert server.check_policy(b"") is not None


@pytest.mark.parametrize(
    "src",
    [
        "from g1_maze_lab.terrains import GENERATED\nclass Pilot: pass\n",
        "import g1_maze_lab.terrains as t\nclass Pilot: pass\n",
        "from g1_maze_lab import terrains\nclass Pilot: pass\n",
        "class Pilot:\n    def act(self, obs):\n        return HAZARDS\n",
    ],
)
def test_pilot_reading_the_map_is_refused(server, src):
    assert "แผนที่จริง" in server.check_python("pilot.py", src.encode())


def test_pilot_checks(server):
    ok = "from g1_maze_lab.autopilot import MazeAutopilot\n\nclass Pilot:\n    pass\n"
    assert server.check_python("pilot.py", ok.encode()) is None
    assert "class Pilot" in server.check_python("pilot.py", b"class Driver:\n    pass\n")
    assert "บรรทัด 2" in server.check_python("pilot.py", b"class Pilot:\n    def act(self)\n")
    assert server.check_python("helper.py", b"X = 1\n") is None  # helpers need no class Pilot
