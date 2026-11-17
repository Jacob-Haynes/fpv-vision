"""scripts/run-laptop.sh on an offline station: a failed reinstall never blocks start-up when the
packages fpv-vision needs are there; only new requirement lines are installed; PATH finds
Homebrew's uv. Fake venv and fake uv; nothing is installed for real."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run-laptop.sh"
BASE, DESKTOP = REPO / "requirements.txt", REPO / "requirements-desktop.txt"

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash script")


def _flat():
    """Both requirement files as the script records them (.req.txt)."""
    desk = [l for l in DESKTOP.read_text().splitlines(keepends=True) if not re.match(r"\s*-r\s", l)]
    return BASE.read_text().splitlines(keepends=True) + desk


def _sha12():
    out = subprocess.run(f"cat {BASE} {DESKTOP} | shasum", shell=True, capture_output=True, text=True).stdout
    return out[:12]


@pytest.fixture
def station(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    py = venv / "bin" / "python"
    py.write_text("""#!/bin/bash
if [ "$1" = "-" ]; then cat >/dev/null; exit ${DEPS_EXIT:-0}; fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then exit 1; fi   # a uv-made venv has no pip
echo "RUN $*"
echo "PATH=$PATH"
""")
    py.chmod(0o755)
    log = tmp_path / "uv.log"
    uv = tmp_path / "uv"
    uv.write_text(f"""#!/bin/bash
echo "CALL $*" >> {log}
for ((i=1; i<=$#; i++)); do if [ "${{!i}}" = "-r" ]; then j=$((i+1)); cat "${{!j}}" >> {log}; fi; done
case " $* " in *" --offline "*) exit ${{UV_OFFLINE_EXIT:-0}};; *) exit ${{UV_EXIT:-0}};; esac
""")
    uv.chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "FPV_VENV": str(venv), "FPV_UV": str(uv),
           "FPV_DATA_DIR": str(tmp_path / "data"), "FPV_SOURCE": "simulate", "MQTT_PASS": "x",
           "TMPDIR": str(tmp_path)}

    def run(*args, **extra):
        r = subprocess.run(["bash", str(SCRIPT), *args], env={**env, **{k: str(v) for k, v in extra.items()}},
                           capture_output=True, text=True, timeout=60)
        return r, (log.read_text() if log.exists() else "")

    return venv, run


def _old_install(venv, prev_lines=None):
    """A venv installed from older requirements: an old stamp, and optionally what it had."""
    (venv / ".req-0123456789ab").touch()
    if prev_lines is not None:
        (venv / ".req.txt").write_text("".join(prev_lines))


def _stamped(venv):
    return sorted(p.name for p in venv.glob(".req-*"))


def test_offline_with_packages_present_starts_anyway(station):
    venv, run = station
    _old_install(venv)
    r, log = run(UV_EXIT=1, UV_OFFLINE_EXIT=1, DEPS_EXIT=0)
    assert r.returncode == 0, r.stderr
    assert "RUN -m fpv_vision" in r.stdout and "starting with what is installed" in r.stderr
    assert "--offline" in log, "tried the local package cache too"
    assert _stamped(venv) == [".req-0123456789ab"], "not stamped: it retries next start"


def test_offline_without_packages_stops_with_a_clear_message(station):
    venv, run = station
    _old_install(venv)
    r, _ = run(UV_EXIT=1, UV_OFFLINE_EXIT=1, DEPS_EXIT=1)
    assert r.returncode == 1 and "RUN" not in r.stdout and "--setup-only" in r.stderr


def test_offline_cache_install_counts(station):
    venv, run = station
    _old_install(venv)
    r, _ = run(UV_EXIT=1, UV_OFFLINE_EXIT=0)
    assert r.returncode == 0 and "local package cache" in r.stdout
    assert _stamped(venv) == [f".req-{_sha12()}"]


def test_only_new_lines_are_installed(station):
    venv, run = station
    lines = _flat()
    new = next(l for l in lines if l.startswith("mss"))
    _old_install(venv, [l for l in lines if l != new])
    r, log = run()
    assert r.returncode == 0, r.stderr
    installed = [l for l in log.splitlines() if not l.startswith("CALL")]
    assert installed == [new.strip()], installed
    assert _stamped(venv) == [f".req-{_sha12()}"]
    assert (venv / ".req.txt").read_text() == "".join(lines)


def test_a_new_line_in_the_base_requirements_is_found_too(station):
    venv, run = station
    lines = _flat()
    new = next(l for l in lines if l.startswith("paho-mqtt"))  # from requirements.txt, via -r
    _old_install(venv, [l for l in lines if l != new])
    r, log = run()
    assert r.returncode == 0, r.stderr
    assert [l for l in log.splitlines() if not l.startswith("CALL")] == [new.strip()]


def test_nothing_new_just_restamps(station):
    venv, run = station
    _old_install(venv, _flat() + ["# an old comment that went away\n"])
    r, log = run(UV_EXIT=1)
    assert r.returncode == 0 and "nothing new to install" in r.stdout and log == ""
    assert _stamped(venv) == [f".req-{_sha12()}"]


def test_up_to_date_venv_from_before_req_txt_records_it_and_installs_nothing(station):
    venv, run = station
    (venv / f".req-{_sha12()}").touch()  # e.g. the HQ Mac's venv today
    r, log = run(UV_EXIT=1)
    assert r.returncode == 0 and "RUN -m fpv_vision" in r.stdout and log == ""
    assert (venv / ".req.txt").read_text() == "".join(_flat())


def test_path_finds_homebrew_and_user_bins(station):
    venv, run = station
    (venv / f".req-{_sha12()}").touch()
    r, _ = run()
    path = next(l for l in r.stdout.splitlines() if l.startswith("PATH="))[5:].split(":")
    assert path[0] == "/opt/homebrew/bin" and any(p.endswith("/.local/bin") for p in path)
    assert any(p.endswith("/.cargo/bin") for p in path) and "/usr/local/bin" in path


def test_setup_only_uses_the_same_install(station):
    venv, run = station
    _old_install(venv)
    r, log = run("--setup-only", UV_EXIT=1, UV_OFFLINE_EXIT=0)
    assert r.returncode == 0, r.stderr
    assert "RUN -m fpv_vision --prefetch" in r.stdout and "--offline" in log


def test_script_never_touches_system_services():
    code = [l.split("#", 1)[0] for l in SCRIPT.read_text().splitlines()]
    code = [re.sub(r'"[^"]*"', '""', l) for l in code]  # messages may mention Homebrew
    for cmd in ("brew", "launchctl", "ollama", "sudo"):
        assert not any(re.search(rf"(^|[;&|(\s]){cmd}\b", l) for l in code), cmd
