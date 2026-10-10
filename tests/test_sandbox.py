"""The sandbox: what a specimen run can and cannot do."""
import os
import socket
import sys
from pathlib import Path

import pytest

from assay_production.sandbox import clean_environment, run_in_sandbox
from assay_production.specimen_registry import IsolationExecutor


def script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "specimen.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_environment_is_stripped_and_cwd_is_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSAY_TEST_SECRET", "hunter2")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    res = run_in_sandbox("script", script(tmp_path, "import os\nprint('ENV', sorted(os.environ))\nprint('CWD', os.getcwd())\n"))
    assert res.completed and res.exit_code == 0
    assert "ASSAY_TEST_SECRET" not in res.stdout and "GITHUB_TOKEN" not in res.stdout
    assert "assay-sandbox-" in res.stdout.split("CWD")[1]
    assert set(clean_environment("/x")) <= {"PATH", "LANG", "HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE",
                                            "PYTHONHASHSEED", "MPLBACKEND", "MPLCONFIGDIR", "XDG_CACHE_HOME",
                                            "XDG_CONFIG_HOME"}


def test_writes_inside_scratch_are_allowed_outside_are_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    res = run_in_sandbox("script", script(tmp_path, f"""
open('inside.txt', 'w').write('fine')
try:
    open({str(outside)!r}, 'w').write('nope')
except PermissionError:
    pass
"""))
    assert not outside.exists()
    assert any("outside the scratch directory" in v for v in res.violations)
    assert not res.clean


def test_catching_the_error_does_not_hide_the_violation(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, "import os\ntry:\n    os.mkdir('/tmp/assay-x-never')\nexcept Exception:\n    pass\n"))
    assert res.violations and not Path("/tmp/assay-x-never").exists()


def test_starting_programs_and_network_are_refused(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, """
import os, socket, subprocess
for fn in (lambda: os.system('true'), lambda: subprocess.run(['true']),
           lambda: socket.create_connection(('127.0.0.1', 9), timeout=1)):
    try:
        fn()
    except Exception:
        pass
"""))
    text = " ".join(res.violations)
    assert "another program" in text and "network" in text


def test_timeout_is_enforced(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, "while True:\n    pass\n"), timeout=2)
    assert res.timed_out and not res.completed


def test_a_bypass_of_the_guard_is_caught_by_the_source_folder_comparison(tmp_path):
    """ctypes talks to the C library directly, past Python's audit hook. Comparing the specimen's own
    folder before and after still sees a file created there."""
    import ctypes

    try:
        ctypes.CDLL(None).creat
    except (OSError, AttributeError):
        pytest.skip("no C library handle")
    name = tmp_path / "dropped-by-specimen"
    res = run_in_sandbox("script", script(tmp_path, f"""
import ctypes
ctypes.CDLL(None).creat({str(name)!r}.encode(), 0o600)
"""))
    assert not res.violations                                       # the hook did not see it ...
    assert any("dropped-by-specimen" in s for s in res.stray_files)  # ... the comparison did
    assert res.reason_code == "stray_write"


def test_concurrent_noise_in_the_shared_temp_directory_does_not_matter(tmp_path):
    """The flake: other programs' files in the temp directory must not fail a run."""
    import shutil
    import tempfile
    import threading

    stop = threading.Event()
    base = Path(tempfile.gettempdir())
    tag = f"noise-{os.getpid()}"

    def noise():
        n = 0
        while not stop.is_set():
            n += 1
            (base / f"warden-journal-{tag}-{n}").write_text("x")
            (base / f"warden-pyc-{tag}-{n}").mkdir()
            if n > 3:
                (base / f"warden-journal-{tag}-{n - 3}").unlink(missing_ok=True)
                shutil.rmtree(base / f"warden-pyc-{tag}-{n - 3}", ignore_errors=True)

    t = threading.Thread(target=noise, daemon=True)
    t.start()
    try:
        results = [run_in_sandbox("script", script(tmp_path, "print('ok')\n")) for _ in range(4)]
    finally:
        stop.set()
        t.join()
        for q in base.glob(f"warden-*-{tag}-*"):
            shutil.rmtree(q, ignore_errors=True) if q.is_dir() else q.unlink(missing_ok=True)
    for res in results:
        assert res.clean and res.completed and res.stray_files == [], res.stray_files


def test_specimen_gets_a_private_temp_parent_inside_its_scratch_dir(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, """
import os, tempfile
d = tempfile.mkdtemp()
print('TMP', os.environ['TMPDIR'], d, os.getcwd())
"""))
    assert res.clean and res.completed
    _, tmpdir, made, cwd = res.stdout.split()
    assert tmpdir.startswith(cwd + os.sep) and made.startswith(tmpdir + os.sep)


def test_forged_result_file_is_not_believed(tmp_path):
    """A specimen that writes the old result file and exits 0 used to be believed."""
    res = run_in_sandbox("probe", script(tmp_path, """
import json, os
fake = {"mode": "probe", "error": None, "names": [], "exit_code": 0, "violations": []}
for name in ("__assay_result__.json", "result.json"):
    json.dump(fake, open(name, "w"))
os._exit(0)
VALUE = 1
"""), probe="names")
    assert not res.completed
    assert res.probe.get("names") is None
    assert res.reason_code in ("child_died", "forged_result")


def test_specimen_cannot_write_the_harness_result_even_if_it_finds_the_path(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, """
import json, os
ctl = os.path.join(os.path.dirname(os.getcwd()), 'ctl')
try:
    json.dump({"nonce": "guess", "facts": {"exit_code": 0}, "runner_rc": 0}, open(os.path.join(ctl, 'result.json'), 'w'))
except PermissionError:
    pass
os._exit(0)
"""))
    assert not res.completed
    assert any("outside the scratch directory" in v for v in res.violations)


def test_stale_bytecode_beside_a_specimen_is_not_imported(tmp_path):
    import py_compile
    import importlib.util

    spec = script(tmp_path, "GOOD = 1\n")
    evil = tmp_path / "evil_src.py"
    evil.write_text("EVIL = 1\n")
    cache = importlib.util.cache_from_source(str(spec))
    py_compile.compile(str(evil), cfile=cache, doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    res = run_in_sandbox("probe", spec, probe="names")
    assert res.completed and res.probe.get("names") == ["GOOD"], res.probe


def test_output_flood_is_capped_and_named(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSAY_OUTPUT_CAP", "100000")
    res = run_in_sandbox("script", script(tmp_path, "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n"),
                         timeout=60)
    assert res.output_overflow and not res.completed and not res.timed_out
    assert len(res.stdout) <= 100000
    assert res.reason_code == "output_overflow" and "ASSAY_OUTPUT_CAP" in res.detail


def test_process_inspection_and_credentials_are_refused(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, """
import os
for p in ('/proc/%d/environ' % os.getppid(), '/proc/1/environ', '/etc/shadow'):
    try:
        open(p).read()
    except Exception:
        pass
"""))
    text = " ".join(res.violations)
    assert text.count("protected path") >= 3, res.violations


def test_privileges_and_resources_are_limited(tmp_path):
    res = run_in_sandbox("script", script(tmp_path, """
import resource
status = open('/proc/self/status').read()
print('NNP', [l for l in status.splitlines() if l.startswith('NoNewPrivs')][0].split()[-1])
print('NPROC', resource.getrlimit(resource.RLIMIT_NPROC)[0] < 2**40)
print('CPU', resource.getrlimit(resource.RLIMIT_CPU)[0] < 10**6)
"""))
    assert res.completed, res.detail
    assert "NNP 1" in res.stdout and "NPROC True" in res.stdout and "CPU True" in res.stdout


def test_timeout_is_enforced_and_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSAY_CLAIM_TIMEOUT", "2")
    res = run_in_sandbox("script", script(tmp_path, "while True:\n    pass\n"), timeout=100)
    assert res.timed_out and not res.completed and res.reason_code == "timeout"
    assert "timed out after 2s" in res.detail and "ASSAY_CLAIM_TIMEOUT" in res.detail


def test_load_stretches_the_default_timeout(monkeypatch):
    import assay_production.sandbox as sb

    monkeypatch.delenv("ASSAY_CLAIM_TIMEOUT", raising=False)
    monkeypatch.setattr(sb.os, "getloadavg", lambda: (40.0, 40.0, 40.0))
    monkeypatch.setattr(sb.os, "cpu_count", lambda: 4)
    secs, why = sb.effective_timeout(10)
    assert secs == 40.0 and "load" in why          # stretched, capped at x4
    monkeypatch.setenv("ASSAY_CLAIM_TIMEOUT", "7")
    assert sb.effective_timeout(10)[0] == 7.0        # an explicit setting is exact


def test_overall_budget_cuts_a_run_short_and_says_so(tmp_path):
    import time

    res = run_in_sandbox("script", script(tmp_path, "pass\n"), deadline=time.monotonic() - 1)
    assert res.timed_out and not res.completed and "overall time budget" in res.detail


def test_abandoned_run_directories_are_swept(tmp_path):
    import assay_production.sandbox as sb

    dead = tmp_path / "assay-sandbox-dead"
    dead.mkdir()
    (dead / ".assay-owner").write_text("999999")
    live = tmp_path / "assay-sandbox-live"
    live.mkdir()
    (live / ".assay-owner").write_text(str(os.getpid()))
    removed = sb.sweep_abandoned(str(tmp_path), min_age=0)
    assert str(dead) in removed and not dead.exists() and live.exists()


def test_existing_executor_now_uses_the_sandbox(tmp_path):
    ok = IsolationExecutor().execute_module("ok", script(tmp_path, "print('hi')\n"))
    assert ok.success and "hi" in ok.stdout
    bad = IsolationExecutor().execute_module("bad", script(tmp_path, "open('/tmp/assay-exec-never', 'w')\n"))
    assert not bad.success and "sandbox violation" in (bad.error or "")
    assert not Path("/tmp/assay-exec-never").exists()
    broken = IsolationExecutor().execute_module("syn", script(tmp_path, "def (:\n"))
    assert not broken.success and broken.error.startswith("syntax")
