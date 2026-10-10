"""Red-team tampers against a copy of the corpus. Each must fail closed, in plain words."""
import itertools
import json
import re
from pathlib import Path

import pytest

from assay_production import manifest_registry as mr
from assay_production import verification as vf
from assay_production.specimen_registry import DuplicateSpecimenId, SpecimenRecord, SpecimenRegistry

from conftest import ROOT, edit, run_verify, run_write

CLAIMS = re.compile(r"(\d+)/(\d+) manifest claims hold")


def failed(proc) -> str:
    assert proc.returncode != 0, "verification passed on a tampered corpus:\n" + proc.stdout[-1500:]
    return proc.stdout + proc.stderr


# ------------------------------------------------------------------ clean run

def test_clean_corpus_passes(corpus):
    proc = run_verify(corpus)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    m = CLAIMS.search(proc.stdout)
    assert m and m.group(1) == m.group(2) and int(m.group(2)) >= 29
    assert "The corpus matches its answer key." in proc.stdout


def test_committed_registry_is_what_write_would_produce(corpus):
    """--write on the clean corpus changes nothing: the committed files are already regenerated."""
    before = {p: (corpus / p).read_bytes() for p in ("MANIFEST.md", "README.md", "assay_production/registry.json")}
    proc = run_write(corpus)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for p, data in before.items():
        assert (corpus / p).read_bytes() == data, p + " changed"


# ---------------------------------------------------------------- finding 1

@pytest.mark.parametrize("old,new,sid,verdict", [
    ("**REFUSE.** A verifier", "**ACCEPT.** A verifier", "fm_3_1_silent_pass", "REFUSE"),
    ("**CONSOLIDATED / LOSSY", "**FAITHFUL", "pair_quorum_state_governance", "CONSOLIDATED_LOSSY"),
])
def test_editing_manifest_prose_verdict_fails(corpus, old, new, sid, verdict):
    edit(corpus / "MANIFEST.md", old, new)
    out = failed(run_verify(corpus))
    assert sid in out and verdict in out
    assert "no longer says" in out and "What to do" in out


def test_editing_manifest_block_verdict_fails(corpus):
    edit(corpus / "MANIFEST.md", '"verdict": "REFUSE"', '"verdict": "ACCEPT"')
    out = failed(run_verify(corpus))
    assert "fm_3_1_silent_pass" in out and "MANIFEST.md block" in out


def _rewrite_block(corpus, change):
    path = corpus / "MANIFEST.md"
    text = path.read_text(encoding="utf-8")
    block_text, problem = mr.extract_block(text)
    assert problem is None
    block = json.loads(block_text)
    change(block)
    path.write_text(mr.replace_block(text, mr.block_to_text(block)), encoding="utf-8")


def test_block_entry_removed_fails(corpus):
    _rewrite_block(corpus, lambda b: b["records"].pop())
    assert "is in the table but not in the MANIFEST.md block" in failed(run_verify(corpus))


def test_block_entry_added_fails(corpus):
    def add(b):
        extra = dict(b["records"][0])
        extra["id"] = "invented_entry"
        b["records"].append(extra)
    _rewrite_block(corpus, add)
    assert "invented_entry is in the MANIFEST.md block but not in the table" in failed(run_verify(corpus))


def test_block_repeating_an_id_fails(corpus):
    _rewrite_block(corpus, lambda b: b["records"].append(dict(b["records"][0])))
    assert "repeats the id" in failed(run_verify(corpus))


def test_deleting_the_machine_block_fails(corpus):
    text = (corpus / "MANIFEST.md").read_text(encoding="utf-8")
    cut = text.index("<!-- ASSAY-MACHINE-BLOCK:BEGIN")
    (corpus / "MANIFEST.md").write_text(text[:cut], encoding="utf-8")
    assert "machine block" in failed(run_verify(corpus))


def test_table_verdict_flip_alone_fails_until_write(corpus):
    """Flipping a verdict in the Python table alone is caught; only the deliberate write command can re-record it."""
    edit(corpus / "assay_production" / "manifest_registry.py", 'verdict="SAME_DESIGN_RESKINNED"',
         'verdict="SAME_DESIGN_COINCIDENCE"')
    out = failed(run_verify(corpus))
    assert "fm_3_5_reskinned_duplicate" in out
    assert "SAME_DESIGN_COINCIDENCE" in out and "MANIFEST.md block" in out
    # The explicit write command re-records it (the prose marker does not contain the verdict token).
    assert run_write(corpus).returncode == 0
    assert run_verify(corpus).returncode == 0
    data = json.loads((corpus / "assay_production" / "registry.json").read_text())
    assert data["fm_3_5_reskinned_duplicate"]["expected_verdict"] == "SAME_DESIGN_COINCIDENCE"


def test_table_flip_that_contradicts_the_prose_cannot_be_written(corpus):
    """--write cannot launder a verdict flip past the human prose."""
    edit(corpus / "assay_production" / "manifest_registry.py", 'prose=_section("### 3.1", "**REFUSE.**")',
         'prose=_section("### 3.1", "**ACCEPT.**")')
    edit(corpus / "assay_production" / "manifest_registry.py", 'verdict="REFUSE"', 'verdict="ACCEPT"')
    proc = run_write(corpus)
    assert proc.returncode != 0 and "refusing to write" in proc.stdout
    assert "no longer says" in proc.stdout


def test_hand_edited_registry_fails(corpus):
    path = corpus / "assay_production" / "registry.json"
    data = json.loads(path.read_text())
    data["fm_3_1_silent_pass"]["expected_verdict"] = "ACCEPT"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    out = failed(run_verify(corpus))
    assert "fm_3_1_silent_pass" in out and "expected_verdict" in out


# ---------------------------------------------------------------- finding 2

TAMPERS = {
    "reference_replaced": ("specimens/reference/sovereign_kernel.py", lambda b: b"x=1\n", "reference_sovereign_kernel", "ACCEPT"),
    "faithful_emptied": ("specimens/pairs/governance_os_security_adapter.py", lambda b: b"", "pair_governance_os_security", "FAITHFUL"),
    "unreachable_emptied": ("specimens/pairs/sre_system_resilience_evaluator_adapter.py", lambda b: b"", "fm_3_4_unreachable_branch", "UNREACHABLE"),
    "quorum_one_liner": ("specimens/pairs/quorum_state_governance_source.py", lambda b: b"x=1", "pair_quorum_state_governance", "CONSOLIDATED_LOSSY"),
    "latin1_byte": ("specimens/pairs/quorum_state_governance_adapter.py", lambda b: b + b"\n# caf\xe9\n", "pair_quorum_state_governance", "CONSOLIDATED_LOSSY"),
    "cr_only": ("specimens/pairs/vanguard-behavioral-simulation-flattened.py", lambda b: b.replace(b". ", b".\r"), "pair_vanguard_behavioral_simulation", "FAITHFUL"),
    "one_char": ("specimens/progressions/uztc/uztc-construct-v1.1-purged.py", lambda b: b + b" ", "uztc_progression", "NOT_MONOTONIC_IMPROVEMENT"),
}


@pytest.mark.parametrize("name", sorted(TAMPERS))
def test_specimen_content_change_fails_and_names_the_verdict(corpus, name):
    rel, change, sid, verdict = TAMPERS[name]
    target = corpus / rel
    original = target.read_bytes()
    target.write_bytes(change(original))
    assert target.read_bytes() != original, "the tamper changed nothing"
    out = failed(run_verify(corpus))
    assert "SHA-256" in out and rel in out
    assert sid in out and f"verdict {verdict}" in out
    assert "can no longer be trusted" in out and "reviewed" in out


def test_unregistered_specimen_file_is_guarded_too(corpus):
    (corpus / "specimens" / "reference" / "wrapper" / "README.md").write_text("changed\n")
    out = failed(run_verify(corpus))
    assert "wrapper/README.md changed" in out and "not a registered specimen" in out


def test_added_specimen_file_fails(corpus):
    (corpus / "specimens" / "reference" / "extra.py").write_text("x = 1\n")
    assert "was added" in failed(run_verify(corpus))


def test_manifest_block_hash_is_checked(corpus):
    path = corpus / "assay_production" / "registry.json"
    data = json.loads(path.read_text())
    for d in data.values():
        d["manifest_block_sha256"] = "0" * 64
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    assert "manifest_block_sha256" in failed(run_verify(corpus))


def test_hash_path_covers_directory_trees(tmp_path):
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "a.py").write_text("a")
    (tmp_path / "d" / "sub" / "b.py").write_text("b")
    one = mr.hash_path(tmp_path, "d")
    (tmp_path / "d" / "__pycache__").mkdir()
    (tmp_path / "d" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    assert mr.hash_path(tmp_path, "d") != one          # caches are part of the hash now
    (tmp_path / "d" / "__pycache__" / "x.pyc").unlink()
    (tmp_path / "d" / "__pycache__").rmdir()
    assert mr.hash_path(tmp_path, "d") == one
    (tmp_path / "d" / "sub" / "b.py").write_text("B")
    assert mr.hash_path(tmp_path, "d") != one           # content is not
    assert mr.hash_path(tmp_path, "missing") is None


# ---------------------------------------------------------------- finding 3

def test_duplicate_id_in_table_fails_and_writer_refuses(corpus):
    registry_file = corpus / "assay_production" / "registry.json"
    before = registry_file.read_bytes()
    edit(corpus / "assay_production" / "manifest_registry.py", 'dict(id="fm_3_2_overclaim"',
         'dict(id="fm_3_1_silent_pass"')
    out = failed(run_verify(corpus))
    assert "duplicate specimen id 'fm_3_1_silent_pass'" in out
    proc = run_write(corpus)
    assert proc.returncode != 0 and "wrote" not in proc.stdout
    assert registry_file.read_bytes() == before, "the writer touched registry.json"


def test_builder_and_registry_raise_on_duplicates():
    specs = [dict(s) for s in mr.MANIFEST_SPECIMENS] + [dict(mr.MANIFEST_SPECIMENS[0])]
    assert any("duplicate" in p for p in mr.table_problems(specs))
    with pytest.raises(DuplicateSpecimenId):
        mr.check_table(specs)
    reg = SpecimenRegistry(ROOT)
    rec = SpecimenRecord("a", "1", "X", "r", "p", "e", None, "s", "pr", "t", "V")
    reg.register(rec)
    with pytest.raises(DuplicateSpecimenId):
        reg.register(rec)


def test_registry_json_with_repeated_key_or_wrong_count_fails(corpus):
    path = corpus / "assay_production" / "registry.json"
    text = path.read_text()
    one = text.index('  "fm_3_1_silent_pass"')
    end = text.index("\n  },", one) + len("\n  },")
    dup = text[one:end]
    path.write_text(text[:end] + "\n" + dup + text[end:])           # the same key twice
    assert "repeats the id" in failed(run_verify(corpus))
    path.write_text(text[:one] + text[end:].lstrip("\n"))           # one entry fewer
    out = failed(run_verify(corpus))
    assert "registry.json" in out and "fm_3_1_silent_pass" in out


def test_write_prints_true_counts(corpus):
    proc = run_write(corpus)
    assert proc.returncode == 0
    n = len(json.loads((corpus / "assay_production" / "registry.json").read_text()))
    assert f"{n} specimens in the file, {len(mr.MANIFEST_SPECIMENS)} rows in the table" in proc.stdout


# ---------------------------------------------------------------- finding 4

def test_statuses_are_honest_and_derived():
    data = json.loads((ROOT / "assay_production" / "registry.json").read_text())
    levels = {}
    for sid, d in data.items():
        v = d["validation"]
        levels.setdefault(v["level"], []).append(sid)
        assert d["manifest_status"] == "ANSWER_KEY_" + v["level"]
        if v["level"] == "UNVALIDATED":
            assert v["reason"] and not v["claims_asserted"], sid
        else:
            assert v["claims_asserted"] and len(v["claims_asserted"]) == len(v["what_was_asserted"]), sid
    # A pair's "faithful" verdict rests on an outside measurement: it must never claim to be proven.
    for sid in ("pair_governance_os_security", "pair_quorum_state_governance", "pair_sre_system_resilience"):
        assert data[sid]["validation"]["level"] == "UNVALIDATED"
    assert data["fm_3_1_silent_pass"]["validation"]["level"] == "VERIFIED_BY_EXECUTION"
    assert data["fm_3_3_flattening_duplicate"]["validation"]["level"] == "VERIFIED_BY_STATIC_CHECK"
    assert "ref.sovereign_kernel.runs_assembled" in data["reference_sovereign_kernel"]["validation"]["claims_asserted"]


def test_typing_a_better_status_by_hand_fails(corpus):
    path = corpus / "assay_production" / "registry.json"
    data = json.loads(path.read_text())
    data["pair_quorum_state_governance"]["manifest_status"] = "ANSWER_KEY_VERIFIED_BY_EXECUTION"
    data["pair_quorum_state_governance"]["validation"]["level"] = "VERIFIED_BY_EXECUTION"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    out = failed(run_verify(corpus))
    assert "pair_quorum_state_governance" in out and "validation status" in out


def test_a_failing_claim_cannot_be_recorded_as_verified(corpus):
    """Break the behaviour (not just the bytes) and re-record: --write must refuse."""
    target = corpus / "specimens" / "progressions" / "uztc" / "uztc-construct-v1.2-validated.py"
    target.write_text(target.read_text() + "\n   def _check_node_compliance(self, a, b):\n       return True\n")
    proc = run_write(corpus)
    assert proc.returncode != 0 and "refusing to write" in proc.stdout
    assert "3.2 overclaim" in proc.stdout


def test_readme_numbers_cannot_drift(corpus):
    edit(corpus / "README.md", "**13** are `UNVALIDATED`", "**2** are `UNVALIDATED`")
    out = failed(run_verify(corpus))
    assert "README" in out and "overclaim or underclaim" in out


def test_todo_count_matches_an_independent_count():
    word = "TO" + "DO"
    hits = 0
    for base in (ROOT / "specimens", ):
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                hits += len(re.findall(r"\b" + word + r"\b", p.read_text(encoding="utf-8", errors="replace")))
    for p in [ROOT / "MANIFEST.md", ROOT / "verify_manifest.py", *(ROOT / "assay_production").glob("*.py")]:
        hits += len(re.findall(r"\b" + word + r"\b", p.read_text(encoding="utf-8", errors="replace")))
    assert vf.count_todo_markers(ROOT) == hits


def test_missing_library_is_a_skip_not_drift(corpus):
    from assay_production.sandbox import SandboxResult

    v = vf.Verifier(corpus)
    res = SandboxResult(True, 1, "", "", False, probe={"error": {"type": "ModuleNotFoundError"}},
                        missing_dependency="matplotlib")
    v.exec_claim("x.runs", "x runs", ["reference_citadel_v1_2"], res, lambda r: True)
    claim = v.report.claims[-1]
    assert claim.skipped and not v.report.failures
    assert "matplotlib" in claim.label


# ---------------------------------------------------------------- finding 5

def test_hostile_specimen_is_stopped_and_fails_verification(corpus, tmp_path):
    marker = Path("/tmp") / f"assay-hostile-{tmp_path.name}"
    assert not marker.exists()
    target = corpus / "specimens" / "reference" / "agent-factory-tactical-agents.py"
    target.write_text(target.read_text() + f"\nopen({str(marker)!r}, 'w').write('pwned')\n")
    proc = run_verify(corpus)
    out = failed(proc)
    assert not marker.exists(), "the hostile specimen wrote outside its scratch directory"
    assert "write outside the scratch directory" in out
    assert "agent-factory-tactical-agents.py" in out
    # And the write command will not bless it.
    proc = run_write(corpus)
    assert proc.returncode != 0 and "refusing to write" in proc.stdout
    assert not marker.exists()


# ---------------------------------------------------------------- finding 6

def test_unrelated_file_does_not_satisfy_3_3(corpus):
    unrelated = (corpus / "specimens" / "pairs" / "quorum_state_governance_source.py").read_bytes()
    (corpus / "specimens" / "reference" / "secure" / "artifact_1.py").write_bytes(unrelated)
    proc = run_write(corpus)             # not a hash problem --write can fix: the claim itself fails
    assert proc.returncode != 0 and "refusing to write" in proc.stdout
    assert "same content once whitespace is ignored" in proc.stdout and "no longer the same code" in proc.stdout
    out = failed(run_verify(corpus))
    assert "fm_3_3_flattening_duplicate" in out and "SAME_CONTENT" in out


def test_same_content_threshold_is_justified_by_the_corpus():
    files = sorted(p for p in (ROOT / "specimens").rglob("*.py") if "__pycache__" not in p.parts)
    text = {p: p.read_text(errors="replace") for p in files}
    real = ({"artifact_3.py", "artifact_1.py"})
    best_other = 0.0
    for a, b in itertools.combinations(files, 2):
        sim = vf.content_similarity(text[a], text[b])
        if {a.name, b.name} == real and a.parent != b.parent:
            assert sim >= vf.SAME_CONTENT_THRESHOLD
        elif {a.name, b.name} != real:
            best_other = max(best_other, sim)
    assert best_other < vf.SAME_CONTENT_THRESHOLD, best_other
    assert vf.content_similarity("class A: pass\n" * 40, "def totally_different():\n    return 7\n" * 40) < 0.5


# ---------------------------------------------------------------- finding 7

def test_failure_messages_are_plain(corpus):
    edit(corpus / "MANIFEST.md", "**REFUSE.** A verifier", "**ACCEPT.** A verifier")
    out = failed(run_verify(corpus))
    for words in ("What happened:", "Verdicts that can no longer be trusted", "What to do:"):
        assert words in out


# ------------------------------------------------- sandbox hardening round

import os
import signal
import subprocess
import sys
import threading
import time

from conftest import _env

SUMMARY = re.compile(r"^ASSAY-SUMMARY: (\{.*\})$", re.M)


def run_verify_env(corpus: Path, extra_env: dict, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    elsewhere = corpus.parent / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    env = _env()
    env.update(extra_env)
    return subprocess.run([sys.executable, str(corpus / "verify_manifest.py"), *args], cwd=elsewhere,
                          capture_output=True, text=True, env=env, timeout=timeout)


def test_summary_line_is_machine_readable_and_matches_the_human_text(corpus):
    proc = run_verify(corpus)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    m = SUMMARY.search(proc.stdout)
    assert m, proc.stdout[-800:]
    data = json.loads(m.group(1))
    assert data["schema"] == "assay-summary/1" and data["all_claims_hold"] is True and data["failures"] == []
    counts = data["entries"]["counts"]
    assert set(counts) == {"VERIFIED_BY_EXECUTION", "VERIFIED_BY_STATIC_CHECK", "UNVALIDATED"}
    assert sum(counts.values()) == data["entries"]["total"] == len(data["entries"]["status"])
    human = re.search(r"\((\d+) verified by execution, (\d+) by static check, (\d+) unvalidated, of (\d+) entries\)",
                      proc.stdout)
    assert human and [int(x) for x in human.groups()] == [counts["VERIFIED_BY_EXECUTION"],
                                                         counts["VERIFIED_BY_STATIC_CHECK"],
                                                         counts["UNVALIDATED"], data["entries"]["total"]]
    assert data["claims"]["passed"] == data["claims"]["total"] and proc.stderr == ""


def test_failure_says_which_claim_and_why_on_stderr_and_in_the_card(corpus):
    target = corpus / "specimens" / "reference" / "citadel_v1.2.py"
    target.write_bytes(target.read_bytes() + b"# edited\n")
    proc = run_verify(corpus)
    out = failed(proc)
    last = proc.stderr.strip().splitlines()[-1]
    assert "NOT PROVEN" in last and "hash.registered" in last and "hash_mismatch" in last
    assert "citadel_v1.2.py" in last
    assert "Reason: hash_mismatch" in proc.stdout
    data = json.loads(SUMMARY.search(proc.stdout).group(1))
    assert data["all_claims_hold"] is False
    assert any(f["claim"] == "hash.registered" and f["reason"] == "hash_mismatch" for f in data["failures"])


def test_a_slow_specimen_is_reported_as_timed_out_with_its_claim(corpus):
    target = corpus / "specimens" / "reference" / "agent-factory-tactical-agents.py"
    target.write_text(target.read_text() + "\nwhile True:\n    pass\n")
    proc = run_verify_env(corpus, {"ASSAY_CLAIM_TIMEOUT": "3"})
    failed(proc)
    last = proc.stderr.strip().splitlines()[-1]
    assert "NOT PROVEN" in last and "timeout" in last and "timed out after 3s" in " ".join(proc.stderr.split())
    assert "ref.agent_factory.runs" in " ".join(proc.stderr.split())


def test_overall_budget_is_configurable_and_reported(corpus):
    proc = run_verify_env(corpus, {"ASSAY_TOTAL_TIMEOUT": "1"})
    failed(proc)
    assert "overall time budget" in proc.stdout and "timeout" in proc.stderr.strip().splitlines()[-1]


def test_symlinked_specimen_with_identical_content_fails(corpus):
    target = corpus / "specimens" / "reference" / "citadel_v1.2.py"
    twin = corpus / "twin_of_citadel.py"
    twin.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(twin)
    assert target.read_bytes() == twin.read_bytes()
    proc = run_verify(corpus)
    out = failed(proc)
    assert "symbolic link" in out and "citadel_v1.2.py" in out
    assert "symlink" in proc.stderr.strip().splitlines()[-1] or "hash_mismatch" in proc.stderr


def test_bytecode_under_specimens_fails(corpus):
    cache = corpus / "specimens" / "pairs" / "__pycache__"
    cache.mkdir()
    (cache / "x.cpython-313.pyc").write_bytes(b"\0\0\0\0")
    out = failed(run_verify(corpus))
    assert "compiled bytecode" in out and "__pycache__" in out


def test_folder_that_is_a_symlink_fails(corpus):
    real = corpus / "specimens" / "pairs"
    moved = corpus / "pairs_elsewhere"
    real.rename(moved)
    real.symlink_to(moved, target_is_directory=True)
    assert "symbolic link" in failed(run_verify(corpus))


def test_verifier_restarts_with_a_minimal_environment(corpus):
    script = corpus / "verify_manifest.py"
    edit(script, "from assay_production import verification  # noqa: E402",
         "from assay_production import verification  # noqa: E402\n"
         "import os\nopen(ROOT / 'envdump.txt', 'a').write(','.join(sorted(os.environ)) + '\\n')")
    proc = run_verify_env(corpus, {"GITHUB_TOKEN": "ghp_secret", "AWS_SECRET_ACCESS_KEY": "s3", "ASSAY_STRICT": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    first, second = (corpus / "envdump.txt").read_text().splitlines()
    assert "GITHUB_TOKEN" in first and "GITHUB_TOKEN" not in second and "AWS_SECRET_ACCESS_KEY" not in second
    assert "ASSAY_MINIMAL_ENV" in second and "ASSAY_STRICT" in second


def test_sigterm_removes_scratch_directories_and_stops_the_child(corpus, tmp_path):
    target = corpus / "specimens" / "reference" / "agent-factory-tactical-agents.py"
    target.write_text(target.read_text() + "\nwhile True:\n    pass\n")
    base = tmp_path / "tmpbase"
    base.mkdir()
    env = _env()
    env.update({"TMPDIR": str(base), "ASSAY_CLAIM_TIMEOUT": "120"})
    proc = subprocess.Popen([sys.executable, str(corpus / "verify_manifest.py")], cwd=corpus.parent, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 120
        # wait until the endless specimen is running: the run directory holds a scratch folder and its harness
        def running():
            return any((d / "ctl" / "result.json").exists() is False and (d / "ctl" / "runner.py").exists()
                       and "agent-factory" in (d / "ctl" / "request.json").read_text()
                       for d in base.glob("assay-sandbox-*") if (d / "ctl" / "request.json").exists())
        while time.time() < deadline and not running():
            time.sleep(0.2)
        assert running(), "the endless specimen never started"
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode != 0
    time.sleep(0.5)
    assert list(base.glob("assay-sandbox-*")) == []
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    assert str(base) not in ps, "a sandbox child is still running"


def test_concurrent_temp_noise_cannot_fail_the_verifier(corpus, tmp_path):
    base = tmp_path / "tmpbase"
    base.mkdir()
    stop = threading.Event()

    def noise():
        n = 0
        while not stop.is_set():
            n += 1
            (base / f"warden-journal-{n}").write_text("x")
            (base / f"warden-pyc-{n}").mkdir()
            if n > 3:
                (base / f"warden-journal-{n - 3}").unlink(missing_ok=True)
                try:
                    (base / f"warden-pyc-{n - 3}").rmdir()
                except OSError:
                    pass

    t = threading.Thread(target=noise, daemon=True)
    t.start()
    try:
        proc = run_verify_env(corpus, {"TMPDIR": str(base)})
    finally:
        stop.set()
        t.join()
    assert proc.returncode == 0, proc.stdout + proc.stderr
