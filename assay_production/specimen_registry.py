"""Versioned specimen registry with isolated execution."""
from __future__ import annotations
import ast, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class IsolationResult:
    specimen_id: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    resource_limited: bool
    success: bool
    error: Optional[str] = None
    def to_dict(self):
        return {"specimen_id": self.specimen_id, "exit_code": self.exit_code,
                "stdout": self.stdout[:8000], "stderr": self.stderr[:8000],
                "timed_out": self.timed_out, "resource_limited": self.resource_limited,
                "success": self.success, "error": self.error}

class IsolationExecutor:
    """Run a specimen as a script, under the sandbox in ``sandbox.py``.

    History, stated plainly: the first version of this class ran the file in a
    subprocess with limits but used the shared temp directory as its working
    directory, had no write guard and no network isolation, and nothing in the
    repository called it (the verifier imported specimens in its own process).
    It now delegates to ``sandbox.run_in_sandbox`` and the verifier uses the
    same sandbox. ``success`` also requires that the guard was not tripped.
    """

    def __init__(self, timeout_seconds=30.0, memory_mb=256, cpu_seconds=15):
        # memory_mb and cpu_seconds are kept so old callers still construct it;
        # the sandbox applies its own (looser) limits because numpy and
        # matplotlib need more address space than 256 MB.
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb
        self.cpu_seconds = cpu_seconds

    def execute_module(self, specimen_id: str, source_path: Path) -> IsolationResult:
        if not source_path.exists():
            return IsolationResult(specimen_id, -1, "", "", False, False, False, f"missing: {source_path}")
        try:
            ast.parse(source_path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as e:
            return IsolationResult(specimen_id, -1, "", str(e), False, False, False, f"syntax: {e}")
        from .sandbox import run_in_sandbox
        res = run_in_sandbox("script", source_path, timeout=self.timeout_seconds)
        problems = list(res.violations) + [f"unexpected change outside the scratch directory: {n}" for n in res.stray_files]
        if res.timed_out:
            return IsolationResult(specimen_id, -1, res.stdout, res.stderr, True, False, False, "timeout")
        if problems:
            return IsolationResult(specimen_id, res.exit_code, res.stdout, res.stderr, False, False, False,
                                   "sandbox violation: " + "; ".join(problems))
        if not res.completed:
            return IsolationResult(specimen_id, res.exit_code, res.stdout, res.stderr, False, False, False,
                                   res.detail or "the sandbox child did not report")
        err = res.error
        return IsolationResult(specimen_id, res.exit_code, res.stdout, res.stderr, False, False,
                               res.exit_code == 0 and not err,
                               f"{err['type']}: {err['message']}" if err else None)

@dataclass
class SpecimenRecord:
    specimen_id: str
    specimen_version: str
    specimen_class: str
    source_revision: str
    path: str
    expected_behavior: str
    failure_mode: Optional[str]
    epistemic_status: str
    provenance: str
    intended_test: str
    expected_verdict: str
    manifest_status: str = "ANSWER_KEY_UNVALIDATED"
    isolation_required: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Added with the content hashes. Older consumers ignore unknown keys.
    sha256: Optional[str] = None                     # SHA-256 of the primary file (or tree)
    companion_sha256: Dict[str, str] = field(default_factory=dict)
    manifest_block_sha256: Optional[str] = None      # SHA-256 of MANIFEST.md's machine block
    validation: Dict[str, Any] = field(default_factory=dict)  # what the verifier really asserted
    def to_dict(self):
        return self.__dict__.copy()


class DuplicateSpecimenId(ValueError):
    """Two specimens claimed the same id. The second would silently replace the first."""

class SpecimenRegistry:
    def __init__(self, root: Path, executor=None):
        self.root = Path(root)
        self.executor = executor or IsolationExecutor()
        self._records = {}
    def register(self, record: SpecimenRecord):
        if record.specimen_id in self._records:
            raise DuplicateSpecimenId(
                f"specimen id {record.specimen_id!r} is registered twice; "
                "the later entry would silently replace the first")
        self._records[record.specimen_id] = record
    def get(self, specimen_id):
        return self._records.get(specimen_id)
    def list_by_class(self, specimen_class):
        return [r for r in self._records.values() if r.specimen_class == specimen_class]
    def execute(self, specimen_id):
        rec = self._records.get(specimen_id)
        if not rec:
            return IsolationResult(specimen_id, -1, "", "", False, False, False, "unknown specimen_id")
        path = self.root / rec.path if not Path(rec.path).is_absolute() else Path(rec.path)
        return self.executor.execute_module(specimen_id, path)
    def save(self, path: Path):
        path.write_text(json.dumps({s: r.to_dict() for s, r in self._records.items()}, indent=2), encoding="utf-8")
    def load(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        for sid, d in data.items():
            self._records[sid] = SpecimenRecord(**d)
