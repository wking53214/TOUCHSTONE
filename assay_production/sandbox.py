"""Run specimen code somewhere it cannot do much harm.

Specimens are real, damaged code. Some of it is untrusted in the plain sense
that nobody has read every line. The verifier has to run it to prove the
answer key, so it runs it here, never in its own process.

What a run gets
---------------
- Its own private directory tree, made fresh for the run::

      assay-sandbox-XXXX/
          work/    the scratch directory: working directory, HOME, and the
                   specimen's private temp parent (work/tmp, which is what
                   TMPDIR points at). The only place a specimen may write.
          ctl/     the harness, the request and the result. The specimen is
                   not allowed to write here and never learns the nonce.

  The whole tree is deleted afterwards, also on timeout, on an exception,
  on SIGTERM / SIGHUP / Ctrl-C and at interpreter exit. A run killed with
  SIGKILL cannot clean up after itself; its children die with it (parent
  death signal) and the next run removes the abandoned tree.
- Two Python processes (both started with ``-I -B``). The *harness* is
  trusted: it holds the supervisor's nonce, never runs specimen code, and is
  the only writer of the result. The *runner* is a child of the harness; it
  runs the specimen, observes what the probe asks for (names defined,
  exception raised, exit code) and hands those facts to the harness over a
  pipe. A file the specimen writes (for example ``__assay_result__.json``)
  is never read as a result, and a result without the nonce is refused.
- A new session and, where the machine allows it, new network and process-ID
  namespaces (``unshare``), so the child has no network and cannot see the
  verifier's ``/proc/<pid>`` entries.
- A stripped environment: PATH, a locale, and a few harmless variables. No
  tokens, keys or any other variable of the caller is passed on.
- ``no_new_privs``, a non-dumpable harness (``/proc/<pid>/environ`` and
  ``mem`` of the harness are not readable), a parent-death signal, and
  limits on CPU time, memory, file size, open files and processes.
- Captured output is capped (``ASSAY_OUTPUT_CAP``, default 1 MiB per stream).
  A specimen that streams more is killed and the failure says so.
- A timeout per run (``ASSAY_CLAIM_TIMEOUT``, default 120 s, stretched on a
  loaded machine) inside an overall budget (``ASSAY_TOTAL_TIMEOUT``, default
  300 s). A run that hits either says it timed out, and says which limit.
- A write guard inside the runner (a Python audit hook, installed before the
  specimen is loaded). Any attempt to create, change, rename or delete a file
  outside the scratch directory, to start another program, to open a network
  connection, or to read the process-inspection files under /proc or the
  usual credential files of the real home directory is refused and recorded
  as a violation. A violation fails the verification even if the specimen
  catches the error.
- Stray-write detection by comparison, not by watching the shared system
  temp directory. Before and after every run the places a specimen could
  plausibly have written are listed: the specimen source tree (the folders
  holding the file and any staged siblings), the sandbox's own control
  directory and the top of the run directory. Anything new, removed or
  changed there is reported. Other programs creating and deleting files in
  /tmp cannot affect the verdict, because /tmp is not watched.

Honest limits
-------------
This is a safety net for accidents and careless code, not a security
boundary. A determined hostile program can get around an audit hook (for
example through the ``ctypes`` module calling the C library directly), and
the after-the-fact comparison only sees the places listed above: a write by
such a program into some other directory (including the shared /tmp) is not
seen. A hostile specimen that finds the report pipe inside its own process
(by listing /proc/self/fd) and writes a well-formed report to it can still
forge the *facts* (names defined, exception raised) for its own run; the
file-write forgery and the early ``os._exit(0)`` are closed, this one is only
made harder. Running as root, with the pid namespace unavailable, the
specimen can still read other processes' memory. Real isolation needs a
container or virtual machine. Run this verifier in a disposable environment
if you add specimens you do not trust.
"""
from __future__ import annotations

import atexit
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_TIMEOUT = 120.0
DEFAULT_TOTAL_TIMEOUT = 300.0
DEFAULT_OUTPUT_CAP = 1024 * 1024

ENV_CLAIM_TIMEOUT = "ASSAY_CLAIM_TIMEOUT"
ENV_TOTAL_TIMEOUT = "ASSAY_TOTAL_TIMEOUT"
ENV_OUTPUT_CAP = "ASSAY_OUTPUT_CAP"

#: Third-party modules a specimen may legitimately need. A run that fails only
#: because one of these is not installed is SKIPPED, never counted as drift.
OPTIONAL_DEPENDENCIES = ("numpy", "matplotlib")

_RESULT_NAME = "result.json"
_REQUEST_NAME = "request.json"
_HARNESS_NAME = "harness.py"
_RUNNER_NAME = "runner.py"
_OWNER_NAME = ".assay-owner"
_SCRATCH_NAME = "work"
_CTL_NAME = "ctl"
_RUN_PREFIX = "assay-sandbox-"
_CTL_EXPECTED = {_RESULT_NAME, _REQUEST_NAME, _HARNESS_NAME, _RUNNER_NAME}
_RUN_EXPECTED = {_SCRATCH_NAME, _CTL_NAME, _OWNER_NAME}

#: Credential locations under the real home directory the guard refuses to read.
_PROTECTED_HOME_PARTS = (".ssh", ".aws", ".gnupg", ".netrc", ".git-credentials", ".docker", ".kube",
                         ".npmrc", ".pypirc", ".config/gh", ".config/gcloud", ".azure")
_PROTECTED_ABSOLUTE = ("/etc/shadow", "/etc/gshadow", "/etc/sudoers", "/root/.ssh", "/root/.aws")

# The program that supervises one run. It is trusted: it never executes the
# specimen. It reads the nonce from its stdin before anything else happens,
# starts the runner, collects the runner's facts from a pipe and writes the
# only result file the verifier will believe.
HARNESS = r'''
import json, os, subprocess, sys

nonce = sys.stdin.readline().strip()
try:
    sys.stdin.close()
except Exception:
    pass
REQ = json.load(open(sys.argv[1], encoding="utf-8"))
try:
    import ctypes
    ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)    # PR_SET_DUMPABLE = 0
except Exception:
    pass
CAP = 1024 * 1024
rfd, wfd = os.pipe()
proc = subprocess.Popen([sys.executable, "-I", "-B", REQ["runner"], sys.argv[1], str(wfd)],
                        pass_fds=(wfd,), stdin=subprocess.DEVNULL, cwd=REQ["scratch"], close_fds=True)
os.close(wfd)
chunks, size = [], 0
while True:
    data = os.read(rfd, 65536)
    if not data:
        break
    size += len(data)
    if size <= CAP:
        chunks.append(data)
rc = proc.wait()
facts, problem = None, ""
if size > CAP:
    problem = "the runner's report was larger than %d bytes" % CAP
elif chunks:
    try:
        facts = json.loads(b"".join(chunks).decode("utf-8", "replace"))
        if not isinstance(facts, dict):
            facts, problem = None, "the runner's report was not a JSON object"
    except ValueError:
        problem = "the runner's report was not valid JSON"
result = {"nonce": nonce, "runner_rc": rc, "facts": facts, "problem": problem}
path = os.path.join(REQ["ctl"], REQ["result_name"])
try:
    os.unlink(path)
except OSError:
    pass
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    json.dump(result, fh)
sys.exit(0)
'''

# The program that runs the specimen. Everything after the guard is installed
# is "specimen time". It reports to the harness over a pipe, never to a file.
RUNNER = r'''
import json, os, re, sys, traceback, importlib.machinery, importlib.util, runpy

REQ = json.load(open(sys.argv[1], encoding="utf-8"))
REPORT_FD = int(sys.argv[2])
SCRATCH = os.path.realpath(REQ["scratch"])
DENY_READ = [os.path.realpath(p) for p in REQ.get("deny_read", [])]
sys.dont_write_bytecode = True
VIOLATIONS = []
_OWN_PID = str(os.getpid())
_PROC_RE = re.compile(r"^/proc/(\d+|self|thread-self)(?:/(.*))?$")
_PROC_DENIED = ("environ", "mem", "cmdline", "fd", "fdinfo", "map_files")


def _inside(p):
    if isinstance(p, int):
        return True
    try:
        real = os.path.realpath(os.fsdecode(p))
    except Exception:
        return False
    return real == SCRATCH or real.startswith(SCRATCH + os.sep) or real == "/dev/null"


def _protected_read(p):
    """True for process-inspection files under /proc and the usual credential files."""
    if isinstance(p, int):
        return False
    try:
        raw = os.fsdecode(p)
        for cand in (raw, os.path.realpath(raw)):
            m = _PROC_RE.match(cand)
            if m:
                pid, rest = m.group(1), (m.group(2) or "")
                first = rest.split("/", 1)[0]
                if pid not in ("self", "thread-self", _OWN_PID) or first in _PROC_DENIED:
                    return True
        real = os.path.realpath(raw)
        for d in DENY_READ:
            if real == d or real.startswith(d + os.sep):
                return True
    except Exception:
        return False
    return False


def _violate(what):
    VIOLATIONS.append(what)
    try:
        os.write(2, ("ASSAY-SANDBOX-VIOLATION: " + what + "\n").encode("utf-8", "replace"))
    except Exception:
        pass
    raise PermissionError("ASSAY sandbox refused: " + what)


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_PATH_EVENTS = {"os.mkdir", "os.remove", "os.rmdir", "os.truncate", "os.chmod", "os.chown",
                "os.utime", "os.mkfifo", "os.mknod", "os.chflags", "os.lchown", "os.setxattr",
                "os.removexattr", "shutil.chown"}
_PAIR_EVENTS = {"os.rename", "os.link", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
                "shutil.move"}
_SPAWN_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
                 "os.fork", "os.forkpty"}
_HARMLESS_PROGRAMS = {"fc-list", "fc-match"}
_NET_EVENTS = {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
               "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}


def _hook(event, args):
    try:
        if event == "open":
            path, mode, flags = args[0], args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                      (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            if writing and not _inside(path):
                _violate("write outside the scratch directory: " + repr(path))
            if _protected_read(path):
                _violate("read of a protected path (process inspection or credentials): " + repr(path))
        elif event in _PATH_EVENTS:
            if not _inside(args[0]):
                _violate(event + " outside the scratch directory: " + repr(args[0]))
        elif event in _PAIR_EVENTS:
            for p in args[:2]:
                if not _inside(p):
                    _violate(event + " outside the scratch directory: " + repr(p))
        elif event == "os.symlink":
            if not _inside(args[1]):
                _violate("os.symlink outside the scratch directory: " + repr(args[1]))
        elif event == "subprocess.Popen" and os.path.basename(str(args[0])) in _HARMLESS_PROGRAMS:
            pass  # matplotlib asks fontconfig which fonts exist; that only reads
        elif event in _SPAWN_EVENTS:
            _violate("tried to start another program (" + event + ")")
        elif event in _NET_EVENTS:
            _violate("tried to use the network (" + event + ")")
    except PermissionError:
        raise
    except Exception:
        pass


def _finish(payload):
    payload["violations"] = list(VIOLATIONS)
    code = payload.get("exit_code", 0)
    if not isinstance(code, int) or not 0 <= code <= 255:
        code = 1
    payload["exit_code"] = code
    data = json.dumps(payload).encode("utf-8")
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    view = memoryview(data)
    while view:
        n = os.write(REPORT_FD, view)
        view = view[n:]
    os._exit(code)


def _load(path):
    # Compiled from the source text every time: an existing .pyc beside the
    # specimen (stale, or planted) is never consulted.
    name = "assay_specimen_" + os.path.basename(path).replace("-", "_").replace(".", "_")
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up here
    with open(path, "rb") as fh:
        source = fh.read()
    code = loader.source_to_code(source, path)
    exec(code, module.__dict__)
    return module


def _err(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:300],
            "module_name": getattr(exc, "name", None)}


try:
    import ctypes
    ctypes.CDLL(None, use_errno=True).prctl(1, 9, 0, 0, 0)    # PR_SET_PDEATHSIG = SIGKILL: die with the harness
except Exception:
    pass
os.chdir(SCRATCH)
sys.addaudithook(_hook)

out = {"mode": REQ["mode"], "error": None}
try:
    path = REQ["path"]
    if REQ["mode"] == "script":
        sys.argv = [path]
        try:
            runpy.run_path(path, run_name="__main__")
            out["exit_code"] = 0
        except SystemExit as exc:
            code = exc.code
            out["exit_code"] = 0 if code in (None, 0) else (code if isinstance(code, int) else 1)
    else:
        module = _load(path)
        probe = REQ["probe"]
        if probe == "names":
            out["names"] = sorted(n for n in dir(module) if not n.startswith("__"))
        elif probe == "uztc_validate":
            construct = module.Universal_Zero_Trust_Construct({})
            try:
                construct.validate_synthesis({"x": 1})
                out["raised"] = None
            except Exception as exc:
                out["raised"] = type(exc).__name__
        elif probe == "capability_denied":
            try:
                module.validate_capabilities({"net.exfiltrate"})
                out["raised"] = None
            except Exception as exc:
                out["raised"] = type(exc).__name__
        else:
            out["error"] = {"type": "UnknownProbe", "message": probe, "module_name": None}
        out.setdefault("exit_code", 0)
except BaseException as exc:  # noqa: BLE001 - everything the specimen does is reported
    out["error"] = _err(exc)
    out["traceback"] = traceback.format_exc()[-1500:]
    out["exit_code"] = 1
_finish(out)
'''


@dataclass
class SandboxResult:
    completed: bool                      # the child ran and gave a report the supervisor believes
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    probe: Dict = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    stray_files: List[str] = field(default_factory=list)
    network_isolated: bool = False
    missing_dependency: Optional[str] = None
    detail: str = ""
    dismissed_strays: List[str] = field(default_factory=list)   # kept for old callers; always empty now
    rewritten: List[str] = field(default_factory=list)          # kept for old callers; always empty now
    output_overflow: bool = False        # the specimen printed more than the cap and was killed
    forged: bool = False                 # the result did not come from the harness, or contradicted the process
    killed_by: str = ""                  # "", "timeout", "output", "overall budget"
    timeout_used: float = 0.0
    timeout_source: str = ""             # where the limit came from (text for the failure message)

    @property
    def clean(self) -> bool:
        """No guard was tripped and nothing unexpected changed outside the scratch dir."""
        return not self.violations and not self.stray_files

    @property
    def error(self) -> Optional[dict]:
        return self.probe.get("error") if self.probe else None

    def problems(self) -> List[Tuple[str, str]]:
        """Why this run cannot be believed, most important first, as (reason code, plain text)."""
        out: List[Tuple[str, str]] = []
        if self.forged:
            out.append(("forged_result", self.detail or "the result did not come from the sandbox harness"))
        if self.violations:
            out.append(("violation", "the specimen was stopped trying to: " + "; ".join(self.violations)))
        if self.stray_files:
            out.append(("stray_write", "the specimen left unexpected changes outside its scratch directory: "
                        + "; ".join(self.stray_files)))
        if self.timed_out:
            out.append(("timeout", self.detail or "timed out"))
        elif self.output_overflow:
            out.append(("output_overflow", self.detail or "output over the cap"))
        elif not self.completed and not self.forged:
            out.append(("child_died", self.detail or "the sandbox child did not report"))
        err = self.error
        if err:
            out.append(("specimen_error", f"it raised {err.get('type')}: {err.get('message')}"))
        return out

    @property
    def reason_code(self) -> str:
        probs = self.problems()
        return probs[0][0] if probs else ""


# ----------------------------------------------------------- configuration

def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def load_factor() -> float:
    """How much to stretch default time limits: 1.0 on an idle machine, up to 4.0 under load."""
    try:
        load = os.getloadavg()[0]
        cpus = os.cpu_count() or 1
    except (OSError, AttributeError):
        return 1.0
    return min(4.0, max(1.0, load / cpus))


def effective_timeout(requested: Optional[float] = None) -> Tuple[float, str]:
    """(seconds, where it came from). ASSAY_CLAIM_TIMEOUT wins and is used exactly as given."""
    env = _env_float(ENV_CLAIM_TIMEOUT)
    if env is not None:
        return env, f"{ENV_CLAIM_TIMEOUT}={env:g}"
    base = float(requested) if requested else DEFAULT_TIMEOUT
    factor = load_factor()
    if factor > 1.0:
        return base * factor, f"{base:g}s stretched x{factor:.1f} for machine load (set {ENV_CLAIM_TIMEOUT} to fix it)"
    return base, f"{base:g}s (set {ENV_CLAIM_TIMEOUT} to change it)"


def total_budget() -> float:
    return _env_float(ENV_TOTAL_TIMEOUT) or DEFAULT_TOTAL_TIMEOUT


def output_cap() -> int:
    env = _env_float(ENV_OUTPUT_CAP)
    return int(env) if env else DEFAULT_OUTPUT_CAP


# ------------------------------------------------------- isolation commands

_UNSHARE: Optional[List[str]] = None


def network_isolation_command() -> List[str]:
    """The ``unshare`` prefix this machine lets us use, else [].

    Best first: no network plus a private process-ID namespace with its own
    /proc (so the child cannot see the verifier's /proc/<pid> files), then no
    network alone.
    """
    global _UNSHARE
    if _UNSHARE is not None:
        return _UNSHARE
    _UNSHARE = []
    exe = shutil.which("unshare")
    if exe:
        for flags in (["-pfn", "--mount-proc", "--kill-child"], ["-rpfn", "--mount-proc", "--kill-child"],
                      ["-pfn", "--mount-proc"], ["-rpfn", "--mount-proc"], ["-n"], ["-rn"]):
            try:
                done = subprocess.run([exe, *flags, "true"], capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if done.returncode == 0:
                _UNSHARE = [exe, *flags]
                break
    return _UNSHARE


def describe_isolation() -> str:
    cmd = network_isolation_command()
    if cmd:
        net = "no network (%s)" % " ".join(cmd)
        if "--mount-proc" in cmd:
            net += ", own process-ID namespace"
        else:
            net += ", process-ID namespace unavailable"
    else:
        net = ("network NOT isolated at operating-system level (unshare unavailable); sockets are still "
               "blocked inside Python")
    return ("separate process in its own session, private scratch and temp directory, stripped environment, "
            "timeout, output cap and resource limits (CPU, memory, file size, processes), no_new_privs, "
            "write guard, %s" % net)


def _libc():
    import ctypes
    return ctypes.CDLL(None, use_errno=True)


def protect_current_process() -> None:
    """Make the calling process harder to inspect and escalate from (best effort, never raises).

    PR_SET_DUMPABLE=0 makes /proc/<pid>/environ and mem unreadable to other
    processes of the same user; PR_SET_NO_NEW_PRIVS=1 is inherited by every
    child. Call it from a command-line entry point, not from a library.
    """
    try:
        libc = _libc()
        libc.prctl(4, 0, 0, 0, 0)     # PR_SET_DUMPABLE
        libc.prctl(38, 1, 0, 0, 0)    # PR_SET_NO_NEW_PRIVS
    except Exception:
        pass


def _count_user_processes() -> int:
    uid, n = os.getuid(), 0
    try:
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                try:
                    if entry.stat().st_uid == uid:
                        n += 1
                except OSError:
                    pass
    except OSError:
        pass
    return n


def _limits(cpu_seconds: int, nproc: int):
    import resource
    try:
        libc = _libc()
    except Exception:
        libc = None

    def apply():
        for name, soft in (("RLIMIT_CPU", cpu_seconds), ("RLIMIT_FSIZE", 32 * 1024 * 1024),
                           ("RLIMIT_NOFILE", 256), ("RLIMIT_CORE", 0),
                           ("RLIMIT_AS", 3 * 1024 * 1024 * 1024), ("RLIMIT_NPROC", nproc)):
            try:
                resource.setrlimit(getattr(resource, name), (soft, soft))
            except (ValueError, OSError):
                pass
        if libc is not None:
            libc.prctl(38, 1, 0, 0, 0)                   # PR_SET_NO_NEW_PRIVS
            libc.prctl(1, int(signal.SIGKILL), 0, 0, 0)  # PR_SET_PDEATHSIG: die with the verifier
    return apply


def clean_environment(scratch: str) -> Dict[str, str]:
    """Only what Python needs. No secrets of the caller get through.

    ``scratch`` is the specimen's scratch directory; its private temp parent
    is ``scratch/tmp``, which TMPDIR points at.
    """
    tmp = os.path.join(scratch, "tmp")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8",
           "HOME": scratch, "TMPDIR": tmp, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "MPLCONFIGDIR": scratch,
           "XDG_CACHE_HOME": scratch, "XDG_CONFIG_HOME": scratch}
    return env


def minimal_parent_environment(extra: Sequence[str] = ()) -> Dict[str, str]:
    """The few variables the verifier itself keeps (everything else, secrets included, is dropped)."""
    keep = {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "HOME", "TMPDIR", "TERM", "TZ", *extra}
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("ASSAY_")}
    env.setdefault("PATH", "/usr/bin:/bin")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASSAY_MINIMAL_ENV"] = "1"
    return env


# ------------------------------------------------------- bookkeeping / cleanup

_ACTIVE: Dict[str, Optional[subprocess.Popen]] = {}
_ACTIVE_LOCK = threading.Lock()
_SWEPT = False


def _kill_group(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _rmtree(path: str) -> None:
    def fix(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=fix)


def cleanup_active() -> None:
    """Kill every sandbox child still running and delete every run directory still on disk."""
    with _ACTIVE_LOCK:
        items = list(_ACTIVE.items())
        _ACTIVE.clear()
    for path, proc in items:
        _kill_group(proc)
        _rmtree(path)


atexit.register(cleanup_active)


def install_signal_cleanup() -> None:
    """Turn SIGTERM / SIGHUP into an orderly exit so scratch directories are removed."""
    def handler(signum, _frame):
        cleanup_active()
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def sweep_abandoned(base: Optional[str] = None, min_age: float = 900.0) -> List[str]:
    """Remove run directories whose owner is gone (a verifier killed with SIGKILL leaves them)."""
    base = base or tempfile.gettempdir()
    removed: List[str] = []
    now = time.time()
    uid = os.getuid()
    try:
        names = os.listdir(base)
    except OSError:
        return removed
    for name in names:
        if not name.startswith(_RUN_PREFIX):
            continue
        path = os.path.join(base, name)
        try:
            st = os.lstat(path)
            if not os.path.isdir(path) or os.path.islink(path) or st.st_uid != uid:
                continue
            with _ACTIVE_LOCK:
                if path in _ACTIVE:
                    continue
            marker = os.path.join(path, _OWNER_NAME)
            try:
                stamp = os.lstat(marker).st_mtime
                owner = int(open(marker, encoding="utf-8").read().strip() or 0)
            except (OSError, ValueError):
                stamp, owner = st.st_mtime, 0
            if now - stamp < min_age:
                continue
            if owner > 0:
                try:
                    os.kill(owner, 0)
                    continue          # the owner is still alive
                except ProcessLookupError:
                    pass
                except PermissionError:
                    continue
            _rmtree(path)
            removed.append(path)
        except OSError:
            continue
    return removed


# ----------------------------------------------------- stray-write detection

def _snapshot(path: Path, limit: int = 20000) -> Dict[str, Tuple]:
    """Relative name -> (kind, size, mtime_ns) for everything under a directory (links not followed)."""
    out: Dict[str, Tuple] = {}
    root = str(path)
    for cur, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs) + list(files):
            full = os.path.join(cur, name)
            try:
                st = os.lstat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, root)
            if os.path.islink(full):
                out[rel] = ("link", os.readlink(full), 0)
            elif os.path.isdir(full):
                out[rel] = ("dir", 0, 0)
            else:
                out[rel] = ("file", st.st_size, st.st_mtime_ns)
            if len(out) >= limit:
                return out
    return out


def _shared_dirs() -> set:
    """Directories other programs write to all the time. Never compared as a whole."""
    names = {"/", "/tmp", "/var/tmp", "/dev/shm", tempfile.gettempdir(), os.environ.get("TMPDIR", "")}
    out = set()
    for n in names:
        if n:
            try:
                out.add(Path(n).resolve())
            except OSError:
                pass
    try:
        out.add(Path.home().resolve())
    except (RuntimeError, OSError):
        pass
    return out


def _snapshot_files(paths: Sequence[Path]) -> Dict[str, Tuple]:
    out: Dict[str, Tuple] = {}
    for p in paths:
        try:
            st = os.lstat(p)
            out[str(p)] = ("file", st.st_size, st.st_mtime_ns)
        except OSError:
            pass
    return out


def _watch_units(target: Path, stage: Sequence[Tuple[Path, str]]) -> List[Tuple[str, Path, List[Path]]]:
    """What to compare before and after: (label, folder, files). A folder is compared as a whole tree
    unless it is a shared one (like /tmp), in which case only the specimen files themselves are."""
    shared = _shared_dirs()
    units: List[Tuple[str, Path, List[Path]]] = []
    seen_dirs: List[Path] = []
    loose: List[Path] = []
    for p in [target, *(Path(s).resolve() for s, _ in stage)]:
        p = Path(p).resolve()
        parent = p.parent
        if parent in shared:
            loose.append(p)
        elif parent not in seen_dirs:
            seen_dirs.append(parent)
            units.append((f"specimen source folder {parent.name}", parent, []))
    if loose:
        units.append(("specimen file", Path("/"), loose))
    return units


def _take(unit: Tuple[str, Path, List[Path]]) -> Dict[str, Tuple]:
    _label, folder, files = unit
    return _snapshot_files(files) if files else _snapshot(folder)


def _diff(label: str, before: Dict[str, Tuple], after: Dict[str, Tuple]) -> List[str]:
    out = []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name), after.get(name)
        if a == b:
            continue
        if a is None:
            out.append(f"{label}: new {b[0]} {name}")
        elif b is None:
            out.append(f"{label}: {a[0]} {name} was removed")
        elif a[0] == "dir" and b[0] == "dir":
            continue
        else:
            out.append(f"{label}: {name} was changed")
    return out


def _signal_name(rc: int) -> str:
    try:
        return signal.Signals(-rc).name
    except (ValueError, TypeError):
        return f"signal {-rc}"


# ------------------------------------------------------------------- running

def run_in_sandbox(mode: str, target: Path, *, probe: str = "", stage: Sequence[Tuple[Path, str]] = (),
                   timeout: Optional[float] = None, python: Optional[str] = None,
                   deadline: Optional[float] = None) -> SandboxResult:
    """Run one specimen under the guard.

    mode "script": run the file as ``__main__`` and report its exit code.
    mode "probe":  import the file and run a named probe (see RUNNER).
    stage: extra files copied beside the target first, as (source, new name).
           When given, ``target`` is also copied, so the specimen finds its
           siblings next to it exactly as its own code expects.
    timeout: seconds for this run (default 120, stretched on a loaded machine;
           ASSAY_CLAIM_TIMEOUT overrides it exactly).
    deadline: a ``time.monotonic()`` value after which no more time is left in
           the overall budget (ASSAY_TOTAL_TIMEOUT); the run is cut short there.
    """
    global _SWEPT
    if not _SWEPT:
        _SWEPT = True
        sweep_abandoned()
    limit, source = effective_timeout(timeout)
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return SandboxResult(False, -1, "", "", True, detail=(
                f"timed out: the overall time budget of {total_budget():g}s ({ENV_TOTAL_TIMEOUT}) was used "
                "up before this run could start"), killed_by="overall budget", network_isolated=False)
        if remaining < limit:
            limit, source = remaining, f"what was left of the overall {total_budget():g}s budget ({ENV_TOTAL_TIMEOUT})"
    run = tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=tempfile.gettempdir())
    with _ACTIVE_LOCK:
        _ACTIVE[run] = None
    proc: Optional[subprocess.Popen] = None
    try:
        return _run(run, mode, Path(target), probe, stage, limit, source, python)
    finally:
        with _ACTIVE_LOCK:
            proc = _ACTIVE.pop(run, None)
        _kill_group(proc)
        _rmtree(run)


def _run(run: str, mode: str, target: Path, probe: str, stage: Sequence[Tuple[Path, str]],
         limit: float, limit_source: str, python: Optional[str]) -> SandboxResult:
    os.chmod(run, 0o700)
    scratch = os.path.join(run, _SCRATCH_NAME)
    ctl = os.path.join(run, _CTL_NAME)
    os.mkdir(scratch, 0o700)
    os.mkdir(os.path.join(scratch, "tmp"), 0o700)   # the specimen's private temp parent
    os.mkdir(ctl, 0o700)
    Path(run, _OWNER_NAME).write_text(str(os.getpid()), encoding="utf-8")
    target = target.resolve()
    watched = _watch_units(target, stage)
    before = [_take(u) for u in watched]
    run_path = target
    if stage:
        for src, name in stage:
            shutil.copyfile(Path(src).resolve(), Path(scratch) / name)
        shutil.copyfile(target, Path(scratch) / target.name)
        run_path = Path(scratch) / target.name
    home = os.path.expanduser("~") if os.environ.get("HOME") else ""
    try:
        import pwd
        home = pwd.getpwuid(os.getuid()).pw_dir or home
    except Exception:
        pass
    deny = list(_PROTECTED_ABSOLUTE)
    for base in {home, os.environ.get("HOME", "")}:
        if base and base != "/":
            deny += [os.path.join(base, part) for part in _PROTECTED_HOME_PARTS]
    request = {"mode": mode, "probe": probe, "path": str(run_path), "scratch": scratch, "ctl": ctl,
               "result_name": _RESULT_NAME, "runner": os.path.join(ctl, _RUNNER_NAME), "deny_read": deny}
    Path(ctl, _REQUEST_NAME).write_text(json.dumps(request), encoding="utf-8")
    Path(ctl, _HARNESS_NAME).write_text(HARNESS, encoding="utf-8")
    Path(ctl, _RUNNER_NAME).write_text(RUNNER, encoding="utf-8")
    nonce = secrets.token_hex(16)
    iso = network_isolation_command()
    cmd = iso + [python or sys.executable, "-I", "-B", os.path.join(ctl, _HARNESS_NAME),
                 os.path.join(ctl, _REQUEST_NAME)]
    cap = output_cap()
    cpu_seconds = int(limit) + 10
    nproc = _count_user_processes() + 64
    proc = subprocess.Popen(cmd, cwd=scratch, env=clean_environment(scratch), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=_limits(cpu_seconds, nproc), start_new_session=True)
    with _ACTIVE_LOCK:
        _ACTIVE[run] = proc
    try:
        proc.stdin.write((nonce + "\n").encode("ascii"))
        proc.stdin.close()
    except OSError:
        pass
    bufs: Dict[str, bytearray] = {"out": bytearray(), "err": bytearray()}
    overflow = threading.Event()

    def drain(stream, key):
        total = 0
        try:
            while True:
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    break
                total += len(chunk)
                room = cap - len(bufs[key])
                if room > 0:
                    bufs[key].extend(chunk[:room])
                if total > cap:
                    overflow.set()
        except (OSError, ValueError):
            pass

    threads = [threading.Thread(target=drain, args=(proc.stdout, "out"), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, "err"), daemon=True)]
    for t in threads:
        t.start()
    started = time.monotonic()
    timed_out, killed_by = False, ""
    while True:
        try:
            proc.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            pass
        if overflow.is_set():
            killed_by = "output"
            _kill_group(proc)
            proc.wait()
            break
        if time.monotonic() - started > limit:
            timed_out, killed_by = True, "timeout"
            _kill_group(proc)
            proc.wait()
            break
    _kill_group(proc)       # anything the specimen left behind in the session
    for t in threads:
        t.join(timeout=2)
    rc = proc.returncode
    out = bytes(bufs["out"]).decode("utf-8", "replace")
    err = bytes(bufs["err"]).decode("utf-8", "replace")

    report, forged, detail = _read_result(ctl, nonce, timed_out, overflow.is_set())
    violations = list(report.get("violations", []))
    for line in err.splitlines():
        if line.startswith("ASSAY-SANDBOX-VIOLATION: "):
            text = line[len("ASSAY-SANDBOX-VIOLATION: "):]
            if text not in violations:
                violations.append(text)
    stray: List[str] = []
    for unit, snap in zip(watched, before):
        stray += _diff(unit[0], snap, _take(unit))
    for name in sorted(set(os.listdir(run)) - _RUN_EXPECTED):
        stray.append(f"sandbox run directory: unexpected {name}")
    for name in sorted(set(os.listdir(ctl)) - _CTL_EXPECTED):
        stray.append(f"sandbox control directory: unexpected {name}")
    err_info = report.get("error") or {}
    missing = None
    if err_info.get("type") == "ModuleNotFoundError":
        root_name = (err_info.get("module_name") or "").split(".")[0]
        if root_name in OPTIONAL_DEPENDENCIES:
            missing = root_name
    exit_code = report.get("exit_code", rc if rc is not None else -1)
    if timed_out:
        load1 = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
        detail = (f"timed out after {limit:g}s of wall-clock time (limit: {limit_source}; machine load "
                  f"{load1:.1f} on {os.cpu_count() or 1} CPU(s)). On a busy machine this can mean the machine "
                  f"was slow rather than that the specimen changed")
    elif killed_by == "output":
        detail = (f"the specimen wrote more than {cap} bytes to stdout or stderr and was killed "
                  f"(cap: {ENV_OUTPUT_CAP}={cap})")
    elif not report and not detail:
        detail = _died_text(rc, err)
    completed = bool(report) and not timed_out and not forged and not overflow.is_set()
    return SandboxResult(completed=completed, exit_code=exit_code, stdout=out, stderr=err, timed_out=timed_out,
                         probe=report, violations=violations, stray_files=stray,
                         network_isolated=bool(iso), missing_dependency=missing, detail=detail,
                         output_overflow=overflow.is_set(), forged=forged,
                         killed_by=killed_by, timeout_used=limit, timeout_source=limit_source)


def _died_text(rc: Optional[int], err: str) -> str:
    how = f"signal {_signal_name(rc)}" if rc is not None and rc < 0 else f"exit {rc}"
    extra = {"SIGXCPU": " (CPU-time limit exceeded)", "SIGXFSZ": " (file-size limit exceeded)",
             "SIGKILL": " (killed, possibly out of memory)"}.get(_signal_name(rc) if rc is not None and rc < 0 else "", "")
    return f"the child process died before reporting ({how}{extra}): {err.strip()[-300:]}"


def _read_result(ctl: str, nonce: str, timed_out: bool, overflow: bool) -> Tuple[Dict, bool, str]:
    """(facts, forged?, explanation). Only the harness's own file, carrying the nonce, is believed."""
    path = os.path.join(ctl, _RESULT_NAME)
    if not os.path.isfile(path) or os.path.islink(path):
        return {}, False, ""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            body = json.loads(fh.read(4 * 1024 * 1024))
    except (OSError, ValueError):
        return {}, True, "the result file is not valid JSON, so it did not come from the sandbox harness"
    if not isinstance(body, dict) or not secrets.compare_digest(str(body.get("nonce", "")), nonce):
        return {}, True, ("the result file does not carry this run's secret nonce, so it was not written by "
                          "the sandbox harness (a specimen forging its own result)")
    facts = body.get("facts")
    rc = body.get("runner_rc")
    if not isinstance(facts, dict):
        if timed_out or overflow:
            return {}, False, ""
        detail = body.get("problem") or ""
        text = (_died_text(rc, "") if isinstance(rc, int) else "the runner gave no report")
        return {}, False, (detail + "; " if detail else "") + "the specimen process ended without giving the " \
            "harness a report (for example it called os._exit before the probe finished): " + text
    if isinstance(rc, int) and facts.get("exit_code") != rc:
        return {}, True, (f"the report says the specimen exited {facts.get('exit_code')} but the process "
                          f"exited {rc}; the report cannot be trusted")
    return facts, False, ""


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
