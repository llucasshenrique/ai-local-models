"""Workspace isolation, execution resource bounding, and benchmark integrity protections."""
import os
import resource
import signal
import stat
import subprocess
import tempfile

def set_resource_limits(cpu_timeout=60, max_fsize_bytes=50_000_000):
    """
    Subprocess preexec hook to enforce POSIX resource limits (CPU and max file size).
    """
    def _apply():
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_timeout + 5, cpu_timeout + 10))
        except (ValueError, OSError):
            pass

        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_fsize_bytes, max_fsize_bytes))
        except (ValueError, OSError):
            pass

    return _apply

def create_isolated_workspace(task, files_override=None):
    """
    Creates a disposable temporary workspace for a trial.
    Protected evaluation files (e.g. test.sh) are set to read-only (0444).
    Source files remain writable.
    """
    d = tempfile.mkdtemp(prefix="llmeval-trial-")
    files = {**task.get("files", {}), **(files_override or {})}
    protected_set = set(task.get("protected", ["test.sh"]))

    for name, body in files.items():
        full_path = os.path.join(d, name)
        os.makedirs(os.path.dirname(full_path) or d, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(body)

        if name in protected_set:
            os.chmod(full_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH) # 0444

    subprocess.run(
        "git init -q && git add -A && git -c user.name=llmeval -c user.email=eval@local commit -qm init",
        shell=True, cwd=d, capture_output=True
    )
    return d

def run_bounded_process(cmd, cwd, timeout=60, env=None):
    """
    Executes a test command with process group isolation, timeout killer, and resource bounding.
    Returns (return_code, timed_out).
    """
    timed_out = False
    p = subprocess.Popen(
        cmd, shell=True, cwd=cwd, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
        preexec_fn=set_resource_limits(cpu_timeout=timeout)
    )
    try:
        code = p.wait(timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()
        code = 124

    return code, timed_out
