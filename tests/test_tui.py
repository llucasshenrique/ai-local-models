"""Drives the real curses TUI in a pseudo-terminal (no GPU, no ollama, throw-away results dir): python3 -m unittest discover -s tests"""
import fcntl, os, pty, re, select, struct, subprocess, sys, tempfile, termios, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TuiTest(unittest.TestCase):
    def test_tabs_render_and_quit_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "LLMEVAL_RESULTS": d, "TERM": "xterm-256color", "PYTHONPATH": ROOT}
            subprocess.run([sys.executable, "-m", "llmeval", "demo-data"], env=env, cwd=ROOT, check=True, capture_output=True)
            pid, fd = pty.fork()
            if pid == 0:
                os.chdir(ROOT); os.execvpe(sys.executable, [sys.executable, "-m", "llmeval", "tui"], env)
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 32, 120, 0, 0))
            buf = b""
            def pump(t):
                nonlocal buf
                end = time.time() + t
                while time.time() < end:
                    if select.select([fd], [], [], 0.2)[0]:
                        try: buf += os.read(fd, 65536)
                        except OSError: return
            pump(1.0)
            for k, wait in (("3", .8), ("4", .8), ("s", 2.0), ("5", .8), ("1", .8)):
                os.write(fd, k.encode()); pump(wait)
            os.write(fd, b"q"); pump(1.0)
            _, status = os.waitpid(pid, 0)
            text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][A-Z0-9]", b"", buf).decode(errors="ignore")
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            for needle in ("Harness", "ornith-agent:9b", "Context fit", "Harnesses", "OK   01-add-function"):
                self.assertIn(needle, text, needle)

if __name__ == "__main__": unittest.main()
