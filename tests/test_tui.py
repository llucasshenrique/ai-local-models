"""Drives the real curses TUI in a pseudo-terminal: arrows, mouse (tabs, buttons, rows, wheel) and prompts.
No GPU, no ollama, throw-away results dir and config copy: python3 -m unittest discover -s tests"""
import fcntl, os, pty, re, select, shutil, struct, subprocess, sys, tempfile, termios, time, unittest
from llmeval import tui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS, COLS = 32, 120
UP, DOWN, RIGHT, LEFT = b"\x1bOA", b"\x1bOB", b"\x1bOC", b"\x1bOD"     # application-cursor mode, as curses enables it (smkx)

def mouse(x, y, press=True):        # SGR mouse report, 1-based; x/y given 0-based like curses
    return f"\x1b[<0;{x + 1};{y + 1}{'M' if press else 'm'}".encode()

class Session:
    def __init__(self, env, cfg):
        self.buf = b""; self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(ROOT); os.execvpe(sys.executable, [sys.executable, "-m", "llmeval", "-c", cfg, "tui"], env)
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0)); self.pump(1.0)
    def pump(self, t=0.5):
        end = time.time() + t
        while time.time() < end:
            if select.select([self.fd], [], [], 0.1)[0]:
                try: self.buf += os.read(self.fd, 65536)
                except OSError: return
    def send(self, data, wait=0.5): os.write(self.fd, data); self.pump(wait)
    def click(self, x, y, wait=0.5):
        # like a human: press and release are separate events, tens of ms apart (ncurses drops events written back-to-back)
        self.send(mouse(x, y), 0.15); self.send(mouse(x, y, False), wait)
    def text(self): return re.sub(rb"\x1b\[[0-9;?<>]*[A-Za-z]|\x1b[()=><][A-Z0-9]?", b"", self.buf).decode(errors="ignore")
    def quit(self):
        self.send(b"q", 1.0); _, status = os.waitpid(self.pid, 0); return os.waitstatus_to_exitcode(status)

class TuiTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(); self.addCleanup(shutil.rmtree, self.d, True)
        self.cfg = os.path.join(self.d, "test.toml"); shutil.copy(os.path.join(ROOT, "evals", "default.toml"), self.cfg)   # the TUI edits this copy
        self.env = {**os.environ, "LLMEVAL_RESULTS": self.d, "TERM": "xterm-256color", "PYTHONPATH": ROOT,
                    "LLMEVAL_FAKE_MODELS": "aaa:1b=1,bbb:1b=2,ccc:1b=3"}
        subprocess.run([sys.executable, "-m", "llmeval", "demo-data"], env=self.env, cwd=ROOT, check=True, capture_output=True)
        self.s = Session(self.env, self.cfg)

    def conf(self): return open(self.cfg).read()

    def test_keyboard_tabs_arrows_and_prompt(self):
        s = self.s
        for k in (b"3", b"5", b"s", b"6", b"1"): s.send(k, 1.5 if k == b"s" else 0.6)
        s.send(b"4", 1.0); s.send(DOWN); s.send(b"\r", 1.5)                 # arrow down to bbb:1b, Enter adds it
        self.assertIn('tag = "bbb:1b"', self.conf())
        for chunk in (b"n", b"zzz:1b\r", b"ornith:9b-q4_K_M\r", b"8192\r"): s.send(chunk)
        s.pump(2.0)
        self.assertIn('tag  = "zzz:1b"', self.conf()); self.assertIn("num_ctx = 8192", self.conf())
        s.send(RIGHT); s.send(LEFT); s.send(UP)                             # tab/arrow keys must not crash
        t = s.text()
        for needle in ("Harness", "Context fit", "Installed ollama models", "OK   01-add-function", "added: bbb:1b"): self.assertIn(needle, t, needle)
        self.assertEqual(s.quit(), 0)

    def test_mouse_tabs_buttons_rows_and_wheel(self):
        s = self.s; spans = tui.tab_spans()
        s.click(spans[3][0] + 2, 0, 1.0)                                    # click the "4 Models" tab
        self.assertIn("[4]Models", s.text())
        y0 = 1 + tui.MODELS_HEADER                                          # first model row
        s.click(10, y0 + 2); s.click(10, y0 + 2, 1.5)                       # select ccc:1b, click again to add
        self.assertIn('tag = "ccc:1b"', self.conf())
        s.click(spans[4][0] + 2, 0, 1.0)                                    # "5 Setup" tab
        s.click(3, ROWS - 2, 2.0)                                           # the "Selfcheck (s)" button
        self.assertIn("OK   01-add-function", s.text())
        s.send(b"\x1b[<64;5;10M"); s.send(b"\x1b[<65;5;10M")               # wheel up / down
        s.click(spans[0][0] + 2, 0, 1.0)                                    # back to Results
        self.assertIn("[1]Results", s.text())
        self.assertEqual(s.quit(), 0)

if __name__ == "__main__": unittest.main()
