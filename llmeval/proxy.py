"""In-process logging reverse proxy in front of ollama: counts model requests and the largest prompt (tokens),
read from ollama's own counters (`prompt_eval_count` native, `prompt_tokens` OpenAI usage). Same numbers for every harness."""
import http.client, http.server, re, socketserver, threading
from . import ollama

PAT = re.compile(rb'"(?:prompt_eval_count|prompt_tokens)"\s*:\s*(\d+)')
CHAT = ("/api/chat", "/api/generate", "/v1/chat", "/v1/completions")

class Proxy:
    def __init__(self, port=11436):
        self.port, self.requests, self.peak, self._lock = port, 0, 0, threading.Lock()
        self.upstream = ollama.host().split("://", 1)[1]
        outer = self
        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *a): pass
            def do_GET(self): outer._fwd(self)
            do_POST = do_PUT = do_DELETE = do_HEAD = do_GET
        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True
        self.server = S(("127.0.0.1", port), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    url = property(lambda self: f"http://127.0.0.1:{self.port}")

    def reset(self):
        with self._lock: self.requests = self.peak = 0

    def stats(self):
        with self._lock: return {"llm_requests": self.requests, "peak_prompt_tokens": self.peak}

    def _fwd(self, h):
        body = h.rfile.read(int(h.headers.get("Content-Length") or 0))
        c = http.client.HTTPConnection(self.upstream, timeout=600)
        c.request(h.command, h.path, body, {k: v for k, v in h.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")})
        r = c.getresponse()
        if h.path.startswith(CHAT):
            with self._lock: self.requests += 1
        h.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): h.send_header(k, v)
        h.send_header("Transfer-Encoding", "chunked"); h.end_headers()
        while True:
            chunk = r.read1(65536)
            if not chunk: break
            for m in PAT.findall(chunk):
                with self._lock: self.peak = max(self.peak, int(m))
            h.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n"); h.wfile.flush()
        h.wfile.write(b"0\r\n\r\n"); c.close()

    def close(self):
        self.server.shutdown(); self.server.server_close()
