#!/usr/bin/env python3
"""Tiny logging reverse proxy in front of ollama so every harness is measured the same way.

Counts model requests and the largest prompt (in tokens) seen, read from ollama's own counters
(`prompt_eval_count` on /api/chat, `prompt_tokens` usage on /v1). Stdlib only.
usage: proxy.py [listen_port=11436] [upstream=127.0.0.1:11434]   (stats: GET /__stats, reset: GET /__reset)
"""
import http.client, http.server, json, re, socketserver, sys, threading

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 11436
UP = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1:11434"
stats = {"requests": 0, "peak_prompt_tokens": 0}
lock = threading.Lock()
PAT = re.compile(rb'"(?:prompt_eval_count|prompt_tokens)"\s*:\s*(\d+)')

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, d):
        b = json.dumps(d).encode(); self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/__stats": return self._json(stats)
        if self.path == "/__reset":
            with lock: stats.update(requests=0, peak_prompt_tokens=0)
            return self._json(stats)
        self.fwd()
    def do_POST(self): self.fwd()
    do_DELETE = do_PUT = do_POST
    def fwd(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        c = http.client.HTTPConnection(UP, timeout=600)
        hdrs = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection", "accept-encoding")}
        c.request(self.command, self.path, body, hdrs)
        r = c.getresponse()
        if self.path.startswith(("/api/chat", "/api/generate", "/v1/chat")):
            with lock: stats["requests"] += 1
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk: break
            for m in PAT.findall(chunk):
                with lock: stats["peak_prompt_tokens"] = max(stats["peak_prompt_tokens"], int(m))
            self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n"); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n"); c.close()

class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    S(("127.0.0.1", PORT), H).serve_forever()
