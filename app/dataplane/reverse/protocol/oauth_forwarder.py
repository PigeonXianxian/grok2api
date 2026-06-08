"""OAuth 回调接收器 — 模仿 CPA 方案

监听 127.0.0.1:56121，收到 OAuth 回调后：
1. 将 code+state 保存到文件（类似 CPA 的 .oauth-xai-*.oauth）
2. 尝试 302 重定向到 grokapi server（如果 server 在线则自动完成交换）
"""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.platform.logging.logger import logger

FORWARDER_PORT = 56121
FORWARDER_HOST = "127.0.0.1"
SAVE_DIR = Path(__file__).resolve().parents[4] / "data" / "oauth_codes"

_forwarder: HTTPServer | None = None
_forwarder_thread: threading.Thread | None = None


class CallbackHandler(BaseHTTPRequestHandler):
    """接收 OAuth 回调，保存 code + 尝试转发。"""

    def do_GET(self):
        qs = parse_qs(urlparse(self.path).query)
        code = qs.get("code", [""])[0]
        state = qs.get("state", [""])[0]
        error = qs.get("error", [""])[0]

        if error:
            self._respond(400, f"<h1>Error</h1><p>{error}</p>")
            return

        if code:
            # 保存到文件
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = SAVE_DIR / f"oauth_{state[:16]}.json"
            data = {"code": code, "state": state, "saved": True}
            save_path.write_text(json.dumps(data))
            logger.info("oauth code saved: path={} code={}...", save_path, code[:20])

            # 尝试重定向到 server
            target = f"http://127.0.0.1:8000/admin/api/xai/oauth/callback-redirect?code={code}&state={state}"
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        self._respond(400, "<h1>Missing code</h1>")

    def _respond(self, status: int, body: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):
        logger.debug("oauth forwarder: {}", format % args)


def start_forwarder() -> bool:
    global _forwarder, _forwarder_thread
    if _forwarder is not None:
        return True
    try:
        _forwarder = HTTPServer((FORWARDER_HOST, FORWARDER_PORT), CallbackHandler)
        _forwarder_thread = threading.Thread(target=_forwarder.serve_forever, daemon=True)
        _forwarder_thread.start()
        logger.info("oauth forwarder started on http://{}:{}", FORWARDER_HOST, FORWARDER_PORT)
        return True
    except OSError as exc:
        logger.warning("oauth forwarder failed: {}", exc)
        return False


def stop_forwarder() -> None:
    global _forwarder
    if _forwarder is not None:
        _forwarder.shutdown()
        _forwarder = None
        logger.info("oauth forwarder stopped")


def get_latest_code() -> dict | None:
    """获取最近保存的 OAuth code。"""
    if not SAVE_DIR.exists():
        return None
    files = sorted(SAVE_DIR.glob("oauth_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    if files:
        return json.loads(files[0].read_text())
    return None


__all__ = ["start_forwarder", "stop_forwarder", "get_latest_code", "FORWARDER_PORT"]
