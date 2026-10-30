import asyncio
import json
import re
import signal
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

MODEL_NAME = "ibm-granite/granite-embedding-small-english-r2"
MAX_BODY_BYTES = 1024 * 1024
MAX_INPUTS = 256

_model = None
_model_lock = threading.Lock()


def get_model():
    """Load the embedding model once, on first use."""
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME)
        return _model


def embed(texts):
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    tokens = sum(len(ids) for ids in model.tokenizer(texts)["input_ids"])
    return vectors.tolist(), tokens


class TunnelHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        asyncio.run(self.do_ASYNC())

    def do_HEAD(self):
        asyncio.run(self.do_ASYNC())

    def do_POST(self):
        asyncio.run(self.do_ASYNC())

    async def do_ASYNC(self):
        """Run supported requests in a per-request event loop."""
        if self.command == "POST" and self.path.split("?", 1)[0] == "/v1/embeddings":
            await self.handle_embeddings()
            return
        if self.command == "POST":
            if self.headers.get_all("X-Tunnel-Original-Method") != ["GET"]:
                await asyncio.to_thread(self.send_error, 501, "Unsupported method ('POST')")
                return
            lengths = self.headers.get_all("Content-Length", [])
            if ("Transfer-Encoding" in self.headers
                    or len(lengths) > 1
                    or (lengths and re.fullmatch(r"0+", lengths[0]) is None)):
                # Do not drain an untrusted body or reuse its connection.
                self.close_connection = True
                await asyncio.to_thread(self.send_error, 400, "Translated GET requires an empty body")
                return
            del self.headers["X-Tunnel-Original-Method"]
            self.command = "GET"
            # BaseHTTPRequestHandler logs requestline, not command.
            self.requestline = f"GET {self.path} {self.request_version}"
        if self.command == "GET":
            # Await the extension hook directly; do_GET would nest asyncio.run.
            await self.handle_GET()
        elif self.command == "HEAD":
            await asyncio.to_thread(super().do_HEAD)

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def json_error(self, status, message):
        self.close_connection = True
        self.send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    async def handle_embeddings(self):
        """OpenAI-compatible POST /v1/embeddings."""
        if "Transfer-Encoding" in self.headers:
            await asyncio.to_thread(self.json_error, 400, "Content-Length body required")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            await asyncio.to_thread(self.json_error, 413 if length > 0 else 400, "Invalid or oversized body")
            return
        raw = await asyncio.to_thread(self.rfile.read, length)
        try:
            inputs = json.loads(raw)["input"]
        except (ValueError, TypeError, KeyError):
            await asyncio.to_thread(self.json_error, 400, "Body must be JSON with an 'input' field")
            return
        if isinstance(inputs, str):
            inputs = [inputs]
        if (not isinstance(inputs, list) or not inputs or len(inputs) > MAX_INPUTS
                or not all(isinstance(t, str) for t in inputs)):
            await asyncio.to_thread(
                self.json_error, 400, f"'input' must be a string or a list of 1-{MAX_INPUTS} strings")
            return
        try:
            vectors, tokens = await asyncio.to_thread(embed, inputs)
        except Exception as exc:
            self.log_error("embedding failed: %r", exc)
            await asyncio.to_thread(self.json_error, 500, "Embedding failed")
            return
        await asyncio.to_thread(self.send_json, 200, {
            "object": "list",
            "model": MODEL_NAME,
            "data": [{"object": "embedding", "index": i, "embedding": v}
                     for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        })

    async def handle_GET(self):
        """Async extension hook shared by direct and restored GET requests."""
        # Keep stdlib path handling and bounded streaming copy, off the loop.
        await asyncio.to_thread(super().do_GET)


async def main():
    handler = partial(TunnelHandler, directory=sys.argv[1])
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    previous_handlers = {
        sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    with ThreadingHTTPServer(("127.0.0.1", 8000), handler) as server:
        try:
            for sig in previous_handlers:
                loop.add_signal_handler(sig, task.cancel)
            serving = asyncio.create_task(asyncio.to_thread(server.serve_forever))
            try:
                # Shield also preserves queued startup: shutdown must not wait
                # for a serve_forever call that cancellation prevented running.
                await asyncio.shield(serving)
            finally:
                async def stop():
                    # shutdown must run in a different thread from serve_forever.
                    await asyncio.to_thread(server.shutdown)
                    await serving

                stopping = asyncio.create_task(stop())
                cancelled = False
                while not stopping.done():
                    try:
                        await asyncio.shield(stopping)
                    except asyncio.CancelledError:
                        # Repeated signals must not interrupt executor cleanup.
                        cancelled = True
                stopping.result()
                if cancelled:
                    raise asyncio.CancelledError
        finally:
            for sig, previous in previous_handlers.items():
                loop.remove_signal_handler(sig)
                signal.signal(sig, previous)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
