"""Exercise resumed downloads and corruption rejection without Docker or a model."""
from pathlib import Path
import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tools.fetch_base import fetch, verify


class DownloadTests(unittest.TestCase):
    def test_resume_and_reject_corrupt_retained_file(self):
        data = bytes(range(256)) * 1024
        item = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                offset = int(self.headers['Range'].removeprefix('bytes=').removesuffix('-'))
                seen.append(offset)
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {offset}-{len(data)-1}/{len(data)}')
                self.end_headers()
                self.wfile.write(data[offset:])
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                path = Path(directory) / 'part'
                path.with_name('part.partial').write_bytes(data[:4137])
                fetch(f'http://127.0.0.1:{server.server_port}/part', path, item)
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(seen, [4137])
                fetch('http://invalid.example/', path, item)
                path.write_bytes(b'X' * len(data))
                with self.assertRaises(ValueError):
                    verify(path, item)
                self.assertEqual(path.read_bytes(), b'X' * len(data))
            finally:
                server.shutdown()
                thread.join()


if __name__ == '__main__':
    unittest.main()
