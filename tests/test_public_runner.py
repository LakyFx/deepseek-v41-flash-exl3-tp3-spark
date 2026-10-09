"""Check tokenizer-only corpus preparation over the bounded loopback lane."""
from pathlib import Path
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from run import Network, prepare_legacy
from runner.corpus import load_legacy, load_workloads


class RunnerTests(unittest.TestCase):
    def test_prepare_uses_tokenizer_without_generation_or_private_inputs(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.path, self.client_address[1], body))
                if self.path != '/tokenize':
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'count': 6000 + len(requests)}).encode())
            def log_message(self, *args):
                pass
        with tempfile.TemporaryDirectory() as directory, ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                workloads = load_workloads(ROOT / 'benchmarks/corpus/workloads/corpus.json')
                destination = Path(directory) / 'prepared'
                prepare_legacy(Network(server.server_port), workloads, destination)
                corpus = load_legacy(destination / 'corpus.json')
                self.assertEqual(len(requests), 8)
                self.assertEqual(len(corpus['code']), 4)
                self.assertEqual(len(corpus['prose']), 4)
                for path, port, body in requests:
                    self.assertEqual(path, '/tokenize')
                    self.assertTrue(20000 <= port <= 20255)
                    self.assertTrue(body['add_generation_prompt'])
                    self.assertEqual(body['chat_template_kwargs']['reasoning_effort'], 'max')
                manifest = json.loads((destination / 'corpus.json').read_text())
                self.assertFalse(manifest['historical_private_inputs_equal'])
                self.assertEqual(manifest['model_generation_requests'], 0)
            finally:
                server.shutdown()
                thread.join()


if __name__ == '__main__':
    unittest.main()
