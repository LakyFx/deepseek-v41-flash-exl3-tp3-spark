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
                raw = self.rfile.read(int(self.headers['Content-Length']))
                body = json.loads(raw) if raw else None
                requests.append((self.path, self.client_address[1], body))
                if self.path not in ('/tokenize', '/reset_prefix_cache'):
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                payload = {'count': 6000 + len(requests)} if self.path == '/tokenize' else {'success': True}
                self.wfile.write(json.dumps(payload).encode())
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
                    self.assertTrue(0 < port < 65536)
                    self.assertTrue(body['add_generation_prompt'])
                    self.assertEqual(body['chat_template_kwargs']['reasoning_effort'], 'max')
                manifest = json.loads((destination / 'corpus.json').read_text())
                self.assertFalse(manifest['historical_private_inputs_equal'])
                self.assertEqual(manifest['model_generation_requests'], 0)
                network = Network(server.server_port)
                self.assertTrue(network.request(network.BASE, 'POST', '/reset_prefix_cache')['success'])
                self.assertTrue(20000 <= requests[-1][1] <= 20255)
            finally:
                server.shutdown()
                thread.join()


if __name__ == '__main__':
    unittest.main()
