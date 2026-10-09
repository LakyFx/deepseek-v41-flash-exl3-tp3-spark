"""CPU checks for installation contracts and public benchmark inputs."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from tools.model import ranges
from tools.launch import build
from tools.gateway import policy
from runner.corpus import load_workloads
from runner.fixtures import initial_body, make_tasks, GOLDEN_CODE
from runner.fixture_eval import evaluate
from runner.metrics import parse, compare, COUNTERS
from runner.profiles import body_for_profile
from runner.suite import Suite


class RecipeTests(unittest.TestCase):
    def test_engram_rank_ranges_match_live_source(self):
        path = ROOT / 'runtime/rootfs/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py'
        tree = ast.parse(path.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_is_prime', 'find_next_prime')]
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
        config = json.loads((ROOT / 'config/model-config.reference.json').read_text())
        c, seen = config['text_config'], set()
        for layer, count in zip(c['engram_layer_ids'], c['engram_num_embeddings']):
            sizes = []
            for _ in range(c['engram_max_ngram_size'] - 1):
                current = c['engram_vocab_size'] - 1
                for _ in range(c['engram_n_heads']):
                    current = namespace['find_next_prime'](current, seen)
                    seen.add(current)
                    sizes.append(current)
            self.assertEqual(sum(sizes), count)
            for rank in range(3):
                self.assertEqual(ranges(config, rank)[str(layer)], [sum(sizes[:rank*8]), sum(sizes[:(rank+1)*8])])

    def test_launcher_preserves_measured_profile_and_worker_flags(self):
        config = json.loads((ROOT / 'config/cluster.example.json').read_text())
        config['roce_cidr'] = '192.0.2.0/24'
        for rank, row in enumerate(config['nodes']):
            row['control_address'] = f'192.0.2.{rank+1}'
            row['control_interface'] = 'management'
        for rank in range(3):
            command = build(config, rank, 'a'*64)
            env = [command[i+1] for i, x in enumerate(command) if x == '--env']
            self.assertIn('DSV41_ENGRAM_DISK_THREADS=32', env)
            self.assertIn('DGX_CAMPAIGN_R_AOT_SHA256='+'a'*64, env)
            self.assertIn('DGX_V089_VERSION=0.8.9X11c', env)
            self.assertIn('DGX_X11_PREFETCH_MIN_ROWS=12', env)
            self.assertEqual('--headless' in command, rank != 0)
            self.assertNotIn('--enable-expert-parallel', command)
            spec = json.loads(command[command.index('--speculative-config')+1])
            self.assertEqual(spec['num_speculative_tokens'], 5)
            self.assertTrue(spec['enable_adaptive_verification'])
            self.assertEqual(command[command.index('--kv-cache-memory-bytes')+1], '8074035200')
            self.assertNotIn('dgx_campaign_runtime.control.CampaignControl', command)

    def test_gateway_forces_both_reasoning_fields(self):
        plain = {'messages': [{'role': 'user', 'content': 'hello'}], 'reasoning_effort': 'none',
                 'chat_template_kwargs': {'thinking': False}}
        image = {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]}
        for body, expected in [(plain, 'max'), (image, 'high')]:
            result = policy(copy.deepcopy(body))
            self.assertEqual(result['reasoning_effort'], expected)
            self.assertEqual(result['chat_template_kwargs']['reasoning_effort'], expected)
            self.assertTrue(result['chat_template_kwargs']['thinking'])
            self.assertTrue(result['chat_template_kwargs']['enable_thinking'])

    def test_none_medium_max_native_mapping(self):
        for profile, expected in [('none', 'none'), ('medium', 'high'), ('max', 'max')]:
            value = body_for_profile({'messages': []}, profile)
            self.assertEqual(value['reasoning_effort'], expected)
            self.assertEqual(value['chat_template_kwargs']['reasoning_effort'], expected)
            self.assertEqual(value['chat_template_kwargs']['thinking'], profile != 'none')

    def test_public_corpus_hashes_and_authored_workflows_match(self):
        rows = load_workloads(ROOT / 'benchmarks/corpus/workloads/corpus.json')
        self.assertEqual(len(rows), 16)
        for task in make_tasks():
            self.assertEqual(initial_body(task, context=rows[task.name]['context']), rows[task.name]['body'])
        suite = Suite(None, {}, rows, campaign='offline', version='0.8.9X11c',
                      monitor=None, save=None, drain=None, cold_reset=None)
        self.assertEqual(suite.allocation['hard_seconds'], 1200)
        self.assertEqual(len(suite.allocation['phases']), 7)

    def test_code_fixture_scores_semantics_and_rejects_unsafe_syntax(self):
        self.assertTrue(evaluate(GOLDEN_CODE)['passed'])
        self.assertFalse(evaluate('import os\ndef merge_events(events): return events')['passed'])

    def test_metrics_reject_counter_reset_and_unknown_coverage(self):
        before = {k: 0.0 for k in COUNTERS}
        before.update(num_requests_running=0.0, num_requests_waiting=0.0)
        after = dict(before, request_generation_tokens_count=1.0, request_generation_tokens_sum=100.0,
                     request_decode_time_seconds_sum=2.0)
        value = compare(before, after, expected_requests=1, same_generation=True, exclusive=True)
        self.assertTrue(value['coverage_verified'])
        self.assertEqual(value['weighted_request_decode_tokens_per_second'], 50.0)
        partial = compare(before, after, expected_requests=2, same_generation=True, exclusive=True)
        self.assertFalse(partial['coverage_verified'])
        self.assertIsNone(partial['weighted_request_decode_tokens_per_second'])
        reset = dict(after, request_generation_tokens_sum=-1.0)
        with self.assertRaises(ValueError):
            compare(before, reset, expected_requests=1, same_generation=True, exclusive=True)


if __name__ == '__main__':
    unittest.main()
