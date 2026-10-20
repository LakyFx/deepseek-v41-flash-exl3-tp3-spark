"""Apply the exact measured 1.2.3 overlay to the pinned extracted source."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Pinned integration anchor changed: '+old[:100])
    return source.replace(old, new)

def patches(files):
    cfg='zig/src/core/lanes/config.zig'
    s=files[cfg]
    s=replace_once(s,'    exact_width: u32 = 1,','    adaptive_learning: bool = false, // 1.2.2xTF opt-in\n    exact_width: u32 = 1,')
    s=replace_once(s,'pub const Config = struct {','pub const Config = struct {\n    adaptive_learning: bool = false,')
    s=replace_once(s,'        if (max_rows < 1) return error.BadMaxRows;',
        '        if (max_rows < 1) return error.BadMaxRows;\n'
        '        if (m.adaptive_learning and (m.batch_rows > 48 or m.max_streams > 16 or m.head_trees)) return error.BadAdaptiveShape;')
    s=replace_once(s,'            .family_width = width,','            .adaptive_learning = m.adaptive_learning,\n            .family_width = width,')
    files[cfg]=s

    engine='zig/src/core/lanes/engine.zig'
    s=files[engine]
    s=replace_once(s,'const fill = @import("fill.zig");','const fill = @import("fill.zig");\nconst adaptive = @import("adaptive.zig");')
    s=replace_once(s,'    log: ?*ev.Log = null,','    log: ?*ev.Log = null,\n    learner: ?*adaptive.Controller = null,')
    s=replace_once(s,'        e.rule.deinit();','        e.rule.deinit();\n        if (e.learner) |learner| e.gpa.destroy(learner);')
    s=replace_once(s,'    /// Streams holding or about to hold a row (paused ones included).',
        '    pub fn learningController(e: *Engine) !?*adaptive.Controller {\n'
        '        if (!e.cfg.adaptive_learning) return null;\n'
        '        if (e.learner == null) {\n'
        '            const learner = try e.gpa.create(adaptive.Controller);\n'
        '            learner.* = .{};\n'
        '            e.learner = learner;\n'
        '        }\n'
        '        return e.learner;\n'
        '    }\n\n'
        '    /// Streams holding or about to hold a row (paused ones included).')
    s=replace_once(s,'        if (plans[0].kind == .head and e.cfg.node_probabilities) try win.allocate(e, &plans);',
        '        if (e.cfg.adaptive_learning or (plans[0].kind == .head and e.cfg.node_probabilities)) try win.allocate(e, &plans);')
    s=replace_once(s,'        try e.backend.verify(&windows, &out);',
        '        const verify_started = e.clock.elapsedMs(.cost);\n'
        '        try e.backend.verify(&windows, &out);\n'
        '        if (e.learner) |learner| learner.verified(e.clock.elapsedMs(.cost) - verify_started);')
    s=replace_once(s,'        try e.backend.verify(windows, out);',
        '        const verify_started = e.clock.elapsedMs(.cost);\n'
        '        try e.backend.verify(windows, out);\n'
        '        if (e.learner) |learner| learner.verified(e.clock.elapsedMs(.cost) - verify_started);')
    s=replace_once(s,'        return .{ .rows = @intCast(rows), .keep = @intCast(o.path.len), .got = o.got };',
        '        if (e.learner) |learner| learner.finished(e.clock.elapsedMs(.overhead), o.got.len);\n'
        '        return .{ .rows = @intCast(rows), .keep = @intCast(o.path.len), .got = o.got };')
    s=replace_once(s,'        try e.rule.observeOverhead(@intCast(plans.len), total, ms);',
        '        try e.rule.observeOverhead(@intCast(plans.len), total, ms);\n'
        '        if (e.learner) |learner| {\n'
        '            var emitted: usize = 0;\n'
        '            for (outcomes) |o| emitted += o.got.len;\n'
        '            learner.finished(ms, emitted);\n'
        '        }')
    s=replace_once(s,'                const kept = path.len - 1;',
        '                const kept = path.len - 1;\n'
        '                if (e.learner) |learner| {\n'
        '                    if (accept.isChain(parents) and (p.kind == .head or p.kind == .copy))\n'
        '                        learner.accepted(@intFromPtr(s), rows - 1, kept);\n'
        '                }')
    files[engine]=s

    win='zig/src/core/lanes/windows.zig'
    s=files[win]
    s=replace_once(s,'const f = ev.f;','const f = ev.f;\nconst adaptive = @import("adaptive.zig");')
    s=replace_once(s,'    const counts = try alloc.allocate(a, fixed, probs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), @max(plans.len, e.cfg.batch_rows));',
        '    const learner = try e.learningController();\n'
        '    const counts = if (learner) |model| block: {\n'
        '        const inputs = try a.alloc(adaptive.Input, plans.len);\n'
        '        for (inputs, plans, fixed, probs) |*input, p, fx, pr| input.* = .{\n'
        '            .key = @intFromPtr(p.stream), .context = p.stream.cache_len, .fixed = fx,\n'
        '            .survivals = pr, .source = if (p.kind == .copy) .copy else .head,\n'
        '            .sampled = if (p.stream.sampling) |sampling| sampling.temperature > 0 else false,\n'
        '            .warm = p.stream.rounds > 1,\n'
        '        };\n'
        '        break :block try model.allocate(a, inputs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), e.cfg.batch_rows);\n'
        '    } else try alloc.allocate(a, fixed, probs, &e.cfg.shared_costs, e.rule.overheadMs(plans.len), @max(plans.len, e.cfg.batch_rows));')
    files[win]=s

    family='zig/src/families/dsv41/lanes.zig'
    s=files[family]
    s=replace_once(s,'    fix_k: ?usize = null,','    adaptive_learning: bool = false, // active online learning, opt-in\n    fix_k: ?usize = null,')
    s=replace_once(s,'            .exact_width = window_rows,','            .adaptive_learning = self.adaptive_learning,\n            .exact_width = window_rows,')
    files[family]=s
    native='zig/src/families/dsv41/native.zig'
    s=files[native]
    s=replace_once(s,'    st.lanes = lanes_mod.Lanes.init(gpa, st.m);',
        '    st.lanes = lanes_mod.Lanes.init(gpa, st.m);\n'
        '    if (getenv("TF_DS_ADAPTIVE_LEARNING")) |value| {\n'
        '        if (!std.mem.eql(u8, value, "0") and !std.mem.eql(u8, value, "1")) return error.BadAdaptiveLearning;\n'
        '        st.lanes.adaptive_learning = std.mem.eql(u8, value, "1");\n'
        '    }')
    files[native]=s
    cli='zig/src/families/dsv41/lanes_main.zig'
    s=files[cli]
    s=replace_once(s,'    ln.fix_k = fix_k;',
        '    ln.fix_k = fix_k;\n'
        '    if (std.c.getenv("TF_DS_ADAPTIVE_LEARNING")) |value| {\n'
        '        const text = std.mem.span(value);\n'
        '        if (!std.mem.eql(u8, text, "0") and !std.mem.eql(u8, text, "1")) return error.BadAdaptiveLearning;\n'
        '        ln.adaptive_learning = std.mem.eql(u8, text, "1");\n'
        '    }')
    files[cli]=s

    api='zig/src/core/engine_api.zig'
    fields=['rounds','cost_updates','acceptance_updates','probes','rejected_times',
            'prediction_samples','prediction_abs_error_microseconds','verify_microseconds',
            'last_rows','last_streams','last_context_tokens','last_predicted_verify_microseconds','last_verify_microseconds']
    declaration='pub const AdaptiveTelemetry = struct {\n'+''.join('    '+name+': u64 = 0,\n' for name in fields)+'};\n\n'
    s=replace_once(files[api],'pub const Status = struct {',declaration+'pub const Status = struct {\n    adaptive: ?AdaptiveTelemetry = null,')
    files[api]=s
    host='zig/src/core/lane_host.zig'
    s=replace_once(files[host],'    token_pool: ?lanes.backend.TokenPool = null,',
        '    adaptive: ?api.AdaptiveTelemetry = null,\n    token_pool: ?lanes.backend.TokenPool = null,')
    s=replace_once(s,'            .prefill_telemetry = h.prefill_telemetry,',
        '            .prefill_telemetry = h.prefill_telemetry,\n            .adaptive = h.adaptive,')
    snapshot='''        if (h.core.learner) |l| h.adaptive = .{
            .rounds = l.rounds, .cost_updates = l.cost_updates,
            .acceptance_updates = l.acceptance_updates, .probes = l.probes,
            .rejected_times = l.rejected_times, .prediction_samples = l.prediction_samples,
            .prediction_abs_error_microseconds = @intFromFloat(@min(1.0e18, l.prediction_error_ms * 1000)),
            .verify_microseconds = @intFromFloat(@min(1.0e18, l.observed_verify_ms * 1000)),
            .last_rows = l.frame.rows, .last_streams = l.frame.count,
            .last_context_tokens = l.frame.context,
            .last_predicted_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0,l.frame.predicted_ms * 1000))),
            .last_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0,(l.frame.verify_ms orelse 0) * 1000))),
        };
'''
    s=replace_once(s,'        h.token_pool = pool;','        h.token_pool = pool;\n'+snapshot)
    files[host]=s
    metrics='zig/src/server/metrics.zig'
    counters={'rounds','cost_updates','acceptance_updates','probes','rejected_times','prediction_samples','prediction_abs_error_microseconds','verify_microseconds'}
    lines='        if (status.adaptive) |adaptive| {\n'
    for name in fields:
        metric='adaptive_'+name+('_total' if name in counters else '')
        kind='counter' if name in counters else 'gauge'
        lines+=f'            try gauge(w, "{metric}", "{kind}", "Adaptive speculation {name}.", adaptive.{name});\n'
    lines+='        }\n'
    files[metrics]=replace_once(files[metrics],'        if (status.token_pool) |p| {',lines+'        if (status.token_pool) |p| {')
    files['zig/src/core/lanes/engine_test.zig'] += (HERE/'engine_test_append.zig').read_text(encoding='utf-8')
    return files

def verify(source):
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['patched_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Adaptive source mismatch: '+name)
    return manifest


def apply(source):
    manifest = json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    files = {}
    for name, expected in manifest['original_sha256'].items():
        raw = (source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Require pristine pinned source: '+name)
        files[name] = raw.decode('utf-8')
    for name, expected in manifest['overlay_sha256'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Overlay drift: '+name)
    updated = patches(files)
    for name in ['adaptive.zig','overhead_check.zig']:
        updated['zig/src/core/lanes/'+name] = (HERE/name).read_text(encoding='utf-8')
    # Verify all proposed bytes before changing the extracted build directory.
    for name, text in updated.items():
        if hashlib.sha256(text.encode('utf-8')).hexdigest() != manifest['patched_sha256'][name]:
            raise ValueError('Generated integration differs from measured source: '+name)
    for name, text in updated.items():
        (source/name).write_text(text,encoding='utf-8',newline='\n')
    verify(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--check',action='store_true')
    args = parser.parse_args()
    if args.check: verify(args.source)
    else: apply(args.source)
    print(json.dumps(dict(status='ADAPTIVE_SOURCE_VERIFIED',version='1.2.3',services_touched=False)))


if __name__ == '__main__':main()
