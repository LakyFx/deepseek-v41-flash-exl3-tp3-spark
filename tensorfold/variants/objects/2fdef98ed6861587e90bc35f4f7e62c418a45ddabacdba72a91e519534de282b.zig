//! 1.2.2xTF: online cost and acceptance learning for verified chain/copy lanes.
//! Inspired by AdaSpark (Liu et al., arXiv:2610.05774); no model-weight training.
//! This module only selects rows. The existing target verifier remains authoritative.
const std = @import("std");
const RefreshBudget = @import("refresh_budget.zig").Budget;
const Table = @import("table.zig").Table;
const Allocator = std.mem.Allocator;

pub const max_rows = 48;
pub const max_streams = 16;
const context_bands = 7; // <=8K,16K,32K,64K,128K,256K, larger
const load_bands = 6; // 1,2,3..4,5..6,7..8,9..16 active streams
const modes = 2; // greedy and sampled: different host sampling costs/acceptance
const bank_size = 16;
const cells = modes * load_bands * context_bands * (max_rows + 1);
const groups = modes * load_bands * context_bands;
pub const Source = enum(u1) { head, copy };

fn clip(x: f64, lo: f64, hi: f64) f64 {
    return @min(@max(x, lo), hi);
}
fn probability(x: f64) f64 {
    return if (std.math.isFinite(x)) clip(x, 0.0001, 0.9999) else 0.5;
}
pub fn contextBand(context: u64) usize {
    var end: u64 = 8192;
    var band: usize = 0;
    while (band + 1 < context_bands and context > end) : (band += 1) end *= 2;
    return band;
}
fn loadBand(streams: usize) usize {
    return if (streams <= 1) 0 else if (streams == 2) 1 else if (streams <= 4) 2 else if (streams <= 6) 3 else if (streams <= 8) 4 else 5;
}
fn group(streams: usize, context: u64, sampled: bool) usize {
    return (@as(usize, @intFromBool(sampled)) * load_bands + loadBand(streams)) * context_bands + contextBand(context);
}

const CostCell = struct {
    values: [bank_size]f64 = @splat(0),
    seen: u32 = 0,
    count: u8 = 0,
    next: u8 = 0,
    median: f64 = 1,

    fn add(c: *CostCell, measured_ratio: f64) void {
        c.seen +|= 1;
        // First sightings can include graph capture/shape initialization.
        if (c.seen <= 2) return;
        c.values[c.next] = measured_ratio;
        c.next = @intCast((@as(usize, c.next) + 1) % bank_size);
        c.count = @intCast(@min(bank_size, @as(usize, c.count) + 1));
        var sorted = c.values;
        std.mem.sort(f64, sorted[0..c.count], {}, std.sort.asc(f64));
        // MoE routing is real variation, not just one-sided interference. Use a
        // median rather than treating every slower expert combination as noise.
        const n: usize = c.count;
        c.median = if (n % 2 == 0) (sorted[n / 2 - 1] + sorted[n / 2]) / 2 else sorted[n / 2];
    }
    fn ratio(c: *const CostCell, prior: f64) f64 {
        // A baseline prior protects sparse buckets; old samples age out naturally.
        const trust = @as(f64, @floatFromInt(c.count)) / (@as(f64, @floatFromInt(c.count)) + 8.0);
        return prior + trust * (c.median - prior);
    }
};

const Average = struct {
    value: f64 = 0,
    count: u32 = 0,
    fn add(a: *Average, value: f64, rate: f64) void {
        if (!std.math.isFinite(value) or value < 0) return;
        a.value = if (a.count == 0) value else a.value + rate * (value - a.value);
        a.count +|= 1;
    }
};
const Rate = struct {
    reward: Average=.{},
    duration: Average=.{},
    fn add(r: *Rate, emitted: usize, ms: f64) void {
        r.reward.add(@floatFromInt(emitted),0.025);
        r.duration.add(ms,0.025);
    }
    fn price(r: *const Rate) ?f64 {
        // Average reward per elapsed time, not the mean of per-round rates.
        // Fast short rounds must not outweigh slow rounds of equal reward.
        return if(r.duration.count>=16 and r.duration.value>0 and r.reward.value>0)
            r.reward.value/r.duration.value else null;
    }
};
const Calibration = struct {
    trials: f64 = 0,
    successes: f64 = 0,
    last_round: u64 = 0,
    fn predict(c: *const Calibration, prior: f64, round: u64) f64 {
        // A suppressed draft must be able to recover on a changed workload.
        // Unobserved buckets gradually return to the current head's confidence.
        const age = round -| c.last_round;
        const retain = 1.0 / (1.0 + @as(f64, @floatFromInt(age)) / 512.0);
        return probability((32.0 * prior + c.successes * retain) / (32.0 + c.trials * retain));
    }
    fn add(c: *Calibration, hit: bool, round: u64) void {
        if (c.trials >= 128) {
            c.trials *= 0.5;
            c.successes *= 0.5;
        }
        c.trials += 1;
        c.successes += @floatFromInt(@intFromBool(hit));
        c.last_round = round;
    }
};
const Guard = struct {
    raw: Average = .{},
    learned: Average = .{},
    fn useLearned(g: *const Guard) bool {
        return g.raw.count >= 32 and g.learned.value <= g.raw.value * 1.01;
    }
    fn observe(g: *Guard, raw: f64, learned: f64, hit: bool) void {
        const x: f64 = @floatFromInt(@intFromBool(hit));
        g.raw.add(-x * @log(probability(raw)) - (1 - x) * @log(1 - probability(raw)), 0.025);
        g.learned.add(-x * @log(probability(learned)) - (1 - x) * @log(1 - probability(learned)), 0.025);
    }
};
const cal_cells = modes * 2 * 16 * 8;
fn calIndex(sampled: bool, source: Source, depth: usize, prior: f64) usize {
    const bin: usize = @intFromFloat(@min(7, @floor(probability(prior) * 8)));
    return (((@as(usize, @intFromBool(sampled)) * 2 + @intFromEnum(source)) * 16 + @min(depth, 15)) * 8) + bin;
}
fn guardIndex(sampled: bool, source: Source) usize {
    return @as(usize, @intFromBool(sampled)) * 2 + @intFromEnum(source);
}

pub const Input = struct {
    key: usize,
    context: u64,
    fixed: u32 = 1,
    survivals: []const f64,
    source: Source = .head,
    sampled: bool = false,
    warm: bool = true,
};
const FrameStream = struct {
    key: usize = 0,
    raw: [max_rows]f64 = @splat(0),
    offered: usize = 0,
    granted: u32 = 0,
    source: Source = .head,
    sampled: bool = false,
};
const Frame = struct {
    streams: [max_streams]FrameStream = @splat(.{}),
    count: usize = 0,
    rows: u32 = 0,
    context: u64 = 0,
    sampled: bool = false,
    warm: bool = false,
    verify_ms: ?f64 = null,
    base_ms: f64 = 0,
    predicted_ms: f64 = 0,
};

pub const Controller = struct {
    refresh: RefreshBudget = .{},
    exploring: bool = false,
    costs: [cells]CostCell = @splat(.{}),
    group_costs: [groups]CostCell = @splat(.{}),
    overhead: [groups]Average = @splat(.{}),
    decode_rate: [groups]Rate = @splat(.{}),
    incumbent: [groups]u32 = @splat(0),
    calibration: [cal_cells]Calibration = @splat(.{}),
    guards: [modes * 2]Guard = @splat(.{}),
    frame: Frame = .{},
    rounds: u64 = 0,
    cost_updates: u64 = 0,
    acceptance_updates: u64 = 0,
    probes: u64 = 0,
    rejected_times: u64 = 0,
    prediction_samples: u64 = 0,
    prediction_error_ms: f64 = 0,
    observed_verify_ms: f64 = 0,

    pub fn predictedCost(c: *const Controller, rows: u32, streams: usize, context: u64, sampled: bool, baseline: *const Table) f64 {
        const base = baseline.get(rows) orelse return 1.0e9;
        if (rows == 0 or rows > max_rows or !std.math.isFinite(base) or base <= 0) return 1.0e9;
        const at = group(streams, context, sampled);
        const cell = &c.costs[at * (max_rows + 1) + rows];
        // A shared correction within this workload bucket protects unmeasured
        // widths from looking artificially cheap after a context/latency shift.
        // Exact-width observations then refine that prior; never borrow across
        // concurrency, context or sampling buckets.
        return base * clip(cell.ratio(c.group_costs[at].ratio(1)), 0.5, 3.0);
    }

    /// All candidate prefixes compete under the current cost and acceptance fit.
    /// Returns allocated draft counts; caller owns the slice.
    pub fn allocate(c: *Controller, gpa: Allocator, inputs: []const Input, baseline: *const Table, fallback_overhead: f64, cap: u32) ![]u32 {
        if (inputs.len == 0 or inputs.len > max_streams or cap == 0 or cap > max_rows) return error.BadAdaptiveShape;
        var probabilities: [max_streams][max_rows]f64 = undefined;
        var counts: [max_streams]u32 = @splat(0);
        var best_counts = counts;
        var rows: u32 = 0;
        var weighted_context: f64 = 0;
        var sampled = false;
        var warm = true;
        c.exploring = false;
        c.frame = .{ .count = inputs.len };
        for (inputs, 0..) |input, i| {
            if (input.fixed == 0 or input.survivals.len >= max_rows) return error.BadAdaptiveShape;
            rows += input.fixed;
            weighted_context += @as(f64, @floatFromInt(input.context)) * @as(f64, @floatFromInt(input.fixed));
            sampled = sampled or input.sampled;
            warm = warm and input.warm;
            // Forced rows do not have observable proposal-acceptance outcomes.
            if (input.fixed > 1) warm = false;
            var f = &c.frame.streams[i];
            f.* = .{ .key = input.key, .offered = input.survivals.len, .source = input.source, .sampled = input.sampled };
            var previous: f64 = 1;
            var reach: f64 = 1;
            for (input.survivals, 0..) |survival, depth| {
                const conditional = probability(if (previous > 0) survival / previous else 0);
                f.raw[depth] = conditional;
                const cal = &c.calibration[calIndex(input.sampled, input.source, depth, conditional)];
                const fitted = cal.predict(conditional, c.rounds);
                reach *= if (c.guards[guardIndex(input.sampled, input.source)].useLearned()) fitted else conditional;
                probabilities[i][depth] = reach;
                previous = @min(previous, @max(0, survival));
            }
        }
        if (rows > cap) return error.BadAdaptiveShape;
        const context: u64 = @intFromFloat(weighted_context / @as(f64, @floatFromInt(rows)));
        var expected: f64 = @floatFromInt(rows);
        var best_rows = rows;
        var best_context = context;
        const group0 = group(inputs.len, context, sampled);
        const over = if (c.overhead[group0].count >= 4) c.overhead[group0].value else @max(0, fallback_overhead);
        const rate = &c.decode_rate[group0];
        // AdaSpark's average-reward price; cold start uses the existing ratio.
        const rho: ?f64 = rate.price();
        var best = c.utility(rows, inputs.len, context, sampled, expected, over, baseline, rho);
        const previous_rows = c.incumbent[group0];
        var previous_score: ?f64 = if (rows == previous_rows) best else null;
        var previous_counts = counts;
        var previous_context = context;
        var probe_score: ?f64 = null;
        var probe_counts = counts;
        var probe_rows: u32 = 0;
        var probe_context = context;
        while (rows < cap) {
            var pick: ?usize = null;
            var pick_p: f64 = 0;
            for (inputs, 0..) |input, i| {
                if (counts[i] >= input.survivals.len) continue;
                const p = probabilities[i][counts[i]];
                if (pick == null or p > pick_p) { pick = i; pick_p = p; }
            }
            const i = pick orelse break;
            counts[i] += 1;
            rows += 1;
            expected += pick_p;
            // Context is a workload feature, independent of the chosen width.
            // Keeping the mandatory-row mean avoids learning different buckets
            // merely because this policy granted more drafts to a long stream.
            const score = c.utility(rows, inputs.len, context, sampled, expected, over, baseline, rho);
            if (rows == previous_rows) { previous_score = score; previous_counts = counts; previous_context = context; }
            const cell = &c.costs[group(inputs.len, context, sampled) * (max_rows + 1) + rows];
            // One near-indifferent neighbouring width, at most every64 rounds.
            // This is still a useful target verify, never a separate model call.
            if (warm and c.refresh.ready() and cell.count < 4 and rows == previous_rows + 1) {
                probe_score = score; probe_counts = counts; probe_rows = rows; probe_context = context;
            }
            if (score > best) { best = score; best_counts = counts; best_rows = rows; best_context = context; }
        }
        const margin = @max(0.000001, @abs(best) * 0.02);
        if (previous_score) |score| if (best - score <= margin) {
            best_counts = previous_counts; best_rows = previous_rows; best_context = previous_context;
        };
        if (probe_score) |score| if (best - score <= @max(0.000001, @abs(best) * 0.04)) {
            best_counts = probe_counts; best_rows = probe_rows; best_context = probe_context; c.probes += 1; c.exploring = true;
        };
        c.incumbent[group0] = best_rows;
        c.frame.rows = best_rows;
        c.frame.context = best_context;
        c.frame.sampled = sampled;
        c.frame.warm = warm;
        c.frame.base_ms = baseline.get(best_rows) orelse 0;
        c.frame.predicted_ms = c.predictedCost(best_rows, inputs.len, best_context, sampled, baseline);
        for (0..inputs.len) |i| c.frame.streams[i].granted = best_counts[i];
        return gpa.dupe(u32, best_counts[0..inputs.len]);
    }

    fn utility(c: *const Controller, rows: u32, streams: usize, context: u64, sampled: bool, expected: f64, over: f64, baseline: *const Table, rho: ?f64) f64 {
        const ms = c.predictedCost(rows, streams, context, sampled, baseline) + over;
        return if (rho) |price| expected - price * ms else expected / ms;
    }

    pub fn verified(c: *Controller, ms: f64) void {
        c.frame.verify_ms = ms;
        if (c.frame.warm and c.exploring) c.refresh.observe(ms, true);
        if (!c.frame.warm) return;
        const base = c.frame.base_ms;
        if (!std.math.isFinite(ms) or !std.math.isFinite(base) or base <= 0 or ms < base * 0.25 or ms > base * 8) {
            c.rejected_times += 1;
            c.frame.verify_ms = null;
            return;
        }
        const bucket = group(c.frame.count, c.frame.context, c.frame.sampled);
        const at = bucket * (max_rows + 1) + c.frame.rows;
        c.costs[at].add(ms / base);
        if (!c.exploring and c.costs[at].seen > 2) c.refresh.observe(ms, false);
        if(c.costs[at].seen>2) c.group_costs[bucket].add(ms/base);
        c.cost_updates += 1;
        c.prediction_samples += 1;
        c.prediction_error_ms += @abs(ms - c.frame.predicted_ms);
        c.observed_verify_ms += ms;
    }

    /// Called before thinking/EOS/output cuts, with target-observed chain outcome.
    /// Rows after the first mismatch are censored, not failed predictions.
    pub fn accepted(c: *Controller, key: usize, proposed: usize, kept: usize) void {
        for (c.frame.streams[0..c.frame.count]) |f| {
            if (f.key != key) continue;
            const n = @min(@min(proposed, f.granted), f.offered);
            for (0..n) |depth| {
                if (depth > kept) break;
                const hit = depth < kept;
                const raw = f.raw[depth];
                const cal = &c.calibration[calIndex(f.sampled, f.source, depth, raw)];
                const fitted = cal.predict(raw, c.rounds);
                c.guards[guardIndex(f.sampled, f.source)].observe(raw, fitted, hit);
                cal.add(hit, c.rounds);
                c.acceptance_updates += 1;
            }
            return;
        }
    }

    pub fn finished(c: *Controller, ms: f64, emitted: usize) void {
        defer c.rounds += 1;
        if (!c.frame.warm or !std.math.isFinite(ms) or ms <= 0) return;
        const verify = c.frame.verify_ms orelse return;
        if (ms < verify or ms > (c.frame.base_ms + 100) * 10) return;
        const at = group(c.frame.count, c.frame.context, c.frame.sampled);
        c.overhead[at].add(ms - verify, 0.1);
        c.decode_rate[at].add(emitted,ms);
    }
};

test "cold prior preserves cap bounds and small batch choices" {
    const gpa = std.testing.allocator;
    var table: Table = .{};
    defer table.deinit(gpa);
    for (1..49) |r| try table.put(gpa, @intCast(r), 30 + @as(f64, @floatFromInt(r)) * 3);
    const c = try gpa.create(Controller);
    defer gpa.destroy(c);
    c.* = .{};
    const inputs = [_]Input{.{ .key = 1, .context = 1024, .survivals = &.{0.9,0.8,0.7,0.6,0.5} }};
    const a = try c.allocate(gpa, &inputs, &table, 8, 16);
    defer gpa.free(a);
    const b = try c.allocate(gpa, &inputs, &table, 8, 48);
    defer gpa.free(b);
    try std.testing.expectEqualSlices(u32, a, b);
    try std.testing.expect(c.frame.rows <= 6);
}

test "online acceptance labels only reached verified positions" {
    const gpa = std.testing.allocator;
    const c = try gpa.create(Controller);
    defer gpa.destroy(c);
    c.* = .{};
    c.frame.count = 1;
    c.frame.streams[0] = .{ .key=7, .offered=5, .granted=3, .raw=@splat(0.9) };
    c.accepted(7, 5, 1);
    try std.testing.expectEqual(@as(u64, 2), c.acceptance_updates);
    try std.testing.expectEqual(@as(f64, 1), c.calibration[calIndex(false,.head,0,0.9)].successes);
    try std.testing.expectEqual(@as(f64, 0), c.calibration[calIndex(false,.head,1,0.9)].successes);
    try std.testing.expectEqual(@as(f64, 0), c.calibration[calIndex(false,.head,2,0.9)].trials);
}

test "cost learning is isolated by context load and sampling" {
    const gpa = std.testing.allocator;
    var table: Table = .{};
    defer table.deinit(gpa);
    try table.put(gpa,8,40);
    const c = try gpa.create(Controller);
    defer gpa.destroy(c);
    c.* = .{};
    c.frame = .{.count=6,.rows=8,.context=100000,.sampled=true,.warm=true,.base_ms=40};
    for (0..30) |_| c.verified(80);
    try std.testing.expect(c.predictedCost(8,6,100000,true,&table)>60);
    try std.testing.expectEqual(@as(f64,40),c.predictedCost(8,1,100000,true,&table));
    try std.testing.expectEqual(@as(f64,40),c.predictedCost(8,6,1024,true,&table));
    try std.testing.expectEqual(@as(f64,40),c.predictedCost(8,6,100000,false,&table));
    c.verified(std.math.nan(f64));
    try std.testing.expectEqual(@as(u64,1),c.rejected_times);
}

test "measured expensive wide verifies actively change the selected width" {
    const gpa=std.testing.allocator;
    var table: Table=.{};
    defer table.deinit(gpa);
    for (1..49) |r| try table.put(gpa,@intCast(r),30+@as(f64,@floatFromInt(r))*10);
    const c=try gpa.create(Controller);
    defer gpa.destroy(c);
    c.*=.{};
    const inputs=[_]Input{.{.key=1,.context=1024,.survivals=&.{0.99,0.98,0.97,0.96,0.95}}};
    const before=try c.allocate(gpa,&inputs,&table,8,48);
    defer gpa.free(before);
    try std.testing.expectEqual(@as(u32,5),before[0]);
    for (1..7) |r| {
        c.frame=.{.count=1,.rows=@intCast(r),.context=1024,.warm=true,.base_ms=table.get(@intCast(r)).?};
        for (0..24) |_| c.verified(c.frame.base_ms*(if(r>=3) @as(f64,3) else 1));
    }
    const after=try c.allocate(gpa,&inputs,&table,8,48);
    defer gpa.free(after);
    try std.testing.expectEqual(@as(u32,1),after[0]);
}

test "miscalibrated confidence learns from real failures and changes allocation" {
    const gpa=std.testing.allocator;
    var table: Table=.{};
    defer table.deinit(gpa);
    // With nearly free extra rows, a low first-token probability can still
    // rationally favour the full chain. This curve has meaningful row cost.
    for (1..49) |r| try table.put(gpa,@intCast(r),30+@as(f64,@floatFromInt(r))*8);
    const c=try gpa.create(Controller);
    defer gpa.destroy(c);
    c.*=.{};
    const inputs=[_]Input{.{.key=9,.context=2048,.survivals=&.{0.9,0.81,0.729,0.6561,0.59049}}};
    const before=try c.allocate(gpa,&inputs,&table,8,48);
    defer gpa.free(before);
    for (0..128) |_| {
        const counts=try c.allocate(gpa,&inputs,&table,8,48);
        defer gpa.free(counts);
        c.accepted(9,counts[0],0);
    }
    try std.testing.expect(c.guards[guardIndex(false,.head)].useLearned());
    const after=try c.allocate(gpa,&inputs,&table,8,48);
    defer gpa.free(after);
    try std.testing.expect(after[0]<before[0]);
}

test "all C1 through C12 preserve mandatory rows and individual prefix bounds" {
    const gpa=std.testing.allocator;
    var table: Table=.{};
    defer table.deinit(gpa);
    for (1..49) |r| try table.put(gpa,@intCast(r),30+@as(f64,@floatFromInt(r))*3);
    const c=try gpa.create(Controller);
    defer gpa.destroy(c);
    c.*=.{};
    var inputs: [12]Input=undefined;
    for (1..13) |n| {
        for (inputs[0..n],0..) |*input,i| input.*=.{.key=i+1,.context=8192*@as(u64,@intCast(i+1)),.survivals=&.{0.9,0.8,0.6,0.3,0.1},.sampled=i%2==0};
        const counts=try c.allocate(gpa,inputs[0..n],&table,8,48);
        defer gpa.free(counts);
        var rows: usize=n;
        for (counts) |count| { try std.testing.expect(count<=5); rows+=count; }
        try std.testing.expect(rows<=48);
        try std.testing.expectEqual(rows,c.frame.rows);
    }
    try std.testing.expect(@sizeOf(Controller)<1024*1024);
}

test "unobserved acceptance calibration returns to current confidence" {
    var cal: Calibration=.{};
    for(0..128) |_| cal.add(false,0);
    try std.testing.expect(cal.predict(0.9,0)<0.2);
    try std.testing.expect(cal.predict(0.9,20000)>0.8);
}

test "average reward price weights slow and fast rounds by elapsed time" {
    var rate: Rate=.{};
    for(0..1024) |i| rate.add(1,if(i%2==0) 1000 else 1);
    const price=rate.price().?;
    try std.testing.expect(price>0.0015 and price<0.0025);
}

test "allocation matches exhaustive cold ratio optimum for competing prefixes" {
    const gpa=std.testing.allocator;
    var table: Table=.{};
    defer table.deinit(gpa);
    for(1..49) |r| try table.put(gpa,@intCast(r),30+@as(f64,@floatFromInt(r))*8);
    const c=try gpa.create(Controller);
    defer gpa.destroy(c);
    c.*=.{};
    const aa=[_]f64{0.8,0.5,0.1};
    const bb=[_]f64{0.7,0.6,0.3};
    const inputs=[_]Input{.{.key=1,.context=16000,.survivals=&aa},.{.key=2,.context=30000,.survivals=&bb}};
    const result=try c.allocate(gpa,&inputs,&table,10,6);
    defer gpa.free(result);
    var optimum: f64=0;
    for(0..4) |a| for(0..4) |b| {
        if(2+a+b>6) continue;
        var expected: f64=2;
        for(aa[0..a]) |p| expected+=p;
        for(bb[0..b]) |p| expected+=p;
        optimum=@max(optimum,expected/(table.get(@intCast(2+a+b)).?+10));
    };
    var actual: f64=2;
    for(aa[0..result[0]]) |p| actual+=p;
    for(bb[0..result[1]]) |p| actual+=p;
    actual/=table.get(@intCast(2+result[0]+result[1])).?+10;
    try std.testing.expectApproxEqAbs(optimum,actual,0.0000001);
}

test "d allocation cannot explore without earned credit and charges selected probe" {
    const a = std.testing.allocator;
    var table: Table = .{};
    defer table.deinit(a);
    // Widths one and two are near indifferent; mandatory width is incumbent.
    try table.put(a, 1, 10);
    try table.put(a, 2, 19);
    const c = try a.create(Controller);
    defer a.destroy(c);
    c.* = .{};
    const inputs = [_]Input{.{ .key = 1, .context = 1024, .survivals = &.{0.9} }};
    c.incumbent[group(1, 1024, false)] = 1;
    var counts = try c.allocate(a, &inputs, &table, 0, 2);
    a.free(counts);
    try std.testing.expectEqual(@as(u64, 0), c.probes);
    c.refresh.observe(2500, false);
    counts = try c.allocate(a, &inputs, &table, 0, 2);
    a.free(counts);
    try std.testing.expectEqual(@as(u64, 1), c.probes);
    try std.testing.expect(c.exploring);
    c.verified(19);
    try std.testing.expectApproxEqAbs(@as(f64, -14), c.refresh.credit_ms, 1e-9);
    counts = try c.allocate(a, &inputs, &table, 0, 2);
    a.free(counts);
    try std.testing.expect(!c.exploring);
}

test "d unknown shape initialization cannot fund exploration" {
    const a = std.testing.allocator;
    const c = try a.create(Controller);
    defer a.destroy(c);
    c.* = .{};
    c.frame = .{ .count=1, .rows=1, .context=1024, .warm=true, .base_ms=10 };
    c.verified(80);
    c.verified(80);
    try std.testing.expectEqual(@as(f64, 0), c.refresh.credit_ms);
    c.verified(10);
    try std.testing.expect(c.refresh.credit_ms > 0);
    const earned = c.refresh.credit_ms;
    c.verified(std.math.nan(f64));
    try std.testing.expectEqual(earned, c.refresh.credit_ms);
}
