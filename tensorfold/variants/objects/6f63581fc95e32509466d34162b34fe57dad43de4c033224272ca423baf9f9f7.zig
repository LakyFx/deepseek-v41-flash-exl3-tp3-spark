//! X1 bounded CPU metadata. Never authorizes restore and never runs per token.
const std = @import("std");
pub const max_entries = 512;
pub const trace_capacity = 128;
pub const ghost_capacity = 64;
pub const Role = enum(u8) { unknown, auxiliary, encoder_tail, prompt_end, decode_end };
pub const Reason = enum(u8) { none, dead, superseded_auxiliary, superseded_tail, unused, fresh, reused, legacy_fallback, upgrade, reset };
pub const Kind = enum(u8) { request, plan, restore, save, dedup, drop, ghost_return, finished };

pub const Entry = struct {
    id: u64 = 0,
    key: u64 = 0, // existing keep hash, never a restore authority or log field
    cohort: u64 = 0,
    parent: u64 = 0,
    successor: u64 = 0,
    created_request: u64 = 0,
    last_return: u64 = 0,
    role: Role = .unknown,
    continuation: bool = false,
    history: u64 = 0,
    history_request: u64 = 0,
    history_lost: bool = false,
};

pub const Request = struct {
    id: u64 = 0,
    namespace_id: u64 = 0,
    cohort: u64 = 0,
    parent: u64 = 0,
    parent_request: u64 = 0,
    input_len: u32 = 0,
    encoder_at: u32 = 0,
    restored_at: u32 = 0,
    continuation: bool = false,
    history: u64 = 0,
    history_observed: bool = false,

    pub fn role(r: Request, at: u32) Role {
        if (at == r.input_len) return .prompt_end;
        if (at > r.input_len) return .decode_end;
        return if (at == r.encoder_at and at != 0) .encoder_tail else .auxiliary;
    }
};

pub const Event = struct {
    kind: Kind = .request,
    reason: Reason = .none,
    role: Role = .unknown,
    request: u64 = 0,
    namespace_id: u64 = 0,
    state: u64 = 0,
    cohort: u64 = 0,
    parent: u64 = 0,
    at: u32 = 0,
    prompt: u32 = 0,
    desired: u32 = 0,
    effective: u32 = 0,
    bytes: u64 = 0,
    baseline: u64 = 0,
    elapsed_ns: u64 = 0,
    prefill_ns: u64 = 0,
    outcome: u8 = 0, // 1 completed, 2 failed, 3 cancelled
};

pub const Ghost = struct {
    compatibility_known: bool = false,
    exact_len: u32 = 0,
    min_len: u32 = 0,
    history: u64 = 0,
    history_request: u64 = 0,
    key: u64 = 0, // internal only: evidence, not a restore authority
    namespace_id: u64 = 0,
    state: u64 = 0,
    cohort: u64 = 0,
    at: u32 = 0,
    last_token: u32 = 0,
    reason: Reason = .none,
};

pub const State = struct {
    enabled: bool = false,
    trace: bool = false,
    slots: u32 = 12,
    serial: u64 = 0,
    request_serial: u64 = 0,
    events: [trace_capacity]Event = @splat(.{}),
    event_start: usize = 0,
    event_count: usize = 0,
    overwritten: u64 = 0,
    ghosts: [ghost_capacity]Ghost = @splat(.{}),
    ghost_next: usize = 0,
    ghost_returns: u64 = 0,
    ghost_hash_skips: u64 = 0,
    ghost_incompatible_skips: u64 = 0,
    choices: u64 = 0,
    sampled_choices: u64 = 0,
    sampled_different: u64 = 0,
    fallbacks: u64 = 0,
    choose_ns: u64 = 0,
    plan_ns: u64 = 0,
    save_ns: u64 = 0,
    drop_ns: u64 = 0,
    mapped: u64 = 0,
    omitted: u64 = 0,
    duplicate_marks: u64 = 0,
    compatible_reservations: u64 = 0,
    continuations: u64 = 0,
    last_reason: Reason = .none,
    last_baseline: u64 = 0,

    pub fn push(s: *State, e: Event) void {
        if (!s.trace) return;
        if (s.event_count == trace_capacity) {
            s.event_start = (s.event_start + 1) % trace_capacity;
            s.event_count -= 1;
            s.overwritten +|= 1;
        }
        s.events[(s.event_start + s.event_count) % trace_capacity] = e;
        s.event_count += 1;
    }

    pub fn fresh(s: *const State, e: Entry, restores: u64) bool {
        return e.id != 0 and restores == 0 and !e.continuation and
            s.serial -| e.id < @as(u64, @min(s.slots, 64)) * 3;
    }

    /// Weak inherited credit is separate from genuine restore counters.
    /// Length contributes only after demonstrated reuse, never by itself.
    pub fn value(s: *const State, e: Entry, restores: u64, at: u32) u64 {
        const age = s.request_serial -| e.last_return;
        if (age >= 512) return 0;
        const hits: u64 = @min(restores, 8);
        const chunks: u64 = @min(at / 2048, 128);
        const direct: u64 = hits * (16 + chunks);
        const inherited: u64 = if (e.continuation) 16 else 0;
        return (direct + inherited) * 256 / (256 + age);
    }
};

pub const Candidate = struct {
    index: usize,
    tier: u8,
    value: u64,
    used: u64,
    reason: Reason,
};

pub fn candidate(s: *const State, e: Entry, restores: u64, at: u32, used: u64, index: usize, superseded_auxiliary: bool, superseded_tail: bool) Candidate {
    const v = s.value(e, restores, at);
    const fresh = s.fresh(e, restores);
    const reason: Reason = if (superseded_auxiliary and v == 0) .superseded_auxiliary
        else if (superseded_tail) .superseded_tail
        else if (v > 0) .reused else if (fresh) .fresh else .unused;
    const tier: u8 = switch (reason) {
        .superseded_auxiliary => 0, .unused => 1, .superseded_tail => 2, .fresh => 3, else => 4,
    };
    return .{ .index = index, .tier = tier, .value = v, .used = used, .reason = reason };
}

pub fn less(a: Candidate, b: Candidate) bool {
    if (a.tier != b.tier) return a.tier < b.tier;
    if (a.value != b.value) return a.value < b.value;
    return a.used < b.used;
}

/// First-stage gate only. Caller must also have an actual successful compatible
/// restore, the same namespace and verified token identity through the old tail.
pub fn tailEligible(role: Role, shared: bool, at: u32, old_len: usize, new_len: usize, window: u32) bool {
    return !shared and (role == .prompt_end or role == .decode_end or role == .encoder_tail) and
        new_len > old_len and at <= old_len and old_len - at <= window;
}

/// Pressure timings use the same Linux monotonic source as the existing planner.
pub fn now() u64 {
    if (@import("builtin").os.tag == .windows) return 0; // CPU test host, never serving
    var ts: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "mixed workload: old one-off before active long and active short; no permanent pin" {
    const s: State = .{ .enabled = true, .serial = 200, .request_serial = 150 };
    const oneoff = candidate(&s, .{ .id = 1 }, 0, 32768, 1, 0, false, false);
    const long = candidate(&s, .{ .id = 2, .last_return = 149, .continuation = true }, 2, 301000, 2, 1, false, false);
    const short = candidate(&s, .{ .id = 3, .last_return = 149 }, 6, 32768, 3, 2, false, false);
    const fresh = candidate(&s, .{ .id = 199 }, 0, 4096, 4, 3, false, false);
    try std.testing.expect(less(oneoff, long));
    try std.testing.expect(less(oneoff, short));
    try std.testing.expect(less(oneoff, fresh));
    var aged = s;
    aged.request_serial = 1000;
    try std.testing.expectEqual(@as(u64, 0), aged.value(.{ .last_return = 149, .continuation = true }, 100, 301000));
}

test "fresh window is bounded, shared hint alone has no value, credit is weak" {
    const s: State = .{ .serial = 100, .request_serial = 50 };
    var count: usize = 0;
    for (1..101) |id| if (s.fresh(.{ .id = id }, 0)) { count += 1; };
    try std.testing.expectEqual(@as(usize, 36), count);
    try std.testing.expectEqual(@as(u64, 0), s.value(.{}, 0, 301000));
    try std.testing.expectEqual(@as(u64, 16), s.value(.{ .continuation = true, .last_return = 50 }, 0, 301000));
    try std.testing.expect(s.value(.{ .last_return = 50 }, 1, 32768) > 16);
    try std.testing.expect(!tailEligible(.auxiliary, false, 5000, 32768, 33000, 128));
    try std.testing.expect(!tailEligible(.prompt_end, true, 5000, 5000, 5300, 128));
    try std.testing.expect(tailEligible(.encoder_tail, false, 32640, 32768, 33000, 128));
}

test "bounded traces account for overwrites and metadata stays below 256KiB at 512 states" {
    var s: State = .{ .trace = true };
    for (0..trace_capacity + 5) |i| s.push(.{ .state = i });
    try std.testing.expectEqual(@as(usize, trace_capacity), s.event_count);
    try std.testing.expectEqual(@as(u64, 5), s.overwritten);
    try std.testing.expect(@sizeOf(State) + max_entries * @sizeOf(Entry) < 256 * 1024);
}
