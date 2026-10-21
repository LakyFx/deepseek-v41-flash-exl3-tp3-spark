//! Adaptation of Flare refresh credit to synchronous host-measured verifies.
//! Copyright 2026 TensorFold recipe contributors.
//! Flare inspiration: NVIDIA/flare-local-inference, Apache-2.0, refresh.rs.
const std = @import("std");

pub const Budget = struct {
    credit_ms: f64 = 0,
    pub const cap_ms = 5.0;
    const income = 0.002 / 0.998;

    pub fn ready(b: *const Budget) bool {
        return b.credit_ms >= cap_ms;
    }
    pub fn observe(b: *Budget, ms: f64, exploratory: bool) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        b.credit_ms = if (exploratory) b.credit_ms - ms else @min(cap_ms, b.credit_ms + ms * income);
    }
    pub fn idle(b: *Budget) void {
        b.credit_ms = @min(0, b.credit_ms);
    }
};

test "no early exploration, capped credit and full episode debt" {
    var b: Budget = .{};
    try std.testing.expect(!b.ready());
    b.observe(2494, false);
    try std.testing.expect(!b.ready());
    b.observe(1.001, false);
    try std.testing.expect(b.ready());
    b.observe(100000, false);
    try std.testing.expectEqual(@as(f64, 5), b.credit_ms);
    b.observe(40, true);
    try std.testing.expectEqual(@as(f64, -35), b.credit_ms);
    b.idle();
    try std.testing.expectEqual(@as(f64, -35), b.credit_ms);
    b.observe(19960, false);
    try std.testing.expect(b.ready());
    b.idle();
    try std.testing.expectEqual(@as(f64, 0), b.credit_ms);
}

test "long-run exploration bounded by ordinary time plus one episode" {
    var b: Budget = .{};
    var ordinary: f64 = 0;
    var exploration: f64 = 0;
    for (0..20000) |i| {
        const ms: f64 = @floatFromInt(7 + i % 131);
        const probe = b.ready();
        b.observe(ms, probe);
        if (probe) exploration += ms else ordinary += ms;
        try std.testing.expect(exploration <= ordinary * (0.002 / 0.998) + 137);
    }
    const saved = b.credit_ms;
    b.observe(std.math.nan(f64), true);
    b.observe(-1, false);
    try std.testing.expectEqual(saved, b.credit_ms);
}
