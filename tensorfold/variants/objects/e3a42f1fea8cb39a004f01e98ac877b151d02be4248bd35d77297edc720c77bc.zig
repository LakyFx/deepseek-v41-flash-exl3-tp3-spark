//! DS41F's EXISTING callback positions. No new chunk, save point or GPU work.
const std = @import("std");

pub const Native = struct {
    len: u32,
    from: u32,
    replay: u32,
    chunk: u32 = 2048,

    pub fn encoder(p: Native) u32 {
        if (p.len == 0) return 0;
        const end = std.mem.alignBackward(u32, @min(p.replay, p.len - 1), 2);
        return if (end > p.from and end <= p.replay) end else 0;
    }

    pub fn effective(p: Native, wanted: u32) u32 {
        const at = wanted / p.chunk * p.chunk;
        const end = p.encoder();
        if (at <= p.from or at >= p.len or at > p.replay or end == 0) return 0;
        // The native loop chunks from the RESTORED start, splitting at encoder.
        // A global-grid mark outside that path was never a callback: omit it.
        return if (at == end or (at < end and (at - p.from) % p.chunk == 0)) at else 0;
    }

    pub fn automatic(p: Native, out: *[2]u32) usize {
        var n: usize = 0;
        const end = p.encoder();
        if (end != 0) { out[n] = end; n += 1; }
        if (p.len > 0) { out[n] = p.len; n += 1; }
        return n;
    }
};

test "native shared cuts map to actual callbacks; restored off-grid marks are omitted" {
    const cold: Native = .{ .len = 10000, .from = 0, .replay = 9872 };
    try std.testing.expectEqual(@as(u32, 2048), cold.effective(2952));
    try std.testing.expectEqual(@as(u32, 4096), cold.effective(4488));
    try std.testing.expectEqual(@as(u32, 4096), cold.effective(5000));
    try std.testing.expectEqual(@as(u32, 8192), cold.effective(9500));
    const warm: Native = .{ .len = 10000, .from = 3000, .replay = 9872 };
    try std.testing.expectEqual(@as(u32, 0), warm.effective(7000));
    var automatic: [2]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 2), warm.automatic(&automatic));
    try std.testing.expectEqualSlices(u32, &.{9872, 10000}, &automatic);
}

test "short continuation and exact hits do not invent encoder callbacks" {
    var out: [2]u32 = undefined;
    const short: Native = .{ .len = 3050, .from = 3000, .replay = 2872 };
    try std.testing.expectEqual(@as(usize, 1), short.automatic(&out));
    try std.testing.expectEqual(@as(u32, 3050), out[0]);
    const exact: Native = .{ .len = 3000, .from = 3000, .replay = 2872 };
    try std.testing.expectEqual(@as(usize, 1), exact.automatic(&out));
    try std.testing.expectEqual(@as(u32, 0), exact.effective(2999));
    const tiny: Native = .{ .len = 1, .from = 0, .replay = 0 };
    try std.testing.expectEqual(@as(usize, 1), tiny.automatic(&out));
}
