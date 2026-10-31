//! Pure row ownership and bounded CED exchange plans; no GPU or timing state.
const std = @import("std");

pub const protocol: u32 = 0x12740001;
pub const min_rows = 768;
pub const max_rows = 8192;
pub const alignment = 16;
pub const Rows = struct {
    first: usize,
    count: usize,
    pub fn end(r: Rows) usize { return r.first + r.count; }
};

pub fn rows(n: usize, rank: usize, world: usize) !Rows {
    if ((world != 3 and world != 4) or rank >= world or n > max_rows) return error.BadRowOwnership;
    const stride = std.mem.alignForward(usize, (n + world - 1) / world, alignment);
    const first = @min(n, rank * stride);
    return .{ .first = first, .count = @min(stride, n - first) };
}

pub fn intersection(a: Rows, first: usize, end: usize) Rows {
    const lo = @min(a.end(), @max(a.first, first));
    const hi = @max(lo, @min(a.end(), end));
    return .{ .first = lo, .count = hi - lo };
}

pub fn eligible(enabled: bool, n: usize, world: usize, two: bool, images: bool) bool {
    return enabled and ((world == 3 and !two) or (world == 4 and two)) and !images and n >= min_rows and n <= max_rows;
}

pub const Stats = struct {
    owned_rows: usize = 0,
    query_rows: u64 = 0,
    front_rows: u64 = 0,
    wide_x_gathers_skipped: u64 = 0,
    index_rows: u64 = 0,
    mhc_rows: u64 = 0,
    gate_rows: u64 = 0,
    partial_rows: u64 = 0,
    gathered_rows: u64 = 0,
    exchanges: u64 = 0,
    sent_bytes: u64 = 0,
};

test "ownership covers every row once including nondivisible CED shapes" {
    for (0..max_rows+1) |n| {
        var at: usize = 0;
        for (0..3) |rank| {
            const r = try rows(n, rank, 3);
            try std.testing.expectEqual(at, r.first);
            if (r.count > 0) try std.testing.expectEqual(@as(usize, 0), r.first % 16);
            at = r.end();
        }
        try std.testing.expectEqual(n, at);
    }
}

test "partial exchange preserves each source rank's exact values in rank order" {
    // Simulate raw all-to-all and the consumer's [world, local_rows, D] layout.
    const d = 3;
    for ([_]usize{ 768, 1024, 1920, 2048, 4095, 4096, 8191, 8192 }) |n| {
        for (0..3) |owner| {
            const ours = try rows(n, owner, 3);
            var output: [3 * 2736 * d]u32 = undefined;
            for (0..3) |source| {
                for (0..ours.count) |r| for (0..d) |column| {
                    const original = @as(u32, @intCast(source * 100000 + (ours.first + r) * d + column));
                    output[(source * ours.count + r) * d + column] = original;
                };
            }
            for (0..ours.count) |r| for (0..d) |column| {
                for (0..3) |source| try std.testing.expectEqual(
                    @as(u32, @intCast(source * 100000 + (ours.first + r) * d + column)),
                    output[(source * ours.count + r) * d + column]);
            };
        }
    }
}

test "CED range gathers only retained rows and supports owners with zero rows" {
    for ([_]usize{ 768, 1024, 1920, 2048, 4095, 4096, 8191, 8192 }) |n| {
        for ([_]usize{ 0, 1, 127, 128, 129, 767 }) |keep| {
            const first = n - keep;
            var at = first;
            var total: usize = 0;
            for (0..3) |rank| {
                const r = intersection(try rows(n, rank, 3), first, n);
                if (r.count > 0) { try std.testing.expectEqual(at, r.first); at = r.end(); }
                total += r.count;
            }
            try std.testing.expectEqual(keep, total);
            try std.testing.expectEqual(n, at);
        }
    }
}

test "decode vision unsupported geometry and short paths remain original" {
    for (0..min_rows) |n| try std.testing.expect(!eligible(true, n, 3, false, false));
    try std.testing.expect(eligible(true, 2048, 3, false, false));
    try std.testing.expect(!eligible(false, 2048, 3, false, false));
    try std.testing.expect(!eligible(true, 2048, 4, false, false));
    try std.testing.expect(!eligible(true, 2048, 3, true, false));
    try std.testing.expect(!eligible(true, 2048, 3, false, true));
}

/// Partition only while D owns all encoder rows. A candidate mask must never
/// cross the CED compaction boundary as an incomplete per-rank view.
pub fn indexEligible(enabled: bool, row_active: bool, compressed: usize, source: usize, layers: usize) bool {
    return enabled and row_active and compressed >= 512 and source >= layers / 2;
}

test "index row ownership is disabled after CED and before sufficient keys" {
    try std.testing.expect(indexEligible(true, true, 512, 32, 64));
    try std.testing.expect(!indexEligible(false, true, 512, 32, 64));
    try std.testing.expect(!indexEligible(true, false, 512, 32, 64));
    try std.testing.expect(!indexEligible(true, true, 511, 32, 64));
    try std.testing.expect(!indexEligible(true, true, 512, 31, 64));
}

test "nonuniform selection rows fill the exact unsliced destination" {
    for ([_]usize{768, 1024, 1999, 2048, 4095, 4096, 8191, 8192}) |n| {
        var got: [8192]u32 = undefined;
        for (0..3) |rank| {
            const r=try rows(n,rank,3);
            for (r.first..r.end()) |j| got[j]=@intCast(j*7+3);
        }
        for (got[0..n],0..) |v,j| try std.testing.expectEqual(@as(u32,@intCast(j*7+3)),v);
    }
}

pub fn attentionNeedsFullX(has_index: bool, has_compressor: bool) bool {
    return has_index or has_compressor;
}
test "only original global-X consumers keep the wide gather" {
    try std.testing.expect(!attentionNeedsFullX(false,false));
    try std.testing.expect(attentionNeedsFullX(true,false));
    try std.testing.expect(attentionNeedsFullX(false,true));
    try std.testing.expect(attentionNeedsFullX(true,true));
}

test "4096 owned rows split137613761344 and reject oversized chunks" {
    const expected=[_]usize{1376,1376,1344};
    for(0..3) |rank| try std.testing.expectEqual(expected[rank],(try rows(4096,rank,3)).count);
    try std.testing.expect(eligible(true,4096,3,false,false));
    try std.testing.expect(!eligible(true,8193,3,false,false));
    try std.testing.expectError(error.BadRowOwnership,rows(8193,0,3));
}

test "four owners partition every AF row and all CED tails without overlap" {
    for (0..max_rows+1) |n| {
        var at:usize=0;
        for (0..4) |rank| {
            const r=try rows(n,rank,4);
            try std.testing.expectEqual(at,r.first);
            if(r.count>0) try std.testing.expectEqual(@as(usize,0),r.first%alignment);
            at=r.end();
        }
        try std.testing.expectEqual(n,at);
        for ([_]usize{0,1,7,127,128,767}) |keep0| {
            const keep=@min(n,keep0);
            var total:usize=0;
            for(0..4) |rank| total+=intersection(try rows(n,rank,4),n-keep,n).count;
            try std.testing.expectEqual(keep,total);
        }
    }
    for(0..4) |rank| try std.testing.expectEqual(@as(usize,1024),(try rows(4096,rank,4)).count);
    try std.testing.expect(eligible(true,4096,4,true,false));
    try std.testing.expect(!eligible(true,4096,4,true,true));
    try std.testing.expect(!eligible(true,767,4,true,false));
    try std.testing.expectError(error.BadRowOwnership,rows(2048,4,4));
}

test "8192 compute rows use four exact 2048 row owners and preserve tails" {
    for(0..4) |rank| {
        const r=try rows(8192,rank,4);
        try std.testing.expectEqual(rank*2048,r.first);
        try std.testing.expectEqual(@as(usize,2048),r.count);
    }
    try std.testing.expect(eligible(true,8192,4,true,false));
    try std.testing.expect(!eligible(true,8193,4,true,false));
    try std.testing.expectError(error.BadRowOwnership,rows(8193,0,4));
}
