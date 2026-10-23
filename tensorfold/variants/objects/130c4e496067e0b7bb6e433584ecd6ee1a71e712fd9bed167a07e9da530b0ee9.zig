//! The loop guard's novelty signal (TensorFold 0.6's call_gate.ThinkLoop, on token ids): a reply still thinking whose
//! reasoning stopped saying anything new. A window of `width` reply tokens is dry when less than `least` of its token
//! `gram`-grams (with the `gram - 1` tokens before it) were in no earlier window; `windows` dry windows in a row fire.
//! A templated loop, each line fresh while its 8-grams recycle, has no exact cycle for `cycleAt` to find; this sees it.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const width: usize = 1024;
pub const gram: usize = 8;
pub const least: f64 = 0.02;

pub const Novelty = struct {
    windows: u32,
    seen: std.AutoHashMapUnmanaged([gram]u32, void) = .empty, // every earlier window's grams
    start: usize = 0, // the open window's first reply token
    dry: u32 = 0, // windows in a row with less than `least` new

    pub fn deinit(n: *Novelty, gpa: Allocator) void {
        n.seen.deinit(gpa);
    }

    /// The share of the grams of `head` then `tail` that no earlier window had, each counted as often as it occurs.
    fn share(n: *const Novelty, head: []const u32, tail: []const u32) f64 {
        const grams = (head.len + tail.len) -| (gram - 1);
        var fresh: usize = 0;
        for (0..grams) |j| {
            var key: [gram]u32 = undefined;
            for (&key, j..) |*k, at| k.* = if (at < head.len) head[at] else tail[at - head.len];
            fresh += @intFromBool(!n.seen.contains(key));
        }
        return @as(f64, @floatFromInt(fresh)) / @as(f64, @floatFromInt(@max(1, grams)));
    }

    /// Count the reply's newest token, the think block still open: at a window's end, the window's share of new grams
    /// (its grams join `seen`); null inside a window.
    pub fn observe(n: *Novelty, gpa: Allocator, reply: []const u32) !?f64 {
        if (reply.len < n.start + width) return null;
        const seq = reply[n.start -| (gram - 1) .. n.start + width];
        const new = n.share(seq, &.{});
        try n.seen.ensureUnusedCapacity(gpa, @intCast(seq.len + 1 - gram));
        for (0..seq.len + 1 - gram) |j| n.seen.putAssumeCapacity(seq[j..][0..gram].*, {});
        n.start += width;
        n.dry = if (new < least) n.dry + 1 else 0;
        return new;
    }

    /// Whether the window `observe` just ended fires: the `windows`-th dry one in a row.
    pub fn fires(n: *const Novelty) bool {
        return n.dry >= n.windows;
    }

    /// Read-only lookahead. Ordinary verify batches cross at most one window; a larger CPU/driver batch also
    /// compares against its earlier full windows so its verdict does not depend on commit batching.
    pub fn firesAt(n: *const Novelty, reply: []const u32, tokens: []const u32) ?usize {
        var start = n.start;
        var dry = n.dry;
        const total = reply.len + tokens.len;
        if (n.dry + (total - start) / width < n.windows) return null;
        while (start + width <= total) : (start += width) {
            const begin = start -| (gram - 1);
            const count = start + width - begin - (gram - 1);
            var fresh: usize = 0;
            for (0..count) |j| {
                const key = joinedGram(reply, tokens, begin + j);
                if (n.seen.contains(key)) continue;
                var earlier = false;
                // Only used for an unusual batch spanning multiple 1024-token windows.
                if (start > n.start) {
                    const from = n.start -| (gram - 1);
                    const end = start - (gram - 1);
                    for (from..end) |k| if (std.mem.eql(u32, &key, &joinedGram(reply, tokens, k))) {
                        earlier = true;
                        break;
                    };
                }
                fresh += @intFromBool(!earlier);
            }
            const share_ = @as(f64, @floatFromInt(fresh)) / @as(f64, @floatFromInt(@max(1, count)));
            dry = if (share_ < least) dry + 1 else 0;
            if (dry >= n.windows) return start + width - reply.len - 1;
        }
        return null;
    }

    fn joinedGram(head: []const u32, tail: []const u32, at: usize) [gram]u32 {
        var out: [gram]u32 = undefined;
        for (&out, at..) |*t, i| t.* = if (i < head.len) head[i] else tail[i - head.len];
        return out;
    }
};

/// Sequences the tests share with the Python reference (call_gate.ThinkLoop driven token by token): fresh tokens, a
/// loop of 13 (longer than `cycleAt`'s periods), and a three-letter LCG text whose grams run out.
pub const reference = struct {
    pub fn fresh(gpa: Allocator, out: *std.ArrayList(u32), from: u32, count: usize) !void {
        for (0..count) |i| try out.append(gpa, 1000 + from + @as(u32, @intCast(i)));
    }

    pub fn loop13(gpa: Allocator, out: *std.ArrayList(u32), count: usize) !void {
        for (0..count) |j| try out.append(gpa, 5000 + @as(u32, @intCast(j % 13)));
    }

    pub fn lcg(gpa: Allocator, out: *std.ArrayList(u32), count: usize) !void {
        var x: u64 = 1;
        for (0..count) |_| {
            x = (1103515245 * x + 12345) % (1 << 31);
            try out.append(gpa, 7000 + @as(u32, @intCast((x >> 16) % 3)));
        }
    }
};

/// Each finished window's share of new grams over `tokens`, token by token.
fn shares(gpa: Allocator, tokens: []const u32, out: *std.ArrayList(f64)) !void {
    var n: Novelty = .{ .windows = 10 };
    defer n.deinit(gpa);
    for (1..tokens.len + 1) |end| if (try n.observe(gpa, tokens[0..end])) |new| try out.append(gpa, new);
}

test "each window's share of new grams is the Python guard's" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    var got: std.ArrayList(f64) = .empty;
    defer got.deinit(gpa);
    try reference.fresh(gpa, &tokens, 0, 2500);
    try reference.loop13(gpa, &tokens, 20000);
    try shares(gpa, tokens.items, &got);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1, 1, 0, 0, 0 }, got.items[0..6]);

    // the LCG text: each window's new grams, as the Python guard counts them (the first window has 1017 grams)
    tokens.clearRetainingCapacity();
    got.clearRetainingCapacity();
    try reference.lcg(gpa, &tokens, 60 * width);
    try shares(gpa, tokens.items, &got);
    const fresh = [_]u16{ 1017, 880, 756, 658, 546, 440, 414, 340, 288, 244, 225, 148, 159, 144, 113, 100, 89, 84, 70, 62, 39, 30, 27, 31, 29, 29, 26, 15, 19, 5, 8, 7, 11, 5, 2, 4, 2, 4, 2, 0, 1, 0, 2, 1, 2, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0 };
    try std.testing.expectEqual(@as(usize, fresh.len), got.items.len);
    for (got.items, fresh, 0..) |new, count, i| try std.testing.expectEqual(@as(f64, @floatFromInt(count)) / @as(f64, if (i == 0) 1017 else 1024), new);
}

test "a batch's window end fires where the token-by-token count does" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    try reference.loop13(gpa, &tokens, 4 * width);
    var n: Novelty = .{ .windows = 2 };
    defer n.deinit(gpa);
    for (1..width + 1) |end| _ = try n.observe(gpa, tokens.items[0..end]);
    try std.testing.expectEqual(@as(?usize, null), n.firesAt(tokens.items[0..width], tokens.items[width .. 2 * width])); // the first dry window
    for (width + 1..2 * width + 1) |end| _ = try n.observe(gpa, tokens.items[0..end]);
    try std.testing.expect(n.dry == 1 and !n.fires());
    const reply = tokens.items[0 .. 2 * width + 5];
    for (2 * width + 1..reply.len + 1) |end| _ = try n.observe(gpa, tokens.items[0..end]);
    try std.testing.expectEqual(@as(?usize, width - 6), n.firesAt(reply, tokens.items[reply.len..]));
    try std.testing.expectEqual(@as(?usize, null), n.firesAt(reply, tokens.items[reply.len .. 3 * width - 1])); // ends after
    for (reply.len + 1..3 * width + 1) |end| _ = try n.observe(gpa, tokens.items[0..end]);
    try std.testing.expect(n.dry == 2 and n.fires());
}
