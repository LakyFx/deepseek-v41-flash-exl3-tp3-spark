//! Bounded, process-local vision dispatch registry. A cancelled ID is sealed
//! against late POSTs; terminal means the HTTP handler has drained its engine.
const std = @import("std");

pub const Entry = struct {
    id: [44]u8,
    expires: i64,
    cancel_requested: std.atomic.Value(bool) = .init(false),
    terminal: bool = false,
};

pub fn validId(id: []const u8) bool {
    if (id.len != 44 or !std.mem.startsWith(u8, id, "dgx-gateway-")) return false;
    for (id[12..]) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

pub const Registry = struct {
    gpa: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayList(*Entry) = .empty,
    limit: usize = 4096,

    pub fn deinit(r: *Registry) void {
        for (r.entries.items) |e| r.gpa.destroy(e);
        r.entries.deinit(r.gpa);
    }

    fn find(r: *Registry, id: []const u8) ?*Entry {
        for (r.entries.items) |e| if (std.mem.eql(u8, &e.id, id)) return e;
        return null;
    }

    fn prune(r: *Registry, now: i64) void {
        var i: usize = 0;
        while (i < r.entries.items.len) {
            const e = r.entries.items[i];
            // No still-valid delayed dispatch can outlive this tombstone.
            if (e.terminal and now > e.expires + 3600) {
                r.gpa.destroy(r.entries.swapRemove(i));
            } else i += 1;
        }
    }

    fn add(r: *Registry, id: []const u8, expires: i64) !*Entry {
        if (r.entries.items.len >= r.limit) return error.Full;
        const e = try r.gpa.create(Entry);
        errdefer r.gpa.destroy(e);
        e.* = .{ .id = id[0..44].*, .expires = expires };
        try r.entries.append(r.gpa, e);
        return e;
    }

    pub fn begin(r: *Registry, io: std.Io, id: []const u8, expires: i64, now: i64) !*Entry {
        if (!validId(id) or expires <= now or expires > now + 3600) return error.Invalid;
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.prune(now);
        if (r.find(id) != null) return error.Sealed;
        return r.add(id, expires);
    }

    pub fn finish(r: *Registry, io: std.Io, e: *Entry) void {
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        e.terminal = true;
    }

    pub fn cancel(r: *Registry, io: std.Io, id: []const u8, expires: i64, now: i64) !bool {
        if (!validId(id) or expires > now + 3600) return error.Invalid;
        r.mutex.lockUncancelable(io);
        defer r.mutex.unlock(io);
        r.prune(now);
        const e = r.find(id) orelse blk: {
            // Cancellation arriving before the POST seals it. If the POST is
            // already expired, begin() rejects it even without a tombstone.
            if (expires <= now) return true;
            const sealed = try r.add(id, expires);
            sealed.terminal = true;
            break :blk sealed;
        };
        e.cancel_requested.store(true, .release);
        return e.terminal;
    }
};

test "active cancellation is not a receipt until the engine handler drained" {
    var r: Registry = .{ .gpa = std.testing.allocator };
    defer r.deinit();
    const id = "dgx-gateway-00000000000000000000000000000001";
    const e = try r.begin(std.testing.io, id, 200, 100);
    try std.testing.expect(!try r.cancel(std.testing.io, id, 200, 100));
    try std.testing.expect(e.cancel_requested.load(.acquire));
    r.finish(std.testing.io, e);
    try std.testing.expect(try r.cancel(std.testing.io, id, 200, 101));
    try std.testing.expectError(error.Sealed, r.begin(std.testing.io, id, 200, 102));
}

test "cancel before dispatch forbids a delayed inference" {
    var r: Registry = .{ .gpa = std.testing.allocator };
    defer r.deinit();
    const id = "dgx-gateway-00000000000000000000000000000002";
    try std.testing.expect(try r.cancel(std.testing.io, id, 200, 100));
    try std.testing.expectError(error.Sealed, r.begin(std.testing.io, id, 200, 101));
    try std.testing.expectError(error.Invalid, r.begin(std.testing.io, id, 200, 4000));
}

test "capacity never evicts a live generation or a valid cancellation seal" {
    var r: Registry = .{ .gpa = std.testing.allocator, .limit = 1 };
    defer r.deinit();
    const one = "dgx-gateway-00000000000000000000000000000001";
    const two = "dgx-gateway-00000000000000000000000000000002";
    const e = try r.begin(std.testing.io, one, 200, 100);
    try std.testing.expectError(error.Full, r.begin(std.testing.io, two, 200, 101));
    try std.testing.expectError(error.Full, r.begin(std.testing.io, two, 4002, 4001));
    r.finish(std.testing.io, e);
    _ = try r.begin(std.testing.io, two, 4002, 4001);
    try std.testing.expectEqual(@as(usize, 1), r.entries.items.len);
}
