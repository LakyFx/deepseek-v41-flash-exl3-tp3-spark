//! TP4 on a physical ring: adjacent xor1 then xor3 exchanges, using the existing RDMA kernels.
const std = @import("std");
const cuda = @import("cuda");
const rdma = @import("rdma.zig");
const Comm = @import("comm.zig").Comm;
pub const cutoff: usize = 512 * 1024;
pub fn configured() !bool {
    const raw = std.c.getenv("TF_DS_NEIGHBOR_RDMA");
    return parseMode(if (raw) |v| std.mem.span(v) else null);
}

fn parseMode(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "1")) return true;
    if (std.mem.eql(u8, value, "0")) return false;
    return error.BadNeighborMode;
}
pub fn gidIndex() !u8 {
    const raw = std.c.getenv("TF_DS_NEIGHBOR_GID_INDEX");
    return parseGid(if (raw) |v| std.mem.span(v) else null);
}

fn parseGid(value: ?[]const u8) !u8 {
    const raw = value orelse return 5;
    if (raw.len == 0) return error.BadNeighborGid;
    for (raw) |c| if (c < '0' or c > '9') return error.BadNeighborGid;
    return std.fmt.parseInt(u8, raw, 10) catch error.BadNeighborGid;
}

pub fn eligible(bytes: usize) bool {
    return bytes > 0 and bytes <= cutoff and bytes % 16 == 0;
}
pub const Gather = struct {
    first: *rdma.Ring,
    second: *rdma.Ring,
    scratch: cuda.DeviceBuffer,

    /// Each stage uses two rails in matching order at both ends of its physical edge.
    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, k: rdma.Kernels, g: u32, comm: *const Comm, stream: cuda.Stream, gid_index: u8) !Gather {
        if (g >= 4 or comm.world != 4) return error.NotFourNodes;
        const first_raw = std.c.getenv("TF_DS_NEIGHBOR_FIRST") orelse return error.NoFirstDevices;
        const second_raw = std.c.getenv("TF_DS_NEIGHBOR_SECOND") orelse return error.NoSecondDevices;
        const first_devices = try devices(std.mem.span(first_raw));
        const second_devices = try devices(std.mem.span(second_raw));
        for (first_devices) |x| for (second_devices) |y| {
            if (std.mem.eql(u8, x, y)) return error.NeighborDevicesOverlap;
        };
        const s: rdma.Settings = .{ .max_bytes = 2 * cutoff, .gid_index = gid_index };
        const first = try rdma.Ring.create(gpa, d, k, &first_devices, g % 2, 2, s);
        errdefer first.destroy(gpa);
        const second = try rdma.Ring.create(gpa, d, k, &second_devices, g / 2, 2, s);
        errdefer second.destroy(gpa);
        var scratch = try cuda.DeviceBuffer.alloc(d, 2 * cutoff);
        errdefer scratch.free();
        const per = @sizeOf(rdma.Info);
        const mine = [4]rdma.Info{ first.info(0), first.info(1), second.info(0), second.info(1) };
        var dev = try cuda.DeviceBuffer.alloc(d, 5 * 4 * per);
        defer dev.free();
        try dev.upload(0, std.mem.sliceAsBytes(&mine));
        try comm.allGather(dev.ptr, dev.ptr + 4 * per, 4 * per, .u8, stream);
        try stream.synchronize();
        var all: [4][4]rdma.Info = undefined;
        try dev.download(4 * per, std.mem.sliceAsBytes(&all));
        var a: [2]rdma.Info = undefined;
        var b: [2]rdma.Info = undefined;
        a[g % 2] = mine[g % 2];
        a[1 - g % 2] = all[g ^ 1][g % 2];
        b[g / 2] = mine[2 + g / 2];
        b[1 - g / 2] = all[g ^ 3][2 + g / 2];
        try first.connect(&a);
        try second.connect(&b);
        try comm.allGather(dev.ptr, dev.ptr + 4 * per, 1, .u8, stream);
        try stream.synchronize();
        first.failure_peer = second;
        second.failure_peer = first;
        try first.start();
        errdefer first.stop();
        try second.start();
        return .{ .first = first, .second = second, .scratch = scratch };
    }
    pub fn close(n: *Gather, gpa: std.mem.Allocator) void {
        n.first.stop();
        n.second.stop();
        n.first.failure_peer = null;
        n.second.failure_peer = null;
        n.first.destroy(gpa);
        n.second.destroy(gpa);
        n.scratch.free();
    }
    pub fn check(n: *const Gather) !void {
        if (n.first.failure() != null or n.second.failure() != null) return error.NeighborTransportFailed;
    }
    /// Opposite rank (xor2) via adjacent xor1 then xor3; only remote slices are copied.
    pub fn relay(n: *const Gather, stream: cuda.Stream, src: u64, dst: u64, bytes: usize) !void {
        if (!eligible(bytes)) return error.NeighborSize;
        try ranges(src, bytes, dst, bytes, n.scratch.ptr, 2 * cutoff, 16);
        const n4: u32 = @intCast(bytes / 16);
        var first_dsts: [2]rdma.Ring.Dst = .{ .{}, .{} };
        var second_dsts: [2]rdma.Ring.Dst = .{ .{}, .{} };
        first_dsts[1 - n.first.rank] = .{ .ptr = n.scratch.ptr, .row4 = n4, .stride4 = n4 };
        second_dsts[1 - n.second.rank] = .{ .ptr = dst, .row4 = n4, .stride4 = n4 };
        try n.check();
        try n.first.gatherInto(stream, src, 1, n4, &first_dsts);
        try n.second.gatherInto(stream, n.scratch.ptr, 1, n4, &second_dsts);
    }
    pub fn pair(n: *const Gather, stream: cuda.Stream, src: u64, dst: u64, bytes: usize) !void {
        if (!eligible(bytes)) return error.NeighborSize;
        try n.check();
        try n.first.gather(stream, src, @intCast(bytes / 16), dst);
    }
    /// Equal-width quarters, final [2, rows, 2*width] byte layout.
    pub fn quarters(n: *const Gather, stream: cuda.Stream, src: u64, dst: u64, rows: usize, width_bytes: usize) !void {
        const bytes = try std.math.mul(usize, rows, width_bytes);
        if (!eligible(bytes) or width_bytes % 16 != 0) return error.NeighborSize;
        // All validation precedes either exchange; no partial fallback.
        const rows2: u32 = @intCast(try std.math.mul(usize, rows, 2));
        const row4: u32 = @intCast(width_bytes / 16);
        const stride4 = 2 * @as(u64, row4);
        const dsts = [_]rdma.Ring.Dst{
            .{ .ptr = dst, .row4 = row4, .stride4 = stride4 },
            .{ .ptr = dst + width_bytes, .row4 = row4, .stride4 = stride4 },
        };
        try n.check();
        try n.first.gather(stream, src, @intCast(bytes / 16), n.scratch.ptr);
        try n.second.gatherInto(stream, n.scratch.ptr, rows2, row4, &dsts);
    }
    pub fn gather(n: *const Gather, stream: cuda.Stream, src: u64, dst: u64, bytes: usize) !void {
        if (!eligible(bytes)) return error.NeighborSize;
        try n.check();
        try n.first.gather(stream, src, @intCast(bytes / 16), n.scratch.ptr);
        try n.second.gather(stream, n.scratch.ptr, @intCast(bytes / 8), dst);
    }
};
/// Larger or unequal diagonal parts travel two physical edges by NCCL, never across a missing diagonal.
pub fn forward(comm: *const Comm, src: u64, dst: u64, counts: [4]usize, dtype: cuda.nccl.DataType, scratch: u64, capacity: usize, stream: cuda.Stream) !void {
    if (comm.world != 4 or comm.rank >= 4) return error.NotFourNodes;
    if (dtype != .f16 and dtype != .bf16) return error.DiagonalType;
    for (counts) |n| {
        if (n == 0 or n > capacity / 2) return error.DiagonalScratchTooSmall;
    }
    const g = comm.rank;
    try ranges(src, counts[g] * 2, dst, counts[g ^ 2] * 2, scratch, capacity, 2);
    const next = (g + 1) % 4;
    const prev = (g + 3) % 4;
    try hop(comm, src, counts[g], next, scratch, counts[prev], prev, dtype, stream);
    try hop(comm, scratch, counts[prev], next, dst, counts[g ^ 2], prev, dtype, stream);
}

fn ranges(src: u64, src_bytes: usize, dst: u64, dst_bytes: usize, scratch: u64, scratch_bytes: usize, alignment: u64) !void {
    const src_end = try std.math.add(u64, src, src_bytes);
    const dst_end = try std.math.add(u64, dst, dst_bytes);
    const scratch_end = try std.math.add(u64, scratch, scratch_bytes);
    if (src == 0 or dst == 0 or scratch == 0 or src % alignment != 0 or dst % alignment != 0 or
        scratch % alignment != 0 or (src < dst_end and dst < src_end) or
        (src < scratch_end and scratch < src_end) or (dst < scratch_end and scratch < dst_end))
        return error.NeighborAlias;
}

fn hop(comm: *const Comm, src: u64, count: usize, next: u32, dst: u64, rcount: usize, prev: u32, dtype: cuda.nccl.DataType, stream: cuda.Stream) !void {
    const lib = &comm.lib;
    try lib.check(lib.api.ncclGroupStart(), "neighbor group start");
    const s = lib.api.ncclSend(src, count, dtype, @intCast(next), comm.comm, stream.handle);
    const r = lib.api.ncclRecv(dst, rcount, dtype, @intCast(prev), comm.comm, stream.handle);
    const end = lib.api.ncclGroupEnd();
    try lib.check(s, "neighbor send");
    try lib.check(r, "neighbor receive");
    try lib.check(end, "neighbor group end");
}

fn devices(raw: []const u8) ![2][]const u8 {
    var it = std.mem.splitScalar(u8, raw, ',');
    const a = it.next() orelse return error.BadNeighborDevices;
    const b = it.next() orelse return error.BadNeighborDevices;
    if (a.len == 0 or b.len == 0 or it.next() != null or std.mem.eql(u8, a, b)) return error.BadNeighborDevices;
    return .{ a, b };
}
test "neighbor stages produce global rank order and bounded dispatch" {
    for (0..4) |g| {
        var output: [4]usize = undefined;
        for (0..2) |p| {
            const owner = if (p == g / 2) g else g ^ 3;
            output[2 * p] = owner & ~@as(usize, 1);
            output[2 * p + 1] = owner | 1;
        }
        try std.testing.expectEqual([4]usize{ 0, 1, 2, 3 }, output);
    }
    try std.testing.expect(eligible(cutoff));
    try std.testing.expect(!eligible(cutoff + 16));
    try std.testing.expect(!eligible(15));
    try std.testing.expect(!eligible(0));
}

test "neighbor mode is off unless explicitly enabled" {
    try std.testing.expect(!try parseMode(null));
    try std.testing.expect(!try parseMode("0"));
    try std.testing.expect(try parseMode("1"));
    for ([_][]const u8{ "", "2", "true", " 1" }) |bad|
        try std.testing.expectError(error.BadNeighborMode, parseMode(bad));
}

test "neighbor GID index accepts site settings and rejects invalid values" {
    try std.testing.expectEqual(@as(u8, 5), try parseGid(null));
    try std.testing.expectEqual(@as(u8, 3), try parseGid("3"));
    try std.testing.expectEqual(@as(u8, 5), try parseGid("5"));
    try std.testing.expectEqual(@as(u8, 0), try parseGid("0"));
    try std.testing.expectEqual(@as(u8, 255), try parseGid("255"));
    for ([_][]const u8{ "", "-1", "+3", "256", "0x3", " 3", "three" }) |bad|
        try std.testing.expectError(error.BadNeighborGid, parseGid(bad));
}

test "a stage requires two distinct device names" {
    const names = try devices("mlx5_0,mlx5_1");
    try std.testing.expectEqualStrings("mlx5_0", names[0]);
    try std.testing.expectEqualStrings("mlx5_1", names[1]);
    for ([_][]const u8{ "", "mlx5_0", "mlx5_0,", ",mlx5_1", "mlx5_0,mlx5_0", "mlx5_0,mlx5_1,mlx5_2" }) |bad| {
        try std.testing.expectError(error.BadNeighborDevices, devices(bad));
    }
}

test "the GPU and verbs declarations compile on the host" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Gather);
}

test "invalid relay shapes and buffers fail before either stage is enqueued" {
    var n: Gather = undefined;
    n.scratch.ptr = 0x100000;
    const stream: cuda.Stream = undefined;
    try std.testing.expectError(error.NeighborSize, n.relay(stream, 16, 32, 0));
    try std.testing.expectError(error.NeighborSize, n.relay(stream, 16, 32, cutoff + 16));
    try std.testing.expectError(error.NeighborAlias, n.relay(stream, 16, 32, 32));
    try std.testing.expectError(error.NeighborAlias, n.relay(stream, 17, 64, 16));
    try std.testing.expectError(error.NeighborAlias, n.relay(stream, n.scratch.ptr, 64, 16));
    try std.testing.expectError(error.Overflow, n.relay(stream, std.math.maxInt(u64) - 15, 32, 16));
}

test "diagonal fallback validates every count before its first NCCL group" {
    var comm: Comm = undefined;
    comm.rank = 0;
    comm.world = 4;
    const stream: cuda.Stream = undefined;
    try std.testing.expectError(error.DiagonalType, forward(&comm, 16, 64, .{ 1, 1, 1, 1 }, .f32, 128, 32, stream));
    try std.testing.expectError(error.DiagonalScratchTooSmall, forward(&comm, 16, 64, .{ 1, 1, 1, 17 }, .f16, 128, 32, stream));
    try std.testing.expectError(error.NeighborAlias, forward(&comm, 16, 64, .{ 1, 1, 1, 1 }, .bf16, 16, 32, stream));
    comm.world = 2;
    try std.testing.expectError(error.NotFourNodes, forward(&comm, 16, 64, .{ 1, 1, 1, 1 }, .f16, 128, 32, stream));
}

const ForwardTest = struct {
    const Call = struct { send: bool, ptr: u64, count: usize, peer: c_int };
    var calls: [4]Call = undefined;
    var used: usize = 0;
    var groups: usize = 0;
    var depth: usize = 0;

    fn start() callconv(.c) c_int {
        std.debug.assert(depth == 0);
        depth = 1;
        groups += 1;
        return 0;
    }
    fn end() callconv(.c) c_int {
        std.debug.assert(depth == 1);
        depth = 0;
        return 0;
    }
    fn send(ptr: u64, count: usize, dtype: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        std.debug.assert(depth == 1 and dtype == .bf16);
        calls[used] = .{ .send = true, .ptr = ptr, .count = count, .peer = peer };
        used += 1;
        return 0;
    }
    fn recv(ptr: u64, count: usize, dtype: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        _ = send(ptr, count, dtype, peer, null, null);
        calls[used - 1].send = false;
        return 0;
    }
};

test "production diagonal fallback forwards unequal parts on physical edges at every rank" {
    const F = ForwardTest;
    var comm: Comm = undefined;
    comm.world = 4;
    comm.comm = null;
    comm.lib.api.ncclGroupStart = F.start;
    comm.lib.api.ncclGroupEnd = F.end;
    comm.lib.api.ncclSend = F.send;
    comm.lib.api.ncclRecv = F.recv;
    const stream: cuda.Stream = .{ .d = undefined, .handle = null };
    const counts = [4]usize{ 3, 3, 5, 5 };
    for (0..4) |g| {
        comm.rank = @intCast(g);
        F.used = 0;
        F.groups = 0;
        try forward(&comm, 16, 64, counts, .bf16, 128, 32, stream);
        const next: c_int = @intCast((g + 1) % 4);
        const prev: c_int = @intCast((g + 3) % 4);
        const expected = [_]F.Call{
            .{ .send = true, .ptr = 16, .count = counts[g], .peer = next },
            .{ .send = false, .ptr = 128, .count = counts[@intCast(prev)], .peer = prev },
            .{ .send = true, .ptr = 128, .count = counts[@intCast(prev)], .peer = next },
            .{ .send = false, .ptr = 64, .count = counts[g ^ 2], .peer = prev },
        };
        try std.testing.expectEqual(@as(usize, 2), F.groups);
        try std.testing.expectEqual(@as(usize, 0), F.depth);
        try std.testing.expectEqualSlices(F.Call, &expected, F.calls[0..F.used]);
    }
}
