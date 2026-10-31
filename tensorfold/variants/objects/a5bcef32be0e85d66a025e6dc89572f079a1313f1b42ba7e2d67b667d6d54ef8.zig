//! The TP ranks' collectives: NCCL opened at run time, its unique id passed from rank 0 over the link. Every model
//! reduction is an all-gather of fp32 partials that each rank then adds in rank order (DESIGN.md section 3), so the
//! ranks hold the same bits without a broadcast.
const std = @import("std");
const cuda = @import("cuda");
const Link = @import("link.zig").Link;

const nccl = cuda.nccl;

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,

    /// Rank 0 makes the id and sends it; the others take it from the link. Needs the CUDA context current.
    pub fn init(link: Link, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id));
        } else {
            var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
            const got = try link.recv(&buf);
            if (got.len != buf.len) return error.BadUniqueId;
            id = std.mem.bytesToValue(nccl.UniqueId, got);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        c.* = undefined;
    }

    /// `count` elements of `dtype` from every rank into `recv`, rank after rank.
    pub fn allGather(c: Comm, send: u64, recv: u64, count: usize, dtype: nccl.DataType, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream.handle), "ncclAllGather");
    }

    /// `count` elements of `dtype` to rank `peer` and `rcount` from it, in one NCCL group (point to point).
    pub fn exchange(c: Comm, send: u64, count: usize, recv: u64, rcount: usize, dtype: nccl.DataType, peer: u32, stream: cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        const s = c.lib.api.ncclSend(send, count, dtype, @intCast(peer), c.comm, stream.handle);
        const r = c.lib.api.ncclRecv(recv, rcount, dtype, @intCast(peer), c.comm, stream.handle);
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try c.lib.check(s, "ncclSend");
        try c.lib.check(r, "ncclRecv");
    }
    /// Route unreduced partials only to the owner of their rows. The consumer
    /// still sums source ranks in the original order; NCCL never sums floats.
    pub fn scatterRows(c: Comm, send: u64, recv: u64, firsts: [3]usize, counts: [3]usize, row_bytes: usize, stream: cuda.Stream) !void {
        if (c.world != 3 or c.rank >= 3) return error.BadRowOwnership;
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        var opened = true;
        errdefer if (opened) { _ = c.lib.api.ncclGroupEnd(); };
        for (0..3) |peer| {
            if (peer == c.rank) continue;
            try c.lib.check(c.lib.api.ncclSend(send + firsts[peer] * row_bytes,
                counts[peer] * row_bytes, .u8, @intCast(peer), c.comm, stream.handle), "ncclSend row partial");
            try c.lib.check(c.lib.api.ncclRecv(recv + peer * counts[c.rank] * row_bytes,
                counts[c.rank] * row_bytes, .u8, @intCast(peer), c.comm, stream.handle), "ncclRecv row partial");
        }
        const result = c.lib.api.ncclGroupEnd();
        opened = false;
        try c.lib.check(result, "ncclGroupEnd row partial");
    }

    /// Complete only the required row range in-place. Each source slice is
    /// disjoint from all receive slices, and empty CED owners send nothing.
    pub fn gatherOwnedRows(c: Comm, buffer: u64, firsts: [3]usize, counts: [3]usize, row_bytes: usize, stream: cuda.Stream) !void {
        if (c.world != 3 or c.rank >= 3) return error.BadRowOwnership;
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        var opened = true;
        errdefer if (opened) { _ = c.lib.api.ncclGroupEnd(); };
        for (0..3) |peer| {
            if (peer == c.rank) continue;
            if (counts[c.rank] > 0) try c.lib.check(c.lib.api.ncclSend(buffer + firsts[c.rank] * row_bytes,
                counts[c.rank] * row_bytes, .u8, @intCast(peer), c.comm, stream.handle), "ncclSend owned rows");
            if (counts[peer] > 0) try c.lib.check(c.lib.api.ncclRecv(buffer + firsts[peer] * row_bytes,
                counts[peer] * row_bytes, .u8, @intCast(peer), c.comm, stream.handle), "ncclRecv owned rows");
        }
        const result = c.lib.api.ncclGroupEnd();
        opened = false;
        try c.lib.check(result, "ncclGroupEnd owned rows");
    }

};
