//! The exact 2D-TP4 split on the prompt path (DESIGN-zig-tp4.md sections 3-4; model.py Comm2D, wo_2d, moe_2d and
//! exl3/experts2d.py on four nodes): node g = 2 p + r is TP2 rank r of pair p. A node computes every output column it
//! holds whole, over its TP2 slice's K in TP2's order, and every exchange only moves bytes, so after an exchange a
//! node holds TP2 rank r's bytes: `quarters` assembles TP2's [2, n, N] gather of the two ranks' partials (which the
//! mixes then add in rank order, as on TP2), `catRank` the rank's whole wo_a output or expert intermediate from the
//! two pairs' parts. Prompt chunks exchange over NCCL across the four (the RDMA rings drop prompt-size bursts on the
//! switch: slots P6, P7b), each as one all-gather of every node's part then copies into place: the bytes Comm2D's
//! NCCL paths (gather4, _p2p) put there. With RDMA rings (Two.rings, ring2d.zig) the exchanges the lane sends over
//! them take them, written straight into place: quarters of fewer than 64 rows (decode windows; TF_DS_2D_PROMPT_NCCL
//! keeps prompt chunks on NCCL) and every cat_rank its column pair's ring takes; the same bytes either way.
//!
//! prompt.zig takes it through thin hooks behind Engine.two; a TP2 engine has none and runs as before. The decode
//! rounds' part is round2d.zig's, on this node state.
const std = @import("std");
const neighbor = @import("neighbor.zig");
const row_state = @import("row_state.zig");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const plan = @import("plan.zig");
const prompt = @import("prompt.zig");
const weights = @import("weights.zig");
const exl3_experts = @import("exl3_experts.zig");
const exl3_experts2d = @import("exl3_experts2d.zig");
const ring2d = @import("ring2d.zig");
const round2d = @import("round2d.zig");
const round_rows = @import("round.zig").max_rows;
const Link = @import("link.zig").Link;
const Comm = @import("comm.zig").Comm;

const nccl = cuda.nccl;

/// The split of node `g` of `world` nodes: TP over 1 or 2, the 2D split over 4.
pub fn split(g: u32, world: u32) !plan.Split {
    if (g >= world) return error.UnsupportedRank;
    return switch (world) {
        1, 2, 3 => .{ .rank = g, .world = world },
        4 => .{ .rank = g % 2, .world = 2, .pair = g / 2 },
        else => error.UnsupportedWorld,
    };
}

/// The links of a run and its NCCL world: rank 0 listens on PORT + g - 1 for follower g, one follower at a time (each
/// follower retries its connect while rank 0 gets to it), so two ranks link exactly as before (rank 1 on PORT).
/// NCCL's unique id goes from rank 0 down every link. (The port's link to N followers replaces this.)
pub const Fan = struct {
    links: [3]?Link = .{ null, null, null },

    pub fn open(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32) !Fan {
        const seconds: u32 = if (world == 3 or world == 4) try startupSeconds(if (std.c.getenv("TF_DS_START_TIMEOUT")) |v| std.mem.span(v) else null) else 300;
        return openTimed(io, ip, port, rank, world, seconds);
    }

    pub fn openTimed(io: std.Io, ip: [4]u8, port: u16, rank: u32, world: u32, seconds: u32) !Fan {
        var f: Fan = .{};
        errdefer f.close();
        if (world > 4) return error.UnsupportedWorld;
        if (rank >= world) return error.UnsupportedRank;
        if (world < 2) return f;
        if (@as(u32, port) + world - 2 > std.math.maxInt(u16)) return error.BadPort;
        if (seconds == 0 or seconds > 86400) return error.BadStartupTimeout;
        const deadline = Link.monotonicNs() + @as(u64, seconds) * std.time.ns_per_s;
        if (rank == 0) {
            for (1..world) |g| {
                const now = Link.monotonicNs();
                if (now >= deadline) return error.StartupTimeout;
                const remaining_ms = (deadline - now + std.time.ns_per_ms - 1) / std.time.ns_per_ms;
                f.links[g - 1] = try Link.listenTimeout(ip, port + @as(u16, @intCast(g - 1)), @intCast(remaining_ms));
            }
        } else {
            f.links[0] = try Link.connect(io, ip, port + @as(u16, @intCast(rank - 1)), seconds);
        }
        return f;
    }

    /// All followers report mode BEFORE NCCL init; rank0 broadcasts a common
    /// verdict. A mixture must not enter different all-gather implementations.
    pub fn checkTransport(f:*const Fan,rank:u32,world:u32,enabled:bool) !void {
        if(world<2) return;
        const ours=[4]u32{row_state.protocol,world,@intFromBool(enabled),0};
        var raw:[16]u8=undefined;
        if(rank==0) {
            var agree=true;
            for(f.links) |maybe| if(maybe) |link| {
                const got=try link.recv(&raw);
                if(got.len!=16 or !std.mem.eql(u8,got,std.mem.asBytes(&ours))) agree=false;
            };
            const verdict=[_]u8{if(agree) 1 else 0};
            for(f.links) |maybe| if(maybe) |link| try link.send(&verdict);
            if(!agree) return error.TransportRankMismatch;
        } else {
            const link=f.links[0].?;
            try link.send(std.mem.asBytes(&ours));
            const got=try link.recv(&raw);
            if(got.len!=1 or got[0]!=1) return error.TransportRankMismatch;
        }
    }

    pub fn close(f: *Fan) void {
        for (&f.links) |*l| {
            if (l.*) |*x| x.close();
            l.* = null;
        }
    }

    /// Comm.init over every link: rank 0 makes the id and sends it to each follower. Needs the CUDA context current.
    pub fn comm(f: *const Fan, rank: u32, world: u32) !Comm {
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (world > 1) {
            if (rank == 0) {
                try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
                for (f.links) |l| if (l) |x| try x.send(std.mem.asBytes(&id));
            } else {
                var buf: [@sizeOf(nccl.UniqueId)]u8 = undefined;
                const got = try f.links[0].?.recv(&buf);
                if (got.len != buf.len) return error.BadUniqueId;
                id = std.mem.bytesToValue(nccl.UniqueId, got);
            }
        } else try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var c: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }
};

pub fn startupSeconds(value: ?[]const u8) error{BadStartupTimeout}!u32 {
    const text = value orelse return 1800;
    if (text.len == 0) return error.BadStartupTimeout;
    for (text) |c| if (c < '0' or c > '9') return error.BadStartupTimeout;
    const seconds = std.fmt.parseInt(u32, text, 10) catch return error.BadStartupTimeout;
    if (seconds == 0 or seconds > 86400) return error.BadStartupTimeout;
    return seconds;
}

test "TP3 startup timeout parses strictly" {
    try std.testing.expectEqual(@as(u32, 1800), try startupSeconds(null));
    try std.testing.expectEqual(@as(u32, 60), try startupSeconds("60"));
    for ([_][]const u8{ "", "0", "-1", " 60", "86401", "999999999999" }) |bad|
        try std.testing.expectError(error.BadStartupTimeout, startupSeconds(bad));
}

test "TP3 fan accepts followers in rank order and closes a partial startup" {
    const Peer = struct {
        fn run(port: u16, rank: u8, result: *bool, expect_close: bool) void {
            var peer = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, port, 3) catch return;
            defer peer.close();
            peer.send(&.{rank}) catch return;
            var buf: [8]u8 = undefined;
            const reply = peer.recv(&buf) catch |err| {
                result.* = expect_close and err == error.PeerDown;
                return;
            };
            result.* = !expect_close and std.mem.eql(u8, reply, "ok");
        }
    };
    const port: u16 = 31000 + @as(u16, @intCast(Link.monotonicNs() % 500));
    var result = [_]bool{ false, false };
    {
        // Start rank2 first: its port is not listened to until rank1 connects.
        const second = try std.Thread.spawn(.{}, Peer.run, .{ port + 1, @as(u8, 2), &result[1], false });
        defer second.join();
        const first = try std.Thread.spawn(.{}, Peer.run, .{ port, @as(u8, 1), &result[0], false });
        defer first.join();
        var fan = try Fan.openTimed(std.testing.io, .{ 127, 0, 0, 1 }, port, 0, 3, 3);
        defer fan.close();
        for (fan.links[0..2], 0..) |maybe, i| {
            var buf: [8]u8 = undefined;
            try std.testing.expectEqualSlices(u8, &.{@intCast(i + 1)}, try maybe.?.recv(&buf));
            try maybe.?.send("ok");
        }
    }
    try std.testing.expect(result[0] and result[1]);
    // No second follower: timeout must close the first accepted socket.
    var closed = false;
    {
        const lone = try std.Thread.spawn(.{}, Peer.run, .{ port + 1000, @as(u8, 1), &closed, true });
        defer lone.join();
        try std.testing.expectError(error.StartupTimeout, Fan.openTimed(std.testing.io, .{ 127, 0, 0, 1 }, port + 1000, 0, 3, 1));
    }
    try std.testing.expect(closed);
}

fn len(r: ?[2]usize) usize {
    const x = r orelse return 0;
    return x[1] - x[0];
}

/// A 2D node's part of the prompt path and the decode rounds: the pairs' widths of each split output and the exchange
/// buffers.
pub const Two = struct {
    neighbor_gather: ?*const neighbor.Gather = null,
    r: usize,
    p: usize,
    heads: usize, // the heads here (its pair's half of the rank's)
    uw: [2]usize, // wo_a's output a pair (its groups x o_lora): pair 0's, pair 1's
    ow: [2]usize, // wo_b's output columns a pair
    gu: [2]usize, // gate / up's columns a pair
    dw: [2]usize, // down's output columns a pair
    ew: [2]usize, // Engram wkv's output columns a pair (0 without an Engram layer)
    hw: [2]usize, // the head's vocabulary columns a pair (253 / 252 blocks of a TP2 half)
    parts: u64, // [4, rows, widest] where the all-gathers land
    pad: u64, // this node's part at the wider pair's width (when the pairs' widths differ)
    part_bytes: usize, // the largest part a node sends
    u_full: u64, // bf16 [cap, uw0 + uw1]: the rank's wo_a output, pair 0's groups then pair 1's
    xd_full: u64, // fp16 [cap * slots, gu0 + gu1]: the rank's expert intermediate, pair 0's blocks then pair 1's
    own: u64, // fp16 [round rows, uw[p]]: a round's half of wo_b's rotated input rows (wo_a's epilogue, round2d.woRot)
    rings: ?*const ring2d.Rings, // the RDMA rings (TF_DS_2D_INTO, TF_DS_2D_GROUPS), when the caller opened them

    /// The widths from the split (both pairs'), and the buffers for chunks of up to `cap` rows (and rounds) from `a`.
    pub fn init(c: *const Config, w: *const weights.Weights, a: *prompt.Arena, s: plan.Split, cap: usize) !Two {
        const pp = s.pair orelse return error.NotA2DSplit;
        const s0 = s.atPair(0);
        const s1 = s.atPair(1);
        const x0 = s0.expertParts(c.*);
        const x1 = s1.expertParts(c.*);
        var t: Two = undefined;
        t.neighbor_gather = null;
        t.r = s.rank;
        t.p = pp;
        t.heads = s.headSpan(c.*)[1];
        t.uw = .{ s0.woAGroups(c.*)[1] * c.o_lora, s1.woAGroups(c.*)[1] * c.o_lora };
        t.ow = .{ len(s0.woBCols(c.*)), len(s1.woBCols(c.*)) };
        t.gu = .{ len(x0.gu), len(x1.gu) };
        t.dw = .{ len(x0.dcols), len(x1.dcols) };
        var engram = false;
        for (w.layers) |lay| engram = engram or lay.engram_wkv != null;
        t.ew = if (engram) .{ len(s0.engramCols(c.*)), len(s1.engramCols(c.*)) } else .{ 0, 0 };
        t.hw = .{ len(s0.headCols(c.*)), len(s1.headCols(c.*)) };
        // the weights loaded here are this split's
        const l0 = w.layers[0];
        if (l0.wq_b.n != t.heads * c.head_dim or l0.groups * c.o_lora != t.uw[pp] or l0.wo_b.n != t.ow[pp] or
            l0.experts.width != t.gu[pp] or l0.experts.down_n != t.dw[pp]) return error.WeightsNotThisSplit;
        // prompt.gather assembles either sublayer's partials: both splits must give the same widths
        if (t.ow[0] != t.dw[0] or t.ow[1] != t.dw[1]) return error.UnequalPartialSplits;
        const slots = c.top_k + 1;
        const biggest = @max(@max(@max(cap * @max(t.ow[0], t.ow[1]) * 4, cap * @max(t.ew[0], t.ew[1]) * 4), @max(cap * @max(t.uw[0], t.uw[1]) * 2, cap * slots * @max(t.gu[0], t.gu[1]) * 2)), @max(t.hw[0], t.hw[1]) * 4);
        t.part_bytes = biggest;
        t.parts = try a.take(4 * biggest);
        t.pad = try a.take(biggest);
        t.u_full = try a.take(cap * (t.uw[0] + t.uw[1]) * 2);
        t.xd_full = try a.take(@max(cap, round_rows) * slots * (t.gu[0] + t.gu[1]) * 2);
        t.own = try a.take(round_rows * t.uw[pp] * 2);
        t.rings = null;
        return t;
    }

    /// Every node's part over NCCL into `parts` [4, n, wmax] (bytes), this node's `src` [n, w[p]] padded to wmax.
    fn allParts(t: *const Two, e: *const prompt.Engine, src: u64, n: usize, w: [2]usize, esize: usize) !usize {
        const wmax = @max(w[0], w[1]);
        if (n * wmax * esize > t.part_bytes) return error.PartTooBig;
        var send = src;
        if (w[t.p] != wmax) {
            try e.ops.copyRows(e.s, src, w[t.p] * esize, t.pad, wmax * esize, w[t.p] * esize, n);
            send = t.pad;
        }
        const bytes=n*wmax*esize;
        if(t.neighbor_gather) |ng| {
            if(neighbor.eligible(bytes)) {
                try ng.gather(e.s,send,t.parts,bytes);
                return wmax;
            }
        }
        try e.comm.allGather(send, t.parts, bytes, .u8, e.s);
        return wmax;
    }

    /// Whether a ring took the exchange: false when there is none or the part does not fit it (nothing sent then).
    fn ringed(r: anyerror!void) !bool {
        r catch |err| switch (err) {
            error.TooLarge, error.NotFloat4 => return false,
            else => return err,
        };
        return true;
    }

    /// quarters (Comm2D.quarters): `src` [n, w[p]] (this node's columns of its TP2 rank's partial) -> `dst`
    /// [2, n, w0 + w1]: rank r's partial, pair 0's columns then pair 1's, for every r.
    pub fn quarters(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (n < 64 and try ringed(rs.quarters(e.s, src, dst, n, w, esize))) return;
        if(t.neighbor_gather) |ng| {
            if(w[0]==w[1] and w[0]*esize%16==0 and neighbor.eligible(n*w[0]*esize)) {
                try ng.quarters(e.s,src,dst,n,w[0]*esize);
                return;
            }
        }
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..4) |g| {
            const rr = g % 2;
            const pg = g / 2;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + rr * n * row + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }


    /// AF's owner-only post/mix/Engram path over all four global ranks.
    /// Preserve two ordered TP2 partials, concatenating columns from both
    /// pairs. Summing four independent partials here would be incorrect.
    pub fn ownedQuarters(t:*const Two,e:*const prompt.Engine,src:u64,dst:u64,n:usize,w:[2]usize) !void {
        var firsts:[4]usize=undefined; var counts:[4]usize=undefined;
        for(0..4) |rank| {
            const r=try row_state.rows(n,rank,4);
            firsts[rank]=r.first; counts[rank]=r.count;
        }
        const rank=e.comm.rank;
        const local=counts[rank];
        const stride=@max(w[0],w[1])*4;
        if(local*stride>t.part_bytes) return error.PartTooBig;
        const ownw=w[t.p]*4;
        // Received slots are contiguous at their SOURCE width, not stride.
        // Only the slot's maximum capacity is padded; no padding is consumed.
        try e.ops.copyRows(e.s,src+firsts[rank]*ownw,ownw,
            t.parts+rank*local*stride,ownw,ownw,local);
        try e.comm.scatterQuarterRows(src,t.parts,t.pad,t.part_bytes,
            firsts,counts,.{w[0]*4,w[1]*4},stride,e.s);
        const row=(w[0]+w[1])*4;
        for(0..4) |source| {
            const rr=source%2; const pp=source/2;
            try e.ops.copyRows(e.s,t.parts+source*local*stride,w[pp]*4,
                dst+rr*local*row+pp*w[0]*4,row,w[pp]*4,local);
        }
    }

    /// cat_rank (Comm2D.cat_rank): `src` [n, w[p]] -> `dst` [n, w0 + w1]: this node's TP2 rank's whole width, pair 0's
    /// part then pair 1's (from its column partner, the same rank of the other pair).
    pub fn catRank(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, w: [2]usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.catRank(e.s, src, dst, n, w, esize))) return;
        if(e.comm.ring_forward and n>=64) {
            const g:usize=e.comm.rank;
            const previous=(g+3)%4;
            const mine=n*w[t.p]*esize;
            const relay=n*w[previous/2]*esize;
            const theirs=n*w[1-t.p]*esize;
            if(@max(mine,@max(relay,theirs))>t.part_bytes) return error.PartTooBig;
            // Only the diagonal column partner is needed. Two physical hops,
            // with exact source widths; no unrelated TP rank's third hop.
            try e.comm.diagonalBytes(src,mine,t.pad,relay,t.parts,theirs,e.s);
            const stride=(w[0]+w[1])*esize;
            try e.ops.copyRows(e.s,src,w[t.p]*esize,dst+t.p*w[0]*esize,
                stride,w[t.p]*esize,n);
            try e.ops.copyRows(e.s,t.parts,w[1-t.p]*esize,dst+(1-t.p)*w[0]*esize,
                stride,w[1-t.p]*esize,n);
            return;
        }
        const wmax = try t.allParts(e, src, n, w, esize);
        const row = (w[0] + w[1]) * esize;
        for (0..2) |pg| {
            const g = t.r + 2 * pg;
            try e.ops.copyRows(e.s, t.parts + g * n * wmax * esize, wmax * esize, dst + pg * w[0] * esize, row, w[pg] * esize, n);
        }
    }

    /// Comm2D.gather (TP2 inside a pair: the DSpark drafter's sublayer partials): `src` [n, width] -> `dst` [2, n, width],
    /// this pair's two ranks in TP2 rank order: over the row pair's ring, else the four-rank all-gather's pair half
    /// (Comm2D.gather4(x)[2p:2p + 2]).
    pub fn pairGather(t: *const Two, e: *const prompt.Engine, src: u64, dst: u64, n: usize, width: usize, esize: usize) !void {
        if (t.rings) |rs| if (try ringed(rs.pairGather(e.s, src, dst, n, width, esize))) return;
        if(t.neighbor_gather) |ng| {
            if(neighbor.eligible(n*width*esize)) {
                try ng.pair(e.s,src,dst,n*width*esize);
                return;
            }
        }
        const wmax = try t.allParts(e, src, n, .{ width, width }, esize);
        try e.ops.copyRows(e.s, t.parts + 2 * t.p * n * wmax * esize, wmax * esize, dst, width * esize, width * esize, 2 * n);
    }

    /// wo_2d after wo_a: the rank's four groups (this node's two in ch.u [n, uw[p]], the partner's two), then wo_b
    /// for this pair's output columns: ch.pa [n, ow[p]] fp32, the matching columns of TP2's partial.
    pub fn woB(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        try t.catRank(e, ch.u, t.u_full, ch.n, t.uw, 2);
        try prompt.mm(e, ch, lay.wo_b, t.u_full, t.uw[0] + t.uw[1], ch.pa, .fp32, lay.wo_b.n);
    }

    /// moe_2d's experts after the routing: this node's gate / up blocks, the rank's whole intermediate assembled, down
    /// for this node's output columns: ch.pm [n, dw[p]] fp32. A chunk of fewer than EXACT_ROWS rows takes the decode
    /// window's fused path and scratch (experts2d.fused_ok), as the rounds do.
    pub fn experts(t: *const Two, e: *const prompt.Engine, ch: *prompt.Chunk, lay: weights.Layer) !void {
        const c = e.c;
        const n = ch.n;
        if (n < exl3_experts.exact_rows) return round2d.experts(t, e, lay.experts, ch.xsd, ch.x, c.hidden, ch.pick, ch.wts, ch.pm, n, c.swiglu_limit);
        try exl3_experts2d.gateUp(e.ex, e.s, lay.experts, ch.xs, ch.x, c.hidden, ch.pick, n, c.swiglu_limit);
        try t.catRank(e, ch.xs.xd, t.xd_full, n * ch.xs.slots, t.gu, 2);
        try exl3_experts2d.down(e.ex, e.s, lay.experts, ch.xs, t.xd_full, ch.pick, ch.wts, ch.pm, n);
    }

    /// Engram's projection summed over the TP2 ranks and cast, kv [n, ew0 + ew1] bf16 = bf16(g0 + g1) (model.py esum
    /// then .to(bf16)), from this node's fp32 columns `src` [n, ew[p]] with less traffic than the quarters: the row
    /// partner's columns over NCCL send / recv, one fp32 add (two operands: the same bits in either order) cast to
    /// bf16, then the column partner's bf16 sums the same way. `tmp` holds n * (ew[p] * 6 + ew[1 - p] * 2) bytes.
    pub fn engramSum(t: *const Two, e: *const prompt.Engine, src: u64, tmp: u64, kv: u64, n: usize) !void {
        const w = t.ew;
        const mine = n * w[t.p];
        const other = tmp;
        const half = tmp + mine * 4;
        const their = half + mine * 2;
        const g: u32 = @intCast(t.r + 2 * t.p);
        try e.comm.exchange(src, mine, other, mine, .f32, g ^ 1, e.s);
        try e.ops.add2Bf16(e.s, src, other, half, mine);
        if(t.neighbor_gather) |_| {
            const counts=[4]usize{n*w[0],n*w[0],n*w[1],n*w[1]};
            try neighbor.forward(e.comm,half,their,counts,.bf16,t.pad,t.part_bytes,e.s);
        } else try e.comm.exchange(half, mine, their, n * w[1 - t.p], .bf16, g ^ 2, e.s);
        const row = (w[0] + w[1]) * 2;
        for (0..2) |pg| {
            const from = if (pg == t.p) half else their;
            try e.ops.copyRows(e.s, from, w[pg] * 2, kv + pg * w[0] * 2, row, w[pg] * 2, n);
        }
    }

    /// Engram's projection summed over the ranks from its column quarters (model.py esum): the gathered width.
    pub fn engramWidth(t: *const Two) usize {
        return t.ew[0] + t.ew[1];
    }
};

test "a node's split: TP over two, the 2D split over four" {
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2 }, try split(1, 2));
    try std.testing.expectEqual(plan.Split{ .rank = 0, .world = 2, .pair = 1 }, try split(2, 4));
    try std.testing.expectEqual(plan.Split{ .rank = 1, .world = 2, .pair = 1 }, try split(3, 4));
    const three = try split(2, 3);
    try std.testing.expectEqual(@as(u32, 2), three.rank);
    try std.testing.expectEqual(@as(u32, 3), three.world);
    try std.testing.expectEqual(@as(?u32, null), three.pair);
    try std.testing.expectError(error.UnsupportedRank, split(3, 3));
    try std.testing.expectError(error.UnsupportedWorld, split(0, 5));
}
