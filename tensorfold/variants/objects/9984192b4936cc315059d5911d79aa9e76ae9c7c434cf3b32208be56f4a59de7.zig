//! The family's one-hop RDMA all-gather of small fp32 slices (TP decode on GB10), host half: the Python build's
//! cuda/rdma_gather.cu v7 in Zig. The staging kernel copies this rank's slice into a slot of pinned host memory and
//! rings a doorbell; the proxy thread splits the slot over the node's RDMA devices and writes the parts, with an
//! immediate, into every peer's receive slot for this rank; a peer's proxy sees each part's receive completion and,
//! once all of a sequence's parts are in, publishes it in that peer's flag; the collecting kernel waits for the flags
//! and copies the slots out in rank order (or straight into the caller's views). The kernel pair is rdma_gather.cu's
//! and runs in graphs like any other kernel; a failed write or `abort` sets an abort word every wait also reads.
const std = @import("std");
const cuda = @import("cuda");
const fabric = @import("fabric");
const seqs = @import("rdma_seq.zig");

const vabi = fabric.verbs_abi;
const Verbs = fabric.verbs.Verbs;
const DevicePtr = cuda.abi.DevicePtr;

pub const max_ranks = seqs.max_ranks;
pub const max_devices = seqs.max_devices;
pub const parsePeerMasks = seqs.peerMasks;
const recv_depth = 256; // receive requests kept posted on every QP (a write with immediate consumes one)
const send_depth = 512;
const cq_depth = 4096;
const stage_threads = 512;
const collect_threads = 256;
const wc_recv_rdma_with_imm = 129;
const host_register_portable = 0x01;
const host_register_devicemap = 0x02;

pub const Error = error{ VerbsFailed, PostFailed, BadMtu, BadInfo, TooLarge, Failed, NoHostRegister } || cuda.Error || fabric.verbs.Error || std.posix.MMapError || std.posix.MadviseError || std.Thread.SpawnError;

/// The served ring has its own GID selection, independent of NCCL's.
pub fn parseGidIndex(text: ?[]const u8) error{InvalidRdmaGidIndex}!u8 {
    const value = text orelse return 5;
    if (value.len == 0) return error.InvalidRdmaGidIndex;
    for (value) |c| if (c < '0' or c > '9') return error.InvalidRdmaGidIndex;
    return std.fmt.parseInt(u8, value, 10) catch error.InvalidRdmaGidIndex;
}

pub fn servedGidIndex() error{InvalidRdmaGidIndex}!u8 {
    const text = if (std.c.getenv("TF_RDMA_GID_INDEX")) |v| std.mem.span(v) else null;
    return parseGidIndex(text) catch |err| {
        std.debug.print("TF_RDMA_GID_INDEX: expected a decimal GID index in 0..255\n", .{});
        return err;
    };
}

/// ibv_mtu is an ordered enum, 1=256 through 5=4096. Both ends use the smaller active MTU.
fn pathMtu(local: c_int, remote: c_int) error{BadMtu}!c_int {
    if (local < 1 or local > 5 or remote < 1 or remote > 5) return error.BadMtu;
    return @min(local, remote);
}

/// The Python build's switches, with their defaults (TF_RDMA_SLOTS, TF_RDMA_STAGE_BLOCKS, ..., TF_RDMA_PDL as served).
pub const Settings = struct {
    slots: u32 = 4,
    max_bytes: usize,
    gid_index: u8 = 5,
    stage_blocks: u32 = 16,
    collect_blocks: u32 = 48,
    collect_grain: u32 = 4,
    pdl: bool = true,
    own_early: bool = true,
    unroll: bool = true,
    fence: bool = true,
    probing: bool = false,
    peer_masks: ?[max_ranks]u8 = null,
};

/// What the other side's queue pairs toward this rank need (a device each), and where this rank's receive ring is.
pub const Info = extern struct {
    devices: u32,
    qpn: [max_devices]u32,
    psn: [max_devices]u32,
    rkey: [max_devices]u32,
    gid: [max_devices][16]u8,
    active_mtu: [max_devices]c_int, // requires the same Info layout on every rank
    recv: u64,
};

/// The kernel pair, from the family's rdma_gather fatbin.
pub const Kernels = struct {
    stage: cuda.Function,
    collect: cuda.Function,
    collect_into: cuda.Function,

    pub fn load(m: cuda.Module) cuda.Error!Kernels {
        return .{ .stage = try m.function("dsv41_rdma_stage"), .collect = try m.function("dsv41_rdma_collect"), .collect_into = try m.function("dsv41_rdma_collect_into") };
    }
};

/// cuMemHostRegister / cuMemHostUnregister, which the runtime's driver table does not hold.
const HostReg = struct {
    lib: std.DynLib,
    register: *const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result,
    unregister: *const fn (?*anyopaque) callconv(.c) cuda.abi.Result,

    fn open() Error!HostReg {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.NoHostRegister;
        errdefer lib.close();
        return .{
            .lib = lib,
            .register = lib.lookup(*const fn (?*anyopaque, usize, c_uint) callconv(.c) cuda.abi.Result, "cuMemHostRegister_v2") orelse return error.NoHostRegister,
            .unregister = lib.lookup(*const fn (?*anyopaque) callconv(.c) cuda.abi.Result, "cuMemHostUnregister") orelse return error.NoHostRegister,
        };
    }
};

extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

/// CPU/GPU/RDMA share these exact physical pages; a decoder child must not inherit them.
/// Apply before CUDA/verbs registration so fork cannot make a later parent write relocate a DMA page.
fn dmaMapping(n: usize) (std.posix.MMapError || std.posix.MadviseError)![]align(std.heap.page_size_min) u8 {
    const mem = try std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    errdefer std.posix.munmap(mem);
    if (@import("builtin").os.tag == .linux)
        try std.posix.madvise(mem.ptr, mem.len, std.os.linux.MADV.DONTFORK);
    return mem;
}

/// Zeroed pages registered with CUDA (mapped, portable). On GB10 the GPU reaches them over the coherent path of its own
/// memory, where cuMemHostAlloc memory goes through an uncached one; mlock keeps them resident (best effort).
const Pinned = struct {
    mem: []align(std.heap.page_size_min) u8,
    dev: DevicePtr,

    fn alloc(d: *const cuda.Driver, hr: *const HostReg, len: usize) Error!Pinned {
        const n = std.mem.alignForward(usize, len, 1 << 16);
        const mem = try dmaMapping(n);
        errdefer std.posix.munmap(mem);
        @memset(mem, 0);
        _ = mlock(mem.ptr, mem.len);
        try d.check(hr.register(mem.ptr, mem.len, host_register_portable | host_register_devicemap), "cuMemHostRegister");
        var dev: DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, mem.ptr, 0), "cuMemHostGetDevicePointer");
        return .{ .mem = mem, .dev = dev };
    }

    fn free(p: *Pinned, hr: *const HostReg) void {
        _ = hr.unregister(p.mem.ptr);
        std.posix.munmap(p.mem);
    }
};

/// One RDMA device: its context, protection domain, one completion queue for every QP, the rings' registrations and an
/// RC queue pair per peer.
const Nic = struct {
    ctx: *vabi.Context,
    pd: *vabi.Pd,
    cq: *vabi.Cq,
    mr_send: *vabi.Mr,
    mr_recv: *vabi.Mr,
    gid: [16]u8,
    active_mtu: c_int,
    qps: [max_ranks]?*vabi.Qp = @splat(null),
    psns: [max_ranks]u32 = @splat(0),
};

const Peer = struct {
    qpn: [max_devices]u32 = @splat(0),
    psn: [max_devices]u32 = @splat(0),
    rkey: [max_devices]u32 = @splat(0),
    gid: [max_devices][16]u8 = @splat(@splat(0)),
    recv: u64 = 0,
};

pub const Ring = struct {
    d: *const cuda.Driver,
    verbs: Verbs,
    v: *const Verbs = undefined,
    k: Kernels,
    s: Settings,
    rank: u32,
    world: u32,
    hr: HostReg,
    send: Pinned,
    recv: Pinned,
    meta: Pinned,
    scalars: cuda.DeviceBuffer, // seq (u64), staged (u32), probe (8 x u64)
    nics: [max_devices]Nic = undefined,
    nd: u32 = 0,
    peers: [max_ranks]Peer = @splat(.{}),
    proxy: ?std.Thread = null,
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    why: [128]u8 = @splat(0),
    tracing: bool = false,
    failure_peer: ?*Ring = null,

    // the metadata block: flags [slots][world], doorbell, sizes [slots], sent [slots], abort word
    fn flags(r: *const Ring) [*]u64 {
        return @ptrCast(@alignCast(r.meta.mem.ptr));
    }
    fn doorbell(r: *const Ring) *u64 {
        return &r.flags()[r.s.slots * r.world];
    }
    fn sizes(r: *const Ring) [*]u64 {
        return r.flags() + r.s.slots * r.world + 1;
    }
    fn sent(r: *const Ring) [*]u64 {
        return r.sizes() + r.s.slots;
    }
    fn abortWord(r: *const Ring) *u64 {
        return &r.sent()[r.s.slots];
    }
    fn metaDev(r: *const Ring, p: anytype) DevicePtr {
        return r.meta.dev + (@intFromPtr(p) - @intFromPtr(r.meta.mem.ptr));
    }

    /// Rings, metadata and queue pairs (in INIT, receives posted) on `devices`; `connect` with every rank's `info` next.
    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, k: Kernels, devices: []const []const u8, rank: u32, world: u32, s: Settings) Error!*Ring {
        if (world < 2 or world > max_ranks or rank >= world or devices.len == 0 or devices.len > max_devices) return error.BadInfo;
        if (s.slots < 2 or s.slots > seqs.max_slots or s.max_bytes == 0 or s.max_bytes % 64 != 0) return error.BadInfo;
        if (s.peer_masks) |masks| {
            const allowed: u8 = (@as(u8, 1) << @intCast(devices.len)) - 1;
            for (masks[0..world], 0..) |mask, peer| {
                if (mask & ~allowed != 0 or (peer == rank and mask != 0) or (peer != rank and mask == 0)) return error.BadInfo;
            }
        }
        const r = try gpa.create(Ring);
        errdefer gpa.destroy(r);
        var verbs = try Verbs.open(null);
        errdefer verbs.close();
        var hr = try HostReg.open();
        errdefer hr.lib.close();
        var send = try Pinned.alloc(d, &hr, s.slots * s.max_bytes);
        errdefer send.free(&hr);
        var recv = try Pinned.alloc(d, &hr, s.slots * world * s.max_bytes);
        errdefer recv.free(&hr);
        var meta = try Pinned.alloc(d, &hr, (s.slots * world + 1 + 2 * s.slots + 1) * 8);
        errdefer meta.free(&hr);
        var scalars = try cuda.DeviceBuffer.alloc(d, 80);
        errdefer scalars.free();
        try scalars.fill8(0, null);
        r.* = .{ .d = d, .verbs = verbs, .k = k, .s = s, .rank = rank, .world = world, .hr = hr, .send = send, .recv = recv, .meta = meta, .scalars = scalars };
        r.v = &r.verbs;
        r.tracing = if (std.c.getenv("TF_RDMA_TRACE")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        errdefer for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        for (devices) |name| {
            r.nics[r.nd] = try r.openNic(name);
            r.nd += 1;
        }
        r.dumpRegistrations("created");
        return r;
    }

    pub fn dumpRegistrations(r: *const Ring, stage_name: []const u8) void {
        if (!r.tracing) return;
        for (r.nics[0..r.nd], 0..) |nic, index| {
            const send_addr: usize = if (nic.mr_send.addr) |v| @intFromPtr(v) else 0;
            const recv_addr: usize = if (nic.mr_recv.addr) |v| @intFromPtr(v) else 0;
            std.log.info("RDMA MR rank {d} {s} device {d}: send={d}/{d} mr={d}/{d} lkey={d}, recv={d}/{d} mr={d}/{d} rkey={d}", .{r.rank,stage_name,index,@intFromPtr(r.send.mem.ptr),r.send.mem.len,send_addr,nic.mr_send.length,nic.mr_send.lkey,@intFromPtr(r.recv.mem.ptr),r.recv.mem.len,recv_addr,nic.mr_recv.length,nic.mr_recv.rkey});
        }
    }

    fn openNic(r: *Ring, name: []const u8) Error!Nic {
        const api = r.v.api;
        const ctx = try r.v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const port = try r.v.port(ctx, 1);
        const active_mtu = try pathMtu(port.active_mtu, port.active_mtu);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const cq = api.ibv_create_cq(ctx, cq_depth, null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        const access = vabi.access_local_write | vabi.access_remote_write;
        const mr_send = api.ibv_reg_mr(pd, r.send.mem.ptr, r.send.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_send);
        const mr_recv = api.ibv_reg_mr(pd, r.recv.mem.ptr, r.recv.mem.len, access) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dereg_mr(mr_recv);
        const gid = try r.v.gid(ctx, 1, r.s.gid_index);
        var nic: Nic = .{ .ctx = ctx, .pd = pd, .cq = cq, .mr_send = mr_send, .mr_recv = mr_recv, .gid = gid.raw, .active_mtu = active_mtu };
        errdefer for (nic.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        for (0..r.world) |p| {
            if (p == r.rank) continue;
            if (r.s.peer_masks) |masks| if (seqs.railIndex(masks[p], r.nd) == null) continue;
            var init: vabi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = recv_depth, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 0 }, .qp_type = vabi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            nic.qps[p] = qp;
            var a: vabi.QpAttr = .{ .qp_state = vabi.qps_init, .port_num = 1, .qp_access_flags = vabi.access_remote_write };
            const m = vabi.mask;
            if (api.ibv_modify_qp(qp, &a, m.state | m.pkey_index | m.port | m.access_flags) != 0) return error.VerbsFailed;
            for (0..recv_depth) |_| try postRecv(ctx, qp);
            nic.psns[p] = (qp.qp_num *% 2654435761) & 0xFF_FFFF;
        }
        return nic;
    }

    fn closeNic(r: *Ring, n: *Nic) void {
        const api = r.v.api;
        for (n.qps) |q| if (q) |qp| {
            _ = api.ibv_destroy_qp(qp);
        };
        _ = api.ibv_dereg_mr(n.mr_recv);
        _ = api.ibv_dereg_mr(n.mr_send);
        _ = api.ibv_destroy_cq(n.cq);
        _ = api.ibv_dealloc_pd(n.pd);
        _ = api.ibv_close_device(n.ctx);
    }

    pub fn destroy(r: *Ring, gpa: std.mem.Allocator) void {
        r.stop();
        for (r.nics[0..r.nd]) |*n| r.closeNic(n);
        r.scalars.free();
        r.meta.free(&r.hr);
        r.recv.free(&r.hr);
        r.send.free(&r.hr);
        r.hr.lib.close();
        r.verbs.close();
        gpa.destroy(r);
    }

    /// What peer `p`'s queue pairs toward this rank need.
    pub fn info(r: *const Ring, p: u32) Info {
        var out: Info = std.mem.zeroes(Info);
        const mask = seqs.deviceMask(r.s.peer_masks, p, r.nd);
        out.devices = @popCount(mask);
        out.recv = @intFromPtr(r.recv.mem.ptr);
        for (r.nics[0..r.nd], 0..) |n, d| {
            const wire = seqs.railIndex(mask, d) orelse continue;
            out.qpn[wire] = if (n.qps[p]) |qp| qp.qp_num else 0;
            out.psn[wire] = n.psns[p];
            out.rkey[wire] = n.mr_recv.rkey;
            out.gid[wire] = n.gid;
            out.active_mtu[wire] = n.active_mtu;
        }
        return out;
    }

    /// `remote[p]`: what rank p's `info(rank)` gave. Moves every QP to RTR and RTS (the Python build's attributes).
    pub fn connect(r: *Ring, remote: []const Info) Error!void {
        if (remote.len != r.world) return error.BadInfo;
        // Validate all peer MTUs before transitioning any QP.
        for (remote, 0..) |ri, p| {
            if (p == r.rank) continue;
            const mask = seqs.deviceMask(r.s.peer_masks, p, r.nd);
            if (ri.devices != @popCount(mask)) return error.BadInfo;
            for (r.nics[0..r.nd], 0..) |n, d| {
                const wire = seqs.railIndex(mask, d) orelse continue;
                _ = try pathMtu(n.active_mtu, ri.active_mtu[wire]);
                if (n.qps[p] == null) return error.BadInfo;
            }
        }
        const api = r.v.api;
        const m = vabi.mask;
        for (remote, 0..) |ri, p| {
            if (p == r.rank) continue;
            const peer = &r.peers[p];
            peer.* = .{ .recv = ri.recv };
            const mask = seqs.deviceMask(r.s.peer_masks, p, r.nd);
            for (r.nics[0..r.nd], 0..) |n, d| {
                const wire = seqs.railIndex(mask, d) orelse continue;
                peer.qpn[d] = ri.qpn[wire];
                peer.psn[d] = ri.psn[wire];
                peer.rkey[d] = ri.rkey[wire];
                peer.gid[d] = ri.gid[wire];
                const qp = n.qps[p].?;
                std.debug.print("RDMA rank {d} peer {d} local device {d} wire rail {d}: gid_index={d} path_mtu={d}\n", .{ r.rank, p, d, wire, r.s.gid_index, try pathMtu(n.active_mtu, ri.active_mtu[wire]) });
                var a: vabi.QpAttr = .{ .qp_state = vabi.qps_rtr, .path_mtu = try pathMtu(n.active_mtu, ri.active_mtu[wire]), .dest_qp_num = ri.qpn[wire], .rq_psn = ri.psn[wire], .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
                a.ah_attr = .{ .grh = .{ .dgid = .{ .raw = ri.gid[wire] }, .sgid_index = r.s.gid_index, .hop_limit = 64 }, .is_global = 1, .port_num = 1 };
                if (api.ibv_modify_qp(qp, &a, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
                a = .{ .qp_state = vabi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = n.psns[p], .max_rd_atomic = 1 };
                if (api.ibv_modify_qp(qp, &a, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
            }
        }
    }

    pub fn start(r: *Ring) Error!void {
        if (r.running.swap(true, .acq_rel)) return;
        r.proxy = try std.Thread.spawn(.{}, proxyLoop, .{r});
    }

    pub fn stop(r: *Ring) void {
        if (r.running.swap(false, .acq_rel)) if (r.proxy) |t| t.join();
        r.proxy = null;
    }

    /// Every wait on this rank gives up from now on (a watchdog lost a peer); `failure` says why.
    pub fn abort(r: *Ring, why: []const u8) void {
        r.fail(why);
    }

    pub fn failure(r: *const Ring) ?[]const u8 {
        if (!r.failed.load(.acquire)) return null;
        return std.mem.sliceTo(&r.why, 0);
    }

    fn fail(r: *Ring, why: []const u8) void {
        if (!r.failed.swap(true, .acq_rel)) {
            const n = @min(why.len, r.why.len - 1);
            @memcpy(r.why[0..n], why[0..n]);
            std.log.err("RDMA rank {d}: proxy failed: {s}", .{ r.rank, why });
        }
        @atomicStore(u64, r.abortWord(), 1, .release);
        if(r.failure_peer) |other| {
            @atomicStore(u64,other.abortWord(),1,.release);
        }
    }

    fn proxyLoop(r: *Ring) void {
        const nd = r.nd;
        var need: u32 = 0;
        for (0..r.world) |p| if (p != r.rank) { need += @popCount(seqs.deviceMask(r.s.peer_masks, p, nd)); };
        var sends: seqs.Sends = .{ .need = need, .slots = r.s.slots };
        var arrivals: seqs.Arrivals = .{ .nd = nd, .masks = r.s.peer_masks };
        var posted: u64 = 0;
        var wc: [64]vabi.Wc = undefined;
        while (r.running.load(.monotonic)) {
            const ring = @atomicLoad(u64, r.doorbell(), .acquire);
            while (posted < ring) {
                const q = posted + 1;
                const slot: usize = @intCast(q % r.s.slots);
                const n = @atomicLoad(u64, &r.sizes()[slot], .acquire);
                if (n > r.s.max_bytes) {
                    std.log.err("RDMA rank {d}: queued bytes {d} exceed slot {d}, seq={d}", .{r.rank,n,r.s.max_bytes,q});
                    r.fail("queued payload exceeds registered slot");
                    return;
                }
                if (r.tracing and q <= 8)
                    std.log.info("RDMA QUEUE rank {d}: seq={d} slot={d} bytes={d} max={d}", .{r.rank,q,slot,n,r.s.max_bytes});
                for (r.nics[0..nd], 0..) |nic, d| {
                    for (0..r.world) |p| {
                        if (p == r.rank) continue;
                        const mask = seqs.deviceMask(r.s.peer_masks, p, nd);
                        const wire = seqs.railIndex(mask, d) orelse continue;
                        const pt = seqs.part(n, @popCount(mask), wire);
                        const off: usize = @intCast(pt.off);
                        const len: usize = @intCast(pt.len);
                        const local = r.send.mem[slot * r.s.max_bytes + off ..][0..len];
                        const at: u64 = @intCast((slot * r.world + r.rank) * r.s.max_bytes + off);
                        if (r.tracing and q <= 8)
                            std.log.info("RDMA WRITE rank {d} device {d} peer {d}: seq={d} local={d} bytes={d} lkey={d} remote={d} rkey={d}", .{r.rank,d,p,q,@intFromPtr(local.ptr),local.len,nic.mr_send.lkey,r.peers[p].recv+at,r.peers[p].rkey[d]});
                        postWrite(nic.ctx, nic.qps[p].?, local, nic.mr_send.lkey, r.peers[p].recv + at, r.peers[p].rkey[d], q) catch {
                            r.fail("ibv_post_send failed");
                            return;
                        };
                    }
                }
                posted = q;
            }
            for (r.nics[0..nd], 0..) |nic, d| {
                const k = nic.ctx.ops.poll_cq(nic.cq, wc.len, &wc);
                if (k < 0) {
                    r.fail("ibv_poll_cq failed");
                    return;
                }
                for (wc[0..@intCast(k)]) |w| {
                    if (w.status != vabi.wc_success) {
                        std.log.err("RDMA rank {d} device {d}: CQ status={d} opcode={d} wr_id={d} qp_num={d} vendor_err={d}", .{ r.rank, d, w.status, w.opcode, w.wr_id, w.qp_num, w.vendor_err });
                        r.fail(r.v.statusText(w.status));
                        return;
                    }
                    if (w.opcode == wc_recv_rdma_with_imm) {
                        const p = r.peerOf(nic, w.qp_num) orelse {
                            r.fail("receive completion on an unknown QP");
                            return;
                        };
                        const got = arrivals.arrive(p, d, w.imm_data);
                        var q = got.from;
                        while (q <= got.to) : (q += 1) @atomicStore(u64, r.flagAt(q, p), q, .release);
                        postRecv(nic.ctx, nic.qps[p].?) catch {
                            r.fail("ibv_post_recv failed");
                            return;
                        };
                    } else if (sends.complete(w.wr_id)) |q| {
                        @atomicStore(u64, &r.sent()[@as(usize, @intCast(q % r.s.slots))], q, .release);
                    }
                }
            }
        }
    }

    fn flagAt(r: *const Ring, q: u64, p: usize) *u64 {
        const slot: usize = @intCast(q % r.s.slots);
        return &r.flags()[slot * r.world + p];
    }
    fn peerOf(r: *const Ring, nic: Nic, qpn: u32) ?usize {
        for (0..r.world) |p| if (nic.qps[p]) |qp| if (qp.qp_num == qpn) return p;
        return null;
    }
    fn scalar(r: *const Ring, offset: usize) DevicePtr {
        return r.scalars.ptr + offset;
    }
    /// recv [world x n4 float4] <- every rank's send [n4 float4] (fp32, 16-byte aligned), on `stream`.
    pub fn gather(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, recv: DevicePtr) Error!void {
        if (r.failure() != null) return error.Failed;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        const own: DevicePtr = if (r.s.own_early) recv + @as(u64, r.rank) * n4 * 16 else 0;
        try r.stage(stream, send, n4, own, n4, n4);
        var a: cuda.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.recv.dev);
        a.add(r.metaDev(r.flags()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(r.s.own_early)));
        a.add(@as(c_int, @intFromBool(r.s.unroll)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const per = collect_threads * r.s.collect_grain;
        const blocks = std.math.clamp((@as(u64, n4) * r.world + per - 1) / per, 1, r.s.collect_blocks);
        try cuda.launch.launch(r.k.collect, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }
    /// A rank's slice in `dsts` (its rows `row4` float4 wide, `stride4` apart; ptr 0: not copied out).
    pub const Dst = struct { ptr: DevicePtr = 0, row4: u32 = 0, stride4: u64 = 0 };
    /// Every rank's [rows, w_p] fp32 slice straight into dsts[p]; `send` is this rank's slice, contiguous. Every rank
    /// names the same widths, so each peer's ring slot holds rows x w_p floats.
    pub fn gatherInto(r: *Ring, stream: cuda.Stream, send: DevicePtr, rows: u32, own_row4: u32, dsts: []const Dst) Error!void {
        if (r.failure() != null) return error.Failed;
        if (dsts.len != r.world) return error.BadInfo;
        const n4 = rows * own_row4;
        if (@as(usize, n4) * 16 > r.s.max_bytes) return error.TooLarge;
        const D = extern struct { ptr: [8]u64, stride4: [8]i64, row4: [8]c_int };
        var dd: D = std.mem.zeroes(D);
        var copy4: u64 = 0;
        for (dsts, 0..) |t, p| {
            if (t.ptr == 0) continue;
            if (@as(usize, rows) * t.row4 * 16 > r.s.max_bytes) return error.TooLarge;
            dd.ptr[p] = t.ptr;
            dd.row4[p] = @intCast(t.row4);
            dd.stride4[p] = @intCast(t.stride4);
            if (p != r.rank) copy4 += @as(u64, rows) * t.row4;
        }
        const mine = dsts[r.rank];
        try r.stage(stream, send, n4, mine.ptr, if (mine.ptr != 0) mine.row4 else n4, if (mine.ptr != 0) mine.stride4 else n4);
        var a: cuda.Args = .{};
        a.add(dd);
        a.add(@as(c_int, @intCast(rows)));
        a.add(r.recv.dev);
        a.add(r.metaDev(r.flags()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const per = collect_threads * r.s.collect_grain;
        const blocks = std.math.clamp((copy4 + per - 1) / per, 1, r.s.collect_blocks);
        try cuda.launch.launch(r.k.collect_into, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = collect_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }
    fn probe(r: *const Ring) DevicePtr {
        return if (r.s.probing) r.scalar(16) else 0;
    }
    fn stage(r: *Ring, stream: cuda.Stream, send: DevicePtr, n4: u32, own: DevicePtr, own_row4: u32, own_stride4: u64) Error!void {
        var a: cuda.Args = .{};
        a.add(send);
        a.add(@as(c_int, @intCast(n4)));
        a.add(r.send.dev);
        a.add(r.metaDev(r.sizes()));
        a.add(r.metaDev(r.doorbell()));
        a.add(r.metaDev(r.sent()));
        a.add(r.scalar(0));
        a.add(@as(c_int, @intCast(r.s.slots)));
        a.add(@as(u64, r.s.max_bytes));
        a.add(r.probe());
        a.add(r.metaDev(r.abortWord()));
        a.add(r.scalar(8));
        a.add(@as(c_int, @intFromBool(r.s.fence)));
        a.add(own);
        a.add(@as(c_int, @intCast(own_row4)));
        a.add(@as(c_longlong, @intCast(own_stride4)));
        a.add(@as(c_int, @intFromBool(r.s.pdl)));
        const blocks = std.math.clamp((@as(u64, n4) + 1023) / 1024, 1, r.s.stage_blocks);
        try cuda.launch.launch(r.k.stage, .{ .grid = .{ .x = @intCast(blocks), .y = 1, .z = 1 }, .block = .{ .x = stage_threads, .y = 1, .z = 1 }, .pdl = r.s.pdl }, stream, &a);
    }
    /// (stage, doorbell to every flag, copy-out) mean ns over the probed gathers and their count; resets the sums.
    pub fn probes(r: *Ring) Error![4]f64 {
        var host: [8]u64 = undefined;
        try r.scalars.download(16, std.mem.sliceAsBytes(&host));
        try r.d.check(r.d.api.cuMemsetD8_v2(r.scalar(16), 0, 64), "cuMemsetD8");
        const n: f64 = if (host[3] > 0) @floatFromInt(host[3]) else 1;
        return .{ @as(f64, @floatFromInt(host[0])) / n, @as(f64, @floatFromInt(host[1])) / n, @as(f64, @floatFromInt(host[2])) / n, @floatFromInt(host[3]) };
    }
};
fn postRecv(ctx: *vabi.Context, qp: *vabi.Qp) Error!void {
    var wr: vabi.RecvWr = .{};
    var bad: ?*vabi.RecvWr = null;
    if (ctx.ops.post_recv(qp, &wr, &bad) != 0) return error.PostFailed;
}
fn postWrite(ctx: *vabi.Context, qp: *vabi.Qp, local: []const u8, lkey: u32, remote: u64, rkey: u32, q: u64) Error!void {
    var sge: vabi.Sge = .{ .addr = @intFromPtr(local.ptr), .length = @intCast(local.len), .lkey = lkey };
    var wr: vabi.SendWr = .{ .wr_id = q, .sg_list = @ptrCast(&sge), .num_sge = @intFromBool(local.len > 0), .opcode = vabi.wr_rdma_write_imm, .send_flags = vabi.send_signaled, .imm_data = seqs.immediate(q), .remote_addr = remote, .rkey = rkey };
    var bad: ?*vabi.SendWr = null;
    if (ctx.ops.post_send(qp, &wr, &bad) != 0) return error.PostFailed;
}
test "every declaration compiles (the GPU and verbs paths run only on the nodes)" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Ring);
    std.testing.refAllDecls(Kernels);
}
test "the info a peer reads is a plain extern struct (it crosses the link as bytes)" {
    try std.testing.expectEqual(@as(usize, 144), @sizeOf(Info));
    try std.testing.expectEqual(@as(usize, 116), @offsetOf(Info, "active_mtu"));
    try std.testing.expectEqual(@as(usize, 136), @offsetOf(Info, "recv"));
}
test "served GID defaults to 5 and accepts explicit 3 without silently accepting malformed settings" {
    try std.testing.expectEqual(@as(u8, 5), try parseGidIndex(null));
    try std.testing.expectEqual(@as(u8, 3), try parseGidIndex("3"));
    try std.testing.expectEqual(@as(u8, 0), try parseGidIndex("0"));
    try std.testing.expectEqual(@as(u8, 255), try parseGidIndex("255"));
    for ([_][]const u8{ "", "-1", "+3", " 3", "3 ", "0x3", "3_0", "256", "invalid" }) |value|
        try std.testing.expectError(error.InvalidRdmaGidIndex, parseGidIndex(value));
}
test "RDMA MTU negotiation is symmetric for every supported port MTU and rejects invalid metadata" {
    for (1..6) |a| for (1..6) |b| {
        try std.testing.expectEqual(@as(c_int, @intCast(@min(a, b))), try pathMtu(@intCast(a), @intCast(b)));
    };
    for ([_]c_int{ -1, 0, 6, 4096 }) |bad| {
        try std.testing.expectError(error.BadMtu, pathMtu(bad, 3));
        try std.testing.expectError(error.BadMtu, pathMtu(3, bad));
    }
}
test "connect programs the negotiated MTU and configured GID and rejects bad peers before QP mutation" {
    const Probe = struct {
        var calls: usize = 0;
        var mtu: c_int = 0;
        var gid: u8 = 0;
        fn modify(_: *vabi.Qp, attr: *vabi.QpAttr, _: c_int) callconv(.c) c_int {
            calls += 1;
            if (attr.qp_state == vabi.qps_rtr) {
                mtu = attr.path_mtu;
                gid = attr.ah_attr.grh.sgid_index;
            }
            return 0;
        }
    };
    var verbs: Verbs = undefined;
    verbs.api.ibv_modify_qp = Probe.modify;
    var r: Ring = undefined;
    r.v = &verbs;
    r.rank = 0;
    r.world = 2;
    r.nd = 1;
    r.s = .{ .max_bytes = 64, .gid_index = 3 };
    r.nics[0].qps[1] = @ptrFromInt(0x1000);
    r.nics[0].psns[1] = 1;
    var peers: [2]Info = @splat(std.mem.zeroes(Info));
    peers[1].devices = 1;
    for ([_][2]c_int{ .{ 3, 5 }, .{ 5, 3 }, .{ 5, 5 } }) |mtus| {
        r.nics[0].active_mtu = mtus[0];
        peers[1].active_mtu[0] = mtus[1];
        Probe.calls = 0;
        try r.connect(&peers);
        try std.testing.expectEqual(@as(usize, 2), Probe.calls);
        try std.testing.expectEqual(@min(mtus[0], mtus[1]), Probe.mtu);
        try std.testing.expectEqual(@as(u8, 3), Probe.gid);
    }
    Probe.calls = 0;
    peers[1].active_mtu[0] = 0;
    try std.testing.expectError(error.BadMtu, r.connect(&peers));
    try std.testing.expectEqual(@as(usize, 0), Probe.calls);
}

test "triangle connects each wire rail to the selected local NIC and remaps remote keys" {
    const Probe = struct {
        var calls: usize = 0;
        var destinations: [4]u32 = @splat(0);
        fn modify(qp: *vabi.Qp, attr: *vabi.QpAttr, _: c_int) callconv(.c) c_int {
            calls += 1;
            if (attr.qp_state == vabi.qps_rtr) destinations[(@intFromPtr(qp) - 0x1000) / 0x100] = attr.dest_qp_num;
            return 0;
        }
    };
    var verbs: Verbs = undefined;
    verbs.api.ibv_modify_qp = Probe.modify;
    var r: Ring = undefined;
    r.v = &verbs;
    r.rank = 0;
    r.world = 3;
    r.nd = 4;
    r.s = .{ .max_bytes = 64, .peer_masks = try parsePeerMasks("0,5,10", 0, 3, 4) };
    for (0..4) |d| {
        r.nics[d].qps = @splat(null);
        r.nics[d].psns = @splat(1);
        r.nics[d].active_mtu = 5;
        r.nics[d].qps[if (d % 2 == 0) 1 else 2] = @ptrFromInt(0x1000 + 0x100 * d);
    }
    var remote: [3]Info = @splat(std.mem.zeroes(Info));
    for (1..3) |p| {
        remote[p].devices = 2;
        for (0..2) |wire| {
            remote[p].active_mtu[wire] = 3;
            remote[p].qpn[wire] = @intCast(p * 100 + wire);
            remote[p].rkey[wire] = @intCast(p * 1000 + wire);
        }
    }
    Probe.calls = 0;
    try r.connect(&remote);
    try std.testing.expectEqual(@as(usize, 8), Probe.calls);
    try std.testing.expectEqualSlices(u32, &.{ 100, 200, 101, 201 }, &Probe.destinations);
    try std.testing.expectEqual(@as(u32, 1000), r.peers[1].rkey[0]);
    try std.testing.expectEqual(@as(u32, 1001), r.peers[1].rkey[2]);
    try std.testing.expectEqual(@as(u32, 2000), r.peers[2].rkey[1]);
    try std.testing.expectEqual(@as(u32, 2001), r.peers[2].rkey[3]);
    try std.testing.expectEqual(@as(u32, 0), r.peers[2].rkey[0]);
    Probe.calls = 0;
    remote[2].devices = 4;
    try std.testing.expectError(error.BadInfo, r.connect(&remote));
    try std.testing.expectEqual(@as(usize, 0), Probe.calls);
}

test "info packs only peer-selected NICs in ascending local device order" {
    var r: Ring = undefined;
    r.nd = 4;
    r.rank = 0;
    r.world = 3;
    r.s = .{ .max_bytes = 64, .peer_masks = try parsePeerMasks("0,5,10", 0, 3, 4) };
    var receive: [64]u8 align(65536) = undefined;
    r.recv.mem = &receive;
    var qps: [4]vabi.Qp = undefined;
    var mrs: [4]vabi.Mr = undefined;
    for (0..4) |d| {
        qps[d].qp_num = @intCast(100 + d);
        mrs[d].rkey = @intCast(200 + d);
        r.nics[d].qps = @splat(null);
        r.nics[d].qps[if (d % 2 == 0) 1 else 2] = &qps[d];
        r.nics[d].psns = @splat(@intCast(300 + d));
        r.nics[d].mr_recv = &mrs[d];
        r.nics[d].gid = @splat(@intCast(d + 1));
        r.nics[d].active_mtu = 3;
    }
    const first = r.info(1);
    const second = r.info(2);
    try std.testing.expectEqual(@as(u32, 2), first.devices);
    try std.testing.expectEqualSlices(u32, &.{ 100, 102, 0, 0 }, &first.qpn);
    try std.testing.expectEqualSlices(u32, &.{ 201, 203, 0, 0 }, &second.rkey);
    try std.testing.expectEqual(@as(u8, 3), first.gid[1][0]);
    try std.testing.expectEqual(@as(u32, 0), r.info(0).devices);
}

test "DMA mapping is absent after fork, ordinary memory stays mapped and parent keeps its contents" {
    if (@import("builtin").os.tag != .linux) return error.SkipZigTest;
    const dma = try dmaMapping(65536);
    defer std.posix.munmap(dma);
    const ordinary = try std.posix.mmap(null, 65536, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(ordinary);
    dma[0] = 17;
    ordinary[0] = 23;
    const pid = std.posix.system.fork();
    if (pid < 0) return error.SystemResources;
    if (pid == 0) {
        var vec: [16]u8 = undefined;
        const absent = std.os.linux.errno(std.os.linux.mincore(dma.ptr, dma.len, &vec)) == .NOMEM;
        const present = std.os.linux.errno(std.os.linux.mincore(ordinary.ptr, ordinary.len, &vec)) == .SUCCESS;
        std.os.linux.exit(if (absent and present and ordinary[0] == 23) 0 else 1);
    }
    var status: c_int = 0;
    try std.testing.expectEqual(pid, std.c.waitpid(pid, &status, 0));
    try std.testing.expectEqual(@as(c_int, 0), status);
    try std.testing.expectEqual(@as(u8, 17), dma[0]);
    dma[0] = 31;
    try std.testing.expectEqual(@as(u8, 31), dma[0]);
}
