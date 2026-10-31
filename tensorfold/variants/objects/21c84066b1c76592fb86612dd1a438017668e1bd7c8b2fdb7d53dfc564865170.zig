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
    ring_forward: bool = false,

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
        if(c.ring_forward) {
            if(c.world!=4 or c.rank>=4) return error.BadRowOwnership;
            const item:usize=switch(dtype) {
                .i8,.u8=>1, .f16,.bf16=>2, .i32,.u32,.f32=>4, .i64,.u64,.f64=>8,
            };
            const bytes=try std.math.mul(usize,count,item);
            const own=recv+c.rank*bytes;
            if(bytes>0 and send!=own) try stream.d.check(
                stream.d.api.cuMemcpyDtoDAsync_v2(own,send,bytes,stream.handle),"TP4 local gather part");
            for(0..3) |step| {
                const source=(c.rank+4-step)%4;
                const incoming=(source+3)%4;
                try c.hop(recv+source*bytes,bytes,(c.rank+1)%4,
                    recv+incoming*bytes,bytes,(c.rank+3)%4,stream);
            }
            return;
        }
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
    /// Exact bytes to/from the diagonal, forwarded on adjacent ring edges.
    /// relay and out must be disjoint and neither may alias send.
    pub fn diagonalBytes(c:Comm,send:u64,bytes:usize,relay:u64,relay_bytes:usize,
        out:u64,out_bytes:usize,stream:cuda.Stream) !void {
        if(c.world!=4 or c.rank>=4 or !c.ring_forward) return error.BadRowOwnership;
        const next=(c.rank+1)%4;const prev=(c.rank+3)%4;
        try c.hop(send,bytes,next,relay,relay_bytes,prev,stream);
        try c.hop(relay,relay_bytes,next,out,out_bytes,prev,stream);
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
    pub fn gatherOwnedRows(c: Comm, buffer: u64, firsts: []const usize, counts: []const usize, row_bytes: usize, stream: cuda.Stream) !void {
        if (firsts.len != c.world or counts.len != c.world or c.rank >= c.world) return error.BadRowOwnership;
        if(c.world==4) {
            // Physical pairs 0/1 and 2/3 first share their owned rows. Then
            // exchange the contiguous pair ranges across 0/3 and 1/2.
            // Same bytes as the three-hop ring, two NCCL launch barriers.
            var starts:[2]usize=undefined;var lengths:[2]usize=undefined;
            var contiguous=true;
            for(0..2) |pair| {
                const a=2*pair;const b=a+1;
                starts[pair]=if(counts[a]>0) firsts[a] else firsts[b];
                lengths[pair]=counts[a]+counts[b];
                if(counts[a]>0 and counts[b]>0 and firsts[a]+counts[a]!=firsts[b]) contiguous=false;
            }
            if(contiguous) {
                const g:usize=c.rank;const peer=g^1;const pair=g/2;
                try c.hop(buffer+firsts[g]*row_bytes,counts[g]*row_bytes,peer,
                    buffer+firsts[peer]*row_bytes,counts[peer]*row_bytes,peer,stream);
                try c.hop(buffer+starts[pair]*row_bytes,lengths[pair]*row_bytes,g^3,
                    buffer+starts[1-pair]*row_bytes,lengths[1-pair]*row_bytes,g^3,stream);
                return;
            }
            // Noncontiguous generic views retain the original exact ring.
            // All-gatherv around the physical ring. Forward only already
            // received byte slices; no floating-point operation or diagonal QP.
            for(0..3) |step| {
                const source=(c.rank+4-step)%4;
                const incoming=(source+3)%4;
                try c.hop(buffer+firsts[source]*row_bytes,counts[source]*row_bytes,(c.rank+1)%4,
                    buffer+firsts[incoming]*row_bytes,counts[incoming]*row_bytes,(c.rank+3)%4,stream);
            }
            return;
        }
        if(c.world!=3) return error.BadRowOwnership;
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

    /// Bytes to one adjacent rank and from the other. Zero counts are omitted
    /// symmetrically, including CED tails whose other owners have no rows.
    fn hop(c:Comm,send:u64,count:usize,to:usize,recv:u64,rcount:usize,from:usize,stream:cuda.Stream) !void {
        try c.lib.check(c.lib.api.ncclGroupStart(),"TP4 group start");
        var opened=true;
        errdefer if(opened) { _=c.lib.api.ncclGroupEnd(); };
        if(count>0) try c.lib.check(c.lib.api.ncclSend(send,count,.u8,@intCast(to),c.comm,stream.handle),"TP4 send");
        if(rcount>0) try c.lib.check(c.lib.api.ncclRecv(recv,rcount,.u8,@intCast(from),c.comm,stream.handle),"TP4 receive");
        const result=c.lib.api.ncclGroupEnd();
        opened=false;
        try c.lib.check(result,"TP4 group end");
    }

    /// [n,w[p]] FP32 quarter -> [4,local_rows,stride_bytes] raw parts.
    /// Adjacent destinations receive only their owned rows. The opposite
    /// destination is relayed in two physical hops with one preallocated buffer.
    /// The caller supplies its own part and assembles [2,local_rows,w0+w1].
    pub fn scatterQuarterRows(c:Comm,src:u64,dst:u64,relay:u64,capacity:usize,
        firsts:[4]usize,counts:[4]usize,width_bytes:[2]usize,stride_bytes:usize,stream:cuda.Stream) !void {
        if(c.world!=4 or c.rank>=4 or width_bytes[0]>stride_bytes or width_bytes[1]>stride_bytes)
            return error.BadRowOwnership;
        const g:usize=c.rank;
        const next=(g+1)%4; const prev=(g+3)%4; const diagonal=g^2;
        const ownw=width_bytes[g/2];
        const pitch=counts[g]*stride_bytes;
        const incoming=counts[next]*width_bytes[prev/2];
        if(incoming>capacity) return error.RowRelayTooSmall;
        // These three exchanges are independent. Preserve each peer's
        // FIFO (owned slice before diagonal relay), then group-end before
        // forwarding the newly received relay. Buffers are unchanged.
        try c.lib.check(c.lib.api.ncclGroupStart(),"TP4 partial group start");
        var opened=true;
        errdefer if(opened) { _=c.lib.api.ncclGroupEnd(); };
        if(counts[next]*ownw>0) try c.lib.check(c.lib.api.ncclSend(src+firsts[next]*ownw,
            counts[next]*ownw,.u8,@intCast(next),c.comm,stream.handle),"TP4 partial send next");
        if(counts[g]*width_bytes[prev/2]>0) try c.lib.check(c.lib.api.ncclRecv(dst+prev*pitch,
            counts[g]*width_bytes[prev/2],.u8,@intCast(prev),c.comm,stream.handle),"TP4 partial recv prev");
        if(counts[prev]*ownw>0) try c.lib.check(c.lib.api.ncclSend(src+firsts[prev]*ownw,
            counts[prev]*ownw,.u8,@intCast(prev),c.comm,stream.handle),"TP4 partial send prev");
        if(counts[g]*width_bytes[next/2]>0) try c.lib.check(c.lib.api.ncclRecv(dst+next*pitch,
            counts[g]*width_bytes[next/2],.u8,@intCast(next),c.comm,stream.handle),"TP4 partial recv next");
        if(counts[diagonal]*ownw>0) try c.lib.check(c.lib.api.ncclSend(src+firsts[diagonal]*ownw,
            counts[diagonal]*ownw,.u8,@intCast(next),c.comm,stream.handle),"TP4 diagonal send");
        if(incoming>0) try c.lib.check(c.lib.api.ncclRecv(relay,incoming,.u8,
            @intCast(prev),c.comm,stream.handle),"TP4 diagonal receive");
        const result=c.lib.api.ncclGroupEnd();opened=false;
        try c.lib.check(result,"TP4 partial group end");
        try c.hop(relay,incoming,next,dst+diagonal*pitch,counts[g]*width_bytes[diagonal/2],prev,stream);
    }

};

// Exercise the PRODUCTION calls with a staged, byte-exact network oracle.
// This verifies addresses, matching counts, empty owners and forwarding; it
// deliberately does not pretend to execute CUDA or verbs on the host.
const TP4Test = struct {
    const Call = struct { send:bool, ptr:u64, count:usize, peer:usize, group:usize };
    var calls:[4][16]Call=undefined;
    var used:[4]usize=@splat(0);
    var groups:[4]usize=@splat(0);
    var rank:usize=0;
    var depth:usize=0;
    fn start() callconv(.c) c_int {
        std.debug.assert(depth==0); depth=1; groups[rank]+=1; return 0;
    }
    fn end() callconv(.c) c_int {
        std.debug.assert(depth==1); depth=0; return 0;
    }
    fn send(p:u64,n:usize,ty:cuda.nccl.DataType,peer:c_int,_:cuda.nccl.Comm,_:cuda.abi.Stream) callconv(.c) c_int {
        std.debug.assert(depth==1 and ty==.u8 and n>0);
        calls[rank][used[rank]] = .{.send=true,.ptr=p,.count=n,.peer=@intCast(peer),.group=groups[rank]-1};
        used[rank]+=1; return 0;
    }
    fn recv(p:u64,n:usize,ty:cuda.nccl.DataType,peer:c_int,c:cuda.nccl.Comm,s:cuda.abi.Stream) callconv(.c) c_int {
        _=send(p,n,ty,peer,c,s); calls[rank][used[rank]-1].send=false; return 0;
    }
    fn reset() void { used=@splat(0); groups=@splat(0); depth=0; }
    fn communicator(g:usize) Comm {
        var c:Comm=undefined; c.world=4; c.rank=@intCast(g); c.comm=null; c.ring_forward=true;
        c.lib.api.ncclGroupStart=start; c.lib.api.ncclGroupEnd=end;
        c.lib.api.ncclSend=send; c.lib.api.ncclRecv=recv; rank=g; return c;
    }
    fn replay(rounds:usize) !void {
        for(0..rounds) |step| {
            var sends:usize=0;var receives:usize=0;
            var consumed:[4][16]bool=@splat(@splat(false));
            for(0..4) |g| for(calls[g][0..used[g]],0..) |call,i| {
                if(call.group!=step) continue;
                try std.testing.expect(call.peer==(g+1)%4 or call.peer==(g+3)%4);
                if(!call.send) {receives+=1;continue;}
                sends+=1;
                var ordinal:usize=0;
                for(calls[g][0..i]) |prior| {
                    if(prior.send and prior.group==step and prior.peer==call.peer) ordinal+=1;
                }
                var matches:usize=0;var receive_ordinal:usize=0;
                for(calls[call.peer][0..used[call.peer]],0..) |other,j| {
                    if(other.group!=step or other.send or other.peer!=g) continue;
                    defer receive_ordinal+=1;
                    if(receive_ordinal!=ordinal) continue;
                    matches+=1;try std.testing.expect(!consumed[call.peer][j]);
                    consumed[call.peer][j]=true;
                    try std.testing.expectEqual(call.count,other.count);
                    const src:[*]const u8=@ptrFromInt(call.ptr);
                    const dst:[*]u8=@ptrFromInt(other.ptr);
                    @memcpy(dst[0..call.count],src[0..call.count]);
                }
                try std.testing.expectEqual(@as(usize,1),matches);
            };
            try std.testing.expectEqual(sends,receives);
            for(0..4) |g| for(calls[g][0..used[g]],0..) |call,j| {
                if(call.group==step and !call.send) try std.testing.expect(consumed[g][j]);
            };
        }
        for(groups) |n| try std.testing.expectEqual(rounds,n);
    }

};

test "TP4 real partial calls route unequal quarters to all owners via adjacent edges" {
    const R=@import("row_state.zig");
    const a=std.testing.allocator;
    const stream:cuda.Stream=.{.d=undefined,.handle=null};
    for([_]usize{1,47,768,1999,4095,4096,8191,8192}) |n| {
        for([_][2]usize{.{12,20},.{20,12},.{12,12}}) |w| {
            var firsts:[4]usize=undefined; var counts:[4]usize=undefined;
            var src:[4][]u8=undefined; var dst:[4][]u8=undefined; var relay:[4][]u8=undefined;
            const stride=@max(w[0],w[1]);
            for(0..4) |g| {
                const r=try R.rows(n,g,4);firsts[g]=r.first;counts[g]=r.count;
                src[g]=try a.alloc(u8,n*w[g/2]);dst[g]=try a.alloc(u8,4*r.count*stride);
                relay[g]=try a.alloc(u8,n*stride);
                for(src[g],0..) |*v,j| v.*=@truncate(g*37+j*13);
                @memset(dst[g],0xcc);@memset(relay[g],0xdd);
            }
            defer for(0..4) |g| { a.free(src[g]);a.free(dst[g]);a.free(relay[g]); };
            TP4Test.reset();
            for(0..4) |g| {
                const ownw=w[g/2];const pitch=counts[g]*stride;
                @memcpy(dst[g][g*pitch..][0..counts[g]*ownw],src[g][firsts[g]*ownw..][0..counts[g]*ownw]);
                const c=TP4Test.communicator(g);
                try c.scatterQuarterRows(@intFromPtr(src[g].ptr),@intFromPtr(dst[g].ptr),
                    @intFromPtr(relay[g].ptr),relay[g].len,firsts,counts,w,stride,stream);
            }
            try TP4Test.replay(2);
            for(0..4) |owner| for(0..4) |source| {
                const actual=dst[owner][source*counts[owner]*stride..][0..counts[owner]*w[source/2]];
                const expected=src[source][firsts[owner]*w[source/2]..][0..actual.len];
                try std.testing.expectEqualSlices(u8,expected,actual);
            };
        }
    }
}

test "TP4 real all-gatherv preserves full and CED-only ranges including empty owners" {
    const R=@import("row_state.zig");const a=std.testing.allocator;
    const stream:cuda.Stream=.{.d=undefined,.handle=null};
    for([_]usize{1,47,768,1999,4096}) |n| for([_]usize{0,1,7,127,128}) |keep0| {
        const first=if(keep0==0) 0 else n-@min(n,keep0);
        const row_bytes=7;
        var buffers:[4][]u8=undefined;var firsts:[4]usize=undefined;var counts:[4]usize=undefined;
        for(0..4) |g| {
            buffers[g]=try a.alloc(u8,n*row_bytes);@memset(buffers[g],0xcc);
            const r=R.intersection(try R.rows(n,g,4),first,n);firsts[g]=r.first;counts[g]=r.count;
            for(r.first*row_bytes..r.end()*row_bytes) |j| buffers[g][j]=@truncate(j*17);
        }
        defer for(buffers) |b| a.free(b);
        TP4Test.reset();
        for(0..4) |g| {
            const c=TP4Test.communicator(g);
            try c.gatherOwnedRows(@intFromPtr(buffers[g].ptr),&firsts,&counts,row_bytes,stream);
        }
        try TP4Test.replay(2);
        for(buffers) |b| for(b,0..) |v,j| {
            const expected:u8=if(j/row_bytes>=first) @truncate(j*17) else 0xcc;
            try std.testing.expectEqual(expected,v);
        };
    };
}

test "TP4 full gather fallback preserves rank order over only neighbor edges" {
    const a=std.testing.allocator; const stream:cuda.Stream=.{.d=undefined,.handle=null};
    for([_]usize{1,16,513,1048577}) |bytes| {
        var buffers:[4][]u8=undefined;
        for(0..4) |g| {
            buffers[g]=try a.alloc(u8,4*bytes);@memset(buffers[g],0xcc);
            for(buffers[g][g*bytes..][0..bytes],0..) |*v,j| v.*=@truncate(g*71+j*13);
        }
        defer for(buffers) |b| a.free(b);
        TP4Test.reset();
        for(0..4) |g| {
            const c=TP4Test.communicator(g);
            // The in-place path avoids the real CUDA local-copy call in this
            // CPU gate; NCCL's entire production forwarding schedule is used.
            const p=@intFromPtr(buffers[g].ptr);
            try c.allGather(p+g*bytes,p,bytes,.u8,stream);
        }
        try TP4Test.replay(3);
        for(buffers) |b| for(0..4) |source| for(0..bytes) |j| {
            try std.testing.expectEqual(@as(u8,@truncate(source*71+j*13)),b[source*bytes+j]);
        };
    }
}

test "TP4 diagonal exchange preserves unequal bytes using exactly two adjacent hops" {
    const a=std.testing.allocator;
    const stream:cuda.Stream=.{.d=undefined,.handle=null};
    for([_]usize{1,47,768,4096,8191,8192}) |n| for([_][2]usize{.{1280,1024},.{1024,1280},.{24,24}}) |w| {
        var src:[4][]u8=undefined;var dst:[4][]u8=undefined;var relay:[4][]u8=undefined;
        for(0..4) |g| {
            src[g]=try a.alloc(u8,n*w[g/2]);dst[g]=try a.alloc(u8,n*w[1-g/2]);
            relay[g]=try a.alloc(u8,n*w[((g+3)%4)/2]);
            for(src[g],0..) |*v,j| v.*=@truncate(g*71+j*13);
            @memset(dst[g],0xcc);@memset(relay[g],0xdd);
        }
        defer for(0..4) |g| {a.free(src[g]);a.free(dst[g]);a.free(relay[g]);};
        TP4Test.reset();
        for(0..4) |g| {
            const c=TP4Test.communicator(g);
            try c.diagonalBytes(@intFromPtr(src[g].ptr),src[g].len,
                @intFromPtr(relay[g].ptr),relay[g].len,@intFromPtr(dst[g].ptr),dst[g].len,stream);
        }
        try TP4Test.replay(2);
        for(0..4) |g| try std.testing.expectEqualSlices(u8,src[g^2],dst[g]);
    };
}

test "TP4 noncontiguous owned slices retain the original three-hop fallback" {
    const a=std.testing.allocator;const stream:cuda.Stream=.{.d=undefined,.handle=null};
    const firsts=[_]usize{0,7,12,20};const counts=[_]usize{3,4,2,5};
    var buffers:[4][]u8=undefined;var want:[25]u8=@splat(0xcc);
    for(0..4) |g| {
        buffers[g]=try a.alloc(u8,25);@memset(buffers[g],0xcc);
        for(firsts[g]..firsts[g]+counts[g]) |i| {buffers[g][i]=@truncate(i*17);want[i]=@truncate(i*17);}
    }
    defer for(buffers) |b| a.free(b);
    TP4Test.reset();
    for(0..4) |g| {
        const c=TP4Test.communicator(g);
        try c.gatherOwnedRows(@intFromPtr(buffers[g].ptr),&firsts,&counts,1,stream);
    }
    try TP4Test.replay(3);
    for(buffers) |b| try std.testing.expectEqualSlices(u8,&want,b);
}

test "Engram row halves survive both real diagonal hops for all four owners" {
    const a=std.testing.allocator;
    const partition=@import("engram_partition.zig");
    const stream:cuda.Stream=.{.d=undefined,.handle=null};
    const width=48; // independent miniature BF16 row with both input halves
    for([_]usize{64,65,127,890,1019,4096,8191,8192}) |n| {
        var full:[4][]u8=undefined;var rel:[4][]u8=undefined;var recv:[4][]u8=undefined;
        for(0..4) |g| {
            const span=partition.rows(n,g/2);
            full[g]=try a.alloc(u8,n*width);
            rel[g]=try a.alloc(u8,partition.rows(n,((g+3)%4)/2).count*width);
            recv[g]=try a.alloc(u8,partition.rows(n,1-g/2).count*width);
            @memset(full[g],0xcc);@memset(rel[g],0xdd);@memset(recv[g],0xee);
            for(span.first..span.first+span.count) |r| for(0..width) |q| {
                full[g][r*width+q]=@truncate((g%2)*71+r*13+q*7);
            };
        }
        defer for(0..4) |g| {a.free(full[g]);a.free(rel[g]);a.free(recv[g]);};
        TP4Test.reset();
        for(0..4) |g| {
            const c=TP4Test.communicator(g);const span=partition.rows(n,g/2);
            try c.diagonalBytes(@intFromPtr(full[g].ptr)+span.first*width,span.count*width,
                @intFromPtr(rel[g].ptr),rel[g].len,@intFromPtr(recv[g].ptr),recv[g].len,stream);
        }
        try TP4Test.replay(2);
        for(0..4) |g| {
            const other=partition.rows(n,1-g/2);
            @memcpy(full[g][other.first*width..][0..recv[g].len],recv[g]);
            for(0..n) |r| for(0..width) |q| {
                const expected:u8=@truncate((g%2)*71+r*13+q*7);
                try std.testing.expectEqual(expected,full[g][r*width+q]);
            };
        }
    }
}
