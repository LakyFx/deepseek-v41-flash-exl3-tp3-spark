//! Rank 0 to rank 1 over TCP (the fabric addresses): length-prefixed frames for each round's step and NCCL's id, as
//! multi.py's Link sends them. One connection, TCP_NODELAY, SO_BUSY_POLL, blocking reads (recvSpin polls first).
const std = @import("std");
const c = std.c;

pub const Link = struct {
    fd: c.fd_t,

    fn socket() !c.fd_t {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        tune(fd);
        return fd;
    }

    /// TCP_NODELAY, and SO_BUSY_POLL: a read polls the NIC's receive queue itself (up to 50 us) instead of waiting for
    /// its interrupt, whose core may be leaving a deep idle state (rank 0 to rank 1, 64-byte frames: round trip 1.0 ms
    /// blocking, 0.72 ms polling the socket, 11 us with SO_BUSY_POLL).
    fn tune(fd: c.fd_t) void {
        const one: c_int = 1;
        const busy_us: c_int = 50;
        _ = c.setsockopt(fd, c.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.BUSY_POLL, std.mem.asBytes(&busy_us), @sizeOf(c_int));
    }

    fn addr(ip: [4]u8, port: u16) c.sockaddr.in {
        return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
    }

    /// Rank 0: listen on `ip`:`port` and take the one peer's connection.
    pub fn listen(ip: [4]u8, port: u16) !Link {
        return listenTimeout(ip, port, null);
    }

    /// Bounded accept used by the experimental multi-peer startup. Null keeps
    /// the original standalone listen contract.
    pub fn listenTimeout(ip: [4]u8, port: u16, timeout_ms: ?u32) !Link {
        const fd = try socket();
        errdefer _ = c.close(fd);
        const one: c_int = 1;
        _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
        var a = addr(ip, port);
        if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.BindFailed;
        if (c.listen(fd, 1) != 0) return error.ListenFailed;
        if (timeout_ms) |ms| try waitReady(fd, c.POLL.IN, monotonicNs() + @as(u64, ms) * std.time.ns_per_ms);
        const peer = c.accept(fd, null, null);
        if (peer < 0) return error.AcceptFailed;
        _ = c.close(fd);
        tune(peer);
        return .{ .fd = peer };
    }

    /// Rank 1: connect to rank 0, retrying for up to `seconds` while it starts.
    pub fn connect(io: std.Io, ip: [4]u8, port: u16, seconds: u32) !Link {
        const deadline = monotonicNs() + @as(u64, seconds) * std.time.ns_per_s;
        while (monotonicNs() < deadline) {
            const fd = try socket();
            const flags = c.fcntl(fd, c.F.GETFL, @as(c_int, 0));
            const nonblock: c_int = @bitCast(std.posix.O{ .NONBLOCK = true });
            if (flags < 0 or c.fcntl(fd, c.F.SETFL, flags | nonblock) != 0) {
                _ = c.close(fd);
                return error.SocketFailed;
            }
            var a = addr(ip, port);
            const rc = c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in));
            var connected = rc == 0;
            if (!connected and c.errno(rc) == .INPROGRESS) {
                const ready = if (waitReady(fd, c.POLL.OUT, deadline)) |_| true else |_| false;
                if (ready) {
                    var socket_error: c_int = 0;
                    var size: c.socklen_t = @sizeOf(c_int);
                    connected = c.getsockopt(fd, c.SOL.SOCKET, c.SO.ERROR, &socket_error, &size) == 0 and socket_error == 0;
                }
            }
            if (connected) {
                if (c.fcntl(fd, c.F.SETFL, flags) != 0) {
                    _ = c.close(fd);
                    return error.SocketFailed;
                }
                return .{ .fd = fd };
            }
            _ = c.close(fd);
            const now = monotonicNs();
            if (now >= deadline) break;
            std.Io.sleep(io, .fromNanoseconds(@intCast(@min(deadline - now, 250 * std.time.ns_per_ms))), .awake) catch {};
        }
        return error.ConnectFailed;
    }

    fn waitReady(fd: c.fd_t, events: i16, deadline: u64) !void {
        var fds = [_]c.pollfd{.{ .fd = fd, .events = events, .revents = 0 }};
        while (true) {
            const now = monotonicNs();
            if (now >= deadline) return error.StartupTimeout;
            const ms = @min((deadline - now + std.time.ns_per_ms - 1) / std.time.ns_per_ms, std.math.maxInt(c_int));
            const rc = c.poll(&fds, 1, @intCast(ms));
            if (rc > 0) {
                if (fds[0].revents & (events | c.POLL.ERR | c.POLL.HUP) != 0) return;
                return error.PeerDown;
            }
            if (rc < 0 and c.errno(rc) != .INTR) return error.PeerDown;
        }
    }

    pub fn close(l: *Link) void {
        _ = c.close(l.fd);
        l.* = undefined;
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = c.send(l.fd, bytes.ptr + done, bytes.len - done, c.MSG.NOSIGNAL);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    fn readAll(l: Link, out: []u8) !void {
        var done: usize = 0;
        while (done < out.len) {
            const n = c.recv(l.fd, out.ptr + done, out.len - done, 0);
            if (n <= 0) {
                if (n < 0 and c.errno(n) == .INTR) continue;
                return error.PeerDown;
            }
            done += @intCast(n);
        }
    }

    /// One frame: its length (u32, little-endian), then its bytes.
    pub fn send(l: Link, bytes: []const u8) !void {
        var len: [4]u8 = undefined;
        std.mem.writeInt(u32, &len, @intCast(bytes.len), .little);
        try l.writeAll(&len);
        try l.writeAll(bytes);
    }

    /// recv, polling the socket without blocking for up to `spin_ns` first: a frame that comes within it is read
    /// without the receiving thread's sleep and wake-up (a core's deepest idle state takes 433 us to leave on GB10; the
    /// follower's frames come a fraction of a millisecond after its GPU goes idle), and after it the read blocks.
    pub fn recvSpin(l: Link, buf: []u8, spin_ns: u64) ![]u8 {
        var len: [4]u8 = undefined;
        var got: usize = 0;
        const t0 = monotonicNs();
        var polls: u32 = 0;
        while (got < len.len) {
            const n = c.recv(l.fd, @as([*]u8, &len) + got, len.len - got, c.MSG.DONTWAIT);
            if (n > 0) {
                got += @intCast(n);
                continue;
            }
            if (n == 0) return error.PeerDown;
            switch (c.errno(n)) {
                .AGAIN => {},
                .INTR => continue,
                else => return error.PeerDown,
            }
            polls +%= 1;
            if (polls % 64 == 0 and monotonicNs() -% t0 > spin_ns) break;
            std.atomic.spinLoopHint();
        }
        if (got < len.len) try l.readAll(len[got..]);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }

    pub fn monotonicNs() u64 {
        var ts: c.timespec = undefined;
        _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    fn timedRead(l: Link, out: []u8, deadline: u64) !void {
        var got: usize = 0;
        while (got < out.len) {
            try waitReady(l.fd, c.POLL.IN, deadline);
            const n = c.recv(l.fd, out.ptr + got, out.len - got, c.MSG.DONTWAIT);
            if (n == 0) return error.PeerDown;
            if (n < 0) {
                switch (c.errno(n)) { .AGAIN, .INTR => continue, else => return error.PeerDown }
            }
            got += @intCast(n);
        }
    }

    /// One absolute deadline covers the header and the complete payload, including partial reads.
    pub fn recvDeadline(l: Link, buf: []u8, deadline: u64) ![]u8 {
        var header: [4]u8 = undefined;
        try l.timedRead(&header, deadline);
        const n = std.mem.readInt(u32, &header, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.timedRead(buf[0..n], deadline);
        return buf[0..n];
    }

    /// Pressure control only: normal decode frame transport is unchanged.
    pub fn sendDeadline(l: Link, bytes: []const u8, deadline: u64) !void {
        var header: [4]u8 = undefined;
        std.mem.writeInt(u32, &header, @intCast(bytes.len), .little);
        for ([_][]const u8{ &header, bytes }) |part| {
            var done: usize = 0;
            while (done < part.len) {
                try waitReady(l.fd, c.POLL.OUT, deadline);
                const n = c.send(l.fd, part.ptr + done, part.len - done, c.MSG.DONTWAIT | c.MSG.NOSIGNAL);
                if (n == 0) return error.PeerDown;
                if (n < 0) {
                    switch (c.errno(n)) { .AGAIN, .INTR => continue, else => return error.PeerDown }
                }
                done += @intCast(n);
            }
        }
    }

    /// The next frame into `buf` (a frame longer than `buf` is refused); its bytes.
    pub fn recv(l: Link, buf: []u8) ![]u8 {
        var len: [4]u8 = undefined;
        try l.readAll(&len);
        const n = std.mem.readInt(u32, &len, .little);
        if (n > buf.len) return error.FrameTooLong;
        try l.readAll(buf[0..n]);
        return buf[0..n];
    }
};

/// "192.0.2.7" as four bytes.
pub fn parseIp(text: []const u8) ![4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    for (&out) |*b| b.* = try std.fmt.parseInt(u8, it.next() orelse return error.BadAddress, 10);
    if (it.next() != null) return error.BadAddress;
    return out;
}

test "addresses parse as four bytes" {
    try std.testing.expectEqual([4]u8{ 192, 0, 2, 7 }, try parseIp("192.0.2.7"));
    try std.testing.expectError(error.BadAddress, parseIp("192.0.2"));
}

test "missing listener peer times out without waiting for a kernel TCP timeout" {
    const began = Link.monotonicNs();
    try std.testing.expectError(error.StartupTimeout, Link.listenTimeout(.{ 127, 0, 0, 1 }, 0, 30));
    try std.testing.expect(Link.monotonicNs() - began < 5 * std.time.ns_per_s);
    try std.testing.expectError(error.ConnectFailed, Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, 0, 0));
}

test "frames cross a loopback link in order" {
    const io = std.testing.io;
    const port: u16 = 29000 + @as(u16, @intCast(@mod(std.Io.Timestamp.now(io, .awake).nanoseconds, 500)));
    const Peer = struct {
        fn run(p: u16) void {
            var l = Link.connect(std.testing.io, .{ 127, 0, 0, 1 }, p, 10) catch return;
            defer l.close();
            l.send("step 1") catch return;
            l.send(&.{}) catch return;
            l.send("polled") catch return;
            l.send("late") catch return;
        }
    };
    const t = try std.Thread.spawn(.{}, Peer.run, .{port});
    var l = try Link.listen(.{ 127, 0, 0, 1 }, port);
    defer l.close();
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("step 1", try l.recv(&buf));
    try std.testing.expectEqual(@as(usize, 0), (try l.recv(&buf)).len);
    try std.testing.expectEqualStrings("polled", try l.recvSpin(&buf, 1000 * std.time.ns_per_ms));
    try std.testing.expectEqualStrings("late", try l.recvSpin(&buf, 0));
    t.join();
}

test "125b ACK reads time out on a peer that sends only a partial header or payload" {
    for ([_][]const u8{ &.{ 17, 0 }, &.{ 17, 0, 0, 0, 10, 1 } }) |partial| {
        var fds: [2]c.fd_t = undefined;
        try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
        defer _ = c.close(fds[0]);
        defer _ = c.close(fds[1]);
        const peer: Link = .{ .fd = fds[0] };
        try std.testing.expectEqual(@as(isize, @intCast(partial.len)), c.send(fds[1], partial.ptr, partial.len, 0));
        var out: [32]u8 = undefined;
        const start = Link.monotonicNs();
        try std.testing.expectError(error.StartupTimeout, peer.recvDeadline(&out, start + 20 * std.time.ns_per_ms));
        try std.testing.expect(Link.monotonicNs() - start < std.time.ns_per_s);
    }
}

test "125b deadline frames preserve exact receipt and reject oversized payloads" {
    var fds: [2]c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &fds));
    defer _ = c.close(fds[0]);
    defer _ = c.close(fds[1]);
    const a: Link = .{ .fd = fds[0] };
    const b: Link = .{ .fd = fds[1] };
    var out: [32]u8 = undefined;
    const deadline = Link.monotonicNs() + std.time.ns_per_s;
    try a.sendDeadline("receipt", deadline);
    try std.testing.expectEqualStrings("receipt", try b.recvDeadline(&out, deadline));
    try a.sendDeadline("too large", deadline);
    try std.testing.expectError(error.FrameTooLong, b.recvDeadline(out[0..1], deadline));
}
