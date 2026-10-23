//! Internal cancellation control; uses the server's existing authentication.
const std = @import("std");
const json = @import("json.zig");
const body = @import("http_body.zig");
const routes = @import("routes.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const Allocator = std.mem.Allocator;
const schema = "tensorfold.vision-control.v1";

fn base(srv: *Server, a: Allocator) !*json.Object {
    const o = try json.newObject(a);
    try o.put(a, "schema", .{ .string = schema });
    try o.put(a, "instance", .{ .string = srv.vision_instance });
    return o;
}

pub fn get(srv: *Server, conn: *Conn, a: Allocator) void {
    const o = base(srv, a) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn post(srv: *Server, conn: *Conn, a: Allocator) void {
    var cx: Cx = .{ .a = a };
    const v = body.readJson(conn, &cx) catch return conn.sendJson(400, "{}");
    const id = text(v, "request_id") orelse return conn.sendJson(400, "{}");
    const instance = text(v, "instance") orelse return conn.sendJson(400, "{}");
    const action = text(v, "action") orelse return conn.sendJson(400, "{}");
    if (!std.mem.eql(u8, action, "cancel")) return conn.sendJson(400, "{}");
    const expires = expiry(v) orelse return conn.sendJson(400, "{}");
    const same = std.mem.eql(u8, instance, srv.vision_instance);
    const stopped = if (same) srv.vision_control.cancel(srv.io, id, expires, std.Io.Clock.real.now(srv.io).toSeconds()) catch
        return conn.sendJson(409, "{}") else true;
    const o = base(srv, a) catch return;
    o.put(a, "requested_instance", .{ .string = instance }) catch return;
    o.put(a, "request_id", .{ .string = id }) catch return;
    o.put(a, "termination_confirmed", .{ .bool = stopped }) catch return;
    o.put(a, "state", .{ .string = if (!same) "instance_replaced" else if (stopped) "terminated" else "cancelling" }) catch return;
    routes.sendValue(conn, a, 200, .{ .object = o });
}

pub fn text(v: json.Value, key: []const u8) ?[]const u8 {
    const field = v.get(key) orelse return null;
    return if (field == .string) field.string else null;
}

pub fn expiry(v: json.Value) ?i64 {
    const field = v.get("expires") orelse return null;
    return if (field == .int) field.int64() else null;
}
