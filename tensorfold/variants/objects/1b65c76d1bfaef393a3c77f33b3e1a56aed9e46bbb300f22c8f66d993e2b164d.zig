//! CPU-only diagnostic, exact pointer comparison. No hashing, GPU calls or sync.
//! Only capture-owned storage is included; per-round contents may change.
const std = @import("std");

pub const Signature = struct {
    values: [768]u64 = undefined,
    n: usize = 0,

    pub fn add(s: *Signature, value: u64) !void {
        if (s.n == s.values.len) return error.GraphStorageSignatureOverflow;
        s.values[s.n] = value;
        s.n += 1;
    }
    pub fn words(s: *Signature, values: []const u64) !void {
        try s.add(values.len);
        for (values) |v| try s.add(v);
    }
    /// Explicit structs of device addresses. Never inspect raw struct bytes or
    /// padding. usize counters, host floats and booleans are not device pointers.
    pub fn storage(s: *Signature, value: anytype, comptime excluded: []const []const u8) !void {
        const info = @typeInfo(@TypeOf(value)).@"struct";
        inline for (info.field_names, info.field_types) |name, FieldType| {
            const skip = comptime blk: {
                for (excluded) |excluded_name| if (std.mem.eql(u8, excluded_name, name)) break :blk true;
                break :blk false;
            };
            if (comptime !skip and FieldType == u64) try s.add(@field(value, name));
        }
    }
    pub fn slice(s: *const Signature) []const u64 { return s.values[0..s.n]; }
    pub fn check(s: *const Signature, captured: []const u64) !void {
        if (!std.mem.eql(u64, s.slice(), captured)) return error.GraphStorageChanged;
    }
};

pub fn enabled() bool {
    if (@import("builtin").is_test) return false;
    const v = std.c.getenv("TF_GRAPH_STORAGE_GUARD") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

test "address mutation refused, contents and usize shape counters excluded" {
    const Fake = struct { buffer: u64, rows: usize, counter: u64, ready: bool };
    var a: Signature = .{};
    try a.storage(Fake{.buffer=100,.rows=1,.counter=9,.ready=false}, &.{"counter"});
    try std.testing.expectEqual(@as(usize,1),a.n);
    var b: Signature = .{};
    try b.storage(Fake{.buffer=100,.rows=8,.counter=19,.ready=true}, &.{"counter"});
    try b.check(a.slice());
    b.values[0] += 1;
    try std.testing.expectError(error.GraphStorageChanged,b.check(a.slice()));
}
test "array length, ownership and overflow are checked" {
    var a: Signature = .{};try a.words(&.{1,2,3});
    var b: Signature = .{};try b.words(&.{1,2});
    try std.testing.expectError(error.GraphStorageChanged,b.check(a.slice()));
    a.n=a.values.len;
    try std.testing.expectError(error.GraphStorageSignatureOverflow,a.add(1));
}
