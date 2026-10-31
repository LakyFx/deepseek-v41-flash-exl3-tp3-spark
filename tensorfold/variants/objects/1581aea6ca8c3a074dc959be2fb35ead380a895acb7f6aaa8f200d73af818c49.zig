//! CPU-only export of the exact native loader order. No CUDA dependencies.
//! Usage: tf-dsv41-plan MODEL_DIR RANK WORLD [--no-dspark]
const std = @import("std");
const config = @import("config.zig");
const plan = @import("plan.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) return error.Usage;
    const rank = try std.fmt.parseInt(u32, args[2], 10);
    const world = try std.fmt.parseInt(u32, args[3], 10);
    if ((world != 3 and world != 4) or rank >= world) return error.ExpectedTP3OrTP4;
    const c = try config.Config.read(a, init.io, args[1]);
    try c.checkWorld(world);
    const draft = args.len == 4;
    if (!draft and !std.mem.eql(u8, args[4], "--no-dspark")) return error.Usage;
    const split:plan.Split=if(world==4)
        .{.rank=rank%2,.world=2,.pair=rank/2} else .{.rank=rank,.world=world};
    const wants = try plan.wants(a, c, split, draft);
    var buf: [65536]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buf);
    for (wants) |w| try stdout.interface.print("{s}\n", .{w.key});
    try stdout.interface.flush();
    return 0;
}
