//! Local CPU timing only. Not model TPS, GPU latency or a performance claim.
const std=@import("std");
const adaptive=@import("adaptive.zig");
const old=@import("allocate.zig");
const Table=@import("table.zig").Table;

test "measure allocator and learning CPU overhead at C1 C6 C12" {
    const gpa=std.testing.allocator;
    var table: Table=.{};
    defer table.deinit(gpa);
    for(1..49) |r| try table.put(gpa,@intCast(r),30+@as(f64,@floatFromInt(r))*3.1);
    const c=try gpa.create(adaptive.Controller);
    defer gpa.destroy(c);
    var inputs: [12]adaptive.Input=undefined;
    const fixed: [12]u32=@splat(1);
    const probabilities=[_]f64{0.9,0.75,0.6,0.4,0.2};
    const probs: [12][]const f64=@splat(&probabilities);
    var arena=std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var checksum: u64=0;
    for([_]usize{1,6,12}) |n| {
        c.*=.{};
        for(inputs[0..n],0..) |*input,i| input.*=.{.key=i+1,.context=65536,.survivals=&probabilities};
        const repetitions=5000;
        const t0=std.Io.Timestamp.now(std.testing.io,.awake);
        for(0..repetitions) |_| {
            _=arena.reset(.retain_capacity);
            const result=try old.allocate(arena.allocator(),fixed[0..n],probs[0..n],&table,8,48);
            checksum+=result[0];
        }
        const t1=std.Io.Timestamp.now(std.testing.io,.awake);
        for(0..repetitions) |_| {
            _=arena.reset(.retain_capacity);
            const result=try c.allocate(arena.allocator(),inputs[0..n],&table,8,48);
            checksum+=result[0];
        }
        const t2=std.Io.Timestamp.now(std.testing.io,.awake);
        for(0..repetitions) |_| {
            _=arena.reset(.retain_capacity);
            const result=try c.allocate(arena.allocator(),inputs[0..n],&table,8,48);
            checksum+=result[0];
            c.verified(60);
            for(inputs[0..n],result) |input,count| c.accepted(input.key,count,@min(2,count));
            c.finished(80,n*3);
        }
        const t3=std.Io.Timestamp.now(std.testing.io,.awake);
        const scale: f64=1000.0*repetitions;
        std.debug.print("CPU_OVERHEAD C={d} original_allocate_us={d:.3} active_allocate_us={d:.3} active_with_feedback_us={d:.3}\n",
            .{n,@as(f64,@floatFromInt(t0.durationTo(t1).nanoseconds))/scale,
                @as(f64,@floatFromInt(t1.durationTo(t2).nanoseconds))/scale,
                @as(f64,@floatFromInt(t2.durationTo(t3).nanoseconds))/scale});
    }
    std.debug.print("CONTROLLER_BYTES={d} checksum={d}\n",.{@sizeOf(adaptive.Controller),checksum});
    try std.testing.expect(checksum>0);
}
