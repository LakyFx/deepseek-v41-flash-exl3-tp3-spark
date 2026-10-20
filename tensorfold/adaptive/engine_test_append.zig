
// 1.2.2xTF integration: actual Engine path, not a duplicate allocator model.
const be = @import("backend.zig");
const AdaptiveClock = struct {
    ms: f64=0,
    fn start(ptr: *anyopaque) void { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms=0; }
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 { const c: *@This()=@ptrCast(@alignCast(ptr)); c.ms+=20; return c.ms; }
    fn clock(c: *@This()) be.Clock { return .{.ptr=c,.vtable=&.{.start=start,.elapsed_ms=elapsed}}; }
};
fn runAdaptiveCases(cases: []const Case, learning: bool, staggered: bool) ![][]u32 {
    var costs: [48]@import("config.zig").Cost = undefined;
    for (&costs, 1..) |*cost, width| cost.* = .{ .width=@intCast(width), .ms=5+@as(f64,@floatFromInt(width))*0.8 };
    var cfg=try Config.init(gpa,.{.adaptive_learning=learning,.exact_width=16,.gpu_tokens=true,.mtp=true,.speculate=true,.speculate_early=false,.drafts=4,.window_costs=&costs,.shared_costs=&costs,.mtp_step_ms=0.5,.hidden_rows=true,.batch_rows=48,.max_streams=12,.draft_streams=true},16,15);
    defer cfg.deinit(gpa);
    var target: fake.Fake=.{.gpa=gpa};
    defer target.deinit();
    var clock: AdaptiveClock=.{};
    var engine=Engine.init(gpa,&cfg,target.backend(),clock.clock());
    defer engine.deinit();
    const streams=try gpa.alloc(sm.Stream,cases.len);
    defer gpa.free(streams);
    const proposers=try gpa.alloc(SuffixLookup,cases.len);
    defer gpa.free(proposers);
    for (cases,streams,proposers) |case,*stream,*proposer| {
        proposer.*=try SuffixLookup.init(gpa,.{.min_match=4});
        stream.*=try sm.Stream.init(gpa,.{.id="adaptive-integration",.prompt=case.prompt,.max_new=case.max_new,.eos=&.{96},.sampling=case.sampling,.drafts=case.drafts,.proposer=proposer.proposer(),.think_budget=case.think_budget,.think_close=&.{90,91,92},.think_end=91});
    }
    defer for (streams,proposers) |*stream,*proposer| { stream.deinit(gpa); proposer.deinit(); };
    const first=if(staggered) @max(1,cases.len/2) else cases.len;
    for (streams[0..first]) |*stream| try engine.addStream(stream);
    var added=first;
    var steps: usize=0;
    while (engine.activeCount()>0 or added<cases.len) : (steps+=1) {
        if (added<cases.len and (steps>=3 or engine.activeCount()==0)) {
            for (streams[added..]) |*stream| try engine.addStream(stream);
            added=cases.len;
        }
        if (steps>1000) return error.AdaptiveIntegrationDidNotFinish;
        try engine.step();
    }
    if (learning) {
        try std.testing.expect(engine.learner!=null);
        try std.testing.expect(engine.learner.?.rounds>0);
        try std.testing.expect(engine.learner.?.acceptance_updates>0);
        // The fake target can reach EOS in its first two cold rounds. Such a
        // completion must not manufacture warm timing samples.
        if(engine.learner.?.rounds>2) {
            try std.testing.expect(engine.learner.?.cost_updates>0);
        } else try std.testing.expectEqual(@as(u64,0),engine.learner.?.cost_updates);
    } else try std.testing.expect(engine.learner==null);
    const out=try gpa.alloc([]u32,cases.len);
    for(out,streams) |*value,*stream| value.*=try gpa.dupe(u32,stream.emitted());
    return out;
}

test "active adaptive engine handles forced thinking closure mixed plain streams and early finish" {
    const cases=[_]Case{
        .{.prompt=&p2,.max_new=48,.think_budget=9},
        .{.prompt=&p1,.max_new=8,.drafts=false},
        .{.prompt=&p2,.max_new=1},
        .{.prompt=&p1,.max_new=52},
    };
    const old=try runAdaptiveCases(&cases,false,false);
    defer free(old);
    const new=try runAdaptiveCases(&cases,true,true);
    defer free(new);
    for(old,new) |a,b| try std.testing.expectEqualSlices(u32,a,b);
}

test "active adaptive engine matches original and serial for C1 through C12 greedy sampled and join leave" {
    for ([_]?Sampling{null,.{.seed=42,.temperature=0.7,.top_k=0,.top_p=0.95}}) |sampling| {
        var cases: [12]Case=undefined;
        for(1..13) |n| {
            for(cases[0..n],0..) |*case,i| case.*=.{.prompt=if(i%2==0) &p1 else &p2,.max_new=@intCast(32+i),.sampling=sampling};
            const reference=try runAdaptiveCases(cases[0..n],false,false);
            defer free(reference);
            const learned=try runAdaptiveCases(cases[0..n],true,true);
            defer free(learned);
            for(reference,learned,0..) |old,new,i| {
                try std.testing.expectEqualSlices(u32,old,new);
                const serial=try run(&.{.{.prompt=cases[i].prompt,.max_new=cases[i].max_new,.sampling=sampling,.drafts=false}});
                defer free(serial);
                try std.testing.expectEqualSlices(u32,serial[0],new);
            }
        }
    }
}
