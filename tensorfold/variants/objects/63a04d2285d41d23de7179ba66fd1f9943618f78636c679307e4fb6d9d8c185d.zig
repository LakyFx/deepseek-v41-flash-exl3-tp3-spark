//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Status = api.Status;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    cache_reset: enum { idle, pending, done, unsupported } = .idle,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    prefill_telemetry: api.PrefillTelemetry = .{},
    adaptive: ?api.AdaptiveTelemetry = null,
    token_pool: ?lanes.backend.TokenPool = null,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    // the request at the queue's front waits for room in the backend's pool (its prompt pass refused ContextFull while
    // other streams ran): admitted again once fewer than this many streams are active
    full_wait: ?usize = null,

    const Mark = struct { at: i96, tokens: u64 };
    const window_ns: i96 = 2 * std.time.ns_per_s;

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up

        /// A completed CUDA event confirms a chunk, without making the engine wait for it.
        fn progress(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            const h = job.host;
            const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
            h.noteLive();
            h.lock();
            defer h.unlock();
            h.prefill_telemetry.progress = at;
            h.prefill_telemetry.pending = job.request.prompt.len -| at;
            if (now > job.began and at > s.cached) {
                h.prefill_rate = @as(f64, @floatFromInt(at - s.cached)) / (@as(f64, @floatFromInt(now - job.began)) / 1e9);
                h.prefill_at = now;
            }
        }

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.request.images.len > 0) return; // a kept state is keyed by tokens: an image's are its placeholder's
            if (job.host.cache) |store| _ = store.keepScoped(job.request.cache_namespace, s.context.items, at, s, job.request.chunks);
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            if (job.stream.reuse_failed) store.resumed(e, job.request.prompt, false) else if (job.stream.cached == e.at) store.resumed(e, job.request.prompt, true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        enforced.cache_salt = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    pub fn start(h: *LaneHost) !void {
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = statusFn, .memory = memoryFn, .keepalive = keepaliveFn, .reset_cache = resetCacheFn } };
    }

    fn resetCacheFn(ctx: *anyopaque) api.ResetCacheError!void {
        const h = self(ctx);
        h.lock();
        defer h.unlock();
        if (h.closing or h.thread == null) return error.Closed;
        if (h.cache_reset != .idle or h.queued.items.len != 0 or h.admitted.items.len != 0) return error.Busy;
        h.cache_reset = .pending;
        h.wake.broadcast(h.io);
        while (h.cache_reset == .pending and !h.closing) h.wake.waitUncancelable(h.io, &h.mutex);
        const result = h.cache_reset;
        h.cache_reset = .idle;
        if (h.closing) return error.Closed;
        if (result == .unsupported) return error.Unsupported;
        std.debug.assert(result == .done);
    }

    /// Only the engine thread may invoke snapshot drop callbacks (which can
    /// send distributed release messages). Submissions are refused until the
    /// waiting caller observes completion.
    fn takeCacheReset(h: *LaneHost) void {
        h.lock();
        defer h.unlock();
        if (h.cache_reset != .pending) return;
        std.debug.assert(h.queued.items.len == 0 and h.admitted.items.len == 0 and h.core.activeCount() == 0);
        if (h.cache) |store| store.clear() catch {
            h.cache_reset = .unsupported;
            h.wake.broadcast(h.io);
            return;
        };
        h.full_wait = null;
        h.cache_reset = .done;
        h.wake.broadcast(h.io);
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        if (h.cache_reset != .idle) {
            h.gpa.destroy(job);
            return error.Busy;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn statusFn(ctx: *anyopaque, out: *Status, stream_tokens: []u32) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        var tokens: u64 = 0;
        for (h.decoded.items) |m| {
            if (m.at >= now - window_ns) tokens += m.tokens;
        }
        const n = @min(stream_tokens.len, h.live_tokens.items.len);
        @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
        out.* = .{
            .running = @intCast(h.admitted.items.len),
            .waiting = @intCast(h.queued.items.len),
            .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
            .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
            .preemptions = 0,
            .streams = n,
            .generation_tokens = h.live_generated,
            .token_pool = h.token_pool,
            .prefill_telemetry = h.prefill_telemetry,
            .adaptive = h.adaptive,
        };
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery.
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            emit(job, .{ .tokens = emitted[job.delivered..] });
            h.noteDecoded(emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, "");
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        job.reported();
        h.gpa.free(job.marks);
        const s = &job.stream;
        var stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .loop_novelty = s.loop_novelty, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        if (job.started) for (&stats.depths, s.depths) |*o, d| {
            o.* = .{ .drafted = d.drafted, .reached = d.reached, .kept = d.kept };
        };
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    fn noteDecoded(h: *LaneHost, n: usize) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        var keep: usize = 0;
        for (h.decoded.items) |m| {
            if (m.at < now - window_ns) continue;
            h.decoded.items[keep] = m;
            keep += 1;
        }
        h.decoded.shrinkRetainingCapacity(keep);
        h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        h.lock();
        if (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return false;
        }
        if (h.full_wait) |n| {
            const active = h.core.activeCount();
            if (active > 0 and active >= n) {
                h.unlock();
                return false;
            }
            h.full_wait = null;
        }
        const job = h.queued.orderedRemove(0);
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.unlock();
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        if (r.images.len == 0) if (h.cache) |store| if (store.lookupScoped(r.cache_namespace, h.gpa, r.prompt, r.history_len, r.shared_prefixes, r.chunks)) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return h.drop(job, "the drafter could not start");
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .loop_guard = r.loop_guard,
            .loop_dry_windows = r.loop_dry_windows,
            .chunks = r.chunks,
            .reuse = reuse,
            .prefill_progress = .{ .ptr = job, .at = Job.progress },
            .images = r.images,
        }) catch {
            job.proposer.deinit();
            return h.drop(job, "out of memory");
        };
        job.started = true;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        job.began = began;
        h.lock();
        h.prefill_telemetry.active = 1;
        h.prefill_rate = 0;
        h.prefill_telemetry.progress = 0;
        h.prefill_telemetry.pending = r.prompt.len;
        h.unlock();
        defer {
            h.lock();
            h.prefill_telemetry.active = 0;
            h.prefill_telemetry.pending = 0;
            h.unlock();
            h.noteLive();
        }
        if (h.loneFits(job)) return h.runLone(job, began);
        h.core.addStream(&job.stream) catch |e| {
            if (e == error.Cancelled) return h.cancel(job);
            if (e == error.ContextFull and h.core.activeCount() > 0) return h.requeue(job);
            return h.drop(job, h.words(e));
        };
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        const cached = @min(job.request.prompt.len, job.stream.cached);
        h.prefill_telemetry.input += job.request.prompt.len;
        h.prefill_telemetry.cached += cached;
        h.prefill_telemetry.computed += job.request.prompt.len - cached;
        h.prefill_telemetry.seconds += @as(f64, @floatFromInt(@max(0, done - began))) / 1e9;
        h.prefill_telemetry.count += 1;
        h.prefill_telemetry.active = 0;
        h.prefill_telemetry.progress = job.request.prompt.len;
        h.prefill_telemetry.pending = 0;
        h.unlock();
        job.reported();
        if (h.cache) |store| store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// An idle backend driver takes a lone drafted request, including sampling when supported.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and h.queued.items.len == 0 and h.cancels.items.len == 0 and h.core.activeCount() == 0;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                host.noteLive();
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.queued.items.len > 0 or host.cancels.items.len > 0 or host.closing;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// A request whose prompt pass the backend's pool cannot hold while other streams run (error.ContextFull, before
    /// anything of it was placed) goes back to the queue's front and waits until one of them ends, as the Python
    /// lane's window waits, instead of failing. Returns false: nothing more is admitted now.
    fn requeue(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.core.discard(&job.stream);
        job.entry = null; // nothing was restored: the store keeps the entry for the next lookup
        h.gpa.free(job.marks);
        job.marks = &.{};
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.began = 0;
        job.prefilled = null;
        job.delivered = 0;
        job.prefill_sent = false;
        h.lock();
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return true;
        };
        h.full_wait = h.core.activeCount();
        h.unlock();
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn noteLive(h: *LaneHost) void {
        const pool = if (h.core.backend.vtable.token_pool) |read| read(h.core.backend.ptr) else null;
        h.lock();
        defer h.unlock();
        h.token_pool = pool;
        if (h.core.learner) |l| h.adaptive = .{
            .rounds = l.rounds, .cost_updates = l.cost_updates,
            .acceptance_updates = l.acceptance_updates, .probes = l.probes,
            .rejected_times = l.rejected_times, .prediction_samples = l.prediction_samples,
            .prediction_abs_error_microseconds = @intFromFloat(@min(1.0e18, l.prediction_error_ms * 1000)),
            .verify_microseconds = @intFromFloat(@min(1.0e18, l.observed_verify_ms * 1000)),
            .last_rows = l.frame.rows, .last_streams = l.frame.count,
            .last_context_tokens = l.frame.context,
            .last_predicted_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0,l.frame.predicted_ms * 1000))),
            .last_verify_microseconds = @intFromFloat(@min(1.0e18, @max(0,(l.frame.verify_ms orelse 0) * 1000))),
        };

        h.live_tokens.clearRetainingCapacity();
        h.live_generated = 0;
        for (h.admitted.items) |job| if (job.started) {
            h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
            h.live_generated += @intCast(job.stream.emitted().len);
        };
    }

    fn admitBatch(h: *LaneHost) void {
        // Failed/cancelled attempts may drain only while no decoder is live.
        while (h.admitOne()) {
            if (h.core.activeCount() > 0) break;
        }
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            h.takeCacheReset();
            h.admitBatch();
            h.noteLive();
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len;
                h.unlock();
                if (left == 0) return;
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0) {
                if (h.core.learner) |l| l.refresh.idle();
                if (h.cancels.items.len == 0 and (h.queued.items.len == 0 or h.admitted.items.len >= h.info_.lanes))
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(100), .clock = .awake } }) catch {};
                h.unlock();
                continue;
            }
            h.unlock();
            if (h.core.learner) |l| l.advanceTo(@intCast(std.Io.Clock.awake.now(h.io).toNanoseconds()));
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        h.unlock();
        h.takeCancels();
    }
};

test "cache reset drops snapshots on engine thread and refuses overlapping work" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snap = struct {
        creator: std.Thread.Id,
        dropped: usize = 0,
        wrong_thread: bool = false,
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 1;
        }
        fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved {
            return ctx;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {}
        fn drop(ctx: *anyopaque, _: pc.Saved) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            s.dropped += 1;
            s.wrong_thread = s.creator == std.Thread.getCurrentId();
        }
        fn event(_: *anyopaque, _: Id, _: *const Event) void {
            unreachable;
        }
    };
    var snap: Snap = .{ .creator = std.Thread.getCurrentId() };
    var store = pc.Store.init(gpa, .{ .ptr = &snap, .vtable = &.{ .bytes = Snap.bytes, .save = Snap.save, .restore = Snap.restore, .drop = Snap.drop } }, .{ .min_prompt = 0 }, 100);
    defer store.deinit();
    try std.testing.expect(store.keep(&.{ 1, 2, 3, 4 }, 3, null, &.{}));
    try std.testing.expect(store.find(&.{ 1, 2, 3, 4 }, &.{}) != null);
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    const e = host.engine();
    try std.testing.expectError(error.Closed, e.resetCache());
    try host.start();
    defer host.stop();
    try e.resetCache();
    try std.testing.expectEqual(@as(usize, 1), snap.dropped);
    try std.testing.expect(!snap.wrong_thread);
    try std.testing.expectEqual(@as(u64, 0), store.held);
    try std.testing.expect(store.find(&.{ 1, 2, 3, 4 }, &.{}) == null);
    try e.resetCache(); // idempotent, no duplicate drop
    try std.testing.expectEqual(@as(usize, 1), snap.dropped);
    host.lock();
    host.cache_reset = .done; // completed reset still owned by its waiter
    host.unlock();
    try std.testing.expectError(error.Busy, e.resetCache());
    const r: Request = .{ .prompt = &.{1}, .max_tokens = 1 };
    try std.testing.expectError(error.Busy, e.submit(1, &r, .{ .ctx = &snap, .event = Snap.event }));
    host.lock();
    host.cache_reset = .idle;
    host.unlock();
    var imprint: api.prompt_imprint.Imprint = undefined;
    store.imprint = &imprint; // clear checks presence before dereferencing
    try std.testing.expectError(error.Unsupported, e.resetCache());
    store.imprint = null;
    try e.resetCache();
}

test "a lane host serves the core's own tokens, in order, and cancels between rounds" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    var box: Box = .{};
    defer box.tokens.deinit(gpa);
    const request: Request = .{ .prompt = &prompt, .max_tokens = 24 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(Reason.length, box.wait());
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &prompt);
    for (box.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(usize, 24), box.tokens.items.len);
    var gone: Box = .{};
    defer gone.tokens.deinit(gpa);
    const long: Request = .{ .prompt = &prompt, .max_tokens = 100000 };
    try e.submit(2, &long, .{ .ctx = &gone, .event = Box.event });
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, gone.wait());

    const CancelPrefill = struct {
        engine: Engine,
        id: Id,
        at: usize,

        fn call(ctx: *anyopaque, _: *lanes.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            if (chunk == c.at) c.engine.cancel(c.id);
        }
    };
    const chunked_prompt = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4 };
    var chunked: Box = .{};
    defer chunked.tokens.deinit(gpa);
    var prefill_cancel = CancelPrefill{ .engine = e, .id = 3, .at = 2 };
    target.prefill_chunks = 10;
    target.prefill_count = 0;
    target.prefill_hook = CancelPrefill.call;
    target.prefill_hook_ctx = &prefill_cancel;
    const chunked_request: Request = .{ .prompt = &chunked_prompt, .max_tokens = 1 };
    try e.submit(3, &chunked_request, .{ .ctx = &chunked, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, chunked.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count()); // its lane released

    // the lone driver's prompt pass (gpu_round.run starts with Backend.opening), cancelled the same way
    const Lone = struct {
        be: lanes.backend.Backend,

        fn run(ctx: *anyopaque, s: *lanes.Stream, _: api.LoneHooks) anyerror!bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            _ = try l.be.opening(gpa, s);
            return error.NotCancelled;
        }
    };
    var lone: Box = .{};
    defer lone.tokens.deinit(gpa);
    var lone_driver = Lone{ .be = target.backend() };
    host.lone = .{ .ctx = &lone_driver, .run = Lone.run };
    prefill_cancel.id = 4;
    target.prefill_count = 0;
    try e.submit(4, &chunked_request, .{ .ctx = &lone, .event = Box.event });
    try std.testing.expectEqual(Reason.cancelled, lone.wait());
    try std.testing.expect(target.prefill_count <= 3);
    try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
}

test {
    _ = @import("lane_host_test.zig");
}

test "prefill counters are available before generated reply completion" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa = a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    const req: api.Request = .{ .prompt = &.{ 1, 2, 3, 4, 5, 6 }, .max_tokens = 100 };
    var stream = try lanes.Stream.init(a, .{ .id = "telemetry", .prompt = req.prompt, .max_new = req.max_tokens });
    defer stream.deinit(a);
    stream.cached = 4;
    const SinkStub = struct {
        fn event(_: *anyopaque, _: Id, ev: *const Event) void {
            std.debug.assert(ev.* == .prefilled);
        }
    };
    var job: LaneHost.Job = .{ .host = &host, .id = 1, .request = &req, .stream = stream, .sink = .{ .ctx = &host, .event = SinkStub.event } };
    const now = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    host.prefilled(&job, now - std.time.ns_per_s);
    var status: api.Status = .{};
    host.engine().status(&status, &.{});
    const p = status.prefill_telemetry.?;
    try std.testing.expectEqual(@as(u64, 6), p.input);
    try std.testing.expectEqual(@as(u64, 4), p.cached);
    try std.testing.expectEqual(@as(u64, 2), p.computed);
    try std.testing.expectEqual(@as(u64, 1), p.count);
    try std.testing.expect(p.seconds >= 1);
    try std.testing.expect(!job.stream.finished);
}

test "f native admission yields to decode and preserves foreground FIFO cancellation" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, .{ .exact_width=8, .gpu_tokens=true, .hidden_rows=true }, 8, 7);
    defer cfg.deinit(a);
    var fake: lanes.fake.Fake = .{ .gpa=a };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, std.testing.io, &core, .{ .lanes=4 });
    defer host.stop();
    defer host.closeAll();
    const Box = struct {
        prefilled: [4]u64 = @splat(0),
        n: usize = 0,
        fn event(ctx: *anyopaque, id: Id, ev: *const Event) void {
            const box: *@This() = @ptrCast(@alignCast(ctx));
            if (ev.* == .prefilled) {
                box.prefilled[box.n] = id;
                box.n += 1;
            }
        }
    };
    var box: Box = .{};
    const foreground: Request = .{ .prompt=&.{1,2,3}, .max_tokens=100 };
    var background = foreground;
    background.background = true;
    const engine = host.engine();
    const sink: Sink = .{ .ctx=&box, .event=Box.event };
    try engine.submit(1, &background, sink);
    try engine.submit(2, &foreground, sink);
    try engine.submit(3, &foreground, sink);
    host.admitBatch();
    try std.testing.expectEqual(@as(usize, 1), core.activeCount());
    try std.testing.expectEqual(@as(usize, 2), host.queued.items.len);
    try std.testing.expectEqual(@as(u64, 2), box.prefilled[0]);
    const steps = core.steps;
    try core.step();
    try std.testing.expect(core.steps > steps);
    host.admitBatch();
    try std.testing.expectEqual(@as(usize, 2), core.activeCount());
    try std.testing.expectEqual(@as(usize, 1), host.queued.items.len);
    try std.testing.expectEqual(@as(u64, 3), box.prefilled[1]);
    engine.cancel(1);
    host.takeCancels();
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len);
    try std.testing.expectEqual(@as(usize, 2), box.n);
}
