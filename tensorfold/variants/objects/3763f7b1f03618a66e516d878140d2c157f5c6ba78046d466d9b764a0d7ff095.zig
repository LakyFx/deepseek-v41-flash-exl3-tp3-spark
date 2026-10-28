//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const x1 = @import("checkpoint_policy.zig");
const x3 = @import("adaptive_prefix.zig");
pub const RequestState = x1.Request;

/// A family's copy of one state.
pub const Saved = *anyopaque;

/// Cache identity only, never part of the model's token stream. Null preserves
/// the original unsalted namespace, including learned on-disk prefixes.
pub const Namespace = ?[32]u8;

pub fn namespaceFor(salt: []const u8) Namespace {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(salt, &digest, .{});
    return digest;
}

/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Optional native checkpoint grid. Null keeps all other families unchanged.
        map_mark: ?*const fn (*anyopaque, ?Saved, usize, u32) u32 = null,
        automatic_marks: ?*const fn (*anyopaque, ?Saved, usize, *[2]u32) usize = null,
        tail_window: u32 = 0,
        /// Retention hint only: called AFTER a successful verified tail continuation.
        /// Never an authority for restore, and never increments actual hit counters.
        continuation_hint: ?*const fn (*anyopaque, Saved, u64) void = null,
        adaptive_hint: ?*const fn (*anyopaque, Saved, u32) void = null,
        ghost_range: ?*const fn (*anyopaque, Saved) [2]u32 = null,
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// Optional storage liveness and request compatibility, checked before longest-prefix selection.
        /// Diagnostic only; called once before removing an already dead entry.
        /// ABI 0 unknown, 1 physical overlap, 2 placement pressure. No token data.
        dead_reason: ?*const fn (ptr: *anyopaque, saved: Saved) u8 = null,
        live: ?*const fn (ptr: *anyopaque, saved: Saved) bool = null,
        compatible: ?*const fn (ptr: *anyopaque, saved: Saved, prompt_len: usize) bool = null,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
    };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Only families saving both complete target state and prompt logits may opt in.
    exact_end: bool = false,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
};

pub const Entry = struct {
    namespace: Namespace = null,
    tokens: []u32, // the state's tokens and `lookahead` more
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    x: x1.Entry = .{},
    reuse: Reuse = .{}, // K2a: CPU observation only; never changes real eviction scoring
};

/// Bounded CPU-only observations. A keep/dedup is NOT a successful restore.
/// Marginal tokens are a conservative lower bound, not measured time saved.
pub const Reuse = struct {
    restores: u64 = 0,
    last_restore: u64 = 0,
    restored_tokens: u64 = 0,
    marginal_lower_bound_tokens: u64 = 0,
    pending_marginal: u32 = 0,
    dedups: u32 = 0,
};

/// Observation units differ: skips count candidates; misses count lookups;
/// loss counters count removed entries. Never add these into one "miss rate".
pub const Reasons = struct {
    image_bypass: u64 = 0,
    image_bypass_prompt_tokens: u64 = 0,
    lookup_candidate: u64 = 0,
    no_eligible_prefix_match: u64 = 0,
    dead_candidate_skips: u64 = 0,
    incompatible_candidate_skips: u64 = 0,
    restore_ok: u64 = 0,
    restore_failed: u64 = 0,
    physical_overlap: u64 = 0,
    placement_pressure: u64 = 0,
    physical_unknown: u64 = 0,
    snapshot_budget: u64 = 0,
    reset_entries: u64 = 0,
};

pub const Audit = struct {
    dead_at_reserve: u64 = 0,
    dead_at_keep: u64 = 0,
    valid_budget_evictions: u64 = 0,
    upgrades: u64 = 0,
    dedups: u64 = 0,
    restored_tokens: u64 = 0,
    marginal_lower_bound_tokens: u64 = 0,
    shadow_decisions: u64 = 0,
    shadow_disagreements: u64 = 0,
    shadow_eligible: u64 = 0,
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0 };

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{}, request: RequestState = .{} };

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers

    audit: Audit = .{},
    reasons: Reasons = .{},
    diagnostics: bool = true,
    prune_dead: bool = true,
    victim_dead: bool = false,
    reuse_eviction: bool = false, // 126b: pressure-only; no timer, GPU work or extra state storage

    normalize_marks: bool = false,
    x: x1.State = .{},
    adaptive: bool = false,
    history: x3.History = .{},

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        var s: Store = .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
        if (!@import("builtin").is_test) {
            if (std.c.getenv("TF_K2A_DIAGNOSTICS")) |v| s.diagnostics = v[0] != '0';
            if (std.c.getenv("TF_K2A_PRUNE_DEAD")) |v| s.prune_dead = v[0] != '0';
            if (std.c.getenv("TF_CACHE_REUSE_EVICTION")) |v| s.reuse_eviction = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_NATIVE_PLAN")) |v| s.normalize_marks = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_BALANCED")) |v| s.x.enabled = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_CACHE_X1_TRACE")) |v| s.x.trace = std.mem.eql(u8, std.mem.span(v), "1");
            if (std.c.getenv("TF_DS_STREAMS")) |v| s.x.slots = std.math.clamp(std.fmt.parseInt(u32, std.mem.span(v), 10) catch 12, 1, 64);
            if (std.c.getenv("TF_CACHE_ADAPTIVE_PREFIX")) |v| s.adaptive = std.mem.eql(u8, std.mem.span(v), "1");
            if (s.adaptive) s.x.enabled = true;
            if (s.x.enabled) s.diagnostics = true;
            if (s.reuse_eviction) s.diagnostics = true; // real restore observations are required
        }
        return s;
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.held = 0;
    }

    /// Drop every reusable state on the owning engine thread, with no live
    /// borrowers. Preserve rules/budget and lifetime counters. Learned disk
    /// states can repopulate RAM, so refuse rather than claim a cold cache.
    pub fn clear(s: *Store) error{LearnedStatesEnabled}!void {
        if (s.imprint != null) return error.LearnedStatesEnabled;
        const count = s.entries.items.len;
        if (s.diagnostics) s.reasons.reset_entries +|= count;
        s.x.last_reason = .reset;
        while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
        s.x.ghosts = @splat(.{});
        s.history = .{};
        s.shared_keys.clearRetainingCapacity();
        s.counts.evicted += count;
        std.debug.assert(s.held == 0);
    }

    fn free(s: *Store, e: *Entry) void {
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.xRemoved(e);
        const t0 = if (s.x.trace) x1.now() else 0;
        s.free(e);
        if (s.x.trace) s.x.drop_ns +|= x1.now() -| t0;
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.spare();
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or std.mem.indexOfScalar(u32, starts, at) != null);
    }

    /// The longest kept state `prompt` resumes exactly: its tokens a prefix of the prompt, at least one row left.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32) ?*Entry {
        return s.findScoped(null, prompt, starts);
    }

    pub fn findScoped(s: *Store, namespace: Namespace, prompt: []const u32, starts: []const u32) ?*Entry {
        var best: ?*Entry = null;
        var alternate: u32 = 0;
        for (s.entries.items) |e| {
            if (!std.meta.eql(e.namespace, namespace)) continue;
            if (e.tokens.len > prompt.len or e.at > prompt.len or (e.at == prompt.len and !s.rules.exact_end) or !s.usable(e.at, starts)) continue;
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) { if (s.diagnostics) s.reasons.dead_candidate_skips +|= 1; if (s.adaptive) s.xLost(e, .dead); continue; };
            if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, e.saved, prompt.len)) { if (s.diagnostics) s.reasons.incompatible_candidate_skips +|= 1; continue; };
            if (best != null and e.at <= best.?.at) {
                // Original lookup skips comparing these tokens. They may match:
                // treating them as alternatives keeps the marginal bound conservative.
                if (s.diagnostics) alternate = @max(alternate, e.at);
                continue;
            }
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (s.diagnostics) if (best) |previous| { alternate = @max(alternate, previous.at); };
            best = e;
        }
        if (s.diagnostics) if (best) |e| { e.reuse.pending_marginal = e.at -| alternate; };
        if (s.diagnostics) {
            if (best != null) s.reasons.lookup_candidate +|= 1 else s.reasons.no_eligible_prefix_match +|= 1;
        }
        return best;
    }

    /// Before a prompt pass: restore the longest state `prompt` resumes (from 0: none or a failed copy) and plan its marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque) !Plan {
        const l = try s.lookup(a, prompt, history_len, shared, starts);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(null, prompt, null, l.marks); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// begin without the restore, for backends that restore inside their own prompt pass and then call `resumed`.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        return s.lookupScoped(null, a, prompt, history_len, shared, starts);
    }

    pub fn lookupScoped(s: *Store, namespace: Namespace, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32) !Lookup {
        const have = s.findScoped(namespace, prompt, starts);
        // Learned files use token-only keys. Salted requests neither read nor
        // write those files, preventing cross-namespace reuse after restart.
        const e = if (namespace == null) s.recall(prompt, starts, have) else have;
        if (e == null) s.counts.misses += 1;
        s.x.request_serial +|= 1;
        const raw = try s.marks(a, prompt, if (e) |x| x.at else 0, history_len, shared, starts, if (e) |x| x.last else &.{});
        const marks_ = try s.fitting(a, try s.actualMarks(a, prompt, e, raw));
        var automatic: [2]u32 = undefined;
        const auto_n = if (s.family.vtable.automatic_marks) |f| f(s.family.ptr, if (e) |x| x.saved else null, prompt.len, &automatic) else 0;
        var request: RequestState = .{ .id = s.x.request_serial, .cohort = s.x.request_serial,
            .namespace_id = sharedKey(namespace, &.{}), .input_len = @intCast(prompt.len),
            .encoder_at = if (auto_n == 2) automatic[0] else 0 };
        if (s.adaptive) request.history = s.history.begin(request.namespace_id, request.id);
        s.x.push(.{ .kind = .request, .request = request.id, .cohort = request.cohort,
            .namespace_id = request.namespace_id, .prompt = request.input_len,
            .at = if (e) |x| x.at else 0, .state = if (e) |x| x.x.id else 0 });
        s.xGhostReturns(namespace, prompt, if (e) |x| x.at else 0, &request);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const raw_at = if (s.rules.planned) floorStart(starts, w) else w;
            const at = if (s.normalize_marks and s.family.vtable.map_mark != null) s.family.vtable.map_mark.?(s.family.ptr, if (e) |x| x.saved else null, prompt.len, raw_at) else raw_at;
            if (at != 0 and std.mem.indexOfScalar(u32, marks_, at) != null) s.noteShared(namespace, prompt[0 .. at + s.rules.lookahead]);
        }
        const reservation = try s.reservationMarks(a, prompt, e, marks_);
        defer a.free(reservation);
        s.reserve(namespace, prompt, e, reservation);
        return .{ .entry = e, .marks = marks_, .request = request };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, namespace: Namespace, prompt: []const u32, from: ?*Entry, marks_: []const u32) void {
        var need: u64 = 0;
        for (marks_, 0..) |at, k| {
            if (s.normalize_marks and std.mem.indexOfScalar(u32, marks_[0..k], at) != null) continue;
            var bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (s.normalize_marks) for (s.entries.items) |e| {
                if (!std.meta.eql(e.namespace, namespace) or e.at != at or e.tokens.len > prompt.len or !std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) continue;
                if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) { continue; };
                const compatible = if (s.family.vtable.compatible) |f| f(s.family.ptr, e.saved, prompt.len) else true;
                // Only a terminal request upgrades an incompatible encoder snapshot.
                // A kept key otherwise deduplicates exactly as keepScoped does.
                if (compatible or !(s.rules.exact_end and at == prompt.len)) {
                    bytes = 0;
                    s.x.compatible_reservations +|= 1;
                } else bytes -|= e.bytes;
                break;
            };
            need +|= bytes;
        }
        while (s.held + need > s.budget) {
            if (!s.evictBudget(namespace, from, prompt)) return;
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        for (s.entries.items, 0..) |e, i| if (e.namespace == null and e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) {
            s.counts.failed += 1;
            if (s.diagnostics) s.reasons.restore_failed +|= 1;
            return s.remove(i);
        };
    }

    fn sharedKey(namespace: Namespace, tokens: []const u32) u64 {
        var hash = std.hash.Wyhash.init(0);
        if (namespace) |digest| hash.update(&digest);
        hash.update(std.mem.sliceAsBytes(tokens));
        return hash.final();
    }

    fn noteShared(s: *Store, namespace: Namespace, tokens: []const u32) void {
        const k = sharedKey(namespace, tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            if (s.diagnostics) s.reasons.restore_failed +|= 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        if (s.diagnostics) s.reasons.restore_ok +|= 1;
        if (s.diagnostics) {
            e.reuse.restores +|= 1;
            e.reuse.last_restore = s.clock;
            e.reuse.restored_tokens +|= e.at;
            e.reuse.marginal_lower_bound_tokens +|= e.reuse.pending_marginal;
            s.audit.restored_tokens +|= e.at;
            s.audit.marginal_lower_bound_tokens +|= e.reuse.pending_marginal;
            e.reuse.pending_marginal = 0;
        }
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            if (k > 0) { // the history's mark always; the others only away from it, each other and the resume point
                if (at - from < s.rules.min_gap) continue;
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// The prompt pass stands at `at`: keep its state for `prompt`, evicting to fit; refused (counted) past the budget.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        return s.keepScoped(null, prompt, at, owner, starts);
    }

    pub fn keepScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32) bool {
        return s.keepRequestScoped(namespace, prompt, at, owner, starts, null);
    }

    pub fn keepRequestScoped(s: *Store, namespace: Namespace, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, request: ?*const RequestState) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const key = sharedKey(namespace, prompt[0..n]);
        var shared = std.mem.indexOfScalar(u64, s.shared_keys.items, key) != null;
        // A stale physical reservation must be replaced, never refreshed by token-key deduplication.
        var old_i: usize = 0;
        while (old_i < s.entries.items.len) {
            const old = s.entries.items[old_i];
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, old.saved)) {
                s.x.last_reason = .dead;
                s.audit.dead_at_keep +|= 1;
                s.noteDead(old);
                s.remove(old_i);
                continue;
            };
            // Encoder-only and complete terminal snapshots can share a token key.
            // Upgrade only at the exact request end; never downgrade a complete state
            // when another longer request keeps an encoder checkpoint at that key.
            if (s.rules.exact_end and at == prompt.len and std.meta.eql(old.namespace, namespace) and old.at == at and std.mem.eql(u32, old.tokens, prompt[0..n])) {
                if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, old.saved, prompt.len)) {
                    s.x.last_reason = .upgrade;
                    s.audit.upgrades +|= 1;
                    shared = shared or old.shared;
                    s.remove(old_i);
                    continue;
                };
            }
            old_i += 1;
        }
        for (s.entries.items) |e| if (std.meta.eql(e.namespace, namespace) and e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n])) {
            if (s.diagnostics) { e.reuse.dedups +|= 1; s.audit.dedups +|= 1; }
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            s.xDedup(e, request);
            return true;
        };
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            if (!s.evictBudget(namespace, null, prompt)) break; // storage may return spare

        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const save_t0 = if (s.x.trace) x1.now() else 0;
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        if (s.x.trace) s.x.save_ns +|= x1.now() -| save_t0;
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .namespace = namespace, .tokens = tokens, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.xSaved(e, key, request);
        s.held += charged;
        s.counts.kept += 1;
        if (shared) s.learn(e, starts);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A shared cut's state written to disk once (--learn): later sessions read it back, after a restart too.
    fn learn(s: *Store, e: *const Entry, starts: []const u32) void {
        if (e.namespace != null) return;
        const im = s.imprint orelse return;
        const write = s.family.vtable.write orelse return;
        const key = imprint.Imprint.keyOf(e.tokens);
        if (im.has(key) or e.bytes > im.cap) return;
        while (!im.fits(e.bytes)) s.unlearn(im, im.victim() orelse return); // the least recently used go first
        write(s.family.ptr, e.saved, im.dir, key) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        im.add(key, e.at, e.tokens, starts, e.bytes) catch |err| return note("learning {d} tokens failed ({s})", .{ e.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{e.at});
    }

    /// Learned state `key` forgotten: its files (the family's), then its index record.
    fn unlearn(s: *Store, im: *imprint.Imprint, key: u64) void {
        if (s.family.vtable.forget) |f| f(s.family.ptr, im.dir, key);
        im.remove(key) catch |err| note("forgetting a learned state failed ({s})", .{@errorName(err)});
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, have: ?*Entry) ?*Entry {
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and s.held + s.spare() + bytes > s.budget) {
            if (!s.evictBudget(null, have, prompt)) return have;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            s.unlearn(im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *Store, prompt: usize, from: u32, kept: u64) void {
        s.flushX1();
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
        if (s.diagnostics) std.log.info("prompt cache k2a: dead_pressure {d}, dead_keep {d}, valid_budget {d}, upgrades {d}, dedups {d}, restored_tokens {d}, marginal_lower_bound_tokens {d}, shadow_decisions {d}, shadow_disagreements {d}, shadow_eligible {d}; image_bypass {d}, image_tokens {d}, lookup_candidate {d}, no_eligible_match {d}, dead_skips {d}, incompatible_skips {d}, restore_ok {d}, restore_failed {d}, physical_overlap {d}, placement_pressure {d}, physical_unknown {d}, snapshot_budget {d}, reset_entries {d}", .{ s.audit.dead_at_reserve, s.audit.dead_at_keep, s.audit.valid_budget_evictions, s.audit.upgrades, s.audit.dedups, s.audit.restored_tokens, s.audit.marginal_lower_bound_tokens, s.audit.shadow_decisions, s.audit.shadow_disagreements, s.audit.shadow_eligible, s.reasons.image_bypass, s.reasons.image_bypass_prompt_tokens, s.reasons.lookup_candidate, s.reasons.no_eligible_prefix_match, s.reasons.dead_candidate_skips, s.reasons.incompatible_candidate_skips, s.reasons.restore_ok, s.reasons.restore_failed, s.reasons.physical_overlap, s.reasons.placement_pressure, s.reasons.physical_unknown, s.reasons.snapshot_budget, s.reasons.reset_entries });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip` (a pass's resume): one a later prompt extends (never a shared cut), oldest first; else the oldest.
    fn victimBut(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.x.last_reason = .none;
        s.x.last_baseline = 0;
        if (s.x.enabled) return s.xVictim(namespace, skip, prompt);
        return s.legacyVictimBut(namespace, skip, prompt);
    }

    fn legacyVictimBut(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.victim_dead = false;
        var best: ?usize = null;
        var shadow: ?usize = null;
        var eligible: u64 = 0;
        // Integrate physical liveness into the existing pressure-only pass.
        // Never drop the borrowed resume entry; no new scan on the decode path.
        for (s.entries.items, 0..) |e, i| {
            if (e == skip) continue;
            if (s.prune_dead) if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) {
                s.victim_dead = true;
                return i;
            };
            if (e.shared) continue;
            const now = std.meta.eql(e.namespace, namespace) and prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens);
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and std.meta.eql(o.namespace, e.namespace) and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on) {
                if (best == null or e.used < s.entries.items[best.?].used) best = i;
                if (s.diagnostics and s.prune_dead) {
                    eligible += 1;
                    if (shadow == null or s.shadowLess(e, s.entries.items[shadow.?])) shadow = i;
                }
            }
        }
        if (best) |i| { s.observeShadow(i, shadow, eligible); return if (s.reuse_eviction and shadow != null) shadow.? else i; }
        var oldest: ?usize = null;
        shadow = null;
        eligible = 0;
        for (s.entries.items, 0..) |e, i| if (e != skip) {
            if (oldest == null or e.used < s.entries.items[oldest.?].used) oldest = i;
            if (s.diagnostics and s.prune_dead) {
                eligible += 1;
                if (shadow == null or s.shadowLess(e, s.entries.items[shadow.?])) shadow = i;
            }
        };
        if (oldest) |i| s.observeShadow(i, shadow, eligible);
        return if (s.reuse_eviction and shadow != null) shadow else oldest;
    }

    fn evictBudget(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) bool {
        const i = s.victimBut(namespace, skip, prompt) orelse return false;
        if (s.victim_dead) s.audit.dead_at_reserve +|= 1 else s.audit.valid_budget_evictions +|= 1;
        if (s.victim_dead) s.noteDead(s.entries.items[i]) else if (s.diagnostics) { s.reasons.snapshot_budget +|= 1; }
        if (s.x.last_reason == .none) s.x.last_reason = if (s.victim_dead) .dead else .legacy_fallback;
        s.remove(i); // same owner thread, family drop and TP3 ordering as before
        s.counts.evicted += 1; // preserve legacy eviction semantics
        return true;
    }

    pub fn noteImageBypass(s: *Store, tokens: usize) void {
        if (!s.diagnostics) return;
        s.reasons.image_bypass +|= 1;
        s.reasons.image_bypass_prompt_tokens +|= tokens;
    }

    fn noteDead(s: *Store, e: *const Entry) void {
        if (!s.diagnostics) return;
        const reason = if (s.family.vtable.dead_reason) |f| f(s.family.ptr, e.saved) else 0;
        switch (reason) {
            1 => s.reasons.physical_overlap +|= 1,
            2 => s.reasons.placement_pressure +|= 1,
            else => s.reasons.physical_unknown +|= 1,
        }
    }

    fn observeShadow(s: *Store, actual: usize, shadow: ?usize, eligible: u64) void {
        if (!s.diagnostics or !s.prune_dead or shadow == null) return;
        s.audit.shadow_decisions +|= 1;
        s.audit.shadow_eligible +|= eligible;
        if (actual != shadow.?) s.audit.shadow_disagreements +|= 1;
    }

    /// Diagnostic hypothesis ONLY, within the same superseded/fallback tier.
    /// Shared anchors last; fresh entries get 64 store-event grace; real reuse
    /// is capped at 8 observations and marginal credit at 128 chunks. Ties LRU.
    fn shadowLess(s: *const Store, a: *const Entry, b: *const Entry) bool {
        if (a.shared != b.shared) return !a.shared;
        const av = s.shadowValue(a);
        const bv = s.shadowValue(b);
        return av < bv or (av == bv and a.used < b.used);
    }

    fn shadowValue(s: *const Store, e: *const Entry) u64 {
        if (e.reuse.restores == 0) return if (s.clock -| e.used < 64) 1 else 0;
        const marginal = e.reuse.marginal_lower_bound_tokens / e.reuse.restores;
        const hits: u64 = @min(e.reuse.restores, 8);
        const chunks: u64 = @min(marginal / 2048, 128);
        const value = hits * (1 + chunks);
        if (!s.reuse_eviction) return value; // baseline diagnostic unchanged
        const age = s.clock -| e.reuse.last_restore;
        // Event-age decay, not a wall-clock TTL. Never permanently pin a
        // historically popular state; only compare within the original tier.
        return if (age >= 512) 0 else value / (1 + age / 64);
    }

    /// Normalize BEFORE fitting, shared registration and reserve. Automatic
    /// callbacks are included once; the native GPU callback path is unchanged.
    fn actualMarks(s: *Store, a: Allocator, prompt: []const u32, e: ?*Entry, raw: []const u32) ![]const u32 {
        const map = s.family.vtable.map_mark orelse return raw;
        if (!s.normalize_marks) return raw;
        const t0 = if (s.x.trace) x1.now() else 0;
        defer if (s.x.trace) { s.x.plan_ns +|= x1.now() -| t0; };
        defer a.free(raw);
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        for (raw) |wanted| {
            const at = map(s.family.ptr, if (e) |v| v.saved else null, prompt.len, wanted);
            s.x.push(.{ .kind = .plan, .request = s.x.request_serial, .desired = wanted, .effective = at });
            if (at == 0) { s.x.omitted +|= 1; continue; }
            if (at != wanted) s.x.mapped +|= 1;
            if (std.mem.indexOfScalar(u32, out.items, at) != null) { s.x.duplicate_marks +|= 1; continue; }
            try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Automatic checkpoints belong in accounting, not in the optional marks
    /// sent to the GPU. This also preserves callbacks if a restore later fails.
    fn reservationMarks(s: *Store, a: Allocator, prompt: []const u32, e: ?*Entry, marks_: []const u32) ![]const u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        try out.appendSlice(a, marks_);
        if (s.normalize_marks) if (s.family.vtable.automatic_marks) |f| {
            var automatic: [2]u32 = undefined;
            const n = f(s.family.ptr, if (e) |v| v.saved else null, prompt.len, &automatic);
            for (automatic[0..n]) |at| {
                if (std.mem.indexOfScalar(u32, out.items, at) != null) { s.x.duplicate_marks +|= 1; continue; }
                try out.append(a, at);
            }
        };
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Capture old tail evidence BEFORE resumed updates e.last. Token comparison
    /// is bounded by the family's window; the earlier prefix was checked by find.
    pub fn resumedRequest(s: *Store, e: *Entry, prompt: []const u32, ok: bool, r: *RequestState) void {
        const present = std.mem.indexOfScalar(*Entry, s.entries.items, e) != null;
        if (!present) return;
        const continuation = ok and x1.tailEligible(e.x.role, e.shared, e.at, e.last.len, prompt.len, s.family.vtable.tail_window) and
            std.mem.eql(u32, e.last[e.at..], prompt[e.at..e.last.len]);
        const id = e.x.id;
        const cohort = e.x.cohort;
        const parent_request = e.x.created_request;
        const at = e.at;
        s.resumed(e, prompt, ok);
        if (!ok) return; // e may have been freed
        e.x.last_return = r.id;
        r.restored_at = at;
        r.parent = id;
        if (s.adaptive and !e.shared and e.x.history != 0 and (continuation or at == r.input_len)) {
            if (!r.history_observed) r.history = s.history.returned(r.namespace_id, r.id, e.x.history, e.x.history_request, false, at == r.input_len);
            r.history_observed = true;
            e.x.history = r.history;
            e.x.history_request = r.id;
            s.adaptiveHint(e);
        }
        if (continuation) {
            r.continuation = true;
            r.parent_request = parent_request;
            r.cohort = if (cohort != 0) cohort else r.id;
            s.x.continuations +|= 1;
        }
        s.x.push(.{ .kind = .restore, .request = r.id, .namespace_id = r.namespace_id,
            .state = id, .cohort = r.cohort, .parent = r.parent, .role = e.x.role,
            .at = at, .prompt = @intCast(prompt.len), .effective = @intFromBool(continuation) });
    }

    fn xSaved(s: *Store, e: *Entry, key: u64, request: ?*const RequestState) void {
        if (!s.x.enabled and !s.x.trace) return;
        s.x.serial +|= 1;
        const r: RequestState = if (request) |v| v.* else .{};
        const role = r.role(e.at);
        const tail = role == .encoder_tail or role == .prompt_end or role == .decode_end;
        e.x = .{ .id = s.x.serial, .key = key, .cohort = r.cohort, .parent = r.parent,
            .created_request = r.id, .role = role, .continuation = tail and r.continuation,
            .last_return = if (tail and r.continuation) s.x.request_serial else 0 };
        if (s.adaptive and tail and !e.shared) { e.x.history = r.history; e.x.history_request = r.id; }
        s.adaptiveHint(e);
        s.xHint(e, r);
        // Only the same request or a verified old request tail can establish
        // succession. Cohort equality by itself is not proof (forks may share it).
        for (s.entries.items) |old| {
            if (old == e or old.at >= e.at or !std.meta.eql(old.namespace, e.namespace)) continue;
            if (old.x.created_request == 0 or !(old.x.created_request == r.id or
                (r.continuation and old.x.created_request == r.parent_request))) continue;
            if (old.x.role == .auxiliary or (old.x.role == .encoder_tail and role == .encoder_tail)) old.x.successor = e.x.id;
            // Retire old historical ends only after an actually saved, verified
            // next-turn prompt end exists. A decode tail alone cannot replace
            // its own exact input prompt. Encoder fallback remains separate.
            if (r.continuation and old.x.created_request == r.parent_request and role == .prompt_end and
                (old.x.role == .prompt_end or old.x.role == .decode_end)) old.x.successor = e.x.id;
        }
        s.x.push(.{ .kind = .save, .request = r.id, .namespace_id = sharedKey(e.namespace, &.{}),
            .state = e.x.id, .cohort = e.x.cohort, .parent = e.x.parent, .role = role,
            .at = e.at, .prompt = r.input_len, .bytes = e.bytes });
    }

    fn xDedup(s: *Store, e: *Entry, request: ?*const RequestState) void {
        if (s.adaptive and e.shared) { e.x.history = 0; e.x.history_request = 0; }
        const r = request orelse {
            if (e.shared) s.adaptiveHint(e);
            return;
        };
        const role = r.role(e.at);
        if (@intFromEnum(role) > @intFromEnum(e.x.role)) e.x.role = role;
        if (r.continuation and role != .auxiliary and !e.shared) {
            e.x.continuation = true;
            e.x.last_return = r.id;
        }
        if (s.adaptive and !e.shared and role != .auxiliary and role != .unknown) { e.x.history = r.history; e.x.history_request = r.id; }
        if (s.adaptive and e.shared) { e.x.history = 0; e.x.history_request = 0; }
        s.adaptiveHint(e);
        s.xHint(e, r.*);
        s.x.push(.{ .kind = .dedup, .request = r.id, .namespace_id = r.namespace_id,
            .state = e.x.id, .cohort = e.x.cohort, .parent = r.parent, .role = e.x.role, .at = e.at });
    }

    fn adaptiveHint(s: *Store, e: *const Entry) void {
        if (!s.adaptive) return;
        const value = if (e.shared or e.x.history == 0) 0 else s.history.density(sharedKey(e.namespace, &.{}), e.x.history, s.x.request_serial, e.at);
        if (s.family.vtable.adaptive_hint) |f| f(s.family.ptr, e.saved, value);
    }

    fn xHint(s: *Store, e: *const Entry, r: RequestState) void {
        // Shared headers, auxiliary marks and mere length are not evidence that
        // this physical tail belongs to a continuing conversation.
        if (!s.x.enabled or !r.continuation or e.shared) return;
        if (r.role(e.at) == .auxiliary or r.role(e.at) == .unknown) return;
        if (s.family.vtable.continuation_hint) |f| f(s.family.ptr, e.saved, e.x.id);
    }

    fn xRemoved(s: *Store, e: *Entry) void {
        if (!s.x.enabled and !s.x.trace) return;
        const namespace_id = sharedKey(e.namespace, &.{});
        s.x.push(.{ .kind = .drop, .request = s.x.request_serial, .namespace_id = namespace_id,
            .state = e.x.id, .cohort = e.x.cohort, .parent = e.x.parent, .role = e.x.role,
            .at = e.at, .bytes = e.bytes, .reason = s.x.last_reason, .baseline = s.x.last_baseline });
        s.xLost(e, s.x.last_reason);
    }

    // Numeric evidence only; no removal, restore or new token hashing.
    fn xLost(s: *Store, e: *Entry, reason: x1.Reason) void {
        if (e.x.history_lost or reason == .reset or reason == .upgrade) return;
        e.x.history_lost = true;
        const namespace_id = sharedKey(e.namespace, &.{});
        if (s.adaptive and !e.shared and e.x.history != 0)
            s.history.evicted(namespace_id, e.x.history, s.x.request_serial);
        if (e.x.id != 0 and e.tokens.len == e.at) {
            const range = if (s.family.vtable.ghost_range) |f| f(s.family.ptr, e.saved) else [2]u32{0,0};
            s.x.ghosts[s.x.ghost_next] = .{ .key = e.x.key, .namespace_id = namespace_id,
                .state = e.x.id, .cohort = e.x.cohort, .at = e.at,
                .history = if (e.shared) 0 else e.x.history, .history_request = e.x.history_request,
                .last_token = e.tokens[e.tokens.len - 1], .reason = reason,
                .compatibility_known = s.family.vtable.ghost_range != null,
                .exact_len = if (s.rules.exact_end) range[0] else 0, .min_len = range[1] };
            s.x.ghost_next = (s.x.ghost_next + 1) % x1.ghost_capacity;
        }
    }

    /// At most four hashes / 524288 tokens per request, not per decode step.
    /// Ghost matches are diagnostic ONLY; all restores still compare real tokens.
    fn xGhostReturns(s: *Store, namespace: Namespace, prompt: []const u32, from: u32, r: *RequestState) void {
        if (!s.x.trace and !s.adaptive) return;
        var hashes: usize = 0;
        var tokens: usize = 0;
        for (0..x1.ghost_capacity) |off| {
            const i = (s.x.ghost_next + x1.ghost_capacity - 1 - off) % x1.ghost_capacity;
            const g = &s.x.ghosts[i];
            if (g.state == 0 or g.namespace_id != r.namespace_id or g.at <= from or g.at > prompt.len or prompt[g.at - 1] != g.last_token) continue;
            if (g.compatibility_known and prompt.len != g.exact_len and prompt.len < g.min_len) {
                s.x.ghost_incompatible_skips +|= 1;
                continue;
            }
            if (hashes >= 4 or tokens + g.at > 524288) { s.x.ghost_hash_skips +|= 1; continue; }
            hashes += 1;
            tokens += g.at;
            if (sharedKey(namespace, prompt[0..g.at]) != g.key) continue;
            if (s.adaptive and g.history != 0) {
                r.history = s.history.returned(r.namespace_id, r.id, g.history, g.history_request, true, false);
                r.history_observed = true;
            }
            s.x.ghost_returns +|= 1;
            s.x.push(.{ .kind = .ghost_return, .request = r.id, .namespace_id = r.namespace_id,
                .state = g.state, .cohort = g.cohort, .at = g.at, .effective = from, .prompt = r.input_len, .reason = g.reason });
            g.state = 0;
            // One known lost prefix per request. Does not claim all misses.
            break;
        }
    }

    /// One bounded pressure scan. No wall-clock TTL and no size-only pinning.
    fn xVictim(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        const t0 = x1.now();
        defer s.x.choose_ns +|= x1.now() -| t0;
        s.victim_dead = false;
        s.x.choices +|= 1;
        if (s.entries.items.len > x1.max_entries) return s.xFallback(namespace, skip, prompt);
        const Ref = struct { id: u64, index: usize };
        var refs: [x1.max_entries]Ref = undefined;
        for (s.entries.items, 0..) |e, i| refs[i] = .{ .id = e.x.id, .index = i };
        const order = struct { fn less(_: void, a: Ref, b: Ref) bool { return a.id < b.id; } }.less;
        std.mem.sort(Ref, refs[0..s.entries.items.len], {}, order);
        var best: ?x1.Candidate = null;
        for (s.entries.items, 0..) |e, i| {
            if (e == skip) continue;
            if (s.prune_dead) if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) {
                s.victim_dead = true;
                s.x.last_reason = .dead;
                return i;
            };
            var superseded = false;
            // An explicit successor was established only by a verified request.
            // Historical terminal ends retain a separate priority from auxiliaries.
            if (!e.shared and e.x.successor != 0) {
                var lo: usize = 0;
                var hi = s.entries.items.len;
                while (lo < hi) {
                    const mid = lo + (hi - lo) / 2;
                    if (refs[mid].id < e.x.successor) lo = mid + 1 else hi = mid;
                }
                if (lo < s.entries.items.len and refs[lo].id == e.x.successor) {
                    const o = s.entries.items[refs[lo].index];
                    superseded = if (s.family.vtable.live) |f| f(s.family.ptr, o.saved) else true;
                }
            }
            if (i % 32 == 0 and x1.now() -| t0 > 2 * std.time.ns_per_ms) return s.xFallback(namespace, skip, prompt);
            const terminal = e.x.role == .prompt_end or e.x.role == .decode_end;
            const candidate = x1.candidate(&s.x, e.x, e.reuse.restores, e.at, e.used, i, superseded and !terminal, superseded and terminal);
            if (best == null or x1.less(candidate, best.?)) best = candidate;
        }
        const chosen = best orelse return null;
        // X2: no observational legacy scan on the engine thread. The actual
        // legacy fallback above remains available for bounded-selection failure.
        s.x.last_reason = chosen.reason;
        return chosen.index;
    }

    fn xFallback(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
        s.x.fallbacks +|= 1;
        s.x.last_reason = .legacy_fallback;
        const victim = s.legacyVictimBut(namespace, skip, prompt);
        if (s.victim_dead) s.x.last_reason = .dead;
        return victim;
    }

    pub fn completedRequest(s: *Store, r: RequestState, cached: u32, elapsed_ns: u64, prefill_ns: u64, outcome: u8) void {
        if (r.id == 0) return; // includes image bypass and queued cancellations
        s.x.push(.{ .kind = .finished, .request = r.id, .namespace_id = r.namespace_id,
            .cohort = r.cohort, .parent = r.parent, .prompt = r.input_len, .at = cached,
            .elapsed_ns = elapsed_ns, .prefill_ns = prefill_ns, .outcome = outcome });
        if (s.adaptive) {
            s.history.completed(r.namespace_id, r.history, r.id, r.input_len, cached, prefill_ns, outcome == 1);
            // Update only this request's saved endpoints, once at completion.
            for (s.entries.items) |e| if (e.x.created_request == r.id) { s.adaptiveHint(e); };
            if (!@import("builtin").is_test and s.x.trace) std.log.info("checkpoint_x3: request={d} namespace={d} history={d} density_us={d} returns={d} forks={d} ghosts={d} cost_samples={d} replaced={d}", .{
                r.id, r.namespace_id, r.history, s.history.density(r.namespace_id, r.history, s.x.request_serial, r.input_len),
                s.history.counters.returns, s.history.counters.forks, s.history.counters.ghosts,
                s.history.counters.cost_samples, s.history.counters.replaced,
            });
        }
        s.flushX1();
    }

    pub fn flushX1(s: *Store) void {
        if (@import("builtin").is_test or !s.x.trace) return;
        // Bounded, numeric metadata, flushed only at prefill/completion boundaries.
        for (0..s.x.event_count) |off| {
            const e = s.x.events[(s.x.event_start + off) % x1.trace_capacity];
            std.log.info("checkpoint_x1: {{\"kind\":\"{s}\",\"reason\":\"{s}\",\"role\":\"{s}\",\"request\":{d},\"namespace\":{d},\"state\":{d},\"cohort\":{d},\"parent\":{d},\"at\":{d},\"prompt\":{d},\"desired\":{d},\"effective\":{d},\"bytes\":{d},\"baseline\":{d},\"elapsed_ns\":{d},\"prefill_ns\":{d},\"outcome\":{d}}}", .{ @tagName(e.kind), @tagName(e.reason), @tagName(e.role), e.request, e.namespace_id, e.state, e.cohort, e.parent, e.at, e.prompt, e.desired, e.effective, e.bytes, e.baseline, e.elapsed_ns, e.prefill_ns, e.outcome });
        }
        s.x.event_count = 0;
        s.x.event_start = 0;
        std.log.info("checkpoint_x1 totals: choices {d}, sampled {d}, different {d}, fallback {d}, mapped {d}, omitted {d}, duplicate_marks {d}, compatible_reservations {d}, continuations {d}, ghost_returns {d}, ghost_hash_skips {d}, ghost_incompatible_skips {d}, overwritten {d}, choose_ns {d}, plan_ns {d}, save_ns {d}, drop_ns {d}", .{ s.x.choices, s.x.sampled_choices, s.x.sampled_different, s.x.fallbacks, s.x.mapped, s.x.omitted, s.x.duplicate_marks, s.x.compatible_reservations, s.x.continuations, s.x.ghost_returns, s.x.ghost_hash_skips, s.x.ghost_incompatible_skips, s.x.overwritten, s.x.choose_ns, s.x.plan_ns, s.x.save_ns, s.x.drop_ns });
    }

};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
}

// K2a regression harness: no CUDA, no IO; real Store with a fake family.
const K2aFake = struct {
    drops: usize = 0,
    live_calls: usize = 0,
    const State = struct { live: bool = true, compatible: bool = true, hint_state: u64 = 0, hints: u64 = 0 };
    fn family(f: *K2aFake, liveness: bool) Snapshots {
        return .{ .ptr = f, .vtable = if (liveness) &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop_, .live = live, .compatible = compatible }
            else &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop_ } };
    }
    fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!Saved {
        const st = try std.testing.allocator.create(State);
        st.* = .{};
        return st;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: Saved) anyerror!void {}
    fn drop_(p: *anyopaque, saved: Saved) void {
        const f: *K2aFake = @ptrCast(@alignCast(p));
        f.drops += 1;
        std.testing.allocator.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn live(p: *anyopaque, saved: Saved) bool {
        const f: *K2aFake = @ptrCast(@alignCast(p));
        f.live_calls += 1;
        return (@as(*State, @ptrCast(@alignCast(saved)))).live;
    }
    fn compatible(_: *anyopaque, saved: Saved, _: usize) bool { return (@as(*State, @ptrCast(@alignCast(saved)))).compatible; }
    fn state(e: *Entry) *State { return @ptrCast(@alignCast(e.saved)); }
    fn add(s: *Store, tokens: []const u32, at: u32) !*Entry {
        const a = std.testing.allocator;
        const e = try a.create(Entry);
        const st = try a.create(State);
        st.* = .{};
        s.clock += 1;
        e.* = .{ .tokens = try a.dupe(u32, tokens), .at = at, .saved = st, .bytes = 1,
            .born = @intCast(tokens.len), .used = s.clock, .last = try a.dupe(u32, tokens) };
        try s.entries.append(a, e);
        s.held += 1;
        return e;
    }
};

test "K2a physical dead state goes before valid LRU, including shared and foreign namespace" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    _ = try K2aFake.add(&s, &.{3}, 1);
    dead.shared = true;
    dead.namespace = namespaceFor("other");
    K2aFake.state(dead).live = false;
    s.reserve(null, &.{9}, null, &.{1});
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expectEqual(@as(u64, 1), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(u64, 0), s.audit.valid_budget_evictions);
    try std.testing.expect(s.entries.items[0] == a);
    try std.testing.expectEqual(@as(u64, 2), s.held);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(@as(u64, 0), s.audit.shadow_decisions);
}

test "K2a reserve preserves borrowed resume even when dead, drops other dead exactly once" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const borrowed = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    _ = try K2aFake.add(&s, &.{3}, 1);
    K2aFake.state(borrowed).live = false;
    K2aFake.state(dead).live = false;
    s.reserve(null, &.{9}, borrowed, &.{1});
    try std.testing.expect(s.entries.items[0] == borrowed);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    // No pressure: no liveness scan and no second drop.
    const calls = f.live_calls;
    s.reserve(null, &.{9}, borrowed, &.{});
    try std.testing.expectEqual(calls, f.live_calls);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
}

test "K2a shadow disagrees but actual valid victim stays LRU; safe off and absent liveness work" {
    for ([_]bool{ true, false }) |liveness| {
        var f: K2aFake = .{};
        var s = Store.init(std.testing.allocator, f.family(liveness), .{}, 3);
        defer s.deinit();
        const a = try K2aFake.add(&s, &.{1}, 1);
        const b = try K2aFake.add(&s, &.{2}, 1);
        _ = try K2aFake.add(&s, &.{3}, 1);
        a.reuse.restores = 8;
        a.reuse.marginal_lower_bound_tokens = 32768;
        try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
        try std.testing.expectEqual(@as(u64, 1), s.audit.shadow_disagreements);
        s.diagnostics = false;
        try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
        try std.testing.expectEqual(@as(u64, 1), s.audit.shadow_disagreements);
        s.reserve(null, &.{9}, null, &.{1});
        try std.testing.expect(s.entries.items[0] == b);
        try std.testing.expectEqual(@as(u64, 1), s.audit.valid_budget_evictions);
    }
}

test "K2a no-dead pressure matches baseline policy across randomized tiers, namespaces and borrowers" {
    var prng = std.Random.DefaultPrng.init(0x4b3241);
    const rng = prng.random();
    for (0..200) |_| {
        var f: K2aFake = .{};
        var s = Store.init(std.testing.allocator, f.family(true), .{}, 8);
        defer s.deinit();
        for (0..8) |j| {
            const e = try K2aFake.add(&s, &.{ @intCast(j % 3), 2, 3 }, @intCast(j % 3 + 1));
            e.shared = rng.boolean();
            e.namespace = if (rng.boolean()) namespaceFor("A") else null;
            e.used = rng.intRangeAtMost(u64, 0, 100);
            e.born = rng.intRangeAtMost(u32, 1, 5);
            e.reuse.restores = rng.intRangeAtMost(u64, 0, 9);
        }
        const skip: ?*Entry = if (rng.boolean()) s.entries.items[rng.uintLessThan(usize, 8)] else null;
        s.prune_dead = false;
        s.diagnostics = false;
        const baseline = s.victimBut(null, skip, &.{ 1, 2, 3, 4 });
        s.prune_dead = true;
        s.diagnostics = true;
        try std.testing.expectEqual(baseline, s.victimBut(null, skip, &.{ 1, 2, 3, 4 }));
    }
}

test "K2a keep dedup is not restore; only successful restore earns tokens and marginal lower bound" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 10);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{ 1, 2 }, 2);
    const best = try K2aFake.add(&s, &.{ 1, 2, 3, 4 }, 4);
    try std.testing.expect(s.keep(&.{ 1, 2, 3, 4, 5 }, 4, null, &.{}));
    try std.testing.expectEqual(@as(u64, 0), best.reuse.restores);
    try std.testing.expectEqual(@as(u32, 1), best.reuse.dedups);
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5 }, &.{}) == best);
    s.resumed(best, &.{ 1, 2, 3, 4, 5 }, true);
    try std.testing.expectEqual(@as(u64, 1), best.reuse.restores);
    try std.testing.expectEqual(@as(u64, 4), best.reuse.restored_tokens);
    try std.testing.expectEqual(@as(u64, 2), best.reuse.marginal_lower_bound_tokens);
    try std.testing.expectEqual(s.clock, best.reuse.last_restore);
    s.resumed(a, &.{ 1, 2, 3, 4, 5 }, false);
    try std.testing.expectEqual(@as(u64, 1), s.counts.failed);
    try std.testing.expectEqual(@as(u64, 4), s.audit.restored_tokens);
    try std.testing.expect(@sizeOf(Reuse) <= 64);
}

test "K2a marginal bound stays conservative without comparing skipped prefixes; incompatible is not dead" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    const best = try K2aFake.add(&s, &.{ 1, 2, 3, 4 }, 4);
    _ = try K2aFake.add(&s, &.{ 9, 9, 9 }, 3); // skipped: may match, conservatively counts as alternative
    const incompatible = try K2aFake.add(&s, &.{ 1, 2 }, 2);
    K2aFake.state(incompatible).compatible = false;
    incompatible.shared = true; // valid shared anchor must survive while an ordinary victim is available
    try std.testing.expect(s.find(&.{ 1, 2, 3, 4, 5 }, &.{}) == best);
    try std.testing.expectEqual(@as(u32, 1), best.reuse.pending_marginal);
    s.reserve(null, &.{8}, best, &.{1});
    try std.testing.expectEqual(@as(u64, 0), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(u64, 1), s.audit.valid_budget_evictions);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, incompatible) != null);
}

test "K2a dead cleanup safe off reproduces old LRU and handles multiple dead with insufficient room" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 3);
    defer s.deinit();
    _ = try K2aFake.add(&s, &.{1}, 1);
    const dead = try K2aFake.add(&s, &.{2}, 1);
    const dead2 = try K2aFake.add(&s, &.{3}, 1);
    K2aFake.state(dead).live = false;
    K2aFake.state(dead2).live = false;
    s.prune_dead = false;
    s.reserve(null, &.{9}, null, &.{1});
    try std.testing.expect(s.entries.items[0] == dead);
    s.prune_dead = true;
    s.reserve(null, &.{9}, null, &.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 2), s.audit.dead_at_reserve);
    try std.testing.expectEqual(@as(usize, 3), f.drops);
}

test "125b reason counters observe decisions without changing lookup and eviction" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 3);
    defer s.deinit();
    const a = try K2aFake.add(&s, &.{ 1, 2 }, 2);
    const dead = try K2aFake.add(&s, &.{ 8, 9 }, 2);
    K2aFake.state(dead).live = false;
    const incompatible = try K2aFake.add(&s, &.{ 3, 4 }, 2);
    K2aFake.state(incompatible).compatible = false;
    try std.testing.expect(s.findScoped(null, &.{ 1, 2, 5 }, &.{}) == a);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.lookup_candidate);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.dead_candidate_skips);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.incompatible_candidate_skips);
    try std.testing.expectEqual(@as(u64, 0), s.reasons.restore_ok);
    s.resumed(a, &.{ 1, 2, 5 }, true);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.restore_ok);
    try std.testing.expect(s.findScoped(null, &.{ 20, 21 }, &.{}) == null);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.no_eligible_prefix_match);
    s.noteImageBypass(200000);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.image_bypass);
    try std.testing.expectEqual(@as(u64, 200000), s.reasons.image_bypass_prompt_tokens);
    try std.testing.expectEqual(@as(u64, 0), s.reasons.snapshot_budget);
    try s.clear();
    try std.testing.expectEqual(@as(u64, 3), s.reasons.reset_entries);
    try std.testing.expectEqual(@as(u64, 0), s.reasons.restore_failed);
}

test "125b death reasons are counted once on removal and unknown stays unknown" {
    var f: K2aFake = .{};
    var family = f.family(true);
    var vt = family.vtable.*;
    vt.dead_reason = struct {
        fn reason(_: *anyopaque, _: Saved) u8 { return 2; }
    }.reason;
    family.vtable = &vt;
    var s = Store.init(std.testing.allocator, family, .{}, 2);
    defer s.deinit();
    const dead = try K2aFake.add(&s, &.{ 1 }, 1);
    K2aFake.state(dead).live = false;
    _ = try K2aFake.add(&s, &.{ 2 }, 1);
    try std.testing.expect(s.evictBudget(null, null, &.{ 9 }));
    try std.testing.expectEqual(@as(u64, 1), s.reasons.placement_pressure);
    try std.testing.expect(s.evictBudget(null, null, &.{ 9 }));
    try std.testing.expectEqual(@as(u64, 1), s.reasons.snapshot_budget);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.placement_pressure);
    vt.dead_reason = null;
    const unknown = try K2aFake.add(&s, &.{ 3 }, 1);
    K2aFake.state(unknown).live = false;
    try std.testing.expect(s.evictBudget(null, null, &.{ 9 }));
    try std.testing.expectEqual(@as(u64, 1), s.reasons.physical_unknown);
    vt.dead_reason = struct {
        fn reason(_: *anyopaque, _: Saved) u8 { return 1; }
    }.reason;
    const overwritten = try K2aFake.add(&s, &.{ 4 }, 1);
    K2aFake.state(overwritten).live = false;
    try std.testing.expect(s.evictBudget(null, null, &.{ 9 }));
    try std.testing.expectEqual(@as(u64, 1), s.reasons.physical_overlap);
    const failed = try K2aFake.add(&s, &.{ 5 }, 1);
    s.resumed(failed, &.{ 5, 6 }, false);
    try std.testing.expectEqual(@as(u64, 1), s.reasons.restore_failed);
    try std.testing.expectEqual(@as(u64, 0), s.reasons.restore_ok);
}


test "126b reusable victim policy remains pressure only and preserves borrowed state" {
    const a=std.testing.allocator;
    var fake: K2aFake=.{};
    var s=Store.init(a,fake.family(true),.{},32);
    defer s.deinit();
    s.reuse_eviction=true;
    const t0=[_]u32{1,10,11};const t1=[_]u32{2,20,21};const t2=[_]u32{3,30,31};
    const hot=try K2aFake.add(&s,&t0,1);
    const cold=try K2aFake.add(&s,&t1,1);
    const borrowed=try K2aFake.add(&s,&t2,1);
    s.clock=100;hot.used=90;hot.reuse=.{.restores=4,.last_restore=90,.marginal_lower_bound_tokens=4*2048*64};
    cold.used=95;borrowed.used=0;
    try std.testing.expectEqual(@as(usize,1),s.victimBut(null,borrowed,&.{}).?);
    try std.testing.expect(K2aFake.state(hot).live and K2aFake.state(cold).live and K2aFake.state(borrowed).live);
    // An old popular entry cannot remain pinned forever by historical hits.
    s.clock=700;
    try std.testing.expectEqual(@as(usize,0),s.victimBut(null,borrowed,&.{}).?);
}

test "126b dead state still wins before reusable or shared states" {
    const a=std.testing.allocator;var fake: K2aFake=.{};
    var s=Store.init(a,fake.family(true),.{},32);defer s.deinit();s.reuse_eviction=true;
    const t0=[_]u32{1,10};const t1=[_]u32{2,20};
    const live_=try K2aFake.add(&s,&t0,1);const dead=try K2aFake.add(&s,&t1,1);
    live_.used=0;K2aFake.state(dead).live=false;
    try std.testing.expectEqual(@as(usize,1),s.victimBut(null,null,&.{}).?);
    try std.testing.expect(s.victim_dead);
}

// X1 exercises the real Store with fake snapshots; no inference benchmark.
const X1Grid = struct {
    fn failSave(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!Saved { return error.InjectedSaveFailure; }
    fn map(_: *anyopaque, _: ?Saved, len: usize, at: u32) u32 {
        const effective = at / 2048 * 2048;
        return if (effective < len and effective <= len -| 128) effective else 0;
    }
    fn automatic(_: *anyopaque, _: ?Saved, len: usize, out: *[2]u32) usize {
        out.* = .{ @intCast(std.mem.alignBackward(usize, len -| 128, 2)), @intCast(len) };
        return 2;
    }
};

test "X1 effective marks and shared keys agree; duplicate reservations are removed" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.map_mark = X1Grid.map;
    vt.automatic_marks = X1Grid.automatic;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 20);
    defer s.deinit();
    s.normalize_marks = true;
    s.x.trace = true;
    const tokens = try std.testing.allocator.alloc(u32, 10000);
    defer std.testing.allocator.free(tokens);
    @memset(tokens, 1);
    const l = try s.lookupScoped(null, std.testing.allocator, tokens, 9500, &.{2952, 4488, 5000}, &.{});
    defer std.testing.allocator.free(l.marks);
    try std.testing.expectEqualSlices(u32, &.{2048, 4096, 8192}, l.marks);
    const reservation = try s.reservationMarks(std.testing.allocator, tokens, null, l.marks);
    defer std.testing.allocator.free(reservation);
    try std.testing.expectEqualSlices(u32, &.{2048, 4096, 8192, 9872, 10000}, reservation);
    try std.testing.expectEqual(@as(u64, 1), s.x.duplicate_marks);
    try std.testing.expectEqual(@as(usize, 2), s.shared_keys.items.len);
    try std.testing.expect(s.keepRequestScoped(null, tokens, 4096, null, &.{}, &l.request));
    try std.testing.expect(s.entries.items[0].shared);
}

test "X1 compatible saved keys and terminal upgrades reserve no duplicate storage" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 3);
    defer s.deinit();
    s.normalize_marks = true;
    const end = try K2aFake.add(&s, &.{1, 2}, 2);
    const other = try K2aFake.add(&s, &.{8, 9}, 2);
    _ = try K2aFake.add(&s, &.{7}, 1);
    s.reserve(null, &.{1, 2}, end, &.{2, 2});
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    K2aFake.state(end).compatible = false;
    s.reserve(null, &.{1, 2}, null, &.{2});
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    try std.testing.expect(s.keep(&.{1, 2}, 2, null, &.{}));
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, other) != null);
    try std.testing.expectEqual(@as(u64, 1), s.audit.upgrades);
    try std.testing.expectEqual(@as(u64, 3), s.held);
}

test "X1 pressure evicts old one-off before active long and short; borrowed dead is preserved" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 4);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .serial = 100, .request_serial = 100 };
    const cold = try K2aFake.add(&s, &.{1}, 1);
    cold.x = .{ .id = 1, .role = .prompt_end };
    const long = try K2aFake.add(&s, &.{2}, 1);
    long.x = .{ .id = 2, .role = .prompt_end, .last_return = 100, .continuation = true };
    long.reuse.restores = 2;
    const short = try K2aFake.add(&s, &.{3}, 1);
    short.x = .{ .id = 3, .role = .prompt_end, .last_return = 100 };
    short.reuse.restores = 5;
    const borrowed = try K2aFake.add(&s, &.{4}, 1);
    borrowed.x.id = 4;
    K2aFake.state(borrowed).live = false;
    s.reserve(null, &.{9}, borrowed, &.{1});
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, long) != null);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, short) != null);
    try std.testing.expect(std.mem.indexOfScalar(*Entry, s.entries.items, borrowed) != null);
    try std.testing.expectEqual(x1.Reason.unused, s.x.last_reason);
}

test "X1 actual tail restore passes bounded credit; shared header and other namespace do not" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true };
    var initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    const l = try s.lookupScoped(null, std.testing.allocator, &.{1, 2, 3, 4, 5, 6}, 0, &.{}, &.{});
    defer std.testing.allocator.free(l.marks);
    var r = l.request;
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &r);
    try std.testing.expect(r.continuation);
    try std.testing.expectEqual(@as(u64, 1), old.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const next = s.entries.items[1];
    try std.testing.expect(next.x.continuation);
    try std.testing.expectEqual(@as(u64, 0), next.reuse.restores);
    try std.testing.expectEqual(old.x.id, next.x.parent);
    try std.testing.expectEqual(next.x.id, old.x.successor);
    // The historical restored end must not out-rank its current descendant
    // merely because only the old object owns the genuine restore counter.
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.superseded_tail, s.x.last_reason);
    const foreign = try s.lookupScoped(namespaceFor("other"), std.testing.allocator, &.{1, 2, 3, 4, 5, 6, 7}, 0, &.{}, &.{});
    defer std.testing.allocator.free(foreign.marks);
    try std.testing.expect(foreign.entry == null);
    try std.testing.expect(!foreign.request.continuation);
    old.shared = true;
    var header: RequestState = .{ .id = 9, .cohort = 9, .input_len = 7 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6, 7}, true, &header);
    try std.testing.expect(!header.continuation);
    try std.testing.expectEqual(@as(u64, 9), header.cohort);
}

test "X1 successor priority requires a live replacement; fallback encoder survives terminal save" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .request_serial = 1 };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 6, .encoder_at = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 2, null, &.{}, &r));
    const aux = s.entries.items[0];
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 4, null, &.{}, &r));
    const encoder = s.entries.items[1];
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const terminal = s.entries.items[2];
    try std.testing.expectEqual(terminal.x.id, aux.x.successor);
    try std.testing.expectEqual(@as(u64, 0), encoder.x.successor);
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.superseded_auxiliary, s.x.last_reason);
    K2aFake.state(terminal).live = false;
    try std.testing.expectEqual(@as(?usize, 2), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(x1.Reason.dead, s.x.last_reason);
}

test "X1 ghost reports a known evicted return without restoring or crossing namespaces" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{ .exact_end = true }, 1);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 3 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3}, 3, null, &.{}, &r));
    try std.testing.expect(s.keepRequestScoped(null, &.{8, 9, 10}, 3, null, &.{}, &r));
    const foreign = try s.lookupScoped(namespaceFor("other"), std.testing.allocator, &.{1, 2, 3}, 0, &.{}, &.{});
    defer std.testing.allocator.free(foreign.marks);
    try std.testing.expectEqual(@as(u64, 0), s.x.ghost_returns);
    const local = try s.lookupScoped(null, std.testing.allocator, &.{1, 2, 3}, 0, &.{}, &.{});
    defer std.testing.allocator.free(local.marks);
    try std.testing.expectEqual(@as(u64, 1), s.x.ghost_returns);
    try std.testing.expect(local.entry == null);
    try std.testing.expectEqual(@as(u64, 0), s.counts.hits);
}

test "X1 can disable retention without disabling plan; bounded entries fall back safely" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 600);
    defer s.deinit();
    for (0..513) |i| _ = try K2aFake.add(&s, &.{@intCast(i + 1)}, 1);
    s.x.enabled = true;
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{900}));
    try std.testing.expectEqual(@as(u64, 1), s.x.fallbacks);
    s.x.enabled = false;
    s.normalize_marks = true;
    try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{900}));
    try std.testing.expect(s.normalize_marks);
}

test "X1 failed save cannot supersede a usable fallback or invent restore credit" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 10);
    defer s.deinit();
    s.x = .{ .enabled = true, .trace = true, .request_serial = 1 };
    const r: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4, .encoder_at = 2 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 2, null, &.{}, &r));
    const fallback = s.entries.items[0];
    vt.save = X1Grid.failSave;
    try std.testing.expect(!s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &r));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), fallback.x.successor);
    try std.testing.expectEqual(@as(u64, 0), fallback.reuse.restores);
    try std.testing.expectEqual(@as(u64, 1), s.x.serial);
}

const X2Hints = struct {
    fn hint(_: *anyopaque, saved: Saved, state: u64) void {
        const st: *K2aFake.State = @ptrCast(@alignCast(saved));
        st.hints += 1;
        st.hint_state = state;
    }
};

test "X2 actual keep and dedup forward verified continuation without fake restores" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    vt.continuation_hint = X2Hints.hint;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 100);
    defer s.deinit();
    s.x.enabled = true;
    const initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(old).hints);
    var r: RequestState = .{ .id = 2, .cohort = 2, .input_len = 6 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &r);
    try std.testing.expect(r.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    const tail = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 1), K2aFake.state(tail).hints);
    try std.testing.expectEqual(tail.x.id, K2aFake.state(tail).hint_state);
    try std.testing.expectEqual(@as(u64, 0), tail.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &r));
    try std.testing.expectEqual(@as(u64, 2), K2aFake.state(tail).hints);
    try std.testing.expectEqual(@as(u64, 0), tail.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6, 7, 8}, 8, null, &.{}, &r));
    const decoded = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 1), K2aFake.state(decoded).hints);
    try std.testing.expectEqual(@as(u64, 0), decoded.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 2, null, &.{}, &r));
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(s.entries.items[s.entries.items.len-1]).hints);
}

test "X2 shared header and failed restore cannot create native continuation hints" {
    var f: K2aFake = .{};
    var vt = f.family(true).vtable.*;
    vt.tail_window = 128;
    vt.continuation_hint = X2Hints.hint;
    var s = Store.init(std.testing.allocator, .{ .ptr = &f, .vtable = &vt }, .{ .exact_end = true }, 100);
    defer s.deinit();
    s.x.enabled = true;
    const initial: RequestState = .{ .id = 1, .cohort = 1, .input_len = 4 };
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4}, 4, null, &.{}, &initial));
    const old = s.entries.items[0];
    old.shared = true;
    var shared: RequestState = .{ .id = 2, .cohort = 2, .input_len = 6 };
    s.resumedRequest(old, &.{1, 2, 3, 4, 5, 6}, true, &shared);
    try std.testing.expect(!shared.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6}, 6, null, &.{}, &shared));
    const tail = s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(tail).hints);
    var failed: RequestState = .{ .id = 3, .cohort = 3, .input_len = 8 };
    s.resumedRequest(tail, &.{1, 2, 3, 4, 5, 6, 7, 8}, false, &failed);
    try std.testing.expect(!failed.continuation);
    try std.testing.expect(s.keepRequestScoped(null, &.{1, 2, 3, 4, 5, 6, 7, 8}, 8, null, &.{}, &failed));
    try std.testing.expectEqual(@as(u64, 0), K2aFake.state(s.entries.items[s.entries.items.len-1]).hints);
}

test "X2 no observational legacy scan every32 choices, but real fallback remains" {
    var f: K2aFake = .{};
    var s = Store.init(std.testing.allocator, f.family(true), .{}, 4);
    defer s.deinit();
    s.x = .{ .enabled = true, .serial = 100, .request_serial = 100 };
    const cold = try K2aFake.add(&s, &.{1}, 1);
    cold.x.id = 1;
    const hot = try K2aFake.add(&s, &.{2}, 1);
    hot.x = .{ .id = 2, .continuation = true, .last_return = 100 };
    hot.reuse.restores = 3;
    for (0..65) |_| try std.testing.expectEqual(@as(?usize, 0), s.victimBut(null, null, &.{9}));
    try std.testing.expectEqual(@as(u64, 0), s.x.sampled_choices);
    try std.testing.expectEqual(@as(u64, 0), s.x.sampled_different);
    try std.testing.expectEqual(@as(u64, 0), s.audit.shadow_decisions);
    try std.testing.expectEqual(@as(u64, 0), s.x.fallbacks);
    // Exercise functional fallback directly; it must still produce a victim.
    try std.testing.expect(s.xFallback(null, null, &.{9}) != null);
    try std.testing.expectEqual(@as(u64, 1), s.x.fallbacks);
}

test "X3 actual cache transfers learned lineage without fake restore counts" {
    const a = std.testing.allocator;
    var k: K2aFake = .{};
    var s = Store.init(a, k.family(true), .{ .planned=false, .exact_end=true }, 32);
    defer s.deinit();
    s.x.enabled = true;
    s.adaptive = true;
    const first = [_]u32{1,2,3,4};
    var r: RequestState = .{ .id=1, .namespace_id=Store.sharedKey(null,&.{}), .input_len=4 };
    s.x.request_serial=1;
    r.history=s.history.begin(r.namespace_id,1);
    try std.testing.expect(s.keepRequestScoped(null,&first,4,null,&.{},&r));
    const e=s.entries.items[0];
    const root=e.x.history;
    var next: RequestState = .{ .id=2, .namespace_id=r.namespace_id, .input_len=5 };
    s.x.request_serial=2;
    next.history=s.history.begin(next.namespace_id,2);
    const longer=[_]u32{1,2,3,4,5};
    s.resumedRequest(e,&longer,true,&next);
    try std.testing.expect(next.continuation);
    try std.testing.expectEqual(root,next.history);
    try std.testing.expectEqual(@as(u64,1),e.reuse.restores);
    try std.testing.expect(s.keepRequestScoped(null,&longer,5,null,&.{},&next));
    const tail=s.entries.items[s.entries.items.len-1];
    try std.testing.expectEqual(root,tail.x.history);
    try std.testing.expectEqual(@as(u64,0),tail.reuse.restores);
    s.completedRequest(next,4,1000,500,1);
    try std.testing.expectEqual(@as(u64,2),s.history.get(r.namespace_id,root).?.completed);
}

test "X3 failed restore cannot promote a branch; reset clears learning" {
    const a = std.testing.allocator;
    var k: K2aFake = .{};
    var s=Store.init(a,k.family(true),.{ .planned=false,.exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var r: RequestState=.{ .id=1,.namespace_id=ns,.input_len=4 };
    r.history=s.history.begin(ns,1);
    const tokens=[_]u32{1,2,3,4};
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&r));
    var next: RequestState=.{ .id=2,.namespace_id=ns,.input_len=5 };
    next.history=s.history.begin(ns,2);
    const longer=[_]u32{1,2,3,4,5};
    s.resumedRequest(s.entries.items[0],&longer,false,&next);
    try std.testing.expect(!next.continuation and next.history==2);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.returns);
    try s.clear();
    try std.testing.expectEqual(@as(u64,0),s.history.counters.requests);
    try std.testing.expect(s.history.get(ns,1)==null);
}

test "X3 actual ghost learns without tracing, never restores or crosses namespaces" {
    const a=std.testing.allocator;
    var k: K2aFake= .{};
    var s=Store.init(a,k.family(true),.{ .exact_end=true },1);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var first: RequestState=.{ .id=1,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=1;
    first.history=s.history.begin(ns,1);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3},3,null,&.{},&first));
    var second: RequestState=.{ .id=2,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=2;
    second.history=s.history.begin(ns,2);
    try std.testing.expect(s.keepRequestScoped(null,&.{8,9,10},3,null,&.{},&second));
    const foreign=try s.lookupScoped(namespaceFor("other"),a,&.{1,2,3},0,&.{},&.{});
    defer a.free(foreign.marks);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.ghosts);
    const wrong=try s.lookupScoped(null,a,&.{9,2,3},0,&.{},&.{});
    defer a.free(wrong.marks);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.ghosts);
    const local=try s.lookupScoped(null,a,&.{1,2,3},0,&.{},&.{});
    defer a.free(local.marks);
    try std.testing.expectEqual(@as(u64,1),s.history.counters.ghosts);
    try std.testing.expectEqual(first.history,local.request.history);
    try std.testing.expect(local.entry==null);
    try std.testing.expectEqual(@as(u64,0),s.counts.hits);
}

test "X3 long ghost plus short restore is one learned return, not a fork" {
    const a=std.testing.allocator;
    var k: K2aFake= .{};
    var s=Store.init(a,k.family(true),.{ .exact_end=true },8);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const ns=Store.sharedKey(null,&.{});
    var first: RequestState=.{ .id=1,.namespace_id=ns,.input_len=3 };
    s.x.request_serial=1;
    first.history=s.history.begin(ns,1);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3},3,null,&.{},&first));
    var second: RequestState=.{ .id=2,.namespace_id=ns,.input_len=5 };
    s.x.request_serial=2;
    second.history=s.history.begin(ns,2);
    s.resumedRequest(s.entries.items[0],&.{1,2,3,4,5},true,&second);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4,5},5,null,&.{},&second));
    s.x.last_reason= .dead;
    s.remove(1); // physical long endpoint gone; short fallback still live
    // Exercise the harder variant: the historical fallback has its own exact
    // old tail, so an ordinary restore would independently qualify as a return.
    a.free(s.entries.items[0].last);
    s.entries.items[0].last=try a.dupe(u32,&.{1,2,3});
    var found=try s.lookupScoped(null,a,&.{1,2,3,4,5,6},0,&.{},&.{});
    defer a.free(found.marks);
    try std.testing.expect(found.entry!=null and found.request.history_observed);
    const before=s.history.counters.returns;
    const history=found.request.history;
    s.resumedRequest(found.entry.?,&.{1,2,3,4,5,6},true,&found.request);
    try std.testing.expect(found.request.continuation);
    try std.testing.expectEqual(history,found.request.history);
    try std.testing.expectEqual(before,s.history.counters.returns);
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
}

// Independent review probes appended only to an owned COPY of the sealed source.
test "X3 fixed review identical sequential replays are not conversation forks" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const tokens=[_]u32{1,2,3,4};
    var ids: [4]u64=undefined;
    for (0..4) |i| {
        var l=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
        defer a.free(l.marks);
        if (l.entry) |e| s.resumedRequest(e,&tokens,true,&l.request);
        try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&l.request));
        s.completedRequest(l.request,l.request.restored_at,1000,500,1);
        ids[i]=l.request.history;
    }
    std.debug.print("REVIEW exact_replays history={any} restores={d} returns={d} forks={d}\n",.{ids,s.counts.hits,s.history.counters.returns,s.history.counters.forks});
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
    try std.testing.expectEqual(@as(u64,1),s.entries.items[0].x.created_request);
    try std.testing.expectEqual(@as(u64,4),s.entries.items[0].x.history_request);
}

test "X3 fixed review first return to physically stale prefix learns its lost history" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const tokens=[_]u32{1,2,3,4};
    const first=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    const previous_history=first.request.history;
    // Native placement invalidates storage; core removal happens later.
    K2aFake.state(s.entries.items[0]).live=false;
    const next_tokens=[_]u32{1,2,3,4,5};
    const next=try s.lookupScoped(null,a,&next_tokens,0,&.{},&.{});
    defer a.free(next.marks);
    try std.testing.expect(next.entry==null);
    const before=s.history.counters.ghosts;
    try std.testing.expect(s.keepRequestScoped(null,&next_tokens,5,null,&.{},&next.request));
    const third=try s.lookupScoped(null,a,&.{1,2,3,4,5,6},0,&.{},&.{});
    defer a.free(third.marks);
    std.debug.print("REVIEW physical_return original={d} next_history={d} observed={} ghosts_before_keep={d} ghosts_next_lookup={d} dead_pruned={d}\n",.{previous_history,next.request.history,next.request.history_observed,before,s.history.counters.ghosts,s.audit.dead_at_keep});
    try std.testing.expect(next.request.history_observed);
    try std.testing.expectEqual(previous_history,next.request.history);
}

const ReviewHint = struct {
    var latest: u32=0;
    var calls: u32=0;
    fn hint(_: *anyopaque,_:Saved,density:u32) void { latest=density;calls+=1; }
};
test "X3 fixed review converting a private checkpoint to shared clears native preference" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var vt=f.family(true).vtable.*;
    vt.adaptive_hint=ReviewHint.hint;
    var s=Store.init(a,.{ .ptr=&f,.vtable=&vt },.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    ReviewHint.latest=0;ReviewHint.calls=0;
    const tokens=[_]u32{1,2,3,4};
    const first=try s.lookupScoped(null,a,&tokens,0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    const before=ReviewHint.latest;
    try std.testing.expect(before>0);
    // A later planned shared cut turns this deduplicated physical key shared.
    s.noteShared(null,&tokens);
    try std.testing.expect(s.keepRequestScoped(null,&tokens,4,null,&.{},&first.request));
    try std.testing.expect(s.entries.items[0].shared);
    std.debug.print("REVIEW shared_conversion before_density={d} after_density={d} callback_count={d}\n",.{before,ReviewHint.latest,ReviewHint.calls});
    try std.testing.expectEqual(@as(u32,0),ReviewHint.latest);
    try std.testing.expectEqual(@as(u64,0),s.entries.items[0].x.history);
}

test "X3 replay of an older endpoint remains reuse then growth follows its learning head" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit();
    s.x.enabled=true; s.adaptive=true;
    const inputs=[_][]const u32{&.{1,2,3,4},&.{1,2,3,4,5,6},&.{1,2,3,4},&.{1,2,3,4,9}};
    var root: u64=0;
    for(inputs,0..) |tokens,i| {
        var l=try s.lookupScoped(null,a,tokens,0,&.{},&.{});
        defer a.free(l.marks);
        if(l.entry) |e| s.resumedRequest(e,tokens,true,&l.request);
        try std.testing.expect(s.keepRequestScoped(null,tokens,@intCast(tokens.len),null,&.{},&l.request));
        if(i==0) root=l.request.history;
        try std.testing.expectEqual(root,l.request.history);
    }
    try std.testing.expectEqual(@as(u64,0),s.history.counters.forks);
    try std.testing.expectEqual(@as(u64,3),s.history.counters.returns);
}

test "X3 stale evidence is single use and does not free snapshots during lookup" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var s=Store.init(a,f.family(true),.{ .exact_end=true },32);
    defer s.deinit(); s.x.enabled=true; s.adaptive=true;
    const first=try s.lookupScoped(null,a,&.{1,2,3,4},0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4},4,null,&.{},&first.request));
    K2aFake.state(s.entries.items[0]).live=false;
    const wrong=try s.lookupScoped(null,a,&.{9,2,3,4,5},0,&.{},&.{});
    defer a.free(wrong.marks);
    try std.testing.expect(!wrong.request.history_observed);
    try std.testing.expectEqual(@as(usize,1),s.entries.items.len);
    const good=try s.lookupScoped(null,a,&.{1,2,3,4,5},0,&.{},&.{});
    defer a.free(good.marks);
    try std.testing.expect(good.request.history_observed);
    try std.testing.expectEqual(first.request.history,good.request.history);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4,5},5,null,&.{},&good.request));
    const again=try s.lookupScoped(null,a,&.{1,2,3,4,6},0,&.{},&.{});
    defer a.free(again.marks);
    try std.testing.expect(!again.request.history_observed);
    try std.testing.expectEqual(@as(u64,1),s.history.counters.ghosts);
}

test "X3 shared promotion clears native preference even without request metadata" {
    const a=std.testing.allocator;
    var f: K2aFake=.{};
    var vt=f.family(true).vtable.*; vt.adaptive_hint=ReviewHint.hint;
    var s=Store.init(a,.{ .ptr=&f,.vtable=&vt },.{ .exact_end=true },32);
    defer s.deinit(); s.x.enabled=true; s.adaptive=true;
    ReviewHint.latest=0;
    const first=try s.lookupScoped(null,a,&.{1,2,3,4},0,&.{},&.{});
    defer a.free(first.marks);
    try std.testing.expect(s.keepRequestScoped(null,&.{1,2,3,4},4,null,&.{},&first.request));
    try std.testing.expect(ReviewHint.latest>0);
    s.noteShared(null,&.{1,2,3,4});
    try std.testing.expect(s.keepScoped(null,&.{1,2,3,4},4,null,&.{}));
    try std.testing.expectEqual(@as(u32,0),ReviewHint.latest);
    try std.testing.expectEqual(@as(u64,0),s.entries.items[0].x.history);
}


test "X4 diagnostic ghost excludes a length that cannot be natively resumed" {
    var fake:K2aFake=.{};
    var vt=fake.family(true).vtable.*;
    vt.ghost_range=struct {
        fn range(_: *anyopaque,_:Saved) [2]u32 { return .{3,std.math.maxInt(u32)}; }
    }.range;
    var s=Store.init(std.testing.allocator,.{.ptr=&fake,.vtable=&vt},.{.exact_end=true},8);
    defer s.deinit();
    s.x.trace=true;
    const old=try K2aFake.add(&s,&.{1,2,3},3);
    old.x.id=1;old.x.key=Store.sharedKey(null,old.tokens);old.x.role=.prompt_end;
    s.xLost(old,.dead);
    var request:RequestState=.{.id=2,.namespace_id=Store.sharedKey(null,&.{}),.input_len=4};
    s.xGhostReturns(null,&.{1,2,3,4},0,&request);
    try std.testing.expectEqual(@as(u64,0),s.x.ghost_returns);
    try std.testing.expectEqual(@as(u64,1),s.x.ghost_incompatible_skips);
    // It is still a valid diagnostic for a future exact replay, not discarded.
    request.input_len=3;
    s.xGhostReturns(null,&.{1,2,3},0,&request);
    try std.testing.expectEqual(@as(u64,1),s.x.ghost_returns);
}
