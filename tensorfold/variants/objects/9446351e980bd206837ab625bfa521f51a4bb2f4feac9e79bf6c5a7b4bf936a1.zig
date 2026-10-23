//! Exact prompt reuse for any family and backend: states kept at prompt-pass chunk ends, found by their tokens.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");

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
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// Optional storage liveness and request compatibility, checked before longest-prefix selection.
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
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

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
    diagnostics: bool = true,
    prune_dead: bool = true,
    victim_dead: bool = false,

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        var s: Store = .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
        if (!@import("builtin").is_test) {
            if (std.c.getenv("TF_K2A_DIAGNOSTICS")) |v| s.diagnostics = v[0] != '0';
            if (std.c.getenv("TF_K2A_PRUNE_DEAD")) |v| s.prune_dead = v[0] != '0';
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
        while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
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
        s.free(e);
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
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, e.saved)) continue;
            if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, e.saved, prompt.len)) continue;
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
        const marks_ = try s.fitting(a, try s.marks(a, prompt, if (e) |x| x.at else 0, history_len, shared, starts, if (e) |x| x.last else &.{}));
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const at = if (s.rules.planned) floorStart(starts, w) else w;
            if (std.mem.indexOfScalar(u32, marks_, at) != null) s.noteShared(namespace, prompt[0 .. at + s.rules.lookahead]);
        }
        s.reserve(namespace, prompt, e, marks_);
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, namespace: Namespace, prompt: []const u32, from: ?*Entry, marks_: []const u32) void {
        var need: u64 = 0;
        for (marks_) |at| need += s.family.vtable.bytes(s.family.ptr, at);
        while (s.held + need > s.budget) {
            if (!s.evictBudget(namespace, from, prompt)) return;
        }
    }

    /// A kept state a peer could not resume goes, so later prompts do not ask for it again.
    pub fn forget(s: *Store, prompt: []const u32, at: u32) void {
        for (s.entries.items, 0..) |e, i| if (e.namespace == null and e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) {
            s.counts.failed += 1;
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
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
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
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        var shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(namespace, prompt[0..n])) != null;
        // A stale physical reservation must be replaced, never refreshed by token-key deduplication.
        var old_i: usize = 0;
        while (old_i < s.entries.items.len) {
            const old = s.entries.items[old_i];
            if (s.family.vtable.live) |f| if (!f(s.family.ptr, old.saved)) {
                s.audit.dead_at_keep +|= 1;
                s.remove(old_i);
                continue;
            };
            // Encoder-only and complete terminal snapshots can share a token key.
            // Upgrade only at the exact request end; never downgrade a complete state
            // when another longer request keeps an encoder checkpoint at that key.
            if (s.rules.exact_end and at == prompt.len and std.meta.eql(old.namespace, namespace) and old.at == at and std.mem.eql(u32, old.tokens, prompt[0..n])) {
                if (s.family.vtable.compatible) |f| if (!f(s.family.ptr, old.saved, prompt.len)) {
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
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .namespace = namespace, .tokens = tokens, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared };
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
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
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
        if (s.diagnostics) std.log.info("prompt cache k2a: dead_pressure {d}, dead_keep {d}, valid_budget {d}, upgrades {d}, dedups {d}, restored_tokens {d}, marginal_lower_bound_tokens {d}, shadow_decisions {d}, shadow_disagreements {d}, shadow_eligible {d}", .{ s.audit.dead_at_reserve, s.audit.dead_at_keep, s.audit.valid_budget_evictions, s.audit.upgrades, s.audit.dedups, s.audit.restored_tokens, s.audit.marginal_lower_bound_tokens, s.audit.shadow_decisions, s.audit.shadow_disagreements, s.audit.shadow_eligible });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip` (a pass's resume): one a later prompt extends (never a shared cut), oldest first; else the oldest.
    fn victimBut(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) ?usize {
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
        if (best) |i| { s.observeShadow(i, shadow, eligible); return i; }
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
        return oldest;
    }

    fn evictBudget(s: *Store, namespace: Namespace, skip: ?*Entry, prompt: []const u32) bool {
        const i = s.victimBut(namespace, skip, prompt) orelse return false;
        if (s.victim_dead) s.audit.dead_at_reserve +|= 1 else s.audit.valid_budget_evictions +|= 1;
        s.remove(i); // same owner thread, family drop and TP3 ordering as before
        s.counts.evicted += 1; // preserve legacy eviction semantics
        return true;
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
        return @min(e.reuse.restores, 8) * (1 + @min(marginal / 2048, 128));
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
    const State = struct { live: bool = true, compatible: bool = true };
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
