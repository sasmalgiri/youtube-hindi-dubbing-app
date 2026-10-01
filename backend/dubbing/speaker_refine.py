"""Speaker refinement at the speech-turn level (shared by the classic
multi-speaker worker and the Hindi dialogue profile)."""
from __future__ import annotations

# pyannote clusters 10 s windows and folds clusters with too little audio
# (min_cluster_size) into the nearest big one, so a story character with
# only a few short lines vanishes into another character's voice. Measured
# on a 10-character test story: a character's own lines sit <= 0.25 apart
# (WeSpeaker cosine distance) while a swallowed minor character sat
# 0.69-0.93 from the cluster it was folded into. Re-clustering each
# cluster's turns (average linkage) and splitting off groups farther than
# REFINE_SPLIT_DIST recovers such characters; voices that are genuinely
# indistinguishable (<= ~0.3) stay together.
REFINE_SPLIT_DIST = 0.6
REFINE_MIN_TURN_SEC = 1.5     # shorter turns give unreliable embeddings
REFINE_MIN_NEW_SEC = 3.0      # a split-off character needs this much speech
REFINE_STRONG_SPLIT_DIST = 0.75  # …unless it is this far from the main group
                                 # (a one-line character: own lines sit <= ~0.25)
REFINE_MOVE_FROM = 0.5        # a line this far from its own cluster…
REFINE_MOVE_TO = 0.45         # …and this close to another one is moved there
REFINE_MOVE_MARGIN = 0.2      # …only if the other cluster is clearly closer


def refine_speakers_by_turns(ranges, turn_embs):
    """ranges: {spk: [(s, e)]}; turn_embs: {(spk, s, e): embedding}.
    Returns (new_ranges, centroids, n_changed).

    1. Reassign: a line far from its own cluster (> REFINE_MOVE_FROM) but
       close to ANOTHER cluster (< REFINE_MOVE_TO, and REFINE_MOVE_MARGIN
       closer) moves there — e.g. one of Raja's lines that pyannote filed
       under the narrator, while Raja's other lines have their own cluster.
    2. Split: within each cluster, lines that form a separate group
       (average linkage > REFINE_SPLIT_DIST) with >= REFINE_MIN_NEW_SEC of
       speech become a new character."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    unit = lambda v: np.asarray(v, dtype=float) / (np.linalg.norm(v) + 1e-9)
    dist = lambda u, v: 1.0 - float(unit(u) @ unit(v))
    items = []   # [spk, (s, e), emb or None]
    for spk, rs in ranges.items():
        for r in rs:
            e = turn_embs.get((spk, r[0], r[1]))
            ok = e is not None and r[1] - r[0] >= REFINE_MIN_TURN_SEC
            items.append([spk, r, unit(e) if ok else None])

    def centroids(exclude=None):
        acc = {}
        for i, (spk, r, e) in enumerate(items):
            if e is None or i == exclude:
                continue
            w = r[1] - r[0]
            c = acc.setdefault(spk, [np.zeros_like(e), 0.0])
            c[0] = c[0] + e * w
            c[1] += w
        return {k: v[0] / v[1] for k, v in acc.items() if v[1] > 0}

    changed = 0
    # 1) reassign stray lines to the cluster they actually belong to
    for i, (spk, r, e) in enumerate(items):
        if e is None:
            continue
        cents = centroids(exclude=i)
        if spk not in cents or len(cents) < 2:
            continue
        d_own = dist(e, cents[spk])
        others = [(dist(e, c), k) for k, c in cents.items() if k != spk]
        d_best, k_best = min(others)
        if d_own > REFINE_MOVE_FROM and d_best < REFINE_MOVE_TO and d_best + REFINE_MOVE_MARGIN < d_own:
            items[i][0] = k_best
            changed += 1

    # 2) split clusters that still hold a separate group of lines
    by_spk = {}
    for spk, r, e in items:
        by_spk.setdefault(spk, []).append((r, e))
    out, cents = {}, {}
    for spk, members in by_spk.items():
        with_e = [(r, e) for r, e in members if e is not None]
        groups = None
        if len(with_e) >= 2:
            lab = fcluster(linkage(np.stack([e for _, e in with_e]), method="average", metric="cosine"),
                           t=REFINE_SPLIT_DIST, criterion="distance")
            if len(set(lab)) > 1:
                groups = {}
                for (r, e), g in zip(with_e, lab):
                    groups.setdefault(int(g), []).append((r, e))
        if not groups:
            out[spk] = sorted(r for r, _ in members)
            if with_e:
                cents[spk] = np.mean([e for _, e in with_e], axis=0)
            continue
        dur = {g: sum(r[1] - r[0] for r, _ in m) for g, m in groups.items()}
        main = max(dur, key=dur.get)
        main_turns = [r for r, _ in groups[main]]
        main_c = np.mean([e for _, e in groups[main]], axis=0)
        k = 0
        for g, mem in sorted(groups.items(), key=lambda kv: -dur[kv[0]]):
            if g == main:
                continue
            far = dist(np.mean([e for _, e in mem], axis=0), main_c) >= REFINE_STRONG_SPLIT_DIST
            if dur[g] >= REFINE_MIN_NEW_SEC or far:
                k += 1
                new = f"{spk}_{chr(ord('a') + k)}"
                out[new] = sorted(r for r, _ in mem)
                cents[new] = np.mean([e for _, e in mem], axis=0)
                changed += 1
            else:
                main_turns += [r for r, _ in mem]   # too little to stand alone
        main_turns += [r for r, e in members if e is None]   # short lines stay
        out[spk] = sorted(main_turns)
        cents[spk] = main_c
    return out, cents, changed



def cut_at_bounds(ranges, seg_bounds):
    """Split each turn at transcript segment boundaries (one piece per line).

    pyannote stitches back-to-back lines of one cluster into a single turn,
    whose embedding is then a blend of two voices."""
    import bisect
    cuts = sorted({float(x) for b in (seg_bounds or []) for x in b})
    if not cuts:
        return ranges
    out = {}
    for spk, rs in ranges.items():
        pieces = []
        for a, b in rs:
            lo, hi = bisect.bisect_right(cuts, a), bisect.bisect_left(cuts, b)
            edges = [a] + cuts[lo:hi] + [b]
            pieces += [(x, y) for x, y in zip(edges, edges[1:]) if y - x > 0.05]
        out[spk] = pieces
    return out


def turn_embeddings(ranges, waveform, sr, emb_fn):
    """{(spk, s, e): embedding} for turns long enough to embed reliably."""
    import numpy as np
    out = {}
    for spk, rs in ranges.items():
        for a, b in rs:
            if b - a < REFINE_MIN_TURN_SEC:
                continue
            try:
                e = emb_fn(waveform[:, int(a * sr):int(b * sr)][None])[0]
            except Exception:
                continue
            if np.all(np.isfinite(e)):
                out[(spk, a, b)] = e
    return out
