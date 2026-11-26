#!/usr/bin/env python3
"""Simulator over moe-trace recordings. No inference code, just counting.

Input:  a directory of *.bin traces written by moe-trace (see run.sh).
Output: tables on stdout; optional JSON with --json (keys such as "zeichen"/"erzeugt"/"lopo_*" are kept
        from the original German analysis so existing data files stay readable).
Run:    python3 analyze.py traces/ [--json out.json]

Questions:
  A  How skewed is the expert distribution? A fixed set of frequent experts in B MiB of VRAM:
     hit rate on unseen prompts (set chosen from all other prompts, leave-one-prompt-out = LOPO).
  B  Reuse between consecutive decode tokens (t-1 -> t), and within a window of the last k tokens.
  C  Verifying k tokens in one batch (MTP/speculation): unique experts per layer / (8k) = bytes per token.
  D  Dropping experts: weight share of the smallest 1/2 experts.
  E  Rough tg estimate from A: token time = GPU part + CPU part; the CPU part shrinks with the hit share.
"""
import glob, json, os, struct, sys
import numpy as np

EXPERT_MIB = 18760 / (40 * 256)          # routed experts total 18,760 MiB (measured)
T_TOKEN_MS, T_CPU_MS = 27.0, 20.0         # measured: ~75 % of the 27 ms token time is CPU/RAM work during tg
BUDGETS = [256, 512, 768, 800, 1024, 1536, 2048, 2600]   # 800 ~ free VRAM at 128k ctx, 2600 ~ free at 64k


def lade(pfad):
    raw = open(pfad, "rb").read()
    off = 0
    phase_tok = []            # phase per token, in order
    topk = {}                 # layer -> list of arrays [n, k]
    wts = {}                  # layer -> list of arrays [n, k] (last version per ubatch)
    erzeugt = []
    while off < len(raw):
        k, a, b, c = struct.unpack_from("<4i", raw, off); off += 16
        if k == 2:
            phase_tok += [a] * b
        elif k == 3:
            erzeugt.append(a)
        elif k in (0, 1):
            n = b * c
            arr = np.frombuffer(raw, dtype=np.int32 if k == 0 else np.float32, count=n, offset=off).reshape(c, b)
            off += 4 * n
            if k == 0:
                topk.setdefault(a, []).append(arr)
                wts.setdefault(a, []).append(None)
            else:
                wts[a][-1] = arr
        else:
            raise ValueError(f"{pfad}: unknown record {k} at {off}")
    # In prefill the last layer only computes the output rows (inp_out_ids), so it has fewer rows: drop it.
    n_tok = len(phase_tok)
    voll = {l for l in topk if sum(len(x) for x in topk[l]) == n_tok}
    if len(voll) < len(topk):
        print(f"  {os.path.basename(pfad)}: layers {sorted(set(topk) - voll)} incomplete (prefill), dropped", file=sys.stderr)
    L = sorted(voll)
    K =np.stack([np.concatenate(topk[l]) for l in L], axis=1)             # [T, L, k]
    W = np.stack([np.concatenate(wts[l]) for l in L], axis=1) if all(x is not None for l in L for x in wts[l]) else None
    ph = np.array(phase_tok)
    assert len(ph) == K.shape[0], f"{pfad}: {len(ph)} marker tokens, {K.shape[0]} routing rows"
    return {"K": K, "W": W, "phase": ph, "erzeugt": len(erzeugt)}


def zaehle(K, n_exp=256):
    T, L, k = K.shape
    c = np.zeros((L, n_exp), np.int64)
    for l in range(L):
        c[l] = np.bincount(K[:, l].ravel(), minlength=n_exp)
    return c


def hot_set(counts, slots):
    """Global top-`slots` (layer, expert) pairs by frequency -> mask [L, E]."""
    flat = counts.ravel()
    idx = np.argsort(-flat, kind="stable")[:slots]
    m = np.zeros(flat.shape, bool); m[idx] = True
    return m.reshape(counts.shape)


def trefferquote(K, maske):
    T, L, k = K.shape
    hits = maske[np.arange(L)[None, :, None], K]
    return float(hits.mean())


def wiederverwendung(K, fenster):
    T, L, k = K.shape
    if T <= fenster: return float("nan")
    tr = []
    for t in range(fenster, T):
        prev = K[t - fenster:t]                                   # [f, L, k]
        for l in range(L):
            s = set(prev[:, l].ravel().tolist())
            tr.append(sum(e in s for e in K[t, l]) / k)
    return float(np.mean(tr))


def union_faktor(K, n):
    """Unique experts per layer over n consecutive tokens / (n*k), non-overlapping windows."""
    T, L, k = K.shape
    f = []
    for t in range(0, T - n + 1, n):
        for l in range(L):
            f.append(len(set(K[t:t + n, l].ravel().tolist())) / (n * k))
    return float(np.mean(f)) if f else float("nan")


def tg_schaetzung(h):
    return 1000.0 / (T_TOKEN_MS - T_CPU_MS * h)


def main():
    d = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    tr = {os.path.basename(p)[:-4]: lade(p) for p in sorted(glob.glob(os.path.join(d, "*.bin")))}
    erg = {"prompts": {}, "A": {}, "B": {}, "C": {}, "D": {}}
    print(f"Expert {EXPERT_MIB:.3f} MiB, uniform distribution: B MiB hits B/{40*256*EXPERT_MIB:.0f}")
    for n, t in tr.items():
        dec = t["phase"] == 1
        erg["prompts"][n] = {"token": int(len(t["phase"])), "decode": int(dec.sum()), "erzeugt": t["erzeugt"]}
        print(f"  {n:16s} tokens {len(t['phase']):6d}  decode {int(dec.sum()):5d}")

    # A: fixed set, leave-one-prompt-out, for all tokens and for decode only
    print("\nA  Fixed set of frequent experts, hits on the held-out prompt (set chosen from the others)")
    print("   Budget  Slots  uniform  LOPO all  LOPO decode  in-sample decode  tg estimate (LOPO decode)")
    n_layer = next(iter(tr.values()))["K"].shape[1]
    for B in BUDGETS:
        # The budget covers all 40 layers; if one is missing (prefill), scale down, otherwise the rate is flattered
        slots = int(B / EXPERT_MIB * n_layer / 40)
        alle, dec, ins = [], [], []
        for n in tr:
            train = [x for m, x in tr.items() if m != n]
            c = sum(zaehle(x["K"]) for x in train)
            mask = hot_set(c, slots)
            alle.append(trefferquote(tr[n]["K"], mask))
            Kd = tr[n]["K"][tr[n]["phase"] == 1]
            if len(Kd): dec.append(trefferquote(Kd, mask))
            ci = zaehle(Kd) if len(Kd) else None
            if ci is not None: ins.append(trefferquote(Kd, hot_set(ci, slots)))
        h = float(np.mean(dec))
        erg["A"][B] = {"slots": slots, "uniform": slots / (n_layer * 256), "lopo_alle": float(np.mean(alle)), "lopo_decode": h,
                       "insample_decode": float(np.mean(ins)), "tg": tg_schaetzung(h), "je_prompt_decode": dec}
        print(f"   {B:6d} {slots:6d}  {slots/(n_layer*256):10.1%}  {np.mean(alle):9.1%}  {h:11.1%}  {np.mean(ins):16.1%}  "
              f"{tg_schaetzung(h):5.1f} t/s (today {1000/T_TOKEN_MS:.1f})")

    # Concentration per layer: share of activations on the top 5 % / 10 % of experts
    call = sum(zaehle(x["K"][x["phase"] == 1]) for x in tr.values())
    s = -np.sort(-call, axis=1); tot = s.sum(1, keepdims=True)
    top5 = (s[:, :13].sum(1) / tot[:, 0]); top10 = (s[:, :26].sum(1) / tot[:, 0])
    print(f"\n   Decode, per layer: top 5 % of experts carry {top5.mean():.1%} (min {top5.min():.1%}, max {top5.max():.1%}), "
          f"Top-10 % {top10.mean():.1%}")
    erg["A"]["konzentration"] = {"top5_mittel": float(top5.mean()), "top10_mittel": float(top10.mean()),
                                 "top5_je_layer": top5.tolist()}

    # B: reuse
    print("\nB  Reuse in decode: share of the 8 experts that were already used in the last f tokens")
    for f in (1, 2, 4, 8, 16):
        v = np.mean([wiederverwendung(t["K"][t["phase"] == 1], f) for t in tr.values() if (t["phase"] == 1).sum() > f])
        erg["B"][f] = float(v)
        print(f"   f={f:2d}: {v:.1%}")

    # C: union for k tokens in a batch
    print("\nC  Batch of n decode tokens: unique experts / (8n) = bytes per token relative to today")
    for n in (1, 2, 3, 4, 8):
        u = np.mean([union_faktor(t["K"][t["phase"] == 1], n) for t in tr.values()])
        erg["C"][n] = float(u)
        print(f"   n={n}: {u:.3f}")

    # D: weights
    W = [t["W"][t["phase"] == 1] for t in tr.values() if t["W"] is not None]
    if W:
        Wd = np.concatenate(W).reshape(-1, 8)
        Ws = np.sort(Wd / Wd.sum(1, keepdims=True), axis=1)
        erg["D"] = {"kleinster": float(Ws[:, 0].mean()), "kleinste2": float(Ws[:, :2].sum(1).mean()),
                    "groesster": float(Ws[:, -1].mean())}
        print(f"\nD  Router weights (normalized, decode): largest {Ws[:,-1].mean():.1%}, smallest {Ws[:,0].mean():.1%}, "
              f"two smallest together {Ws[:,:2].sum(1).mean():.1%}")
    if out_json:
        json.dump(erg, open(out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
