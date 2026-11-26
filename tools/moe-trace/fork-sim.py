#!/usr/bin/env python3
"""Replay the strategy of the fork llama-wackMall (src/llama-expert-tier.cpp, maybe_update) on routing traces.

Input:  a directory of *.bin traces written by moe-trace (see run.sh); imports the loader from analyze.py.
Output: per VRAM budget and start mode: hit rate, swaps per token, estimated tg; optional JSON (--json).
Run:    python3 fork-sim.py traces/ [--json out.json]

Reimplemented from reading the fork at commit 39d867b:
  - S = slots per layer, the same number in every layer (not distributed globally as in analyze.py A).
  - Per update (after each ubatch): score = score * 0.999 + count, then at most ONE swap per layer:
    coldest slot si against the hottest cold expert ec, if the slot is empty or
    (dwell[si] >= 32 and score[ec] > 1.5 * score[si]).
  - A swap uploads the expert slice synchronously (ggml_backend_tensor_set): gate + up + down,
    EXPERT_MIB, at 12.4 GB/s (pinned H2D, measured) ~0.155 ms, serial to compute.
  - Prefill (ubatch > TMAX = 16) only counts (build_moe_count) and hits nothing. Here modeled as updates in
    blocks of UB tokens.
Assumption: dwell counts updates. Hits are counted in decode only.

Start modes:
  leer   slots empty, filled with one expert per layer and update (LLAMA_EXPERT_ADAPT without heat CSV)
  seed   heat CSV from the other prompts (leave-one-prompt-out), then adaptive
  fest   like seed but without swapping (baseline: value of adapting)
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import lade, zaehle, EXPERT_MIB, T_TOKEN_MS, T_CPU_MS

DECAY, HYST, DWELL = 0.999, 1.5, 32
H2D_GBS = 12.4
T_SWAP_MS = EXPERT_MIB * 1024 * 1024 / (H2D_GBS * 1e9) * 1e3
UB = 4096
BUDGETS = [800, 1024, 1172, 2048, 2600]          # 1172 ~ fork default S = 16


def slots_fuer(budget_mib, n_layer_modell=40):
    return int(budget_mib // (EXPERT_MIB * n_layer_modell))


def simuliere(K, phase, S, seed_counts=None, adapt=True, n_exp=256):
    T, L, k = K.shape
    slot = -np.ones((L, S), np.int64)
    dwell = np.zeros((L, S), np.int64)
    score = np.zeros((L, n_exp))
    if seed_counts is not None:
        for l in range(L):
            top = np.argsort(-seed_counts[l], kind="stable")[:S]
            slot[l, :len(top)] = top
            score[l] = seed_counts[l] * 1.0          # assumption: the heat CSV sets the initial scores
    hot = np.zeros((L, n_exp), bool)
    for l in range(L):
        hot[l, slot[l][slot[l] >= 0]] = True

    hits = n_dec = swaps = 0
    swaps_dec = 0
    t = 0
    while t < T:
        if phase[t] == 1:
            blk = slice(t, t + 1)
        else:
            e = t
            while e < T and phase[e] != 1 and e - t < UB:
                e += 1
            blk = slice(t, e)
        Kb = K[blk]                                    # [n, L, k]
        dec = phase[t] == 1
        if dec:
            hits += int(hot[np.arange(L)[None, :, None], Kb].sum())
            n_dec += Kb.shape[0]
        # update()
        for l in range(L):
            cnt = np.bincount(Kb[:, l].ravel(), minlength=n_exp)
            score[l] = score[l] * DECAY + cnt
            dwell[l] += 1
            if not adapt:
                continue
            s_sc = np.where(slot[l] >= 0, score[l][np.maximum(slot[l], 0)], 0.0)
            si = int(np.argmin(s_sc)); smin = s_sc[si]
            kalt = np.where(hot[l], -1.0, score[l])
            ec = int(np.argmax(kalt)); sc = kalt[ec]
            if sc <= 0:
                continue
            eold = slot[l, si]
            if eold < 0 or (dwell[l, si] >= DWELL and sc > HYST * smin):
                if eold >= 0:
                    hot[l, eold] = False
                hot[l, ec] = True
                slot[l, si] = ec
                dwell[l, si] = 0
                swaps += 1
                swaps_dec += dec
        t = blk.stop
    h = hits / max(1, n_dec * L * k)
    return {"treffer": h, "tausche": swaps, "tausche_decode": swaps_dec, "decode_token": n_dec,
            "tausche_je_token": swaps_dec / max(1, n_dec)}


def tg(h, tausche_je_token):
    return 1000.0 / (T_TOKEN_MS - T_CPU_MS * h + tausche_je_token * T_SWAP_MS)


def main():
    d = sys.argv[1]
    tr = {os.path.basename(p)[:-4]: lade(p) for p in sorted(glob.glob(os.path.join(d, "*.bin")))}
    counts = {n: zaehle(t["K"]) for n, t in tr.items()}
    print(f"Swap {EXPERT_MIB:.3f} MiB = {T_SWAP_MS:.3f} ms at {H2D_GBS} GB/s, baseline {1000/T_TOKEN_MS:.1f} t/s")
    erg = {}
    for B in BUDGETS:
        S = slots_fuer(B)
        erg[B] = {"S": S}
        print(f"\nBudget {B} MiB, S = {S} per layer")
        print(f"  {'Start':6s} {'Hits':>8s} {'Swaps/Tok':>11s} {'tg':>7s}   per prompt (hits)")
        for art in ("fest", "seed", "leer"):
            res = {}
            for n, t in tr.items():
                seed = None
                if art in ("fest", "seed"):
                    seed = sum(c for m, c in counts.items() if m != n)
                res[n] = simuliere(t["K"], t["phase"], S, seed, adapt=(art != "fest"))
            nd = sum(r["decode_token"] for r in res.values())
            h = sum(r["treffer"] * r["decode_token"] for r in res.values()) / nd
            tpt = sum(r["tausche_decode"] for r in res.values()) / nd
            v = tg(h, tpt)
            erg[B][art] = {"treffer": h, "tausche_je_token": tpt, "tg": v, "je_prompt": res}
            print(f"  {art:6s} {h:8.1%} {tpt:11.2f} {v:7.1f}   " +
                  " ".join(f"{n}:{r['treffer']:.0%}" for n, r in res.items()))
    if "--json" in sys.argv:
        json.dump(erg, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
