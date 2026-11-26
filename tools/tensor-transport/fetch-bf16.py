#!/usr/bin/env python3
"""Fetch selected tensors from the BF16 GGUF on Hugging Face via HTTP range requests into a sparse file.

The output file has the full size of the shard but only the header and the chosen tensors occupy disk space.
gguf-py's GGUFReader can open it normally afterwards. Needs curl.

Input:  target path and one or more layer prefixes (e.g. blk.20); only *_exps and ffn_gate_inp tensors are fetched.
Output: sparse GGUF at the target path (re-running resumes: header is only fetched if the file is missing).
Run:    python3 -I fetch-bf16.py data/bf16/Qwen3.6-35B-A3B-BF16-00001.gguf blk.20 blk.5
"""
import os, sys, struct, subprocess

URL = ("https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/"
       "BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf")
SIZE = 49913715424
HEAD = 48 << 20  # 48 MiB is enough for metadata + tensor infos (asserted below)

def fetch(target, start, end):
    """Write bytes [start, end) via curl directly to the right offset."""
    with open(target, "r+b") as f:
        f.seek(start)
        p = subprocess.Popen(["curl", "-sfL", "--retry", "5", "-r", f"{start}-{end-1}", URL],
                             stdout=subprocess.PIPE)
        n = 0
        while True:
            b = p.stdout.read(1 << 22)
            if not b: break
            f.write(b); n += len(b)
        if p.wait() != 0 or n != end - start:
            sys.exit(f"Download {start}-{end} incomplete: {n} bytes, exit {p.returncode}")

def read_head(b):
    o = 0
    def u(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, b, o); o += struct.calcsize("<" + fmt); return v[0]
    def s():
        nonlocal o
        n = u("Q"); v = b[o:o+n].decode(); o += n; return v
    gr = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def wert(t):
        nonlocal o
        if t == 8: return s()
        if t == 9:
            et = u("I"); n = u("Q")
            if et == 8:
                for _ in range(n): s()
            else: o += gr[et] * n
            return None
        v = b[o:o+gr[t]]; o += gr[t]; return v
    assert b[:4] == b"GGUF"; o = 4
    ver = u("I"); nt = u("Q"); nkv = u("Q")
    align = 32
    for _ in range(nkv):
        k = s(); t = u("I"); v = wert(t)
        if k == "general.alignment": align = struct.unpack("<I", v)[0]
    tensors = []
    for _ in range(nt):
        name = s(); nd = u("I"); dims = [u("Q") for _ in range(nd)]; typ = u("I"); off = u("Q")
        tensors.append((name, dims, typ, off))
    data_off = (o + align - 1) // align * align
    return tensors, data_off

def main():
    target, prefixes = sys.argv[1], sys.argv[2:]
    if not os.path.exists(target):
        with open(target, "wb") as f: f.truncate(SIZE)
        fetch(target, 0, HEAD)
    with open(target, "rb") as f: head = f.read(HEAD)
    tensors, data_off = read_head(head)
    assert data_off < HEAD, "header larger than assumed"
    for name, dims, typ, off in tensors:
        if not any(name.startswith(p + ".") for p in prefixes): continue
        if "_exps" not in name and "ffn_gate_inp" not in name: continue
        n = 1
        for d in dims: n *= d
        nb = n * (2 if typ == 30 else 4)  # 30 = BF16, otherwise F32 (router)
        start = data_off + off
        print(f"{name} type {typ} {dims} {nb/2**20:.0f} MiB @ {start}", flush=True)
        fetch(target, start, start + nb)
    print("done")

if __name__ == "__main__":
    main()
