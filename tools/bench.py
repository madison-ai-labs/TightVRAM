#!/usr/bin/env python3
"""One-command benchmark: system info, RAM bandwidth, llama-bench grid, optional llama-server replay, HTML report.

Works on any llama.cpp backend (CUDA, Vulkan, ROCm, Metal, CPU). Linux gets the most detail; macOS and
Windows get what llama-bench itself reports plus basic system info. Python 3.8+, standard library only.

    python3 tools/bench.py --model ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
    python3 tools/bench.py --model M.gguf --full --replay
    python3 tools/bench.py --model M.gguf --replay --label spec-on --server-extra "-md draft.gguf --draft-max 16"

Where llama.cpp is: --llama <checkout> (uses <checkout>/build/bin, default $LLAMA_SRC or ~/src/llama.cpp),
or --bin <dir with llama-bench / llama-server>. Falls back to llama-bench on PATH.

Discrete GPU too small for the model: give --ncmoe (e.g. "40" or "32,40"), the number of layers whose
experts stay in system RAM. iGPU / UMA / CPU: leave it at 0 (default).

Output: results/<label>-<YYYYmmdd-HHMMSS>/
    system.json   hardware, OS, driver, llama.cpp build      run.json     exact commands, settings
    stream.txt    RAM bandwidth (STREAM, Linux/macOS + gcc)   bench.jsonl  llama-bench raw results
    replay.jsonl  per-request results (--replay)              server.log   llama-server log (--replay)
    report.html   self-contained report vs. the RTX 3070 reference (tools/report.py)
    summary.md    short Markdown summary to paste somewhere
Nothing is uploaded anywhere. system.json contains no host name, user name or IP; check it before sharing anyway.
"""
import argparse, json, os, platform, re, shlex, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXE = ".exe" if os.name == "nt" else ""


def sh(cmd, timeout=30):
    """Run a command, return stdout or '' (never raises): system probing is best-effort."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


# --------------------------------------------------------------------------- system info

def cpu_info():
    c = {"logical": os.cpu_count()}
    if sys.platform.startswith("linux"):
        ls = sh(["lscpu"])
        m = re.search(r"^Model name:\s*(.+)$", ls, re.M)
        c["model"] = m.group(1).strip() if m else platform.processor()
        cores = {l for l in sh(["lscpu", "-p=core,socket"]).splitlines() if l and not l.startswith("#")}
        c["physical"] = len(cores) or c["logical"]
        m = re.search(r"^CPU max MHz:\s*([\d.]+)", ls, re.M)
        if m: c["max_mhz"] = round(float(m.group(1)))
        m = re.search(r"^Flags:\s*(.+)$", ls, re.M)
        flags = set(m.group(1).split()) if m else set()
        isa = ["avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512_vnni", "avx512_bf16", "avx_vnni", "amx_tile",
               "amx_bf16", "asimd", "sve", "i8mm"]
        c["isa"] = [f for f in isa if f in flags]
        c["governor"] = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or None
        c["epp"] = read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference") or None
        c["boost"] = read("/sys/devices/system/cpu/cpufreq/boost") or None
    elif sys.platform == "darwin":
        c["model"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        c["physical"] = int(sh(["sysctl", "-n", "hw.physicalcpu"]) or 0) or c["logical"]
        c["perf_cores"] = int(sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]) or 0) or None
    else:
        c["model"] = platform.processor()
        c["physical"] = c["logical"]
    return c


def ram_info():
    r = {}
    if sys.platform.startswith("linux"):
        m = re.search(r"MemTotal:\s*(\d+)", read("/proc/meminfo"))
        if m: r["total_gib"] = round(int(m.group(1)) / 2**20, 1)
        dmi = sh(["sudo", "-n", "dmidecode", "-t", "memory"])
        if dmi:
            dims = [d for d in dmi.split("\n\n") if "Memory Device" in d and re.search(r"Size:\s*\d", d)]
            def field(d, k):
                m = re.search(rf"^\s*{k}:\s*(.+)$", d, re.M); return m.group(1).strip() if m else None
            r["modules"] = len(dims)
            if dims:
                for k, key in (("Type", "type"), ("Size", "module_size"), ("Speed", "rated_speed"),
                               ("Configured Memory Speed", "configured_speed"), ("Rank", "rank"),
                               ("Form Factor", "form_factor")):
                    vals = sorted({field(d, k) for d in dims} - {None})
                    if vals: r[key] = ", ".join(vals)
        else:
            r["dmidecode"] = "unavailable (needs passwordless sudo; run once with sudo -v before, or skip)"
    elif sys.platform == "darwin":
        r["total_gib"] = round(int(sh(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1)
    return r


def gpu_info():
    gpus = []
    if shutil.which("nvidia-smi"):
        q = "name,driver_version,memory.total,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current"
        for line in sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]).splitlines():
            v = [x.strip() for x in line.split(",")]
            if len(v) == 7:
                gpus.append({"vendor": "NVIDIA", "name": v[0], "driver": v[1], "vram_mib": int(float(v[2])),
                             "pcie": f"Gen{v[3]} x{v[5]} max, Gen{v[4]} x{v[6]} now (idle links downclock)"})
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", sh(["nvidia-smi"]))
        if cuda and gpus: gpus[0]["cuda_driver_api"] = cuda.group(1)
        nvcc = re.search(r"release ([\d.]+)", sh(["nvcc", "--version"]))
        if nvcc and gpus: gpus[0]["cuda_toolkit"] = nvcc.group(1)
    if sys.platform.startswith("linux"):
        for dev in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            if read(dev / "vendor") != "0x1002" or not re.fullmatch(r"card\d+", dev.parent.name):
                continue
            g = {"vendor": "AMD", "pci_id": read(dev / "device")}
            for k, f in (("vram_mib", "mem_info_vram_total"), ("gtt_mib", "mem_info_gtt_total")):
                v = read(dev / f)
                if v.isdigit(): g[k] = int(v) // 2**20
            g["driver_kernel"] = Path(os.path.realpath(dev / "driver")).name
            gpus.append(g)
    vk = sh(["vulkaninfo", "--summary"])
    if vk:
        devs = re.findall(r"deviceName\s*=\s*(.+)", vk)
        drv = re.findall(r"driverInfo\s*=\s*(.+)", vk)
        types = re.findall(r"deviceType\s*=\s*(.+)", vk)
        api = re.findall(r"apiVersion\s*=\s*(.+)", vk)
        vlist = [{"name": d.strip(), "type": (types[i].strip() if i < len(types) else None),
                  "driver": (drv[i].strip() if i < len(drv) else None),
                  "api": (api[i].strip() if i < len(api) else None)} for i, d in enumerate(devs)]
        vlist = [v for v in vlist if "llvmpipe" not in v["name"].lower()]
        if vlist:
            gpus.append({"vendor": "Vulkan", "devices": vlist})
    if sys.platform == "darwin":
        m = re.search(r"Chipset Model:\s*(.+)", sh(["system_profiler", "SPDisplaysDataType"], 60))
        if m: gpus.append({"vendor": "Apple", "name": m.group(1).strip(), "memory": "unified"})
    return gpus


def os_info():
    o = {"system": platform.system(), "kernel": platform.release(), "machine": platform.machine(),
         "python": platform.python_version()}
    rel = read("/etc/os-release")
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)', rel, re.M)
    if m: o["distro"] = m.group(1)
    if sys.platform == "darwin": o["distro"] = "macOS " + platform.mac_ver()[0]
    if sys.platform.startswith("linux"):
        o["thp"] = (re.search(r"\[(\w+)\]", read("/sys/kernel/mm/transparent_hugepage/enabled")) or [None, None])[1]
        o["swap_gib"] = round(int((re.search(r"SwapTotal:\s*(\d+)", read("/proc/meminfo")) or [0, "0"])[1]) / 2**20, 1)
        cmd = read("/proc/cmdline")
        o["kernel_args_gpu_mem"] = " ".join(a for a in cmd.split() if re.match(r"(amdgpu\.|ttm\.|amd_iommu|iommu)", a)) or None
    return o


def llama_info(src, bindir):
    l = {}
    if src and (Path(src) / ".git").exists():
        l["commit"] = sh(["git", "-C", str(src), "rev-parse", "HEAD"])
        l["dirty"] = bool(sh(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no"]))
        l["describe"] = sh(["git", "-C", str(src), "describe", "--tags", "--always"])
    cache = None
    for cand in (bindir.parent.parent / "CMakeCache.txt", bindir.parent / "CMakeCache.txt"):
        if cand.exists(): cache = cand; break
    if cache:
        flags = {}
        for line in cache.read_text(errors="replace").splitlines():
            m = re.match(r"^(GGML_[A-Z0-9_]+|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|AMDGPU_TARGETS|GPU_TARGETS):[A-Z]+=(.*)$", line)
            if m and m.group(2) not in ("", "OFF") and not m.group(1).endswith(("_DIR", "_EXECUTABLE", "_LIBRARY", "_INCLUDE_DIR")):
                flags[m.group(1)] = m.group(2)
        l["cmake"] = flags
    try:  # llama.cpp prints --version to stderr
        r = subprocess.run([str(bindir / ("llama-server" + EXE)), "--version"], capture_output=True, text=True, timeout=30)
        ver = " ".join((r.stdout + r.stderr).split())
        m = re.search(r"version: .*", ver)
        if m: l["version"] = m.group(0)[:200]
    except (OSError, subprocess.SubprocessError):
        pass
    return l


def model_info(path, do_hash):
    p = Path(path)
    m = {"file": p.name, "size_bytes": p.stat().st_size}
    if do_hash:
        import hashlib
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
        m["sha256"] = h.hexdigest()
    return m


# --------------------------------------------------------------------------- steps

def run_stream(out, threads_list):
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc or os.name == "nt":
        (out / "stream.txt").write_text("skipped: no gcc\n"); print("  skipped: no C compiler"); return
    bindir = REPO / ".build"; bindir.mkdir(exist_ok=True)
    exe = bindir / "stream"
    if not exe.exists():
        flags = ["-O3", "-fopenmp"] + (["-march=native"] if platform.machine() in ("x86_64", "AMD64") else ["-mcpu=native"])
        r = subprocess.run([cc, *flags, str(HERE / "stream.c"), "-o", str(exe)], capture_output=True, text=True)
        if r.returncode:
            (out / "stream.txt").write_text("skipped: stream.c did not compile (needs OpenMP)\n" + r.stderr[-2000:])
            print("  skipped: stream.c did not compile (needs gcc with OpenMP; on macOS: brew install gcc libomp)"); return
    lines = []
    for t in threads_list:
        env = dict(os.environ, OMP_NUM_THREADS=str(t), OMP_PROC_BIND="spread")
        print(f"  STREAM, {t} threads", flush=True)
        lines.append(subprocess.run([str(exe)], capture_output=True, text=True, env=env).stdout.strip())
    (out / "stream.txt").write_text("\n".join(lines) + "\n")
    print("  " + "\n  ".join(lines))


def run_bench(out, bindir, a, configs, commands):
    exe = bindir / ("llama-bench" + EXE)
    jl = open(out / "bench.jsonl", "w"); log = open(out / "bench.log", "w")
    n = len(configs)
    for i, (nc, ub, kv) in enumerate(configs, 1):
        cmd = [str(exe), "-m", a.model, "-ngl", str(a.ngl), "-b", str(ub), "-ub", str(ub), "-fa", a.fa,
               "-t", str(a.threads), "-ctk", kv, "-ctv", kv, "-p", a.prompts, "-n", str(a.gen), "-r", str(a.reps),
               "-o", "jsonl"]
        if nc: cmd[5:5] = ["-ncmoe", str(nc)]
        cmd += shlex.split(a.bench_extra)
        commands.append(cmd)
        print(f"  [{i}/{n}] n-cpu-moe={nc} ubatch={ub} kv={kv}", flush=True)
        log.write(f"== {time.strftime('%T')} {shlex.join(cmd)}\n"); log.flush()
        t0 = time.time()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
        rows = [l for l in r.stdout.splitlines() if l.startswith("{")]
        for l in rows:
            d = json.loads(l)
            test = f"pp{d['n_prompt']}" if d["n_gen"] == 0 else f"tg{d['n_gen']}"
            print(f"      {test:>8}  {d['avg_ts']:8.1f} ± {d['stddev_ts']:.1f} t/s")
            jl.write(l + "\n")
        jl.flush()
        if r.returncode:
            msg = f"FAIL rc={r.returncode} n-cpu-moe={nc} ubatch={ub} kv={kv} (OOM? see bench.log)"
            print("      " + msg); log.write(msg + "\n")
        log.write(f"== done in {time.time()-t0:.0f} s\n"); log.flush()


def wait_health(url, proc, timeout):
    t_end = time.time() + timeout
    while time.time() < t_end:
        if proc.poll() is not None: return False
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200: return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_replay(out, bindir, src, a, commands):
    exe = bindir / ("llama-server" + EXE)
    url = f"http://127.0.0.1:{a.port}"
    cmd = [str(exe), "-m", a.model, "-ngl", str(a.ngl), "-fa", "on", "-c", str(a.ctx), "-np", "1",
           "-t", str(a.threads), "-ub", str(a.server_ub), "-b", str(a.server_ub), "--host", "127.0.0.1", "--port", str(a.port)]
    nc = int(a.ncmoe.split(",")[-1])
    if nc: cmd[5:5] = ["--n-cpu-moe", str(nc)]
    cmd += shlex.split(a.server_extra)
    commands.append(cmd)
    print("  starting llama-server (loading the model can take a few minutes)", flush=True)
    with open(out / "server.log", "w") as slog:
        proc = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT)
        try:
            if not wait_health(url, proc, a.load_timeout):
                print("  llama-server did not come up, see server.log"); return
            env = dict(os.environ, LLAMA_URL=url, LLAMA_API_KEY="")
            env.pop("LLAMA_CA", None)
            if src: env["LLAMA_SRC"] = str(src)
            for sc in a.scenarios.split(","):
                if sc in ("agent", "document", "repetitive") and not (src and (Path(src) / "tools" / "server").is_dir()):
                    print(f"  replay: {sc} skipped, it uses llama.cpp source files as prompt material: pass --llama <checkout>")
                    continue
                rcmd = [sys.executable, str(HERE / "replay.py"), SCENARIO.get(sc, sc), "--out", str(out / "replay.jsonl"),
                        "--label", a.label, "--runs", str(a.replay_runs)]
                commands.append(["LLAMA_URL=" + url] + rcmd[1:])
                print(f"  replay: {sc}", flush=True)
                subprocess.run(rcmd, env=env)
        finally:
            proc.terminate()
            try: proc.wait(30)
            except subprocess.TimeoutExpired: proc.kill()


# English names for replay.py's (German) scenario names
SCENARIO = {"interactive": "interaktiv", "document": "dokument", "repetitive": "repetitiv", "novel": "neuartig"}


def find_bins(a):
    src = Path(os.path.expanduser(a.llama)) if a.llama else Path(os.environ.get("LLAMA_SRC", "~/src/llama.cpp")).expanduser()
    for cand in ([Path(os.path.expanduser(a.bin))] if a.bin else []) + [src / "build" / "bin", src / "build" / "bin" / "Release"]:
        if (cand / ("llama-bench" + EXE)).exists():
            return (src if src.exists() else None), cand
    w = shutil.which("llama-bench")
    if w: return (src if src.exists() else None), Path(w).parent
    sys.exit(f"llama-bench not found in {src}/build/bin or on PATH; pass --llama <checkout> or --bin <dir>")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with the child processes' output
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Defaults are the --quick grid. Every value ends up in run.json and the report.")
    ap.add_argument("--model", required=True, help="GGUF file")
    ap.add_argument("--label", default=None, help="name for this run (default: model + backend)")
    ap.add_argument("--llama", help="llama.cpp checkout (default $LLAMA_SRC or ~/src/llama.cpp)")
    ap.add_argument("--bin", help="directory with llama-bench / llama-server, if not <checkout>/build/bin")
    ap.add_argument("--out", default=str(REPO / "results"), help="results root (default ./results)")
    g = ap.add_argument_group("llama-bench grid")
    g.add_argument("--full", action="store_true", help="bigger grid: ubatch 512..4096, KV f16+q8_0, pp up to 16k, 5 reps")
    g.add_argument("--ncmoe", default="0", help="comma list of --n-cpu-moe values (0 = all on the GPU / UMA)")
    g.add_argument("--ubatch", default=None, help="comma list (quick: 512,2048; full: 512,1024,2048,4096)")
    g.add_argument("--kv", default=None, help="comma list of KV cache types (quick: f16; full: f16,q8_0)")
    g.add_argument("--prompts", default=None, help="llama-bench -p (quick: 512,4096; full: 512,4096,16384)")
    g.add_argument("--gen", type=int, default=128, help="llama-bench -n")
    g.add_argument("--reps", type=int, default=None, help="llama-bench -r (quick 3, full 5)")
    g.add_argument("--threads", type=int, default=None, help="-t (default: physical cores)")
    g.add_argument("--ngl", default="99")
    g.add_argument("--fa", default="on", help="flash attention for llama-bench (on/off; old builds want 1/0)")
    g.add_argument("--bench-extra", default="", help="extra llama-bench arguments, quoted")
    r = ap.add_argument_group("llama-server replay")
    r.add_argument("--replay", action="store_true", help="also start llama-server and replay real request shapes")
    r.add_argument("--scenarios", default="interactive,agent",
                   help="comma list: interactive, agent (15k prefix + tool turns), document (32k/64k, slow), repetitive, novel")
    r.add_argument("--replay-runs", type=int, default=3)
    r.add_argument("--ctx", type=int, default=32768, help="llama-server -c (document needs 72k)")
    r.add_argument("--server-ub", type=int, default=2048, help="llama-server -ub/-b")
    r.add_argument("--server-extra", default="", help="extra llama-server arguments, e.g. a draft model")
    r.add_argument("--port", type=int, default=8099)
    r.add_argument("--load-timeout", type=int, default=900)
    s = ap.add_argument_group("other")
    s.add_argument("--no-stream", action="store_true", help="skip the RAM bandwidth test")
    s.add_argument("--no-bench", action="store_true", help="skip llama-bench (e.g. replay only)")
    s.add_argument("--hash", action="store_true", help="SHA256 the model file (a minute for 20 GB)")
    a = ap.parse_args()

    a.model = str(Path(os.path.expanduser(a.model)).resolve())
    if not Path(a.model).exists(): sys.exit(f"model not found: {a.model}")
    a.ubatch = a.ubatch or ("512,1024,2048,4096" if a.full else "512,2048")
    a.kv = a.kv or ("f16,q8_0" if a.full else "f16")
    a.prompts = a.prompts or ("512,4096,16384" if a.full else "512,4096")
    a.reps = a.reps or (5 if a.full else 3)
    cpu = cpu_info()
    a.threads = a.threads or cpu.get("physical") or os.cpu_count()
    if "document" in a.scenarios and a.ctx < 72000: a.ctx = 73728
    src, bindir = find_bins(a)
    a.label = a.label or re.sub(r"[^A-Za-z0-9._-]+", "-", Path(a.model).stem)[:40]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) / f"{a.label}-{stamp}"; out.mkdir(parents=True)
    print(f"# results → {out}")

    print("# system info")
    sysinfo = {"cpu": cpu, "ram": ram_info(), "gpu": gpu_info(), "os": os_info(), "llama_cpp": llama_info(src, bindir),
               "model": model_info(a.model, a.hash)}
    (out / "system.json").write_text(json.dumps(sysinfo, indent=1))
    print(f"  {cpu.get('model')}, {cpu.get('physical')}c/{cpu.get('logical')}t, "
          f"{sysinfo['ram'].get('total_gib')} GiB RAM, GPUs: {[g.get('name') or g.get('vendor') for g in sysinfo['gpu']]}")

    commands = []
    run = {"label": a.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "args": {k: v for k, v in vars(a).items()
           if k not in ("model", "out", "llama", "bin")}, "model_file": Path(a.model).name, "commands": commands}
    def save_run():
        run["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        # paths in commands: keep only file names, so no home directory ends up in a shared report
        run["commands_shown"] = [" ".join(shlex.quote(Path(x).name if os.sep in x and Path(x).exists() else x) for x in c)
                                 for c in commands]
        (out / "run.json").write_text(json.dumps({k: v for k, v in run.items() if k != "commands"}, indent=1))

    try:
        if not a.no_stream:
            print("# RAM bandwidth (STREAM)")
            tl = sorted({1, cpu.get("physical") or 1, cpu.get("logical") or 1})
            run_stream(out, tl)
        if not a.no_bench:
            configs = [(int(nc), int(ub), kv) for nc in a.ncmoe.split(",") for ub in a.ubatch.split(",") for kv in a.kv.split(",")]
            print(f"# llama-bench: {len(configs)} configurations, -p {a.prompts} -n {a.gen} -r {a.reps}, {a.threads} threads")
            run_bench(out, bindir, a, configs, commands)
        if a.replay:
            print("# replay against llama-server")
            run_replay(out, bindir, src, a, commands)
    except KeyboardInterrupt:
        print("\n# interrupted, writing the report from what is there")
    save_run()

    print("# report")
    subprocess.run([sys.executable, str(HERE / "report.py"), str(out), "-o", str(out / "report.html"),
                    "--summary", str(out / "summary.md")])
    print(f"# done: open {out / 'report.html'}")


if __name__ == "__main__":
    main()
