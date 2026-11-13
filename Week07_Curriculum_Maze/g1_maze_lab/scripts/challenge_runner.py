"""Instructor: run the uploaded teams one after another and keep the leaderboard page up to date.

    uv run python scripts/challenge_runner.py --seed 13                 # new uploads + teams not run on seed 13
    uv run python scripts/challenge_runner.py --seed 13 --all           # every team, again
    uv run python scripts/challenge_runner.py --seed 13 --teams tiger panda
    uv run python scripts/challenge_runner.py --seed 13 --no-replay       # faster, no replay videos
    uv run python scripts/challenge_runner.py --seed 1 --watch --no-replay  # practice: run every upload as it comes in
    uv run python scripts/challenge_runner.py --level 9 --seed 13       # level 7 (default, or CHALLENGE_LEVEL) / 9 / final

Each team runs scripts/challenge.py (same rules, one robot). submissions/status.json shows waiting / running /
done / error on the upload page; a team whose run crashes gets "error" with the last error line.
Teams run in upload order (first come, first run). Each run is recorded to submissions/_replays/<team>_L<level>_seed<seed>.mp4
(the Newton viewer window opens for it), so the whole class can watch every run afterwards on /replay.
A run that takes longer than --timeout seconds of wall clock (default 6 x the level's time limit + 300 s: a 600 s run
takes ~40 min with driver 535) or whose simulation stops moving for CHALLENGE_STALL seconds (default 300: a pilot that
hangs) is stopped: "error", with the replay of what it did until then.
--watch keeps going: every few seconds it runs whatever is waiting (Ctrl+C to stop), for a practice round.
--follow-round (with --watch): level and seed come from submissions/round.json (set on the server's /admin page)
and are read again before every run, so changing the round on /admin re-runs every team on the new maze.
--parallel N (or CHALLENGE_PARALLEL): N teams at the same time, each its own simulator process. One G1 run leaves the
GPU mostly idle (the CPU side of launching the kernels is the limit), so 3 at once take about as long as one.
Live view: each running team writes submissions/_live/<team>/ (the server's /live page).

CHALLENGE_SANDBOX=docker runs every team in a throw-away container (scripts/sandbox/Dockerfile) instead of on this
machine: no network, read-only system, only the team's own files (a copy) writable, the GPU passed through. A pilot
cannot read the other teams' files or anything else on the server. Settings: CHALLENGE_SANDBOX_IMAGE (default
maze-sandbox:1), CHALLENGE_WARP_CACHE (a compiled-kernel cache copied into every run; fill it with --warm-cache),
CHALLENGE_ASSET_MIRROR (a folder with the robot assets Isaac Lab downloaded on the host: <folder>/https/<host>/...).
"""

import argparse
import contextlib
import copy
import csv
import fcntl
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from challenge_levels import DEFAULT_LEVEL, LEVELS, level_info  # noqa: E402

SUBMISSIONS = os.environ.get("CHALLENGE_DIR", os.path.join(HERE, "..", "submissions"))
SANDBOX = os.environ.get("CHALLENGE_SANDBOX", "")
PROJECT = os.path.abspath(os.path.join(HERE, ".."))
LB_FIELDS = ["team", "seed", "result", "time_s", "cells_to_goal", "falls", "finished_at", "best_left_m", "route_m", "level"]
# log lines that look like errors but are handled (Isaac Lab falls back): never the reason a run failed
HARMLESS = ("Conditional graph nodes require", "sensor CUDA graph capture failed", "Could not register GL buffer")
STALL = float(os.environ.get("CHALLENGE_STALL", 300))  # [s] wall clock without the simulation moving: the pilot hangs
STARTUP = 600  # [s] wall clock allowed before the first simulated step (Isaac Lab + the robot + the maze load)


def default_timeout(limit):
    """Wall-clock limit of one run: the simulation runs at ~0.27 x real time here (driver 535, replay recorded), a bit
    slower with several runs at once, so a full 600 s run takes ~40 min; 6 x + 300 s leaves room for slow pilots."""
    return 6 * float(limit) + 300


@contextlib.contextmanager
def status_lock():
    """Same lock as challenge_server.py: an upload and a state change never overwrite each other."""
    with open(os.path.join(SUBMISSIONS, ".status.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load():
    try:
        with open(os.path.join(SUBMISSIONS, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save(st):
    tmp = os.path.join(SUBMISSIONS, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUBMISSIONS, "status.json"))


def set_state(team, **kw):
    with status_lock():
        st = load()
        st.setdefault(team, {}).update(kw)
        save(st)


def follow_round(args):
    """--follow-round: level / seed of submissions/round.json (the server's /admin page). False = no seed set."""
    try:
        with open(os.path.join(SUBMISSIONS, "round.json")) as f:
            r = json.load(f)
    except (OSError, ValueError):
        r = {}
    level = str(r.get("level", args.level))
    args.level = level if level in LEVELS else args.level
    seed = str(r.get("seed", "" if args.seed is None else args.seed))
    if not seed.isdigit():
        return False
    args.seed = int(seed)
    args.timeout = args.timeout_arg or default_timeout(os.environ.get("CHALLENGE_TIME", level_info(args.level)["time"]))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int)
    p.add_argument("--level", default=os.environ.get("CHALLENGE_LEVEL", DEFAULT_LEVEL))
    p.add_argument("--teams", nargs="*")
    p.add_argument("--all", action="store_true", help="run every team, also those already done")
    p.add_argument("--viz", default="", help="default: newton when recording replays")
    p.add_argument("--no-replay", action="store_true", help="do not record replay videos")
    p.add_argument("--timeout", type=float, default=0,
                   help="wall-clock limit per run [s] (default: 6 x the level's time limit + 300)")
    p.add_argument("--watch", action="store_true", help="keep running new uploads until Ctrl+C (practice round)")
    p.add_argument("--follow-round", action="store_true", help="level / seed from submissions/round.json (/admin)")
    p.add_argument("--parallel", type=int, default=int(os.environ.get("CHALLENGE_PARALLEL", 1)),
                   help="teams run at the same time, each in its own simulator (default 1, or CHALLENGE_PARALLEL)")
    p.add_argument("--warm-cache", action="store_true",
                   help="sandbox: one run of the pretrained policy + default pilot that fills CHALLENGE_WARP_CACHE")
    args = p.parse_args()
    if args.seed is None and not args.follow_round:
        p.error("--seed is required (or --follow-round)")
    args.parallel = max(1, args.parallel)
    args.timeout_arg = args.timeout
    limit = float(os.environ.get("CHALLENGE_TIME", level_info(args.level)["time"]))
    if args.timeout <= 0:
        args.timeout = default_timeout(limit)
    os.makedirs(SUBMISSIONS, exist_ok=True)
    if args.warm_cache:
        return warm_cache(args)
    stop_leftover_containers()
    with status_lock():  # a runner stopped with Ctrl+C mid-run left a team "running": queue it again
        st = load()
        for s in st.values():
            if s.get("state") == "running":
                s.update(state="waiting", msg="")
        save(st)
    workers = {}
    try:
        if not args.watch:
            if args.follow_round and not follow_round(args):
                p.error("round.json has no seed: set one on /admin or pass --seed")
            teams = pending(args)
            print(f"[RUNNER] level {args.level} seed {args.seed}: {', '.join(teams) or 'nothing to run'}", flush=True)
            while teams or workers:
                reap(workers)
                while teams and len(workers) < args.parallel:
                    start(workers, teams.pop(0), args)
                time.sleep(1)
            return
        print(f"[RUNNER] watching {os.path.abspath(SUBMISSIONS)} (level {args.level}, seed {args.seed}"
              f"{', following round.json' if args.follow_round else ''}, {args.parallel} at a time"
              f"{', sandbox ' + SANDBOX if SANDBOX else ''}), Ctrl+C to stop", flush=True)
        while True:
            reap(workers)
            if len(workers) < args.parallel and (not args.follow_round or follow_round(args)):
                for team in pending(args, busy=workers)[: args.parallel - len(workers)]:
                    start(workers, team, args)  # first come, first run; a team that re-uploads goes to the back
            time.sleep(5)
    except KeyboardInterrupt:
        stop_all(workers)
        print("[RUNNER] stopped, the teams that were running are back in the queue", flush=True)


def pending(args, busy=()):
    """Teams to run, in upload order: new uploads + teams not run on this round yet (or --teams / --all)."""
    st = load()
    teams = args.teams or sorted((d for d in os.listdir(SUBMISSIONS)
                                  if os.path.isdir(os.path.join(SUBMISSIONS, d)) and d[0] != "_"),
                                 key=lambda t: (st.get(t, {}).get("uploaded_at", 0), t))  # first come, first run
    if not args.teams and not args.all:
        teams = [t for t in teams if st.get(t, {}).get("state", "waiting") == "waiting"
                 or st.get(t, {}).get("seed") != args.seed or str(st.get(t, {}).get("level", "7")) != args.level]
    return [t for t in teams if t not in busy]


# ----------------------------------------------------------------------------- workers
STOPPING = threading.Event()
PROCS = {}  # team -> (Popen, container name or None) of the runs in progress, for stop_all()
PROCS_LOCK = threading.Lock()


def start(workers, team, args):
    """Run one team in a worker thread (with a copy of args: --follow-round may change the round meanwhile)."""
    set_state(team, state="running", msg="", started=time.strftime("%H:%M:%S"))
    th = threading.Thread(target=run_team, args=(team, copy.copy(args)), name=f"run-{team}", daemon=True)
    workers[team] = th
    th.start()


def reap(workers):
    for team in [t for t, th in workers.items() if not th.is_alive()]:
        workers.pop(team)


def kill(proc, container):
    if container:  # killing the docker client does not stop the container
        subprocess.run(["docker", "kill", container], capture_output=True)
    with contextlib.suppress(ProcessLookupError):  # the whole group (the simulator has helper processes)
        os.killpg(proc.pid, 9)


def stop_all(workers):
    """Ctrl+C / pm2 stop: stop every run in progress and put those teams back in the queue."""
    STOPPING.set()
    with PROCS_LOCK:
        running = list(PROCS.values())
    for proc, container in running:
        kill(proc, container)
    for th in workers.values():
        th.join(timeout=15)
    for team in workers:
        set_state(team, state="waiting", msg="")


def stop_leftover_containers():
    """A runner killed hard (pm2 kill, a reboot) can leave its containers running: stop them before starting.
    Only this data folder's (label maze.dir): another runner on the machine (a test) keeps its runs."""
    if SANDBOX != "docker":
        return
    ids = subprocess.run(["docker", "ps", "-q", "--filter", f"label=maze.dir={os.path.abspath(SUBMISSIONS)}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "kill", *ids], capture_output=True)
        print(f"[RUNNER] stopped {len(ids)} container(s) left by an earlier runner", flush=True)


def run_team(team, args):
    """One run of one team (in a worker thread): challenge.py on this machine or in the sandbox, then the results."""
    log_dir = os.path.join(SUBMISSIONS, "_logs")
    replay_dir = os.path.join(SUBMISSIONS, "_replays")
    live_dir = os.path.join(SUBMISSIONS, "_live", team)  # the server's /live page, one folder per running team
    for d in (log_dir, replay_dir, live_dir):
        os.makedirs(d, exist_ok=True)
    script_args = ["--team", team, "--seed", str(args.seed), "--level", args.level]
    viz = args.viz or ("" if args.no_replay else "newton")
    if viz:
        script_args += ["--viz", viz]
    name = f"{team}_L{args.level}_seed{args.seed}"
    video = os.path.join(replay_dir, name + ".mp4")
    for old in (name + ".mp4", name + ".json", name + ".trace.json"):  # the replay always shows the latest run
        if os.path.exists(os.path.join(replay_dir, old)):
            os.remove(os.path.join(replay_dir, old))
    log = os.path.join(log_dir, name + ".log")
    print(f"[RUNNER] {team}: start (level {args.level}, seed {args.seed})", flush=True)
    t0 = time.time()
    work = None
    try:
        if SANDBOX == "docker":
            work = new_workdir(name)
            shutil.copytree(os.path.join(SUBMISSIONS, team), os.path.join(work, "sub", team),
                            ignore=shutil.ignore_patterns("exported", "__pycache__"))
            part = os.path.join(work, "out", "replay.part.mp4")
            trace = os.path.join(work, "out", "trace.json")
            env = {"CHALLENGE_DIR": "/work/sub", "MAZE_LIVE_DIR": "/live", "MAZE_TRACE": "/work/out/trace.json"}
            if not args.no_replay:
                env["MAZE_REC_MP4"] = "/work/out/replay.part.mp4"
            cname = container_name(name)
            code, stopped = run_logged(sandbox_cmd(cname, work, env, script_args, live=live_dir), log, args.timeout,
                                       team, container=cname, live=live_dir)
            if not STOPPING.is_set():
                merge_leaderboard(os.path.join(work, "sub", "leaderboard.csv"), team, args)
        else:
            part = video + ".part.mp4"
            trace = os.path.join(replay_dir, name + ".trace.part.json")
            env = dict(os.environ, CHALLENGE_DIR=SUBMISSIONS, MAZE_LIVE_DIR=live_dir, MAZE_TRACE=trace)
            if not args.no_replay:  # challenge.py writes the video while it runs
                env["MAZE_REC_MP4"] = part
            cmd = [sys.executable, os.path.join(HERE, "challenge.py")] + script_args
            code, stopped = run_logged(cmd, log, args.timeout, team, env=env, live=live_dir)
        if STOPPING.is_set():  # the runner is stopping: stop_all() puts the team back in the queue
            return
        finish(team, args, log, code, stopped, part, trace, video, name, t0, live_dir)
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)


def finish(team, args, log, code, stopped, part, trace, video, name, t0, live_dir):
    """The run is over (``stopped``: None, "timeout" or "stall"): result, replay (also of a run stopped half way)."""
    text = open(log, errors="replace").read()
    res = re.findall(r"\[CHALLENGE\] RESULT team=\S+ seed=\d+: (.*?)\s+\(wall clock", text)
    if res:
        save_replay(team, args, part, trace, video, name, res[-1])
        set_state(team, state="done", msg=res[-1], seed=args.seed, level=args.level)
        print(f"[RUNNER] {team}: {res[-1]}  ({time.time() - t0:.0f} s)", flush=True)
        return
    errs = [ln.strip() for ln in text.splitlines() if re.search(r"(Error|Exception)\b", ln)
            and not any(h in ln for h in HARMLESS)]
    why = errs[-1] if errs else f"exit code {code}"
    if code in (137, -9) and not stopped:  # SIGKILL: docker kill / out of memory, not the team's code
        why = "this run was stopped by the system, not by the team's code: the instructor can re-run it"
    sim_t = _read_json_file(os.path.join(live_dir, "state.json")).get("t")
    at = f" at t={sim_t:.0f} s" if isinstance(sim_t, (int, float)) else ""
    if stopped == "timeout":
        why = f"stopped{at}: over the {args.timeout / 60:.0f} min wall-clock limit (is the pilot very slow?)"
    elif stopped == "stall":
        why = f"stopped{at}: the simulation did not move for {STALL:.0f} s (is the pilot stuck?)"
    save_replay(team, args, part, trace, video, name, f"STOPPED{at}: " + why.split(": ", 1)[-1] if stopped else None)
    set_state(team, state="error", msg=why[:200], seed=args.seed, level=args.level)
    print(f"[RUNNER] {team}: ERROR {why[:160]}", flush=True)


def save_replay(team, args, part, trace, video, name, result):
    """Video (remuxed to a normal MP4: the browser needs its length up front) + 2D trace + the caption for /replay.
    ``result`` None: no replay (the run crashed), the partial files are removed."""
    replay_dir = os.path.dirname(video)
    has_video = result is not None and os.path.exists(part) and os.path.getsize(part) > 0 and remux(part, video)
    has_trace = result is not None and os.path.exists(trace)
    if result is not None and not has_video and not args.no_replay:
        print(f"[RUNNER] {team}: no replay video (see the log)", flush=True)
    if has_trace:  # 2D replay (the maze + the pose at 10 Hz), played on /replay when there is no video
        shutil.move(trace, os.path.join(replay_dir, name + ".trace.json"))
    for leftover in (part, trace):
        with contextlib.suppress(OSError):
            os.remove(leftover)
    if has_video or has_trace:
        with open(os.path.join(replay_dir, name + ".json"), "w") as f:  # caption for /replay
            json.dump({"team": team, "seed": args.seed, "level": args.level, "result": result,
                       "video": name + ".mp4" if has_video else None,
                       "trace": name + ".trace.json" if has_trace else None,
                       "at": time.strftime("%H:%M:%S")}, f, ensure_ascii=False)


def remux(src, dst):
    """challenge.py writes a fragmented MP4 (playable even when the run is killed); copy it into a normal MP4 with the
    index up front (no re-encoding, a few seconds). Falls back to the fragmented file if ffmpeg fails."""
    try:
        import imageio_ffmpeg

        tmp = dst + ".tmp.mp4"
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-c", "copy",
                            "-movflags", "+faststart", tmp], capture_output=True, timeout=300)
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, dst)
            return True
        print(f"[RUNNER] remux failed: {r.stderr.decode(errors='replace')[-300:]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[RUNNER] remux failed: {e}", flush=True)
    with contextlib.suppress(OSError):
        os.remove(dst + ".tmp.mp4")
    try:
        os.replace(src, dst)
        return True
    except OSError:
        return False


def _read_json_file(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_logged(cmd, log, timeout, team, container=None, env=None, live=None):
    """Run ``cmd`` with its output in ``log``. Returns (exit code, why it was stopped: None / "timeout" / "stall").
    Stopped after ``timeout`` s, or when ``live``/state.json (written ~10 x per simulated second) has not changed for
    STALL s (STARTUP s before the first simulated step)."""
    t0 = time.time()
    state = os.path.join(live, "state.json") if live else None
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        with PROCS_LOCK:
            PROCS[team] = (proc, container)
        try:
            while True:
                try:
                    return proc.wait(timeout=5), None
                except subprocess.TimeoutExpired:
                    pass
                now = time.time()
                why = "timeout" if now - t0 > timeout else None
                if state and not why:
                    try:
                        moved = os.path.getmtime(state)
                    except OSError:
                        moved = 0
                    if moved < t0:  # not this run's yet: still loading
                        why = "stall" if now - t0 > STARTUP else None
                    elif now - moved > STALL:
                        why = "stall"
                if why:
                    kill(proc, container)
                    return proc.wait(), why
        finally:
            with PROCS_LOCK:
                PROCS.pop(team, None)


# ----------------------------------------------------------------------------- sandbox (CHALLENGE_SANDBOX=docker)
def new_workdir(name):
    """A fresh folder for one run: sub/<team>/ (a copy of the upload), out/ (replay), home/, warp/ (kernel cache)."""
    work = os.path.join(SUBMISSIONS, "_runs", name)
    shutil.rmtree(work, ignore_errors=True)
    for d in ("sub", "out", "home", "cache", "tmp"):
        os.makedirs(os.path.join(work, d))
    mirror = os.environ.get("CHALLENGE_ASSET_MIRROR", "")
    if mirror:  # Isaac Lab looks for remote assets under $TMPDIR/https/<host>/... (and locks them: a writable copy)
        shutil.copytree(os.path.join(mirror, "https"), os.path.join(work, "tmp", "https"))
    warm = os.environ.get("CHALLENGE_WARP_CACHE", "")
    if warm and os.path.isdir(warm):
        shutil.copytree(warm, os.path.join(work, "warp"), symlinks=True)
    else:
        os.makedirs(os.path.join(work, "warp"))
    return work


def container_name(name):
    return "maze-run-" + re.sub(r"[^A-Za-z0-9_.-]", "-", name)[:60] + f"-{os.getpid()}"


def sandbox_cmd(cname, work, env, script_args, warp=None, live=None):
    """docker run ... for one run: no network, read-only system, only /work (this run) and /live writable."""
    venv_py = os.path.join(PROJECT, ".venv", "bin", "python")
    py_home = os.path.dirname(os.path.dirname(os.path.realpath(venv_py)))  # the uv-managed interpreter
    # the venv names it through uv's version alias (cpython-3.12-... -> cpython-3.12.14-...): mount it there too
    py_alias = os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(venv_py), os.readlink(venv_py))))
    isaaclab = os.path.realpath(os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab")))
    cmd = ["docker", "run", "--rm", "--name", cname, "--label", f"maze.dir={os.path.abspath(SUBMISSIONS)}",
           "--network", "none", "--read-only",
           "--tmpfs", "/tmp:rw,exec,size=8g", "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--shm-size", "4g", "--pids-limit", "8192",
           "--memory", os.environ.get("CHALLENGE_SANDBOX_MEMORY", "32g"),
           "--cpus", os.environ.get("CHALLENGE_SANDBOX_CPUS", "12")]
    for dev in sorted(glob.glob("/dev/nvidia*")):
        if not os.path.isdir(dev):
            cmd += ["--device", dev]
    # headless OpenGL for the replay viewer: Mesa on a non-NVIDIA render node (e.g. the CPU's integrated GPU);
    # Mesa has no driver for the NVIDIA node and would pick it first
    def vendor(dev):
        try:
            with open(f"/sys/class/drm/{os.path.basename(dev)}/device/vendor") as f:
                return f.read().strip()
        except OSError:
            return ""

    mesa = [d for d in sorted(glob.glob("/dev/dri/renderD*")) if vendor(d) != "0x10de"]  # 0x10de = NVIDIA
    for dev in mesa:
        cmd += ["--device", dev]
    if mesa:
        cmd += ["--group-add", str(os.stat(mesa[0]).st_gid)]
    libdir = "/usr/lib/x86_64-linux-gnu"
    for lib in ("libcuda.so", "libcuda.so.1", "libnvidia-ptxjitcompiler.so.1", "libnvidia-nvvm.so.4", "libnvidia-ml.so.1"):
        path = os.path.join(libdir, lib)
        if os.path.exists(path):  # the host's driver libraries (they must match the kernel module)
            cmd += ["-v", f"{os.path.realpath(path)}:{path}:ro"]
    for ro in (PROJECT, isaaclab, py_home):
        cmd += ["-v", f"{ro}:{ro}:ro"]
    if os.path.normpath(py_alias) != py_home:
        cmd += ["-v", f"{py_home}:{os.path.normpath(py_alias)}:ro"]
    cmd += ["-v", f"{os.path.abspath(work)}:/work"]
    if live:  # this team's folder for the /live page
        cmd += ["-v", f"{os.path.abspath(live)}:/live"]
    if warp:
        cmd += ["-v", f"{os.path.abspath(warp)}:/work/warp"]
    full_env = {"HOME": "/work/home", "XDG_CACHE_HOME": "/work/cache", "WARP_CACHE_PATH": "/work/warp", "TMPDIR": "/work/tmp",
                "PYTHONDONTWRITEBYTECODE": "1", "LD_LIBRARY_PATH": libdir, "MPLCONFIGDIR": "/work/cache/mpl",
                "GIT_PYTHON_REFRESH": "quiet"}  # GitPython is imported but no git is needed (or installed)
    for k in ("CHALLENGE_TIME", "CHALLENGE_CUDA_GRAPH", "MAZE_CAM", "MAZE_REC_EVERY", "ISAACSIM_ASSET_ROOT"):
        if k in os.environ:
            full_env[k] = os.environ[k]
    full_env.update(env)
    for k, v in full_env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["-w", "/work", os.environ.get("CHALLENGE_SANDBOX_IMAGE", "maze-sandbox:1"),
            os.path.join(PROJECT, ".venv", "bin", "python"), os.path.join(PROJECT, "scripts", "challenge.py")]
    return cmd + script_args


def merge_leaderboard(path, team, args):
    """Copy the run's result row (written inside the sandbox) to the real leaderboard, after checking it."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    good = []
    for r in rows:
        try:
            ok = (r["team"] == team and r["seed"] == str(args.seed) and r["level"] == args.level
                  and r["result"] in ("GOAL", "DNF") and int(r["falls"]) >= 0 and float(r["best_left_m"]) >= 0
                  and (r["result"] == "DNF" or float(r["time_s"]) > 0))
        except (KeyError, TypeError, ValueError):
            ok = False
        if ok:
            good.append([r[k] for k in LB_FIELDS])
    if not good:
        return
    real = os.path.join(SUBMISSIONS, "leaderboard.csv")
    with status_lock():
        new = not os.path.exists(real)
        with open(real, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LB_FIELDS)
            w.writerows(good[-1:])  # one run = one row


def warm_cache(args):
    """Fill CHALLENGE_WARP_CACHE with one trusted run (pretrained policy, default pilot): every later run starts
    from a copy of it instead of compiling the simulator's GPU kernels again."""
    warm = os.environ.get("CHALLENGE_WARP_CACHE", "")
    if SANDBOX != "docker" or not warm:
        raise SystemExit("--warm-cache needs CHALLENGE_SANDBOX=docker and CHALLENGE_WARP_CACHE=<folder>")
    os.makedirs(warm, exist_ok=True)
    os.makedirs(os.path.join(SUBMISSIONS, "_live"), exist_ok=True)
    work = new_workdir("_warmup")
    shutil.rmtree(os.path.join(work, "warp"))
    os.makedirs(os.path.join(work, "warp"))
    seed = args.seed if args.seed is not None else 1
    # the pretrained policy, copied: playing a checkpoint exports it next to itself (the project is read-only)
    os.makedirs(os.path.join(work, "sub", "_warmup"))
    shutil.copy(os.path.join(PROJECT, "pretrained", "g1_rough", "model_pretrained.pt"),
                os.path.join(work, "sub", "_warmup", "policy.pt"))
    script_args = ["--seed", str(seed), "--level", args.level, "--policy", "/work/sub/_warmup/policy.pt"]
    script_args += ["--viz", args.viz] if args.viz else []
    log = os.path.join(SUBMISSIONS, "_logs", "_warmup.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    env = {"CHALLENGE_TIME": os.environ.get("CHALLENGE_TIME", "20")}
    print(f"[RUNNER] warming the kernel cache {warm} (log {log})", flush=True)
    t0 = time.time()
    code, _ = run_logged(sandbox_cmd(container_name("_warmup"), work, env, script_args, warp=warm), log, 3600,
                         "_warmup", container=container_name("_warmup"))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[RUNNER] warm-up exit code {code} after {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
