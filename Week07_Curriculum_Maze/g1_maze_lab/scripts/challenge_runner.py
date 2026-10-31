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
A run that takes longer than --timeout seconds of wall clock (a pilot that hangs or is very slow) is stopped: "error".
--watch keeps going: every few seconds it runs whatever is waiting (Ctrl+C to stop), for a practice round.
"""

import argparse
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from challenge_levels import DEFAULT_LEVEL, level_info  # noqa: E402

SUBMISSIONS = os.environ.get("CHALLENGE_DIR", os.path.join(HERE, "..", "submissions"))


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--level", default=os.environ.get("CHALLENGE_LEVEL", DEFAULT_LEVEL))
    p.add_argument("--teams", nargs="*")
    p.add_argument("--all", action="store_true", help="run every team, also those already done")
    p.add_argument("--viz", default="", help="default: newton when recording replays")
    p.add_argument("--no-replay", action="store_true", help="do not record replay videos")
    p.add_argument("--timeout", type=float, default=0,
                   help="wall-clock limit per run [s] (default: 3 x the level's time limit + 180)")
    p.add_argument("--watch", action="store_true", help="keep running new uploads until Ctrl+C (practice round)")
    args = p.parse_args()
    limit = float(os.environ.get("CHALLENGE_TIME", level_info(args.level)["time"]))
    if args.timeout <= 0:
        args.timeout = 3 * limit + 180
    os.makedirs(SUBMISSIONS, exist_ok=True)
    with status_lock():  # a runner stopped with Ctrl+C mid-run left a team "running": queue it again
        st = load()
        for s in st.values():
            if s.get("state") == "running":
                s.update(state="waiting", msg="")
        save(st)
    if not args.watch:
        return run_queue(args, quiet=False)
    print(f"[RUNNER] watching {os.path.abspath(SUBMISSIONS)} (level {args.level}, seed {args.seed}), Ctrl+C to stop",
          flush=True)
    try:
        while True:
            run_queue(args, quiet=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("[RUNNER] stopped", flush=True)


def run_queue(args, quiet):
    st = load()
    teams = args.teams or sorted((d for d in os.listdir(SUBMISSIONS)
                                  if os.path.isdir(os.path.join(SUBMISSIONS, d)) and d[0] != "_"),
                                 key=lambda t: (st.get(t, {}).get("uploaded_at", 0), t))  # first come, first run
    if not args.teams and not args.all:
        # waiting = new upload, or not run on this seed yet
        teams = [t for t in teams if st.get(t, {}).get("state", "waiting") == "waiting"
                 or st.get(t, {}).get("seed") != args.seed or str(st.get(t, {}).get("level", "7")) != args.level]
    if teams or not quiet:
        print(f"[RUNNER] level {args.level} seed {args.seed}: {', '.join(teams) or 'nothing to run'}", flush=True)
    if args.watch:  # one team per pass: a team that re-uploads meanwhile goes to the back of the queue again
        teams = teams[:1]
    log_dir = os.path.join(SUBMISSIONS, "_logs")
    replay_dir = os.path.join(SUBMISSIONS, "_replays")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(replay_dir, exist_ok=True)
    for team in teams:
        set_state(team, state="running", msg="", started=time.strftime("%H:%M:%S"))
        cmd = [sys.executable, os.path.join(HERE, "challenge.py"), "--team", team, "--seed", str(args.seed),
               "--level", args.level]
        env = dict(os.environ, CHALLENGE_DIR=SUBMISSIONS)
        viz = args.viz or ("" if args.no_replay else "newton")
        if viz:
            cmd += ["--viz", viz]
        name = f"{team}_L{args.level}_seed{args.seed}"
        video = os.path.join(replay_dir, name + ".mp4")
        for old in (name + ".mp4", name + ".json"):  # the replay always shows the latest run
            if os.path.exists(os.path.join(replay_dir, old)):
                os.remove(os.path.join(replay_dir, old))
        if not args.no_replay:  # challenge.py writes the video while it runs
            env["MAZE_REC_MP4"] = video + ".part.mp4"
        log = os.path.join(log_dir, name + ".log")
        t0 = time.time()
        timed_out = False
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                code = proc.wait(timeout=args.timeout)
            except KeyboardInterrupt:  # Ctrl+C: stop this run too, the team waits for the next runner
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, 9)
                set_state(team, state="waiting", msg="")
                raise
            except subprocess.TimeoutExpired:  # stop the whole process group (the simulator has helper processes)
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, 9)
                code = proc.wait()
        text = open(log, errors="replace").read()
        res = re.findall(r"\[CHALLENGE\] RESULT team=\S+ seed=\d+: (.*?)\s+\(wall clock", text)
        if res:
            if os.path.exists(video + ".part.mp4"):
                os.replace(video + ".part.mp4", video)
                with open(os.path.join(replay_dir, name + ".json"), "w") as f:  # caption for /replay
                    json.dump({"team": team, "seed": args.seed, "level": args.level, "result": res[-1], "video": name + ".mp4",
                               "at": time.strftime("%H:%M:%S")}, f, ensure_ascii=False)
            elif not args.no_replay:
                print(f"[RUNNER] {team}: no replay video (see the log)", flush=True)
            set_state(team, state="done", msg=res[-1], seed=args.seed, level=args.level)
            print(f"[RUNNER] {team}: {res[-1]}  ({time.time() - t0:.0f} s)", flush=True)
        else:
            errs = [ln.strip() for ln in text.splitlines() if re.search(r"(Error|Exception)\b", ln)]
            why = errs[-1] if errs else f"exit code {code}"
            if timed_out:
                why = f"หยุดเพราะใช้เวลาเกิน {args.timeout:.0f} วินาที (pilot ค้างหรือช้าเกินไป?)"
            if os.path.exists(video + ".part.mp4"):
                os.remove(video + ".part.mp4")
            set_state(team, state="error", msg=why[:200], seed=args.seed, level=args.level)
            print(f"[RUNNER] {team}: ERROR {why[:160]}", flush=True)


if __name__ == "__main__":
    main()
