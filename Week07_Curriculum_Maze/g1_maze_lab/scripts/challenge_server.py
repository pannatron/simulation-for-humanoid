"""Challenge 7.5 upload page + leaderboard (standard library + torch for the policy check).

    uv run python scripts/challenge_server.py           # students:  http://<this-machine-ip>:8000
                                                        # projector: http://<this-machine-ip>:8000/board
    CHALLENGE_PORT=8080 CHALLENGE_SEED=13 uv run python scripts/challenge_server.py
    CHALLENGE_LEVEL=9 CHALLENGE_SEED=13 uv run python scripts/challenge_server.py   # level 7 (default) / 9 / final

Teams upload policy.pt (required), pilot.py (optional) and helper .py files their pilot imports. Each upload is
checked: the policy must have the same network as G1 Rough, every .py file must compile. Files land in
submissions/<team>/ (a new upload replaces the old one). The instructor runs the queue with
scripts/challenge_runner.py, which updates submissions/status.json; results go to submissions/leaderboard.csv.
Runs are played in upload order (the queue on the pages). Once the runner has recorded them, /replay plays every
team's run on the projector, from the last place up to the winner.

Note: pilot.py runs on this machine. Use it on a classroom network only, and read a pilot before running it
(or run the queue with CHALLENGE_SANDBOX=docker, see challenge_runner.py).

Online (behind a reverse proxy / tunnel), every setting below is optional:
    CHALLENGE_HOST=127.0.0.1           listen on this address only
    CHALLENGE_UPLOAD_CODE=<code>       class code: an upload needs it (strangers cannot queue runs)
    CHALLENGE_TEAM_PIN=1               the first upload of a team sets its PIN, later uploads need the same PIN
    CHALLENGE_ADMIN_KEY=<key>          instructor page /admin?key=<key>: level, seed, uploads open / closed, re-run
The round (level, seed, uploads open) set on /admin is kept in submissions/round.json and overrides
CHALLENGE_LEVEL / CHALLENGE_SEED. /live shows the run in progress on a map (written by challenge.py, MAZE_LIVE_DIR).
"""

import ast
import contextlib
import csv
import fcntl
import glob
import hashlib
import hmac
import html
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS = os.environ.get("CHALLENGE_DIR", os.path.join(HERE, "..", "submissions"))
PORT = int(os.environ.get("CHALLENGE_PORT", 8000))
HOST = os.environ.get("CHALLENGE_HOST", "0.0.0.0")
sys.path.insert(0, HERE)
from challenge_levels import DEFAULT_LEVEL, LEVELS, level_info  # noqa: E402

SEED = os.environ.get("CHALLENGE_SEED", "")  # announced seed of the round (shown on the pages)
LEVEL = os.environ.get("CHALLENGE_LEVEL", DEFAULT_LEVEL)  # announced level of the round
LIMIT = os.environ.get("CHALLENGE_TIME", str(level_info(LEVEL)["time"]))
UPLOAD_CODE = os.environ.get("CHALLENGE_UPLOAD_CODE", "")
TEAM_PIN = os.environ.get("CHALLENGE_TEAM_PIN", "") == "1"
ADMIN_KEY = os.environ.get("CHALLENGE_ADMIN_KEY", "")
LIVE_DIR = os.path.join(SUBMISSIONS, "_live")
MAX_BYTES = 60 * 1024 * 1024
REF_POLICY = os.path.join(HERE, "..", "pretrained", "g1_rough", "model_pretrained.pt")
TEAM_RE = r"[A-Za-z0-9][A-Za-z0-9_\-]{1,31}"
HELPER_RE = r"[A-Za-z][A-Za-z0-9_]{0,40}\.py"
# the real map of the maze: a pilot may only use what the robot senses (obs)
MAP_RE = rb"\b(GENERATED|HAZARDS|HARD_CELLS)\b|g1_maze_lab\.terrains|from\s+g1_maze_lab\s+import\s+[^\n]*\bterrains\b"


# ----------------------------------------------------------------------------- checks
def _actor_shapes(path_or_file):
    import torch

    ck = torch.load(path_or_file, map_location="cpu", weights_only=True)  # weights only: no code from the file
    sd = ck.get("actor_state_dict") or ck.get("model_state_dict")
    if not isinstance(sd, dict):
        raise ValueError("no actor_state_dict inside")
    return {k: tuple(v.shape) for k, v in sd.items() if hasattr(v, "shape")}


try:
    REF_SHAPES = _actor_shapes(REF_POLICY)
except Exception as e:  # torch missing or no reference: skip the network check
    print(f"[UPLOAD] policy network check off ({e})", flush=True)
    REF_SHAPES = None


def check_policy(data: bytes):
    if len(data) < 1000:
        return "policy.pt is empty or missing"
    if REF_SHAPES is None:
        return None
    import io

    try:
        shapes = _actor_shapes(io.BytesIO(data))
    except Exception as e:
        return f"cannot open policy.pt: it must be an rsl_rl model_*.pt checkpoint ({str(e)[:80]})"
    first = next(iter(shapes.items()), None)
    if shapes != REF_SHAPES:
        ref_first = next(k for k in REF_SHAPES if k.endswith("0.weight"))
        got = shapes.get(ref_first)
        return (f"this policy's network does not match G1 Rough: its first layer takes {got[1] if got else '?'} inputs "
                f"but this challenge needs {REF_SHAPES[ref_first][1]} (was it trained from the G1 Rough or G1-Maze task?)"
                if first else "this policy has no network inside")
    return None


def check_python(name, data: bytes):
    try:
        compile(data.decode("utf-8"), name, "exec")
    except SyntaxError as e:
        return f"{name} line {e.lineno}: {e.msg}"
    except UnicodeDecodeError:
        return f"{name} must be a UTF-8 text file"
    if name == "pilot.py" and not re.search(rb"^class\s+Pilot\b", data, re.M):
        return "pilot.py must define class Pilot"
    m = re.search(MAP_RE, data)
    if m:
        line = data[: m.start()].count(b"\n") + 1
        return (f"{name} line {line}: {m.group(0).decode(errors='ignore')} is not allowed "
                "(that is the real map of the maze: a pilot may only use what the robot senses, obs)")
    return None


def _module_available(name, local):
    """Importable in a run: the standard library, the course's .venv (the sandbox mounts the same one), scripts/
    (on sys.path here and in challenge.py), or one of the team's own files."""
    if name in local or name in sys.stdlib_module_names:
        return True
    try:
        return importlib.util.find_spec(name) is not None  # finds a top-level module without importing it
    except (ImportError, ValueError):
        return False


def code_warnings(files):
    """Problems the upload checks let through but a run would only show after its wait in the queue: an import the
    server does not have, a Pilot without act() / reset(), an __init__ that cannot take the four arguments.
    ``files``: {file name: bytes}. Imports inside ``try:`` are optional on purpose and not reported."""
    local = {os.path.splitext(n)[0] for n in files}
    out = []
    for name, data in files.items():
        try:
            tree = ast.parse(data.decode("utf-8"))
        except Exception:  # noqa: BLE001 (syntax errors are reported by check_python)
            continue
        optional = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try) for b in t.body for n in ast.walk(b)}
        seen = set()
        for node in ast.walk(tree):
            if id(node) in optional:
                continue
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for top in {m.split(".")[0] for m in mods} - seen:
                seen.add(top)
                if not _module_available(top, local):
                    out.append(f"{name} line {node.lineno}: import {top} - the server does not have it (the standard "
                               "library, numpy, scipy, torch, networkx, shapely, PIL and your own helper files are "
                               "there): the run would stop at that line")
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Pilot"), None)
        if name != "pilot.py" or cls is None or cls.bases:  # a subclass may inherit the methods
            continue
        methods = {n.name: n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for meth, when in (("act", "at its first step"), ("reset", "at the robot's first fall")):
            if meth not in methods:
                out.append(f"pilot.py: class Pilot has no {meth}(): the run would stop {when}")
        init = methods.get("__init__")
        if init and not init.args.vararg and len(init.args.args) - 1 < 4:
            out.append("pilot.py: Pilot.__init__ must take (num_envs, num_cells, cell_size, device, start_cell, "
                       "goal_cell) (see DefaultPilot): the run would stop at the start")
    return out


# ----------------------------------------------------------------------------- data
@contextlib.contextmanager
def status_lock():
    """status.json is written by this server (uploads) and by challenge_runner.py: one writer at a time."""
    os.makedirs(SUBMISSIONS, exist_ok=True)
    with open(os.path.join(SUBMISSIONS, ".status.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_status():
    try:
        with open(os.path.join(SUBMISSIONS, "status.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_status(st):
    tmp = os.path.join(SUBMISSIONS, "status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUBMISSIONS, "status.json"))


def _read_json(name, default):
    try:
        with open(os.path.join(SUBMISSIONS, name)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(name, data):
    tmp = os.path.join(SUBMISSIONS, name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(SUBMISSIONS, name))


def round_cfg():
    """The round: level, seed (str, "" = any) and whether uploads are open. round.json (set on /admin) wins over
    the environment; challenge_runner.py --follow-round reads the same file."""
    r = _read_json("round.json", {})
    level = str(r.get("level", LEVEL))
    if level not in LEVELS:
        level = LEVEL
    seed = str(r.get("seed", SEED))
    limit = os.environ.get("CHALLENGE_TIME", str(level_info(level)["time"]))
    return {"level": level, "seed": seed, "open": bool(r.get("open", True)), "limit": limit}


def _pin_hash(team, pin):
    return hashlib.sha256(f"maze-pin:{team}:{pin}".encode()).hexdigest()


def board_data():
    """Best run per team (GOAL fastest first, then DNF closest to the goal, then fewer falls) + every team's status."""
    rnd = round_cfg()
    SEED, LEVEL, LIMIT = rnd["seed"], rnd["level"], rnd["limit"]  # noqa: N806  (this request's round)
    rows = []
    path = os.path.join(SUBMISSIONS, "leaderboard.csv")
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if (r.get("level") or "7") == LEVEL]  # this level only
    if SEED:  # this round only: every team on the announced maze
        rows = [r for r in rows if r["seed"] == SEED]
    best, runs = {}, {}
    for r in rows:
        runs[r["team"]] = runs.get(r["team"], 0) + 1
        if r["result"] == "GOAL":
            key = (0, float(r["time_s"]), int(r["falls"]))
        else:
            left = r.get("best_left_m")
            key = (1, float(left) if left else 99.0, int(r["falls"]))
        if r["team"] not in best or key < best[r["team"]][0]:
            best[r["team"]] = (key, r)
    ranked = []
    for k, (_, r) in enumerate(sorted(best.values(), key=lambda x: x[0]), 1):
        ranked.append({"rank": k, "team": r["team"], "goal": r["result"] == "GOAL", "time": r["time_s"],
                       "left": r.get("best_left_m") or "?", "falls": r["falls"], "seed": r["seed"],
                       "runs": runs[r["team"]]})
    teams = sorted(d for d in os.listdir(SUBMISSIONS) if os.path.isdir(os.path.join(SUBMISSIONS, d)) and d[0] != "_") \
        if os.path.isdir(SUBMISSIONS) else []
    st = load_status()
    status = {}
    for t in teams:  # a team run on another seed is still waiting for this round
        s = st.get(t, {"state": "waiting"})
        same = (not SEED or str(s.get("seed", SEED)) == SEED) and str(s.get("level", LEVEL)) == LEVEL
        status[t] = s if same or s.get("state") == "running" else {"state": "waiting"}
    running = sorted((t for t, s in st.items() if s.get("state") == "running"), key=lambda t: st[t].get("started", ""))
    queue = sorted((t for t, s in status.items() if s.get("state") == "waiting"),
                   key=lambda t: (st.get(t, {}).get("uploaded_at", 0), t))  # same order as challenge_runner.py
    return {"ranked": ranked, "status": status, "running": running, "queue": queue, "replays": replays(SEED, LEVEL),
            "seed": SEED, "level": LEVEL, "limit": LIMIT, "open": rnd["open"], "updated": time.strftime("%H:%M:%S")}


def replays(SEED=SEED, LEVEL=LEVEL):  # noqa: N803
    """{team: caption} of the recorded runs of this round (written by challenge_runner.py)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(SUBMISSIONS, "_replays", "*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        have = any(r.get(k) and os.path.exists(os.path.join(SUBMISSIONS, "_replays", r[k])) for k in ("video", "trace"))
        if (not SEED or str(r.get("seed")) == SEED) and str(r.get("level", "7")) == LEVEL and have:
            out[r["team"]] = r
    return out


# ----------------------------------------------------------------------------- pages
HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Prompt:wght@500;600;700&family=IBM+Plex+Sans+Thai:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0E1621;--card:#16212F;--line:#26364a;--text:#F4F3EE;--soft:#B8C4D0;--dim:#7d8da0;--green:#7AC943;
--orange:#F2A65A;--red:#E8604C;--gold:#F2C14E;--silver:#C9D1DA;--bronze:#D08B5B}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);
font-family:'IBM Plex Sans Thai',system-ui,sans-serif}}
h1,h2,h3{{font-family:'Prompt','IBM Plex Sans Thai',system-ui,sans-serif;margin:0}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px 18px 60px}}
.top{{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;margin-bottom:22px}}
.eyebrow{{color:var(--green);letter-spacing:3px;text-transform:uppercase;font-weight:600;font-size:.85rem}}
.chips span{{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:999px;
padding:6px 14px;margin:4px 6px 0 0;color:var(--soft);font-size:.9rem}} .chips b{{color:var(--text)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin:18px 0}}
.podium{{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:16px;align-items:end;margin:8px 0 6px}}
.p{{border-radius:18px;padding:20px 16px;text-align:center;background:linear-gradient(180deg,#1d2b3d,#16212F);
border:1px solid var(--line)}} .p .medal{{font-size:2.6rem;line-height:1}} .p .name{{font-family:Prompt,sans-serif;
font-size:1.35rem;margin:8px 0 2px;word-break:break-all}} .p .score{{font-size:1.6rem;font-weight:600}}
.p1{{border-color:var(--gold);box-shadow:0 0 32px rgba(242,193,78,.18);padding-top:34px}} .p1 .score{{color:var(--gold)}}
.p2 .score{{color:var(--silver)}} .p3 .score{{color:var(--bronze)}} .p.empty{{opacity:.35}}
table{{width:100%;border-collapse:collapse;font-size:1rem}} th,td{{padding:10px 10px;text-align:left;
border-bottom:1px solid var(--line)}} th{{color:var(--dim);font-weight:500;font-size:.85rem;text-transform:uppercase;letter-spacing:1px}}
td.rank{{font-family:Prompt,sans-serif;font-size:1.2rem;width:56px}} .ok{{color:var(--green);font-weight:600}}
.dnf{{color:var(--orange);font-weight:600}} .badge{{display:inline-block;border-radius:999px;padding:3px 10px;font-size:.8rem;
font-weight:600}} .b-waiting{{background:#24344a;color:var(--soft)}} .b-running{{background:#1f4d06;color:#cfeebb;
animation:pulse 1.2s infinite}} .b-done{{background:#203a2a;color:var(--green)}} .b-error{{background:#4a2323;color:#ffb3a6}}
@keyframes pulse{{50%{{opacity:.55}}}}
.live{{background:#1f4d06;border-radius:14px;padding:12px 18px;margin:0 0 6px;display:flex;align-items:center;gap:12px}}
.dot{{width:12px;height:12px;border-radius:50%;background:var(--green);animation:pulse 1s infinite}}
label{{display:block;margin:14px 0 6px;color:var(--soft);font-weight:500}}
input[type=text]{{width:100%;max-width:360px;padding:11px 14px;border-radius:12px;border:1px solid #33465c;
background:var(--bg);color:var(--text);font-size:1rem}}
.file{{display:block;border:2px dashed #33465c;border-radius:14px;padding:14px 16px;background:var(--bg);color:var(--soft)}}
.file input{{color:var(--soft);font-size:.95rem;width:100%}} .hint{{color:var(--dim);font-size:.85rem;margin-top:4px}}
button{{margin-top:18px;padding:12px 28px;border:0;border-radius:12px;background:var(--green);color:#0E1621;
font-weight:700;font-size:1.05rem;cursor:pointer;font-family:Prompt,sans-serif}} button:hover{{filter:brightness(1.08)}}
.msg{{padding:14px 18px;border-radius:14px;background:#203a2a;border:1px solid #2f5c3d;margin:14px 0}}
.msg.err{{background:#3a1f1f;border-color:#6b2f2f}} .msg .wr{{display:inline-block;margin-top:8px;color:var(--orange);font-weight:600}} .grid2{{display:grid;grid-template-columns:1.1fr 1fr;gap:18px}}
@media(max-width:820px){{.grid2,.podium{{grid-template-columns:1fr}} .p1{{order:-1}}}}
.big .p .name{{font-size:2rem}} .big .p .score{{font-size:2.4rem}} .big table{{font-size:1.35rem}} .big h1{{font-size:3rem}}
small.u,.u{{color:var(--dim)}}
.queue{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 10px}} .queue .u{{margin-right:4px}}
.q{{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 14px;color:var(--soft)}}
.q b{{color:var(--orange);margin-right:4px}}
a.play{{display:inline-block;margin-left:6px;width:30px;height:30px;line-height:30px;text-align:center;border-radius:50%;
background:#24344a;color:var(--green);text-decoration:none;font-size:.85rem}} a.play:hover{{background:#1f4d06}}
a.replaybar{{display:block;background:linear-gradient(90deg,#1f4d06,#2c5f12);border-radius:14px;padding:14px 20px;
margin:0 0 10px;color:var(--text);text-decoration:none;font-family:Prompt,sans-serif;font-size:1.15rem}}
nav.nav{{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 18px}} nav.nav a{{color:var(--soft);text-decoration:none;
padding:6px 14px;border-radius:999px;border:1px solid var(--line);font-size:.9rem}} nav.nav a:hover{{color:var(--text);
border-color:var(--green)}} a.livebtn{{color:var(--text);margin-left:auto;background:#1f4d06;border-radius:999px;
padding:4px 14px;text-decoration:none;font-size:.95rem}}
</style></head><body><div class="wrap {cls}"><nav class="nav"><a href="/">Submit</a><a href="/board">Leaderboard</a>
<a href="/live">Live</a><a href="/replay">Replay</a></nav>"""


def podium_html(ranked):
    slots = {1: ("🥇", "p1"), 2: ("🥈", "p2"), 3: ("🥉", "p3")}
    cards = []
    for place in (2, 1, 3):
        medal, cls = slots[place]
        r = next((x for x in ranked if x["rank"] == place), None)
        if r is None:
            cards.append(f'<div class="p {cls} empty"><div class="medal">{medal}</div><div class="name">–</div></div>')
            continue
        score = f'{r["time"]} s' if r["goal"] else f'{r["left"]} m to go'
        cards.append(f'<div class="p {cls}"><div class="medal">{medal}</div><div class="name">{html.escape(r["team"])}</div>'
                     f'<div class="score">{score}</div><small class="u">{r["falls"]} falls</small></div>')
    return '<div class="podium">' + "".join(cards) + "</div>"


def badge(team, state, data):
    labels = {"waiting": "waiting", "running": "running", "done": "done", "error": "error"}
    if state == "waiting" and team in data["queue"]:
        return f'<span class="badge b-waiting">queue #{data["queue"].index(team) + 1}</span>'
    b = f'<span class="badge b-{state}">{labels.get(state, state)}</span>'
    if team in data["replays"]:
        b += f' <a class="play" href="/replay?team={html.escape(team)}" title="Watch the replay">▶</a>'
    return b


def table_html(data):
    st = data["status"]
    rows = []
    seen = set()
    for r in data["ranked"]:
        seen.add(r["team"])
        res = (f'<span class="ok">{r["time"]} s</span>' if r["goal"]
               else f'<span class="dnf">DNF · {r["left"]} m to go</span>')
        s = st.get(r["team"], {}).get("state", "done")
        rows.append(f'<tr><td class="rank">{r["rank"]}</td><td>{html.escape(r["team"])}</td><td>{res}</td>'
                    f'<td>{r["falls"]}</td><td>{r["runs"]}</td><td>{badge(r["team"], s, data)}</td></tr>')
    q = data["queue"]  # running first, then the queue in order, then errors
    for team, s in sorted(st.items(), key=lambda x: (x[1].get("state") != "running", x[0] not in q,
                                                     q.index(x[0]) if x[0] in q else 0)):
        if team in seen:
            continue
        state = s.get("state", "waiting")
        note = html.escape(s.get("msg", ""))[:140]
        rows.append(f'<tr><td class="rank">–</td><td>{html.escape(team)}</td><td><small class="u">{note or "no result yet"}</small></td>'
                    f'<td></td><td>0</td><td>{badge(team, state, data)}</td></tr>')
    if not rows:
        return '<small class="u">No uploads yet</small>'
    return ("<table><tr><th>#</th><th>Team</th><th>Result</th><th>Falls</th><th>Runs</th><th>Status</th></tr>"
            + "".join(rows) + "</table>")


def live_html(data):
    out = ""
    if data["running"]:
        names = ", ".join(f'<a style="color:var(--text)" href="/live?team={html.escape(t)}">{html.escape(t)}</a>'
                          for t in data["running"])
        out += (f'<div class="live"><span class="dot"></span><b>Running:</b> {names}'
                '<a class="livebtn" href="/live">▶ Watch live</a></div>')
    if data["queue"]:
        names = "".join(f'<span class="q"><b>{k}</b> {html.escape(t)}</span>' for k, t in enumerate(data["queue"], 1))
        out += f'<div class="queue"><span class="u">Up next</span>{names}</div>'
    if data["replays"] and not data["running"] and not data["queue"]:
        out += ('<a class="replaybar" href="/replay">▶ Watch every team\'s replay, last place first, the winner at the end '
                f'({len(data["replays"])} teams)</a>')
    return out


def chips(data):
    seed = f'<span>Level <b>{html.escape(data["level"])}</b></span>'
    seed += f'<span>seed <b>{html.escape(data["seed"])}</b></span>' if data["seed"] else ""
    if level_info(data["level"])["cells"]:
        seed += f'<span>Map <b>{level_info(data["level"])["cells"]} x {level_info(data["level"])["cells"]}</b></span>'
    if not data.get("open", True):
        seed += '<span style="color:var(--orange)">Uploads closed</span>'
    return (f'<div class="chips">{seed}<span>Time limit <b>{data["limit"]} s</b></span>'
            f'<span>Updated <b id="upd">{data["updated"]}</b></span></div>')


REFRESH_JS = """<script>
setInterval(async () => {
  try { const r = await fetch(location.pathname === '/board' ? '/board?part=1' : '/?part=1');
        if (r.ok) document.getElementById('live').innerHTML = await r.text(); } catch (e) {}
}, 5000);
</script>"""


def board_page():
    data = board_data()
    return (HEAD.format(title="Leaderboard · Challenge 7.5", cls="big")
            + '<div class="top"><div><div class="eyebrow">Challenge 7.5 · G1 Hazard Maze</div><h1>Leaderboard</h1></div>'
            + chips(data) + "</div><div id=\"live\">" + board_part(data) + "</div>" + REFRESH_JS + "</div></body></html>")


def board_part(data=None):
    data = data or board_data()
    return live_html(data) + podium_html(data["ranked"]) + '<div class="card">' + table_html(data) + "</div>"


REPLAY_CSS = """<style>
.wrap.replay{max-width:1500px} .stage{display:grid;grid-template-columns:1fr 330px;gap:20px;align-items:start}
.screen{position:relative;background:#000;border-radius:18px;overflow:hidden;aspect-ratio:16/9}
.screen video{width:100%;height:100%;display:block}
.tag{position:absolute;right:18px;top:16px;background:rgba(14,22,33,.85);border-radius:14px;padding:10px 18px;
text-align:right} .tag .n{font-family:Prompt,sans-serif;font-size:1.7rem} .tag .k{color:var(--soft);font-size:.95rem}
.reveal{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;
background:rgba(14,22,33,.82);gap:6px;text-align:center} .reveal.on{display:flex}
.reveal .rk{font-family:Prompt,sans-serif;font-size:2rem;color:var(--soft)} .reveal .nm{font-family:Prompt,sans-serif;
font-size:4.2rem;line-height:1.1} .reveal .rs{font-size:2.2rem;font-weight:600} .reveal .medal{font-size:5rem;line-height:1}
.intro{position:absolute;inset:0;display:none;align-items:center;justify-content:center;flex-direction:column;
background:#0E1621;gap:10px} .intro.on{display:flex} .intro .k{color:var(--green);letter-spacing:3px;font-weight:600}
.intro .nm{font-family:Prompt,sans-serif;font-size:4.5rem}
.ctrl{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:14px}
.ctrl button{margin:0;padding:9px 18px;font-size:.95rem;background:#24344a;color:var(--text)}
.ctrl button.on{background:var(--green);color:#0E1621} .ctrl .sp{flex:1}
.list{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:10px}
.item{display:flex;gap:12px;align-items:center;padding:12px;border-radius:12px;cursor:pointer}
.item:hover{background:#1d2b3d} .item.cur{background:#1f4d06} .item .r{font-family:Prompt,sans-serif;font-size:1.3rem;
width:34px;text-align:center;color:var(--dim)} .item .t{flex:1;word-break:break-all;font-weight:600}
.item .s{font-size:.85rem;color:var(--soft)} .item.hid .s{visibility:hidden}
.seek{display:flex;align-items:center;gap:12px;margin-top:12px} .seek span{font-variant-numeric:tabular-nums;color:var(--soft);
min-width:46px;text-align:center} .seek input{flex:1;accent-color:var(--green);height:6px;cursor:pointer}
#player:fullscreen,#player.fsx{background:#000;display:flex;flex-direction:column;padding:10px 14px}
#player.fsx{position:fixed;inset:0;z-index:50} #player:fullscreen .screen,#player.fsx .screen{flex:1;aspect-ratio:auto;border-radius:10px}
#player:fullscreen video,#player.fsx video{object-fit:contain}
@media(max-width:900px){.stage{grid-template-columns:1fr}}
</style>"""


TRACE_JS = r"""
// a 2D replay from <run>.trace.json (written by challenge.py, MAZE_TRACE) with the same API as a <video>
class TracePlayer{
 constructor(cv){ this.cv=cv; this.paused=true; this.playbackRate=1; this.d=null; this.t=0; this.raf=0; }
 set src(url){ this.pause(); this.d=null; this.t=0; this.url=url; fetch(url).then(r=>r.json()).then(d=>{ if(this.url!==url) return;
  this.d=d; const f=d.frames; this.end=f.length?f[f.length-1][0]:0; this.draw(); if(this.want) this.play(); }); }
 play(){ if(!this.d){ this.want=true; return Promise.resolve(); } this.want=false; if(this.t>=this.end) this.t=0;
  this.paused=false; this.last=performance.now(); cancelAnimationFrame(this.raf); this.raf=requestAnimationFrame(()=>this.tick());
  this.onplay&&this.onplay(); return Promise.resolve(); }
 pause(){ this.want=false; if(this.paused) return; this.paused=true; cancelAnimationFrame(this.raf); this.onpause&&this.onpause(); }
 get currentTime(){ return this.t; }
 set currentTime(v){ this.t=Math.max(0,Math.min(+v||0,this.end||0)); this.last=performance.now(); this.draw(); }
 get duration(){ return this.d?this.end:NaN; }
 tick(){ const now=performance.now(); this.t+=(now-this.last)/1000*this.playbackRate; this.last=now;
  if(this.t>=this.end){ this.t=this.end; this.draw(); this.paused=true; this.onended&&this.onended(); return; }
  this.draw(); this.raf=requestAnimationFrame(()=>this.tick()); }
 draw(){ const cv=this.cv,dpr=window.devicePixelRatio||1,w=Math.round(cv.clientWidth*dpr),h=Math.round(cv.clientHeight*dpr);
  if(w&&(cv.width!==w||cv.height!==h)){ cv.width=w; cv.height=h; }
  if(!this.d){ drawMap(cv,null,null,'Loading…'); return; }
  const f=this.d.frames; let lo=0,hi=f.length-1; while(lo<hi){ const m=(lo+hi+1)>>1; if(f[m][0]<=this.t) lo=m; else hi=m-1; }
  const trail=[]; let falls=f.length?f[0][4]:0;
  for(let k=0;k<=lo;k+=5){ if(f[k][4]!==falls){ falls=f[k][4]; trail.push(null); } trail.push([f[k][1],f[k][2]]); }
  const fr=f[lo]||[0,0,0,0,0,0,0,0],nx=f[lo+1]; let x=fr[1],y=fr[2],yaw=fr[3];
  if(nx&&nx[4]===fr[4]&&nx[0]>fr[0]){  // between two 10 Hz samples (not across a fall: that one jumps to the start)
   const a=Math.min(1,Math.max(0,(this.t-fr[0])/(nx[0]-fr[0]))),dy=Math.atan2(Math.sin(nx[3]-yaw),Math.cos(nx[3]-yaw));
   x+=(nx[1]-x)*a; y+=(nx[2]-y)*a; yaw+=dy*a; }
  trail.push([x,y]);
  drawMap(cv,this.d.maze,{x,y,yaw,flash:fr[5],trail});
  const ctx=cv.getContext('2d'),s=Math.min(cv.width,cv.height)/720,m=this.d.maze,tt=Math.min(this.t,this.end||0);
  ctx.fillStyle='rgba(14,22,33,.85)'; ctx.fillRect(10*s,10*s,250*s,128*s); ctx.textAlign='left'; ctx.textBaseline='top';
  ctx.fillStyle='#7AC943'; ctx.font=`${13*s}px sans-serif`; ctx.fillText(`L${m.level} · seed ${m.seed} · ${m.n}x${m.n}`,22*s,20*s);
  const goal=this.d.goal!=null&&tt>=this.d.goal; ctx.fillStyle=goal?'#7AC943':'#F4F3EE'; ctx.font=`600 ${30*s}px Prompt,sans-serif`;
  ctx.fillText(goal?`GOAL ${this.d.goal.toFixed(1)} s`:`${tt.toFixed(1)} s`,22*s,38*s);
  ctx.fillStyle='#B8C4D0'; ctx.font=`${13*s}px sans-serif`;
  ctx.fillText(`to goal ${fr[6].toFixed(1)} m · closest ${fr[7].toFixed(1)} m`,22*s,84*s);
  ctx.fillText(`falls ${fr[4]}`,22*s,106*s); }
}
"""


def replay_page(query=""):
    """Projector: every team's recorded run, from the last place up to the winner (the result shows at the end)."""
    data = board_data()
    ranked = [r for r in reversed(data["ranked"]) if r["team"] in data["replays"]]
    # runs without a place (stopped half way: wall-clock limit, pilot stuck) first, then last place ... the winner
    order = [{"rank": None, "team": t} for t in sorted(data["replays"]) if t not in {r["team"] for r in ranked}]
    order += ranked
    one = re.search(r"team=(" + TEAM_RE + ")", query)
    items = []
    for r in order:
        rp = data["replays"][r["team"]]
        def url(name):  # ?v=<mtime>: a re-run gets a new URL (Cloudflare caches the videos)
            path = os.path.join(SUBMISSIONS, "_replays", name or "")
            return f"/replays/{name}?v={int(os.path.getmtime(path))}" if name and os.path.isfile(path) else None

        items.append({"team": r["team"], "rank": r["rank"], "video": url(rp.get("video")), "trace": url(rp.get("trace")),
                      "result": rp["result"], "goal": rp["result"].startswith("GOAL"),
                      "board": (f'{r["time"]} s' if r.get("goal") else f'{r["left"]} m to go') if r["rank"] else ""})
    start = next((k for k, it in enumerate(items) if one and it["team"] == one.group(1)), 0)
    body = ('<div class="top"><div><div class="eyebrow">Challenge 7.5 Replay</div><h1>Replay</h1></div>'
            + chips(data) + "</div>")
    if not items:
        return (HEAD.format(title="Replay · Challenge 7.5", cls="replay big") + body
                + '<div class="card">No replays yet: every team\'s clip appears here once its run has finished.</div>'
                + '<a style="color:var(--green)" href="/board">Back to /board</a></div></body></html>')
    return (HEAD.format(title="Replay · Challenge 7.5", cls="replay") + REPLAY_CSS + body
            + '<div class="stage"><div id="player"><div class="screen"><video id="v" muted playsinline preload="auto"></video>'
            + '<canvas id="tc" style="display:none;width:100%;height:100%"></canvas>'
            + '<div class="tag"><div class="n" id="tn"></div><div class="k" id="tk"></div></div>'
            + '<div class="intro" id="intro"><div class="k" id="ik"></div><div class="nm" id="inm"></div></div>'
            + '<div class="reveal" id="rev"><div class="medal" id="rm"></div><div class="rk" id="rk"></div>'
            + '<div class="nm" id="rn"></div><div class="rs" id="rs"></div><div class="u" id="rx"></div></div></div>'
            + '<div class="seek"><span id="ct">0:00</span><input type="range" id="sk" min="0" max="1000" step="1" value="0" '
            + 'aria-label="Position in the clip"><span id="dt">–</span></div>'
            + '<div class="ctrl"><button id="prev">◀ Previous</button><button id="back5">⟲ 5 s</button>'
            + '<button id="pp">⏸ Pause</button><button id="fwd5">5 s ⟳</button>'
            + '<button id="next">Next ▶</button><span class="sp"></span><span class="u">Speed</span>'
            + '<button data-s="0.5">0.5x</button><button data-s="1">1x</button><button data-s="2">2x</button>'
            + '<button data-s="4">4x</button><button data-s="8">8x</button><button id="auto">Autoplay next</button>'
            + '<button id="fs">⛶ Full screen</button></div></div>'
            + '<div class="list" id="list"></div></div>'
            + '<p class="u">Last place first, the winner at the end; each result is revealed when its clip ends. A clip is the team\'s latest run '
            + '(its clock is the time in the maze, whatever the playback speed). Drag the time bar to go back or skip · '
            + 'Keys: Space play / pause, J / L back / forward 5 s, ← → previous / next team, F full screen '
            + '<a style="color:var(--green)" href="/board">Back to /board</a></p>'
            + "<script>" + MAP_JS + TRACE_JS + "const ITEMS=" + json.dumps(items, ensure_ascii=False) + f";let cur={start};"
            + r"""
const vid=document.getElementById('v'),tcv=document.getElementById('tc'),tp=new TracePlayer(tcv),$=id=>document.getElementById(id);
let v=vid;
let speed=+(localStorage.getItem('rs')||2),auto=localStorage.getItem('ra')!=='0',seen=new Set(),timer=null;
const medal=r=>r===1?'🥇':r===2?'🥈':r===3?'🥉':'';
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function list(){ $('list').innerHTML=ITEMS.map((it,k)=>`<div class="item ${k===cur?'cur':''} ${seen.has(k)?'':'hid'}" data-k="${k}">
 <span class="r">${seen.has(k)&&it.rank?it.rank:'?'}</span><div class="t">${esc(it.team)}<div class="s">${esc(it.board||it.result)}</div></div></div>`).join('');
 document.querySelectorAll('.item').forEach(e=>e.onclick=()=>load(+e.dataset.k)); }
function ui(){ document.querySelectorAll('[data-s]').forEach(b=>b.classList.toggle('on',+b.dataset.s===speed));
 $('auto').classList.toggle('on',auto); $('pp').textContent=v.paused?'▶ Play':'⏸ Pause'; }
function load(k){ clearTimeout(timer); cur=(k+ITEMS.length)%ITEMS.length; const it=ITEMS[cur];
 $('rev').classList.remove('on'); $('tn').textContent=it.team; $('tk').textContent=`Clip ${cur+1} of ${ITEMS.length}`;
 $('ik').textContent=cur===ITEMS.length-1?'Last team':`Team ${cur+1} of ${ITEMS.length}`;
 $('inm').textContent=it.team; $('intro').classList.add('on'); v.pause();
 v=it.video?vid:tp; vid.style.display=it.video?'block':'none'; tcv.style.display=it.video?'none':'block';
 v.src=it.video||it.trace; v.playbackRate=speed; list();
 timer=setTimeout(()=>{ $('intro').classList.remove('on'); v.playbackRate=speed; v.play().catch(()=>{}); ui(); },1800); }
vid.onended=tp.onended=()=>{ const it=ITEMS[cur]; seen.add(cur); list();
 $('rm').textContent=medal(it.rank); $('rk').textContent=it.rank?`Rank ${it.rank}`:'';
 $('rn').textContent=it.team; $('rs').textContent=it.goal?it.result.replace(/^GOAL\s*/,'Goal in ').replace(/, falls (\d+)/,' · $1 falls')
   :it.result.replace(/^DNF.*closest ([\d.]+) m.*falls (\d+).*/,'Did not finish · $1 m to go · $2 falls')
     .replace(/^STOPPED( at t=\d+ s)?: (.*)/,(m,a,w)=>`Stopped${a?a.replace(' t=',' '):''} · no result · ${w}`);
 $('rs').style.color=it.goal?'var(--green)':'var(--orange)';
 const best=(it.board.match(/[\d.]+/)||[''])[0];
 $('rx').textContent=best&&!it.result.includes(best+' ')?`Not this team\'s best run: the board uses ${it.board}`:'';
 $('rev').classList.add('on'); ui(); if(auto&&cur<ITEMS.length-1) timer=setTimeout(()=>load(cur+1),5000); };
vid.onplay=vid.onpause=tp.onplay=tp.onpause=ui;
$('prev').onclick=()=>load(cur-1); $('next').onclick=()=>load(cur+1);
$('pp').onclick=()=>{ if($('rev').classList.contains('on')) return load(cur); v.paused?v.play():v.pause(); };
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>{ speed=+b.dataset.s; v.playbackRate=speed;
 try{localStorage.setItem('rs',speed)}catch(e){} ui(); });
$('auto').onclick=()=>{ auto=!auto; try{localStorage.setItem('ra',auto?'1':'0')}catch(e){} ui(); };
// seek bar: drag to watch any moment again (the video and the 2D replay both seek)
const sk=$('sk'),player=$('player'); let drag=false;
const fmt=s=>{ s=Math.max(0,s||0); return `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`; };
function bar(){ const d=v.duration,c=v.currentTime; if(!drag) sk.value=d>0?Math.round(c/d*1000):0;
 $('ct').textContent=fmt(c); $('dt').textContent=d>0?fmt(d):'–'; requestAnimationFrame(bar); }
function seeking(){ clearTimeout(timer); $('rev').classList.remove('on'); $('intro').classList.remove('on'); ui(); }
sk.oninput=()=>{ drag=true; const d=v.duration; if(d>0) v.currentTime=sk.value/1000*d; seeking(); };
sk.onchange=()=>{ drag=false; };
function jump(s){ const d=v.duration; if(!(d>0)) return; v.currentTime=Math.max(0,Math.min(d,v.currentTime+s)); seeking(); }
$('back5').onclick=()=>jump(-5); $('fwd5').onclick=()=>jump(5);
const isFs=()=>document.fullscreenElement===player||player.classList.contains('fsx');
function fsDone(){ $('fs').textContent=isFs()?'✕ Exit full screen':'⛶ Full screen'; setTimeout(()=>tp.draw(),60); }
$('fs').onclick=()=>{
 if(isFs()){ if(document.fullscreenElement) document.exitFullscreen(); else { player.classList.remove('fsx'); fsDone(); } return; }
 const go=player.requestFullscreen||player.webkitRequestFullscreen,css=()=>{ player.classList.add('fsx'); fsDone(); };
 const p=go?go.call(player):null; if(p&&p.catch) p.catch(css); else if(!go) css(); };
document.addEventListener('fullscreenchange',fsDone); addEventListener('resize',()=>tp.draw());
document.addEventListener('keydown',e=>{ if(e.target===sk&&e.key.startsWith('Arrow')) return;
 if(e.key===' '){e.preventDefault();$('pp').click();}
 if(e.key==='ArrowRight')load(cur+1); if(e.key==='ArrowLeft')load(cur-1);
 if(e.key==='j'||e.key==='J') jump(-5); if(e.key==='l'||e.key==='L') jump(5);
 if(e.key==='f'||e.key==='F') $('fs').click();
 if(e.key==='Escape'&&player.classList.contains('fsx')){ player.classList.remove('fsx'); fsDone(); } });
bar(); load(cur);
</script></div></body></html>""")


UPLOAD_JS = """<script>
(() => { const f = document.getElementById('up'); if (!f || !window.FormData) return;
 f.addEventListener('submit', e => { e.preventDefault();
  const b = f.querySelector('button[type=submit]'), p = document.getElementById('prog'), x = new XMLHttpRequest();
  b.disabled = true; b.textContent = 'Uploading...'; p.style.display = 'block'; p.textContent = 'Uploading...';
  x.open('POST', '/upload');
  x.upload.onprogress = ev => { if (ev.lengthComputable) p.textContent = 'Uploading ' + Math.round(ev.loaded / ev.total * 100)
    + ' % (' + (ev.loaded / 1e6).toFixed(1) + ' of ' + (ev.total / 1e6).toFixed(1) + ' MB): keep this page open'; };
  x.upload.onload = () => { p.textContent = 'Checking the files on the server...'; };
  x.onload = () => { document.open(); document.write(x.responseText); document.close(); window.scrollTo(0, 0); };
  x.onerror = x.ontimeout = () => { b.disabled = false; b.textContent = 'Submit';
    p.textContent = 'The upload did not reach the server (network?): try again'; };
  x.send(new FormData(f)); });
})();
</script>"""


def student_page(msg=""):
    data = board_data()
    closed = ('' if data["open"] else
              '<div class="msg err">Uploads are closed: the instructor is running the round. See the results below or <a style="color:var(--green)" '
              'href="/live">watch live</a></div>')
    code = ('<label>Class code</label><input type="text" name="code" required autocomplete="off" '
            'placeholder="given in class">' if UPLOAD_CODE else "")
    pin = ('<label>Team PIN</label><input type="text" name="pin" required pattern="[0-9A-Za-z]{4,12}" '
           'autocomplete="off" placeholder="4-12 letters or digits, you choose it">'
           '<div class="hint">Your first upload sets the PIN; later uploads need the same PIN (so no other team can overwrite yours)</div>'
           if TEAM_PIN else "")
    return (HEAD.format(title="Challenge 7.5 · Submit", cls="")
            + '<div class="top"><div><div class="eyebrow">Challenge 7.5 · G1 Hazard Maze</div><h1>Submit to the leaderboard</h1></div>'
            + chips(data) + "</div>" + msg + closed
            + '<div class="grid2"><div class="card"><h3>Upload</h3>'
            + '<form method="post" action="/upload" enctype="multipart/form-data" id="up">'
            + f'<label>Team name</label><input type="text" name="team" required pattern="{TEAM_RE}" placeholder="e.g. team_tiger">'
            + '<div class="hint">Letters, digits, _ or -, 2-32 characters. Uploading again with the same name replaces your files</div>'
            + pin + code
            + '<label>policy.pt (required)</label><span class="file"><input type="file" name="policy" accept=".pt" required></span>'
            + '<div class="hint">rsl_rl checkpoint trained from G1 Rough (a model_*.pt file)</div>'
            + '<label>pilot.py (optional)</label><span class="file"><input type="file" name="pilot" accept=".py"></span>'
            + '<div class="hint">Must define class Pilot; without one the default autopilot drives</div>'
            + '<label>Helper .py files your pilot imports (optional)</label><span class="file"><input type="file" name="helpers" accept=".py" multiple></span>'
            + '<div class="hint">Up to 5 files. A pilot may not keep data from one run for the next</div>'
            + '<button type="submit">Submit</button><div class="hint" id="prog" style="display:none"></div></form>'
            + UPLOAD_JS + '</div>'
            + '<div class="card"><h3>Rules in short</h3><ul style="line-height:1.8;color:var(--soft)">'
            + "<li>Fastest to the goal wins; without a goal, the team that got closest wins</li><li>A fall, a wall crash or 10 s stuck sends the robot back to the start; the clock keeps running</li>"
            + "<li>A pilot may only use what the robot senses during the run, never the map</li><li>Touching a red spike bed counts as a fall: back to the start</li><li>Every team runs the same maze; the instructor announces the level and the seed</li>"
            + "<li>Uploads run automatically in queue order; watch them at <a style=\"color:var(--green)\" href=\"/live\">/live</a></li></ul>"
            + '<small class="u">Projector view: <a style="color:var(--green)" href="/board">/board</a></small></div></div>'
            + '<h2 style="margin-top:12px">Current ranking</h2><div id="live">' + board_part(data) + "</div>" + REFRESH_JS
            + "</div></body></html>")


# ----------------------------------------------------------------------------- live view
def _live_team(team=""):
    """The team the /live page shows: the one asked for, else the first running, else the last that ran."""
    st = load_status()
    running = sorted((t for t, s in st.items() if s.get("state") == "running"), key=lambda t: st[t].get("started", ""))
    if team and re.fullmatch(TEAM_RE, team) and os.path.isdir(os.path.join(LIVE_DIR, team)):
        return team, running
    if running:
        return running[0], running
    done = [(os.path.getmtime(p), os.path.basename(os.path.dirname(p)))
            for p in glob.glob(os.path.join(LIVE_DIR, "*", "state.json"))]
    return (max(done)[1] if done else ""), running


def live_data(team="", since=""):
    """A run in progress, from the files challenge.py writes to submissions/_live/<team>/ (MAZE_LIVE_DIR):
    maze.json once per run (walls, hazards, start, goal) and state.json several times a second (pose, clock)."""
    team, running = _live_team(team)
    out = {"running": running, "team": team, "state": None, "frame": False}
    if not team:
        return out
    folder = os.path.join(LIVE_DIR, team)
    try:
        with open(os.path.join(folder, "state.json")) as f:
            out["state"] = json.load(f)
    except (OSError, ValueError):
        pass
    run = (out["state"] or {}).get("run", "")
    if run and run != since:
        try:
            with open(os.path.join(folder, "maze.json")) as f:
                maze = json.load(f)
            if maze.get("run") == run:
                out["maze"] = maze
        except (OSError, ValueError):
            pass
    try:  # the 3D camera: frame_id changes with every new frame (the page asks for /live.jpg?f=<frame_id>)
        mtime = os.stat(os.path.join(folder, "frame.jpg")).st_mtime_ns
        out["frame"] = time.time() - mtime / 1e9 < 10
        out["frame_id"] = str(mtime)
    except OSError:
        pass
    return out


MAP_JS = r"""
const C={wall:'#F4F3EE',floor:'#16212F',grid:'#22324a',ice:'#3f8fe0',rough:'#8a6a4a',ramp:'#F2A65A',gap:'#05080c',
 spike:'#E8604C',start:'#F2A65A',goal:'#7AC943',robot:'#F2C14E',trail:'rgba(242,193,78,.55)'};
// the maze (walls, hazards, start / goal) + the robot (st: x, y, yaw, flash, trail) on canvas cv, square, centred
function drawMap(cv,maze,st,empty){
 const ctx=cv.getContext('2d'); ctx.setTransform(1,0,0,1,0,0); ctx.fillStyle='#0E1621'; ctx.fillRect(0,0,cv.width,cv.height);
 const S=Math.min(cv.width,cv.height),ox=(cv.width-S)/2,oy=(cv.height-S)/2;
 if(!maze){ ctx.fillStyle='#7d8da0'; ctx.font=`${S/26}px 'IBM Plex Sans Thai',sans-serif`; ctx.textAlign='center';
  ctx.fillText(empty||'',cv.width/2,cv.height/2); return; }
 const n=maze.n,c=maze.cell,pad=S*0.03,k=(S-2*pad)/(n*c);
 const X=x=>ox+pad+(x+c/2)*k, Y=y=>oy+S-pad-(y+c/2)*k;           // maze frame: cell (i,j) centred at (i*c, j*c)
 const cell=(i,j,col,inset=0.04)=>{ ctx.fillStyle=col; ctx.fillRect(X(i*c-c/2+inset*c),Y(j*c+c/2-inset*c),(1-2*inset)*c*k,(1-2*inset)*c*k); };
 for(let i=0;i<n;i++)for(let j=0;j<n;j++) cell(i,j,C.floor,0);
 ctx.strokeStyle=C.grid; ctx.lineWidth=1; for(let i=0;i<=n;i++){ ctx.beginPath(); ctx.moveTo(X(i*c-c/2),Y(-c/2)); ctx.lineTo(X(i*c-c/2),Y(n*c-c/2)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X(-c/2),Y(i*c-c/2)); ctx.lineTo(X(n*c-c/2),Y(i*c-c/2)); ctx.stroke(); }
 cell(maze.start[0],maze.start[1],'rgba(242,166,90,.35)'); cell(maze.goal[0],maze.goal[1],'rgba(122,201,67,.45)');
 (maze.ice||[]).forEach(([i,j])=>cell(i,j,'rgba(63,143,224,.45)',0.08));
 (maze.rough||[]).forEach(([i,j])=>{ cell(i,j,'rgba(138,106,74,.22)',0.04); ctx.fillStyle=C.rough;
  [[-.25,-.2],[.2,-.28],[0,.05],[-.28,.27],[.27,.22]].forEach(([u,v])=>ctx.fillRect(X(i*c+u*c),Y(j*c+v*c),0.07*c*k,0.07*c*k)); });
 (maze.ramp||[]).forEach(([i,j,a,h])=>{ const g=a===0?ctx.createLinearGradient(X(i*c-c/2),0,X(i*c+c/2),0):ctx.createLinearGradient(0,Y(j*c-c/2),0,Y(j*c+c/2));
  g.addColorStop(0,'rgba(242,166,90,0)'); g.addColorStop(.5,`rgba(242,166,90,${Math.min(.9,h*3)})`); g.addColorStop(1,'rgba(242,166,90,0)');
  ctx.fillStyle=g; ctx.fillRect(X(i*c-c/2),Y(j*c+c/2),c*k,c*k); });
 (maze.gap||[]).forEach(([i,j,a,w])=>{ ctx.fillStyle=C.gap; if(a===0) ctx.fillRect(X(i*c-w/2),Y(j*c+c/2),w*k,c*k); else ctx.fillRect(X(i*c-c/2),Y(j*c+w/2),c*k,w*k); });
 (maze.spike||[]).forEach(([i,j,a,x0,y0,x1,y1])=>{ ctx.fillStyle='rgba(232,96,76,.55)'; ctx.fillRect(X(x0),Y(y1),(x1-x0)*k,(y1-y0)*k);
  ctx.fillStyle=C.spike; for(let x=x0+.08;x<x1;x+=.15)for(let y=y0+.08;y<y1;y+=.15){ ctx.beginPath(); ctx.arc(X(x),Y(y),Math.max(1,.035*k),0,7); ctx.fill(); } });
 ctx.strokeStyle=C.wall; ctx.lineWidth=Math.max(2,0.1*k); ctx.lineCap='round';
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){ const m=maze.walls[i][j],x0=i*c-c/2,y0=j*c-c/2;
  const seg=(a,b,cc,d)=>{ ctx.beginPath(); ctx.moveTo(X(a),Y(b)); ctx.lineTo(X(cc),Y(d)); ctx.stroke(); };
  if(m&1) seg(x0+c,y0,x0+c,y0+c); if(m&2) seg(x0,y0+c,x0+c,y0+c); if(m&4) seg(x0,y0,x0,y0+c); if(m&8) seg(x0,y0,x0+c,y0); }
 ctx.font=`600 ${0.32*c*k}px Prompt,sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle';
 ctx.fillStyle=C.start; ctx.fillText('S',X(maze.start[0]*c),Y(maze.start[1]*c)); ctx.fillStyle=C.goal; ctx.fillText('G',X(maze.goal[0]*c),Y(maze.goal[1]*c));
 if(!st) return;
 ctx.strokeStyle=C.trail; ctx.lineWidth=Math.max(1.5,.04*k); ctx.beginPath(); let pen=false;
 (st.trail||[]).forEach(p=>{ if(!p){pen=false;return;} pen?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])); pen=true; });
 ctx.stroke();
 const rx=X(st.x),ry=Y(st.y),r=Math.max(5,.22*k);
 ctx.fillStyle=st.flash?'#E8604C':C.robot; ctx.beginPath(); ctx.arc(rx,ry,r,0,7); ctx.fill();
 ctx.strokeStyle='#0E1621'; ctx.lineWidth=Math.max(2,.06*k); ctx.beginPath(); ctx.moveTo(rx,ry);
 ctx.lineTo(rx+Math.cos(st.yaw)*r*1.1,ry-Math.sin(st.yaw)*r*1.1); ctx.stroke(); }
"""

LIVE_JS = MAP_JS + r"""
const $=id=>document.getElementById(id),cv=$('map'),mini=$('mini'),pip=$('pip'),img=$('cam3d'),stage=$('stage');
const Q=new URLSearchParams(location.search);
let maze=null,run='',st=null,team='',want=Q.get('team')||'',view='2d',frameId='',hasFrame=false,loading=false;
try{ view=localStorage.getItem('live_view')||'2d'; }catch(e){}
if(Q.get('view')==='2d'||Q.get('view')==='3d') view=Q.get('view');
const escT=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function setUrl(k,v){ const u=new URL(location.href); u.searchParams.set(k,v); history.replaceState(null,'',u); }
function teams(list){ $('teams').innerHTML=list.length>1?'<span class="u">Running now</span>'+list.map(t=>
  `<a class="q${t===team?' on':''}" href="?team=${encodeURIComponent(t)}" data-t="${escT(t)}">${escT(t)}</a>`).join(''):'';
 document.querySelectorAll('#teams a').forEach(a=>a.onclick=e=>{ e.preventDefault(); want=a.dataset.t; run=''; frameId='';
  setUrl('team',want); }); }
const isFs=()=>document.fullscreenElement===stage||stage.classList.contains('fsx');
function size(c,w,h){ const d=window.devicePixelRatio||1; c.style.width=w+'px'; c.style.height=h+'px';
 c.width=Math.round(w*d); c.height=Math.round(h*d); }
function fit(){
 if(view==='2d'){ let s;
  if(isFs()) s=Math.min(innerWidth,innerHeight);
  else { const r=stage.getBoundingClientRect(); s=Math.max(280,Math.min(r.width-24,innerHeight-Math.max(0,r.top+scrollY)-40)); }
  size(cv,s,s); }
 const mw=Math.max(200,mini.parentElement.clientWidth||270); size(mini,mw,mw);
 const ps=Math.round(Math.min(innerWidth,innerHeight)*0.3); size(pip,ps,ps); draw(); }
let empty='No run in progress';
function draw(){ const s=st&&maze&&st.run===maze.run?st:null;
 if(view==='2d'){ drawMap(cv,maze,s,empty); return; }
 drawMap(mini,maze,s,empty); if(isFs()) drawMap(pip,maze,s,''); }
function setView(v){ view=v; try{ localStorage.setItem('live_view',v); }catch(e){}
 stage.classList.toggle('v3d',v==='3d'); $('lvgrid').classList.toggle('v3d',v==='3d');
 document.querySelectorAll('.seg button').forEach(b=>b.classList.toggle('on',b.dataset.v===v));
 setUrl('view',v); frameId=''; fit(); }
function hud(){ const s=st&&maze&&st.run===maze.run?st:null;
 const tm=s?(s.goal!=null?`GOAL ${s.goal.toFixed(1)} s`:`${s.t.toFixed(1)} s`):'–';
 const col=s&&s.goal!=null?'var(--green)':s&&maze&&maze.limit-s.t<30?'var(--red)':'var(--text)';
 $('tm').textContent=$('ovtm').textContent=tm; $('tm').style.color=$('ovtm').style.color=col;
 $('tn').textContent=$('ovt').textContent=maze?maze.team:'–';
 $('lv').textContent=maze?`Level ${maze.level} · seed ${maze.seed} · ${maze.n}x${maze.n}`:'';
 $('dg').textContent=s?`${s.dist.toFixed(1)} m`:'–'; $('bs').textContent=s?`${s.best.toFixed(1)} m`:'–';
 $('fl').textContent=s?s.falls:'–'; $('rt').textContent=maze?`${maze.route_m.toFixed(0)} m`:'–';
 $('ovd').textContent=s?`${s.dist.toFixed(1)} m to go · ${s.falls} falls`:'';
 $('why').textContent=$('ovw').textContent=s&&s.flash?`${s.why} → back to the start`:''; $('res').textContent=s&&s.result?s.result:'';
 $('stt').textContent=s?(s.done?'Finished':'Running'):'Waiting for the next run'; }
function frame(d){  // 3D: a new camera frame when there is one (each frame has its own URL: Cloudflare caches it)
 hasFrame=!!d.frame; stage.classList.toggle('nof',!hasFrame);
 $('b3d').title=hasFrame?'':'3D frames come while a robot runs and its replay is recorded';
 $('wait3d').textContent=st&&!st.done?'Waiting for 3D frames…':empty;
 if(view!=='3d'||!hasFrame||!d.frame_id||d.frame_id===frameId||loading) return;
 loading=true; const id=d.frame_id,n=new Image();
 n.onload=()=>{ img.src=n.src; frameId=id; loading=false; }; n.onerror=()=>{ loading=false; };
 n.src=`/live.jpg?team=${encodeURIComponent(d.team)}&f=${id}`; }
async function tick(){ try{ const r=await fetch(`/live.json?team=${encodeURIComponent(want)}&since=${encodeURIComponent(run)}`,{cache:'no-store'});
  if(r.ok){ const d=await r.json(); if(d.team!==team) frameId=''; team=d.team; teams(d.running);
  const starting=d.running.includes(d.team)&&(!d.state||d.state.done);  // the simulator is still loading
  if(starting){ maze=null; run=''; empty=`Preparing the maze for ${d.team} (about a minute)`; }
  else if(!d.state){ maze=null; run=''; empty='No run in progress'; }
  if(d.maze&&!starting){ maze=d.maze; run=maze.run; } st=starting?null:d.state;
  frame(d); draw(); hud(); } }catch(e){} setTimeout(tick,st&&!st.done?250:2000); }
document.querySelectorAll('.seg button').forEach(b=>b.onclick=()=>setView(b.dataset.v));
$('fsbtn').onclick=()=>{
 if(isFs()){ if(document.fullscreenElement) document.exitFullscreen(); else { stage.classList.remove('fsx'); fsDone(); } return; }
 const go=stage.requestFullscreen||stage.webkitRequestFullscreen;
 const css=()=>{ stage.classList.add('fsx'); fsDone(); };  // no element fullscreen (iPhone): fill the window
 const p=go?go.call(stage):null; if(p&&p.catch) p.catch(css); else if(!go) css(); };
function fsDone(){ $('fsbtn').textContent=isFs()?'✕ Exit full screen':'⛶ Full screen'; fit(); }
document.addEventListener('fullscreenchange',fsDone);
addEventListener('keydown',e=>{ if(/INPUT|TEXTAREA/.test(e.target.tagName)) return;
 if(e.key==='f'||e.key==='F') $('fsbtn').click(); else if(e.key==='2') setView('2d'); else if(e.key==='3') setView('3d');
 else if(e.key==='Escape'&&stage.classList.contains('fsx')){ stage.classList.remove('fsx'); fsDone(); } });
addEventListener('resize',fit); setView(view); tick();
"""

LIVE_CSS = """<style>.wrap{max-width:1400px} .lv{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:20px;align-items:start}
.viewbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:0 0 12px}
.seg{display:inline-flex;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:3px}
.seg button{margin:0;padding:8px 18px;font-size:.95rem;background:transparent;color:var(--soft);border-radius:999px}
.seg button.on{background:var(--green);color:#0E1621} #fsbtn{margin:0;padding:9px 18px;font-size:.95rem;background:#24344a;color:var(--text)}
#stage{position:relative;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:12px;display:flex;
justify-content:center;align-items:center;min-height:300px} #stage canvas#map{display:block;border-radius:10px}
#cam3d{display:none;width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:10px}
#wait3d{display:none;position:absolute;inset:0;align-items:center;justify-content:center;color:var(--soft);font-size:1.2rem}
#stage.v3d canvas#map{display:none} #stage.v3d #cam3d{display:block} #stage.v3d.nof #wait3d{display:flex}
#stage.v3d.nof #cam3d{visibility:hidden} #pip,#ov{display:none}
#stage:fullscreen,#stage.fsx{background:#000;border:0;border-radius:0;padding:0}
#stage.fsx{position:fixed;inset:0;z-index:50;margin:0}
#stage:fullscreen #cam3d,#stage.fsx #cam3d{width:100vw;height:100vh;aspect-ratio:auto;border-radius:0}
#stage:fullscreen.v3d #pip,#stage.fsx.v3d #pip{display:block;position:absolute;right:18px;bottom:18px;border-radius:12px;
box-shadow:0 0 0 2px rgba(255,255,255,.25)}
#stage:fullscreen:not(.v3d) #ov,#stage.fsx:not(.v3d) #ov{display:block;position:absolute;left:20px;top:20px;
background:rgba(14,22,33,.85);border-radius:14px;padding:12px 20px;min-width:220px}
#ov b{font-family:Prompt,sans-serif;font-size:1.3rem} #ovtm{font-family:Prompt,sans-serif;font-size:2.4rem;display:block}
#ovw{color:var(--orange);font-weight:600} .fsclose{display:none} #stage.fsx .fsclose{display:block;position:absolute;right:16px;
top:16px;z-index:2;margin:0;padding:8px 14px;background:rgba(14,22,33,.85);color:var(--text)}
#minibox{display:none;margin-top:14px} .lv.v3d #minibox{display:block} #mini{display:block;border-radius:10px}
.stat{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--line)} .stat span:first-child{color:var(--soft)}
#tm{font-family:Prompt,sans-serif;font-size:2.6rem;margin:6px 0} #why{color:var(--orange);min-height:1.4em;font-weight:600}
#res{color:var(--green);font-weight:600} .lg{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.85rem;color:var(--soft);margin-top:12px}
.lg i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
#teams a.q{text-decoration:none} #teams a.q.on{background:#1f4d06;color:var(--text)}
@media(max-width:900px){.lv{grid-template-columns:1fr}}</style>"""


def live_page():
    data = board_data()
    return (HEAD.format(title="Live · Challenge 7.5", cls="") + LIVE_CSS
            + '<div class="top"><div><div class="eyebrow">Challenge 7.5 · Live</div><h1 id="tn">–</h1>'
            + '<div class="u" id="lv"></div></div>' + chips(data) + "</div>"
            + '<div class="queue" id="teams"></div>'
            + '<div class="viewbar"><div class="seg"><button data-v="2d">2D map</button><button data-v="3d" id="b3d">3D camera</button></div>'
            + '<button id="fsbtn">⛶ Full screen</button><span class="u">Keys: 2 / 3 switch view, F full screen</span></div>'
            + '<div class="lv" id="lvgrid"><div id="stage"><canvas id="map"></canvas><img id="cam3d" alt="3D camera">'
            + '<div id="wait3d"></div><canvas id="pip"></canvas>'
            + '<div id="ov"><b id="ovt"></b><span id="ovtm"></span><span class="u" id="ovd"></span><div id="ovw"></div></div>'
            + '<button class="fsclose" onclick="document.getElementById(\'fsbtn\').click()">✕</button></div>'
            + '<div class="card" style="margin:0">'
            + '<div class="u" id="stt">–</div><div id="tm">–</div><div id="why"></div>'
            + '<div class="stat"><span>To the goal</span><b id="dg">–</b></div>'
            + '<div class="stat"><span>Closest so far</span><b id="bs">–</b></div>'
            + '<div class="stat"><span>Falls</span><b id="fl">–</b></div>'
            + '<div class="stat"><span>Shortest route</span><b id="rt">–</b></div><div id="res"></div>'
            + '<div id="minibox"><canvas id="mini"></canvas></div>'
            + '<div class="lg"><span><i style="background:#7AC943"></i>Goal</span><span><i style="background:#F2A65A"></i>Start / ramp</span>'
            + '<span><i style="background:#E8604C"></i>Spike bed</span><span><i style="background:#3f8fe0"></i>Ice</span>'
            + '<span><i style="background:#05080c;border:1px solid #33465c"></i>Trench</span><span><i style="background:#8a6a4a"></i>Rough floor</span></div>'
            + '</div></div><p class="u">The map is drawn from the robot\'s position in the actual simulation; the 3D camera is the simulator\'s own renderer '
            + '(it updates as fast as the simulation runs). Full clips after the run: <a style="color:var(--green)" href="/replay">/replay</a></p>'
            + "<script>" + LIVE_JS + "</script></div></body></html>")


# ----------------------------------------------------------------------------- admin (instructor)
def _admin_token():
    return hashlib.sha256(f"maze-admin:{ADMIN_KEY}".encode()).hexdigest()


def admin_page(note=""):
    rnd = round_cfg()
    st = load_status()
    teams = sorted(d for d in os.listdir(SUBMISSIONS) if os.path.isdir(os.path.join(SUBMISSIONS, d)) and d[0] != "_")
    opts = "".join(f'<option value="{k}"{" selected" if k == rnd["level"] else ""}>{k} ({v["cells"]} x {v["cells"]}, '
                   f'{v["time"]} s{", random start/goal" if v["random_ends"] else ""})</option>' for k, v in LEVELS.items())
    rows = []
    for t in sorted(teams, key=lambda t: st.get(t, {}).get("uploaded_at", 0)):
        s = st.get(t, {})
        files = sorted(os.listdir(os.path.join(SUBMISSIONS, t)))
        links = " ".join(f'<a style="color:var(--green)" href="/admin/file?team={t}&name={f}">{html.escape(f)}</a>'
                         for f in files if f.endswith(".py"))
        rows.append(
            f'<tr><td>{html.escape(t)}</td><td><span class="badge b-{s.get("state", "waiting")}">{s.get("state", "waiting")}</span></td>'
            f'<td><small class="u">{html.escape(str(s.get("msg", "")))[:120]}</small></td><td>{s.get("uploaded", "")}</td>'
            f'<td>{links} <a style="color:var(--soft)" href="/admin/log?team={t}">log</a></td>'
            f'<td><form method="post" action="/admin" style="display:inline"><input type="hidden" name="team" value="{t}">'
            f'<button name="action" value="rerun" style="margin:0;padding:5px 12px;font-size:.85rem">Re-run</button> '
            + (f'<button name="action" value="reset_pin" style="margin:0;padding:5px 12px;font-size:.85rem;background:#24344a;color:var(--soft)" '
               f'onclick="return confirm(\'Reset the PIN of team {t}? Its next upload sets a new PIN\')">Reset PIN</button> '
               if TEAM_PIN else "")
            + f'<button name="action" value="delete" style="margin:0;padding:5px 12px;font-size:.85rem;background:#4a2323;color:#ffb3a6" '
            f'onclick="return confirm(\'Delete team {t} and all its results?\')">Delete</button></form></td></tr>')
    return (HEAD.format(title="Admin · Challenge 7.5", cls="")
            + '<div class="top"><div><div class="eyebrow">Instructor</div><h1>Round settings</h1></div></div>'
            + (f'<div class="msg">{html.escape(note)}</div>' if note else "")
            + '<div class="card"><form method="post" action="/admin"><input type="hidden" name="action" value="round">'
            + '<label>Level</label><select name="level" style="padding:10px;border-radius:10px;background:var(--bg);color:var(--text);'
            + f'border:1px solid #33465c">{opts}</select>'
            + f'<label>seed (empty = show every seed)</label><input type="text" name="seed" value="{html.escape(rnd["seed"])}" pattern="[0-9]*">'
            + f'<label><input type="checkbox" name="open" value="1"{" checked" if rnd["open"] else ""}> Uploads open</label>'
            + '<div class="hint">Changing the seed or the level queues every team that has not run on it yet. '
            + 'For the real round, close uploads first, then enter the announced seed.</div>'
            + '<button type="submit">Save</button></form></div>'
            + '<div class="card"><form method="post" action="/admin" style="float:right"><button name="action" value="rerun_all" '
            + 'style="margin:0" onclick="return confirm(\'Re-run every team on this round?\')">Re-run all teams</button></form>'
            + f'<h3>Teams ({len(teams)})</h3><table><tr><th>Team</th><th>Status</th><th>Message</th><th>Uploaded</th><th>Files</th><th></th></tr>'
            + "".join(rows) + "</table></div></div></body></html>")


def admin_action(form):
    action = form.get("action", "")
    if action == "round":
        level = form.get("level", LEVEL)
        seed = form.get("seed", "").strip()
        if level not in LEVELS or not re.fullmatch(r"\d{0,9}", seed):
            return "invalid level or seed"
        with status_lock():
            _write_json("round.json", {"level": level, "seed": seed, "open": form.get("open") == "1"})
        print(f"[ADMIN] round: level {level} seed {seed or '-'} open {form.get('open') == '1'}", flush=True)
        return f"Saved: level {level}, seed {seed or '-'}, uploads {'open' if form.get('open') == '1' else 'closed'}"
    team = form.get("team", "")
    if action in ("rerun", "rerun_all"):
        with status_lock():
            st = load_status()
            names = [t for t in st if t == team] if action == "rerun" else list(st)
            for t in names:
                if st[t].get("state") != "running":
                    st[t].update(state="waiting", msg="")
            save_status(st)
        return f"Queued again: {', '.join(names) or '-'}"
    if action == "reset_pin" and re.fullmatch(TEAM_RE, team):  # a team forgot its PIN: results and files stay
        with status_lock():
            pins = _read_json("_teams.json", {})
            pins.pop(team, None)
            _write_json("_teams.json", pins)
        print(f"[ADMIN] PIN reset: {team}", flush=True)
        return f"PIN of team {team} reset: its next upload sets a new PIN"
    if action == "delete" and re.fullmatch(TEAM_RE, team):
        with status_lock():
            st = load_status()
            if st.get(team, {}).get("state") == "running":
                return f"team {team} is running: it cannot be deleted now"
            st.pop(team, None)
            save_status(st)
            pins = _read_json("_teams.json", {})
            pins.pop(team, None)
            _write_json("_teams.json", pins)
            shutil.rmtree(os.path.join(SUBMISSIONS, team), ignore_errors=True)
            for old in glob.glob(os.path.join(SUBMISSIONS, "_replays", f"{team}_*")):
                os.remove(old)
            path = os.path.join(SUBMISSIONS, "leaderboard.csv")
            if os.path.exists(path):
                with open(path) as f:
                    rows = list(csv.reader(f))
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerows([r for k, r in enumerate(rows) if k == 0 or (r and r[0] != team)])
        print(f"[ADMIN] deleted team {team}", flush=True)
        return f"Deleted team {team}"
    return "unknown action"


# ----------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def _send(self, body, status=200, ctype="text/html; charset=utf-8", headers=(), cache="no-store"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", cache)
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def _json(self, data):
        return self._send(json.dumps(data, ensure_ascii=False), ctype="application/json")

    def _is_admin(self):
        if not ADMIN_KEY:
            return False
        m = re.search(r"(?:^|;\s*)maze_admin=([0-9a-f]{64})", self.headers.get("Cookie", ""))
        return bool(m) and hmac.compare_digest(m.group(1), _admin_token())

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = {k: v[-1] for k, v in parse_qs(query).items()}
        if "part=1" in query:
            return self._send(board_part())
        if path == "/board":
            return self._send(board_page())
        if path == "/replay":
            return self._send(replay_page(query))
        if path.startswith("/replays/") and path.endswith(".trace.json"):
            name = path[len("/replays/"):]
            if not re.fullmatch(TEAM_RE + r"(_L\w+)?_seed\d+\.trace\.json", name):
                return self._send("not found", 404, "text/plain")
            try:
                with open(os.path.join(SUBMISSIONS, "_replays", name), "rb") as f:
                    return self._send(f.read(), ctype="application/json")
            except OSError:
                return self._send("not found", 404, "text/plain")
        if path.startswith("/replays/"):
            return self._video(path[len("/replays/"):])
        if path == "/board.json":
            return self._json(board_data())
        if path == "/live":
            return self._send(live_page())
        if path == "/live.json":
            return self._json(live_data(q.get("team", ""), q.get("since", "")))
        if path == "/live.jpg":
            team, _ = _live_team(q.get("team", ""))
            # ?f=<frame id>: one URL per frame, so Cloudflare can serve it to every viewer from one download
            cache = "public, max-age=60, immutable" if re.fullmatch(r"\d{1,24}", q.get("f", "")) else "no-store"
            try:
                with open(os.path.join(LIVE_DIR, team, "frame.jpg"), "rb") as f:
                    return self._send(f.read(), ctype="image/jpeg", cache=cache)
            except OSError:
                return self._send("no frame", 404, "text/plain")
        if path.startswith("/admin"):
            return self._admin_get(path, q)
        return self._send(student_page())

    def _admin_get(self, path, q):
        if not ADMIN_KEY:
            return self._send("not found", 404, "text/plain")
        if "key" in q:  # /admin?key=... -> cookie, then a clean URL (the key does not stay in the address bar)
            if not hmac.compare_digest(q["key"], ADMIN_KEY):
                return self._send("wrong key", 403, "text/plain")
            secure = "; Secure" if self.headers.get("X-Forwarded-Proto") == "https" else ""
            return self._send("", 303, headers=[("Location", "/admin"), (
                "Set-Cookie", f"maze_admin={_admin_token()}; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000{secure}")])
        if not self._is_admin():
            return self._send("open /admin?key=... first", 403, "text/plain; charset=utf-8")
        if path == "/admin":
            return self._send(admin_page())
        team = q.get("team", "")
        if not re.fullmatch(TEAM_RE, team):
            return self._send("bad team", 400, "text/plain")
        if path == "/admin/file":  # read a pilot before running it
            name = q.get("name", "")
            if not re.fullmatch(HELPER_RE, name):
                return self._send("bad name", 400, "text/plain")
            try:
                with open(os.path.join(SUBMISSIONS, team, name), "rb") as f:
                    return self._send(f.read(), ctype="text/plain; charset=utf-8")
            except OSError:
                return self._send("not found", 404, "text/plain")
        if path == "/admin/log":
            logs = sorted(glob.glob(os.path.join(SUBMISSIONS, "_logs", f"{team}_L*.log")), key=os.path.getmtime)
            if not logs:
                return self._send("no log yet", 404, "text/plain")
            with open(logs[-1], "rb") as f:
                return self._send(f.read()[-200_000:], ctype="text/plain; charset=utf-8")
        return self._send("not found", 404, "text/plain")

    def _video(self, name):
        """mp4 with Range support (the browser seeks and changes speed smoothly)."""
        if not re.fullmatch(TEAM_RE + r"(_L\w+)?_seed\d+\.mp4", name):
            return self._send("not found", 404)
        path = os.path.join(SUBMISSIONS, "_replays", name)
        if not os.path.exists(path):
            return self._send("not found", 404)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(size - int(m.group(2)), 0)
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=86400")  # the page asks for ?v=<mtime>: a re-run = new URL
        self.send_header("Content-Length", str(end - start + 1))
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _error(self, text, status=400):
        return self._send(student_page(f'<div class="msg err">✗ {html.escape(text)}</div>'), status)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        if self.path == "/admin":
            if not self._is_admin() or size > 10_000:
                return self._send("forbidden", 403, "text/plain")
            form = {k: v[-1] for k, v in parse_qs(self.rfile.read(size).decode(errors="ignore")).items()}
            return self._send(admin_page(admin_action(form)))
        if self.path != "/upload":
            return self._error("not found", 404)
        if size > MAX_BYTES:
            return self._error("the upload is larger than 60 MB")
        if not round_cfg()["open"]:
            self.rfile.read(size)
            return self._error("uploads are closed")
        raw = b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + self.rfile.read(size)
        msg = BytesParser(policy=HTTP).parsebytes(raw)
        fields, helpers = {}, []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            item = (part.get_filename(), part.get_payload(decode=True) or b"")
            if name == "helpers":
                if item[0]:
                    helpers.append(item)
            else:
                fields[name] = item
        team = fields.get("team", (None, b""))[1].decode(errors="ignore").strip()
        if not re.fullmatch(TEAM_RE, team):
            return self._error("team name: letters, digits, _ or -, 2-32 characters")
        if UPLOAD_CODE:
            code = fields.get("code", (None, b""))[1].decode(errors="ignore").strip()
            if not hmac.compare_digest(code.encode(), UPLOAD_CODE.encode()):
                return self._error("wrong class code")
        pin = fields.get("pin", (None, b""))[1].decode(errors="ignore").strip()
        if TEAM_PIN:
            if not re.fullmatch(r"[0-9A-Za-z]{4,12}", pin):
                return self._error("the PIN must be 4-12 letters or digits")
            known = _read_json("_teams.json", {}).get(team)
            if known and not hmac.compare_digest(known, _pin_hash(team, pin)):
                return self._error(f"team {team} is taken and the PIN does not match (your team but you forgot the PIN? ask the instructor to reset it)")
        policy = fields.get("policy", (None, b""))[1]
        err = check_policy(policy)
        if err:
            return self._error(err)
        pilot = fields.get("pilot", (None, b""))[1]
        if pilot:
            err = check_python("pilot.py", pilot)
            if err:
                return self._error(err)
        if len(helpers) > 5:
            return self._error("at most 5 helper files")
        for fname, data in helpers:
            fname = os.path.basename(fname)
            if not re.fullmatch(HELPER_RE, fname) or fname == "pilot.py":
                return self._error(f"file name {fname} is not allowed (letters, digits and _, ending in .py, not pilot.py)")
            err = check_python(fname, data)
            if err:
                return self._error(err)
        if load_status().get(team, {}).get("state") == "running":
            return self._error(f"team {team} is running right now: upload again when this run has finished")
        folder = os.path.join(SUBMISSIONS, team)
        warns = code_warnings({**({"pilot.py": pilot} if pilot else {}),
                               **{os.path.basename(f): d for f, d in helpers}})
        if not pilot and os.path.exists(os.path.join(folder, "pilot.py")):
            warns.append("no pilot.py in this upload: it replaces your earlier files, so the default autopilot drives "
                         "this run (upload pilot.py again to keep yours)")
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):  # a new upload replaces everything (also exported/ and __pycache__/ of old runs)
            path = os.path.join(folder, old)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        saved = [("policy.pt", policy)] + ([("pilot.py", pilot)] if pilot else []) + \
            [(os.path.basename(f), d) for f, d in helpers]
        for fname, data in saved:
            with open(os.path.join(folder, fname), "wb") as f:
                f.write(data)
        for old in glob.glob(os.path.join(SUBMISSIONS, "_replays", f"{team}_seed*.*")) + glob.glob(
                os.path.join(SUBMISSIONS, "_replays", f"{team}_L*_seed*.*")):  # replays of the old upload
            os.remove(old)
        with status_lock():
            st = load_status()
            st[team] = {"state": "waiting", "msg": "", "uploaded": time.strftime("%H:%M:%S"), "uploaded_at": time.time()}
            save_status(st)
            if TEAM_PIN:
                pins = _read_json("_teams.json", {})
                pins.setdefault(team, _pin_hash(team, pin))
                _write_json("_teams.json", pins)
        names = ", ".join(n for n, _ in saved)
        print(f"[UPLOAD] {team}: {names} ({len(policy) / 1e6:.1f} MB)", flush=True)
        warn_html = "".join(f'<br><span class="wr">⚠ {html.escape(w)}</span>' for w in warns)
        return self._send(student_page(f'<div class="msg">✓ Uploaded for team <b>{html.escape(team)}</b>: {html.escape(names)} '
                                       f'<br><small class="u">It runs automatically in queue order; the result appears in the table below</small>'
                                       f'{warn_html}</div>'))

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    os.makedirs(SUBMISSIONS, exist_ok=True)
    print(f"[UPLOAD] students: http://{HOST}:{PORT}   projector: http://{HOST}:{PORT}/board   -> "
          f"{os.path.abspath(SUBMISSIONS)}", flush=True)
    print(f"[UPLOAD] class code {'on' if UPLOAD_CODE else 'off'}, team PIN {'on' if TEAM_PIN else 'off'}, "
          f"admin {'/admin?key=...' if ADMIN_KEY else 'off'}, round {round_cfg()}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
