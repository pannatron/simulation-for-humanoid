"""Challenge 7.5 upload page + leaderboard (standard library + torch for the policy check).

    uv run python scripts/challenge_server.py           # students:  http://<this-machine-ip>:8000
                                                        # projector: http://<this-machine-ip>:8000/board
    CHALLENGE_PORT=8080 CHALLENGE_SEED=13 uv run python scripts/challenge_server.py

Teams upload policy.pt (required), pilot.py (optional) and helper .py files their pilot imports. Each upload is
checked: the policy must have the same network as G1 Rough, every .py file must compile. Files land in
submissions/<team>/ (a new upload replaces the old one). The instructor runs the queue with
scripts/challenge_runner.py, which updates submissions/status.json; results go to submissions/leaderboard.csv.
Runs are played in upload order (the queue on the pages). Once the runner has recorded them, /replay plays every
team's run on the projector, from the last place up to the winner.

Note: pilot.py runs on this machine. Use it on a classroom network only, and read a pilot before running it.
"""

import csv
import glob
import html
import json
import os
import re
import shutil
import time
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS = os.environ.get("CHALLENGE_DIR", os.path.join(HERE, "..", "submissions"))
PORT = int(os.environ.get("CHALLENGE_PORT", 8000))
SEED = os.environ.get("CHALLENGE_SEED", "")  # announced seed of the round (shown on the pages)
LIMIT = os.environ.get("CHALLENGE_TIME", "300")
MAX_BYTES = 60 * 1024 * 1024
REF_POLICY = os.path.join(HERE, "..", "pretrained", "g1_rough", "model_pretrained.pt")
TEAM_RE = r"[A-Za-z0-9][A-Za-z0-9_\-]{1,31}"
HELPER_RE = r"[A-Za-z][A-Za-z0-9_]{0,40}\.py"


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
        return "policy.pt ว่างหรือไม่ได้แนบมา"
    if REF_SHAPES is None:
        return None
    import io

    try:
        shapes = _actor_shapes(io.BytesIO(data))
    except Exception as e:
        return f"เปิด policy.pt ไม่ได้ ต้องเป็นไฟล์ model_*.pt ของ rsl_rl ({str(e)[:80]})"
    first = next(iter(shapes.items()), None)
    if shapes != REF_SHAPES:
        ref_first = next(k for k in REF_SHAPES if k.endswith("0.weight"))
        got = shapes.get(ref_first)
        return (f"policy นี้ network ไม่ตรงกับ G1 Rough: ชั้นแรกรับ {got[1] if got else '?'} ค่า "
                f"แต่ด่านนี้ต้องรับ {REF_SHAPES[ref_first][1]} ค่า (เทรนจาก task G1 Rough หรือ G1-Maze แล้วหรือยัง?)"
                if first else "policy นี้ไม่มี network ข้างใน")
    return None


def check_python(name, data: bytes):
    try:
        compile(data.decode("utf-8"), name, "exec")
    except SyntaxError as e:
        return f"{name} บรรทัด {e.lineno}: {e.msg}"
    except UnicodeDecodeError:
        return f"{name} ต้องเป็นไฟล์ข้อความ UTF-8"
    if name == "pilot.py" and not re.search(rb"^class\s+Pilot\b", data, re.M):
        return "pilot.py ต้องมี class Pilot"
    return None


# ----------------------------------------------------------------------------- data
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


def board_data():
    """Best run per team (GOAL fastest first, then DNF closest to the goal, then fewer falls) + every team's status."""
    rows = []
    path = os.path.join(SUBMISSIONS, "leaderboard.csv")
    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.DictReader(f))
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
        status[t] = s if not SEED or str(s.get("seed", SEED)) == SEED or s.get("state") == "running" else {"state": "waiting"}
    running = next((t for t, s in st.items() if s.get("state") == "running"), None)
    queue = sorted((t for t, s in status.items() if s.get("state") == "waiting"),
                   key=lambda t: (st.get(t, {}).get("uploaded_at", 0), t))  # same order as challenge_runner.py
    return {"ranked": ranked, "status": status, "running": running, "queue": queue, "replays": replays(),
            "seed": SEED, "limit": LIMIT, "updated": time.strftime("%H:%M:%S")}


def replays():
    """{team: caption} of the recorded runs of this round (written by challenge_runner.py)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(SUBMISSIONS, "_replays", "*.json"))):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if (not SEED or str(r.get("seed")) == SEED) and os.path.exists(os.path.join(SUBMISSIONS, "_replays", r["video"])):
            out[r["team"]] = r
    return out


# ----------------------------------------------------------------------------- pages
HEAD = """<!doctype html><html lang="th"><head><meta charset="utf-8">
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
.msg.err{{background:#3a1f1f;border-color:#6b2f2f}} .grid2{{display:grid;grid-template-columns:1.1fr 1fr;gap:18px}}
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
</style></head><body><div class="wrap {cls}">"""


def podium_html(ranked):
    slots = {1: ("🥇", "p1"), 2: ("🥈", "p2"), 3: ("🥉", "p3")}
    cards = []
    for place in (2, 1, 3):
        medal, cls = slots[place]
        r = next((x for x in ranked if x["rank"] == place), None)
        if r is None:
            cards.append(f'<div class="p {cls} empty"><div class="medal">{medal}</div><div class="name">–</div></div>')
            continue
        score = f'{r["time"]} s' if r["goal"] else f'เหลือ {r["left"]} m'
        cards.append(f'<div class="p {cls}"><div class="medal">{medal}</div><div class="name">{html.escape(r["team"])}</div>'
                     f'<div class="score">{score}</div><small class="u">ล้ม {r["falls"]} ครั้ง</small></div>')
    return '<div class="podium">' + "".join(cards) + "</div>"


def badge(team, state, data):
    labels = {"waiting": "รอรัน", "running": "กำลังรัน", "done": "รันแล้ว", "error": "error"}
    if state == "waiting" and team in data["queue"]:
        return f'<span class="badge b-waiting">คิวที่ {data["queue"].index(team) + 1}</span>'
    b = f'<span class="badge b-{state}">{labels.get(state, state)}</span>'
    if team in data["replays"]:
        b += f' <a class="play" href="/replay?team={html.escape(team)}" title="ดูรีเพลย์">▶</a>'
    return b


def table_html(data):
    st = data["status"]
    rows = []
    seen = set()
    for r in data["ranked"]:
        seen.add(r["team"])
        res = (f'<span class="ok">{r["time"]} s</span>' if r["goal"]
               else f'<span class="dnf">ไม่ถึง เหลือ {r["left"]} m</span>')
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
        rows.append(f'<tr><td class="rank">–</td><td>{html.escape(team)}</td><td><small class="u">{note or "ยังไม่มีผล"}</small></td>'
                    f'<td></td><td>0</td><td>{badge(team, state, data)}</td></tr>')
    if not rows:
        return '<small class="u">ยังไม่มีทีมส่งงาน</small>'
    return ("<table><tr><th>#</th><th>ทีม</th><th>ผล</th><th>ล้ม</th><th>รอบที่รัน</th><th>สถานะ</th></tr>"
            + "".join(rows) + "</table>")


def live_html(data):
    out = ""
    if data["running"]:
        out += f'<div class="live"><span class="dot"></span><b>กำลังรัน:</b> {html.escape(data["running"])}</div>'
    if data["queue"]:
        names = "".join(f'<span class="q"><b>{k}</b> {html.escape(t)}</span>' for k, t in enumerate(data["queue"], 1))
        out += f'<div class="queue"><span class="u">คิวถัดไป</span>{names}</div>'
    if data["replays"] and not data["running"] and not data["queue"]:
        out += ('<a class="replaybar" href="/replay">▶ ดูรีเพลย์ทุกทีม เรียงจากอันดับท้ายขึ้นไปหาแชมป์ '
                f'({len(data["replays"])} ทีม)</a>')
    return out


def chips(data):
    seed = f'<span>seed <b>{html.escape(data["seed"])}</b></span>' if data["seed"] else ""
    return (f'<div class="chips">{seed}<span>เวลาสูงสุด <b>{data["limit"]} วินาที</b></span>'
            f'<span>อัปเดต <b id="upd">{data["updated"]}</b></span></div>')


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
@media(max-width:900px){.stage{grid-template-columns:1fr}}
</style>"""


def replay_page(query=""):
    """Projector: every team's recorded run, from the last place up to the winner (the result shows at the end)."""
    data = board_data()
    order = [r for r in reversed(data["ranked"]) if r["team"] in data["replays"]]
    order += [{"rank": None, "team": t} for t in sorted(data["replays"]) if t not in {r["team"] for r in order}]
    one = re.search(r"team=(" + TEAM_RE + ")", query)
    items = []
    for r in order:
        rp = data["replays"][r["team"]]
        items.append({"team": r["team"], "rank": r["rank"], "video": "/replays/" + rp["video"],
                      "result": rp["result"], "goal": rp["result"].startswith("GOAL"),
                      "board": (f'{r["time"]} s' if r.get("goal") else f'เหลือ {r["left"]} m') if r["rank"] else ""})
    start = next((k for k, it in enumerate(items) if one and it["team"] == one.group(1)), 0)
    body = ('<div class="top"><div><div class="eyebrow">Challenge 7.5 Replay</div><h1>รีเพลย์</h1></div>'
            + chips(data) + "</div>")
    if not items:
        return (HEAD.format(title="Replay · Challenge 7.5", cls="replay big") + body
                + '<div class="card">ยังไม่มีรีเพลย์ ครูรัน <code>challenge_runner.py</code> ก่อน แล้วคลิปของแต่ละทีมจะขึ้นที่นี่</div>'
                + '<a style="color:var(--green)" href="/board">กลับไป /board</a></div></body></html>')
    return (HEAD.format(title="Replay · Challenge 7.5", cls="replay") + REPLAY_CSS + body
            + '<div class="stage"><div><div class="screen"><video id="v" muted playsinline></video>'
            + '<div class="tag"><div class="n" id="tn"></div><div class="k" id="tk"></div></div>'
            + '<div class="intro" id="intro"><div class="k" id="ik"></div><div class="nm" id="inm"></div></div>'
            + '<div class="reveal" id="rev"><div class="medal" id="rm"></div><div class="rk" id="rk"></div>'
            + '<div class="nm" id="rn"></div><div class="rs" id="rs"></div><div class="u" id="rx"></div></div></div>'
            + '<div class="ctrl"><button id="prev">◀ ก่อนหน้า</button><button id="pp">⏸ หยุด</button>'
            + '<button id="next">ถัดไป ▶</button><span class="sp"></span><span class="u">ความเร็ว</span>'
            + '<button data-s="1">1x</button><button data-s="2">2x</button><button data-s="4">4x</button>'
            + '<button data-s="8">8x</button><button id="auto">เล่นต่ออัตโนมัติ</button></div></div>'
            + '<div class="list" id="list"></div></div>'
            + '<p class="u">เรียงจากอันดับท้ายขึ้นไปหาแชมป์ ผลของแต่ละทีมเฉลยตอนจบคลิป คลิปคือรอบล่าสุดที่ครูรัน '
            + '(ตัวเลขบนคลิปคือเวลาในด่านจริง ไม่ขึ้นกับความเร็วที่เปิดดู) '
            + '<a style="color:var(--green)" href="/board">กลับไป /board</a></p>'
            + "<script>const ITEMS=" + json.dumps(items, ensure_ascii=False) + f";let cur={start};"
            + r"""
const v=document.getElementById('v'),$=id=>document.getElementById(id);
let speed=+(localStorage.getItem('rs')||2),auto=localStorage.getItem('ra')!=='0',seen=new Set(),timer=null;
const medal=r=>r===1?'🥇':r===2?'🥈':r===3?'🥉':'';
const esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function list(){ $('list').innerHTML=ITEMS.map((it,k)=>`<div class="item ${k===cur?'cur':''} ${seen.has(k)?'':'hid'}" data-k="${k}">
 <span class="r">${seen.has(k)&&it.rank?it.rank:'?'}</span><div class="t">${esc(it.team)}<div class="s">${esc(it.board||it.result)}</div></div></div>`).join('');
 document.querySelectorAll('.item').forEach(e=>e.onclick=()=>load(+e.dataset.k)); }
function ui(){ document.querySelectorAll('[data-s]').forEach(b=>b.classList.toggle('on',+b.dataset.s===speed));
 $('auto').classList.toggle('on',auto); $('pp').textContent=v.paused?'▶ เล่น':'⏸ หยุด'; }
function load(k){ clearTimeout(timer); cur=(k+ITEMS.length)%ITEMS.length; const it=ITEMS[cur];
 $('rev').classList.remove('on'); $('tn').textContent=it.team; $('tk').textContent=`คลิปที่ ${cur+1} จาก ${ITEMS.length}`;
 $('ik').textContent=cur===ITEMS.length-1?'ทีมสุดท้าย':`ทีมที่ ${cur+1} จาก ${ITEMS.length}`;
 $('inm').textContent=it.team; $('intro').classList.add('on'); v.src=it.video; v.playbackRate=speed; list();
 timer=setTimeout(()=>{ $('intro').classList.remove('on'); v.playbackRate=speed; v.play().catch(()=>{}); ui(); },1800); }
v.onended=()=>{ const it=ITEMS[cur]; seen.add(cur); list();
 $('rm').textContent=medal(it.rank); $('rk').textContent=it.rank?`อันดับ ${it.rank}`:'';
 $('rn').textContent=it.team; $('rs').textContent=it.goal?it.result.replace(/^GOAL\s*/,'ถึงเส้นชัย ').replace(/, falls (\d+)/,' ล้ม $1 ครั้ง')
   :it.result.replace(/^DNF.*closest ([\d.]+) m.*falls (\d+).*/,'ไม่ถึง เหลือ $1 m ล้ม $2 ครั้ง');
 $('rs').style.color=it.goal?'var(--green)':'var(--orange)';
 const best=(it.board.match(/[\d.]+/)||[''])[0];
 $('rx').textContent=best&&!it.result.includes(best+' ')?`รอบนี้ไม่ใช่รอบที่ดีที่สุด บนตารางใช้ ${it.board}`:'';
 $('rev').classList.add('on'); ui(); if(auto&&cur<ITEMS.length-1) timer=setTimeout(()=>load(cur+1),5000); };
v.onplay=v.onpause=ui;
$('prev').onclick=()=>load(cur-1); $('next').onclick=()=>load(cur+1);
$('pp').onclick=()=>{ if($('rev').classList.contains('on')) return load(cur); v.paused?v.play():v.pause(); };
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>{ speed=+b.dataset.s; v.playbackRate=speed;
 try{localStorage.setItem('rs',speed)}catch(e){} ui(); });
$('auto').onclick=()=>{ auto=!auto; try{localStorage.setItem('ra',auto?'1':'0')}catch(e){} ui(); };
document.addEventListener('keydown',e=>{ if(e.key===' '){e.preventDefault();$('pp').click();}
 if(e.key==='ArrowRight')load(cur+1); if(e.key==='ArrowLeft')load(cur-1); });
load(cur);
</script></div></body></html>""")


def student_page(msg=""):
    data = board_data()
    return (HEAD.format(title="Challenge 7.5 · ส่งงาน", cls="")
            + '<div class="top"><div><div class="eyebrow">Challenge 7.5 · G1 Hazard Maze</div><h1>ส่งงานขึ้น Leaderboard</h1></div>'
            + chips(data) + "</div>" + msg
            + '<div class="grid2"><div class="card"><h3>อัปโหลด</h3>'
            + '<form method="post" action="/upload" enctype="multipart/form-data">'
            + f'<label>ชื่อทีม</label><input type="text" name="team" required pattern="{TEAM_RE}" placeholder="เช่น team_tiger">'
            + '<div class="hint">ภาษาอังกฤษ ตัวเลข _ หรือ - ยาว 2-32 ตัว ส่งซ้ำชื่อเดิม = แทนไฟล์เก่า</div>'
            + '<label>policy.pt (ต้องมี)</label><span class="file"><input type="file" name="policy" accept=".pt" required></span>'
            + '<div class="hint">checkpoint ของ rsl_rl ที่เทรนจาก G1 Rough (ไฟล์ model_*.pt)</div>'
            + '<label>pilot.py (ถ้ามี)</label><span class="file"><input type="file" name="pilot" accept=".py"></span>'
            + '<div class="hint">ต้องมี class Pilot ไม่ส่ง = ใช้ autopilot ตั้งต้น</div>'
            + '<label>ไฟล์ .py ที่ pilot ใช้ด้วย (ถ้ามี)</label><span class="file"><input type="file" name="helpers" accept=".py" multiple></span>'
            + '<div class="hint">เลือกได้หลายไฟล์ ไม่เกิน 5 ไฟล์ ห้ามเขียนไฟล์เก็บข้อมูลข้ามรอบ</div>'
            + '<button type="submit">ส่งงาน</button></form></div>'
            + '<div class="card"><h3>กติกาสั้น ๆ</h3><ul style="line-height:1.8;color:var(--soft)">'
            + "<li>ถึงเส้นชัยเร็วสุดชนะ ไม่ถึงดูว่าใครเหลือระยะน้อยกว่า</li><li>ล้ม ชนกำแพง หรือค้าง 10 วินาที กลับไปเริ่มใหม่ นาฬิกาเดินต่อ</li>"
            + "<li>pilot ใช้ได้แค่ข้อมูลที่หุ่นเห็นตอนวิ่ง ห้ามอ่านแผนที่</li><li>ทุกทีมวิ่งด่านเดียวกัน ครูประกาศ seed ตอนแข่ง</li></ul>"
            + '<small class="u">หน้าจอสำหรับโปรเจกเตอร์: <a style="color:var(--green)" href="/board">/board</a></small></div></div>'
            + '<h2 style="margin-top:12px">อันดับตอนนี้</h2><div id="live">' + board_part(data) + "</div>" + REFRESH_JS
            + "</div></body></html>")


# ----------------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status=200):
        b = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if "part=1" in query:
            return self._send(board_part())
        if path == "/board":
            return self._send(board_page())
        if path == "/replay":
            return self._send(replay_page(query))
        if path.startswith("/replays/"):
            return self._video(path[len("/replays/"):])
        if path == "/board.json":
            b = json.dumps(board_data(), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        return self._send(student_page())

    def _video(self, name):
        """mp4 with Range support (the browser seeks and changes speed smoothly)."""
        if not re.fullmatch(TEAM_RE + r"_seed\d+\.mp4", name):
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
        if self.path != "/upload":
            return self._error("ที่อยู่ไม่ถูกต้อง", 404)
        if size > MAX_BYTES:
            return self._error("ไฟล์ใหญ่เกิน 60 MB")
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
            return self._error("ชื่อทีมใช้ได้แค่ a-z A-Z 0-9 _ - ยาว 2-32 ตัว")
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
            return self._error("ไฟล์ประกอบได้ไม่เกิน 5 ไฟล์")
        for fname, data in helpers:
            fname = os.path.basename(fname)
            if not re.fullmatch(HELPER_RE, fname) or fname == "pilot.py":
                return self._error(f"ชื่อไฟล์ {fname} ใช้ไม่ได้ (ตัวอักษรอังกฤษ ตัวเลข _ ลงท้าย .py และห้ามชื่อ pilot.py)")
            err = check_python(fname, data)
            if err:
                return self._error(err)
        folder = os.path.join(SUBMISSIONS, team)
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):  # a new upload replaces everything (also exported/ and __pycache__/ of old runs)
            path = os.path.join(folder, old)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        saved = [("policy.pt", policy)] + ([("pilot.py", pilot)] if pilot else []) + \
            [(os.path.basename(f), d) for f, d in helpers]
        for fname, data in saved:
            with open(os.path.join(folder, fname), "wb") as f:
                f.write(data)
        for old in glob.glob(os.path.join(SUBMISSIONS, "_replays", f"{team}_seed*.*")):  # replays of the old upload
            os.remove(old)
        st = load_status()
        st[team] = {"state": "waiting", "msg": "", "uploaded": time.strftime("%H:%M:%S"), "uploaded_at": time.time()}
        save_status(st)
        names = ", ".join(n for n, _ in saved)
        print(f"[UPLOAD] {team}: {names} ({len(policy) / 1e6:.1f} MB)", flush=True)
        return self._send(student_page(f'<div class="msg">✓ ส่งงานทีม <b>{html.escape(team)}</b> แล้ว: {html.escape(names)} '
                                       f'<br><small class="u">รอครูรัน ผลจะขึ้นในตารางด้านล่าง</small></div>'))

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    os.makedirs(SUBMISSIONS, exist_ok=True)
    print(f"[UPLOAD] students: http://0.0.0.0:{PORT}   projector: http://0.0.0.0:{PORT}/board   -> "
          f"{os.path.abspath(SUBMISSIONS)}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
