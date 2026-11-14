"""Challenge 7.5: submit from the terminal (same as the upload form on the server's web page).

    python scripts/submit.py --team team_tiger --pin tiger2026 --code <class code> \
        --policy logs/rsl_rl/<experiment>/<run>/model_XXXX.pt --pilot pilot.py --helper tuning.py --wait
    python scripts/submit.py --status --team team_tiger     # where the team is: queue, running, result
    python scripts/submit.py --status                       # the round: ranking + queue

Standard library only: plain ``python3`` works too (no uv / Isaac Lab needed on the machine that submits).
Settings can also come from the environment: MAZE_URL (default https://maze.mcp-digitalstudio.com), MAZE_TEAM,
MAZE_PIN, MAZE_CODE. Without a PIN or a class code the script asks for them (the PIN is not echoed).

A new upload replaces all the team's files (send pilot.py and the helpers again), puts the team at the end of the
queue and is refused while the team is running. Exit code 0 = accepted (or, with --wait, the run is done), 1 = not.
"""

import argparse
import getpass
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid

DEFAULT_URL = "https://maze.mcp-digitalstudio.com"
TEAM_RE = r"[A-Za-z0-9][A-Za-z0-9_\-]{1,31}"
PIN_RE = r"[0-9A-Za-z]{4,12}"
HELPER_RE = r"[A-Za-z][A-Za-z0-9_]{0,40}\.py"
MAX_BYTES = 60 * 1024 * 1024
AGENT = "maze-submit/1.0"


def fail(msg):
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


def request(url, data=None, headers=None, timeout=60):
    """(status, body text) of a GET / POST; an HTTP error status is returned, not raised."""
    req = urllib.request.Request(url, data=data, headers={"User-Agent": AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except urllib.error.URLError as e:
        fail(f"cannot reach {url}: {e.reason}")


def multipart(fields, files):
    """multipart/form-data body: fields {name: text}, files [(field name, file name, bytes)]."""
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(), value.encode(), b"\r\n"]
    for name, fname, data in files:
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{fname}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n".encode(), data, b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def page_message(page):
    """The ✓ / ✗ box of the upload page, as text."""
    m = re.search(r'<div class="msg[^"]*">(.*?)</div>', page, re.S)
    if not m:
        return None
    text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<br\s*/?>", "\n", m.group(1))))
    return "\n".join(line.strip() for line in text.strip().splitlines())


def board(url):
    code, body = request(url + "/board.json")
    if code != 200:
        fail(f"{url}/board.json answered HTTP {code}")
    return json.loads(body)


def team_line(b, team, url):
    """One line: where the team is in this round."""
    st = b["status"].get(team)
    rank = next((r for r in b["ranked"] if r["team"] == team), None)
    best = ""
    if rank:
        best = (f"rank {rank['rank']} of {len(b['ranked'])}, best run on this maze: "
                + (f"GOAL {rank['time']} s" if rank["goal"] else f"DNF {rank['left']} m to go") + f", falls {rank['falls']}")
    if not st:
        return f"{team}: not uploaded on this server" + (f" ({best})" if best else "")
    state = st.get("state", "waiting")
    if state == "waiting":
        pos = b["queue"].index(team) + 1 if team in b["queue"] else "?"
        return f"{team}: queue #{pos} (uploaded {st.get('uploaded', '?')})" + (f" · {best}" if best else "")
    if state == "running":
        return f"{team}: running since {st.get('started', '?')} · watch {url}/live?team={team}"
    if state == "done":
        return f"{team}: done: {st.get('msg', '')}" + (f" · {best}" if best else "") + f" · replay {url}/replay?team={team}"
    return f"{team}: error: {st.get('msg', '')}"


def show_status(url, team):
    b = board(url)
    print(f"Round: level {b['level']}, seed {b['seed'] or '-'}, limit {b['limit']} s, "
          f"uploads {'open' if b['open'] else 'CLOSED'} (updated {b['updated']})")
    if team:
        print(team_line(b, team, url))
        return
    for r in b["ranked"]:
        res = f"GOAL {r['time']} s" if r["goal"] else f"DNF {r['left']} m to go"
        print(f"  {r['rank']:>2}. {r['team']:<24} {res:<20} falls {r['falls']}")
    if b["running"]:
        print("  running: " + ", ".join(b["running"]))
    if b["queue"]:
        print("  queue:   " + ", ".join(f"{i}. {t}" for i, t in enumerate(b["queue"], 1)))


def wait(url, team, every=20):
    """Print each change of the team's status until its run is done (or an error); Ctrl+C stops waiting."""
    last = None
    try:
        while True:
            b = board(url)
            line = team_line(b, team, url)
            if line != last:
                print(time.strftime("%H:%M:%S"), line, flush=True)
                last = line
            state = b["status"].get(team, {}).get("state")
            if state in ("done", "error"):
                return state == "done"
            time.sleep(every)
    except KeyboardInterrupt:
        print("\n(stopped waiting: the run goes on, check again with --status)")
        return True


def main():
    p = argparse.ArgumentParser(description="Submit to the Challenge 7.5 server (like its upload form).")
    p.add_argument("--url", default=os.environ.get("MAZE_URL", DEFAULT_URL), help=f"server (default {DEFAULT_URL})")
    p.add_argument("--team", default=os.environ.get("MAZE_TEAM", ""), help="team name: letters, digits, _ or -, 2-32")
    p.add_argument("--pin", default=os.environ.get("MAZE_PIN", ""), help="team PIN, 4-12 letters / digits (set by the first upload)")
    p.add_argument("--code", default=os.environ.get("MAZE_CODE", ""), help="class code (from the instructor)")
    p.add_argument("--policy", help="rsl_rl checkpoint: logs/rsl_rl/<experiment>/<run>/model_XXXX.pt")
    p.add_argument("--pilot", help="pilot.py (optional: without it the default autopilot drives)")
    p.add_argument("--helper", action="append", default=[], help="a .py file the pilot imports (repeat, up to 5)")
    p.add_argument("--wait", action="store_true", help="after the upload, follow the team until its run is done")
    p.add_argument("--status", action="store_true", help="only show the round / the team's status, upload nothing")
    a = p.parse_args()
    url = a.url.rstrip("/")

    if a.status:
        return show_status(url, a.team)
    if not a.policy:
        p.error("--policy is required (or --status)")
    if not re.fullmatch(TEAM_RE, a.team or ""):
        fail("--team: letters, digits, _ or -, 2-32 characters (e.g. team_tiger)")
    pin = a.pin or getpass.getpass(f"Team PIN for {a.team} (4-12 letters or digits): ")
    if not re.fullmatch(PIN_RE, pin):
        fail("the PIN must be 4-12 letters or digits")
    code = a.code or input("Class code: ").strip()

    paths = [("policy", a.policy)] + ([("pilot", a.pilot)] if a.pilot else []) + [("helpers", h) for h in a.helper]
    if len(a.helper) > 5:
        fail("at most 5 helper files")
    files, total = [], 0
    for field, path in paths:
        if not os.path.isfile(path):
            fail(f"no such file: {path}")
        name = os.path.basename(path)
        if field == "helpers" and (not re.fullmatch(HELPER_RE, name) or name == "pilot.py"):
            fail(f"helper file name {name}: letters, digits and _, ending in .py, not pilot.py")
        with open(path, "rb") as f:
            data = f.read()
        total += len(data)
        files.append((field, name if field != "pilot" else "pilot.py", data))
    if total > MAX_BYTES:
        fail(f"the upload is {total / 1e6:.1f} MB, the server takes 60 MB at most")
    if a.pilot and b"class Pilot" not in files[1][2]:
        print("! pilot.py has no 'class Pilot' line: the server will refuse it", file=sys.stderr)

    body, ctype = multipart({"team": a.team, "pin": pin, "code": code}, files)
    print(f"Uploading {', '.join(f[1] for f in files)} ({total / 1e6:.1f} MB) as team {a.team} to {url} ...", flush=True)
    status, page = request(url + "/upload", data=body, headers={"Content-Type": ctype}, timeout=300)
    msg = page_message(page)
    if status != 200 or not msg or not msg.startswith("✓"):
        fail(msg.lstrip("✗ ") if msg else f"HTTP {status} from {url}/upload")
    print(msg)
    b = board(url)
    print(team_line(b, a.team, url))
    if a.wait:
        sys.exit(0 if wait(url, a.team) else 1)


if __name__ == "__main__":
    main()
