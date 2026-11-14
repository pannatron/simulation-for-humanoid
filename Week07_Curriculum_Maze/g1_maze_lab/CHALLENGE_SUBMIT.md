# Challenge 7.5: how to submit (students)

Server: **https://maze.mcp-digitalstudio.com**

| Page | What it shows |
|---|---|
| [`/`](https://maze.mcp-digitalstudio.com/) | **Submit** (the upload form) + the queue and the ranking of the current round |
| [`/board`](https://maze.mcp-digitalstudio.com/board) | Leaderboard: podium + table, refreshes by itself |
| [`/live`](https://maze.mcp-digitalstudio.com/live) | The runs in progress: 2D map or 3D camera, full screen (keys `2` / `3` / `F`) |
| [`/replay`](https://maze.mcp-digitalstudio.com/replay) | Every team's latest run: seek bar, ±5 s, 0.5x-8x (keys `Space`, `J` / `L`, `←` / `→`, `F`) |

Your files run on the server, one team after another, on the maze of the current round (level + seed shown at the
top of every page). Nothing has to be installed on your side: a browser is enough to upload.

## 1. What you need

| | Where it comes from |
|---|---|
| **Class code** | **`maze8071`** (without it the upload is refused; if it stops working, the instructor has changed it: ask in class) |
| **Team name** | You choose it: letters, digits, `_` or `-`, 2-32 characters, e.g. `team_tiger` |
| **Team PIN** | You choose it on your first upload: 4-12 letters or digits. Write it down and share it only with your team |
| **policy.pt** (required) | Your trained checkpoint: `logs/rsl_rl/<experiment>/<run>/model_XXXX.pt` (any file name is fine) |
| **pilot.py** (optional) | Your autopilot, `class Pilot` (copy `DefaultPilot` from `scripts/challenge.py` and improve it). Without it the default autopilot drives |
| **helper .py files** (optional) | Up to 5 files that `pilot.py` imports |

The policy file must be the **rsl_rl checkpoint** (`model_XXXX.pt`, the one with `actor_state_dict` inside), trained
from G1 Rough (same network: the upload checks it). Not the exported `exported/policy.pt` or `policy.onnx`.

## 2. Test on your machine first

The final 10 x 10 maze is only on the server, but levels 7 and 9 have the same hazards. Random start / goal like the
final:

```bash
cd Week07_Curriculum_Maze/g1_maze_lab
uv run python scripts/challenge.py --policy logs/rsl_rl/<experiment>/<run>/model_XXXX.pt --pilot pilot.py \
    --level 9 --random-ends --seed 1 --viz newton
```

Nothing goes to the leaderboard. The last line is your result: `RESULT ... GOAL 123.4 s, falls 1` or
`DNF (closest 12.3 m to the goal ...)`. Try a few seeds.

## 3. Upload (the form on `/`)

1. Open https://maze.mcp-digitalstudio.com
2. **Team name**: the same name every time (it is your team on the leaderboard)
3. **Team PIN**: the first upload of a team name sets its PIN; every later upload of that name needs the same PIN, so
   nobody else can replace your files
4. **Class code**: `maze8071`
5. **policy.pt**: choose your `model_XXXX.pt`
6. **pilot.py** and **helper files**: if you have them
7. **Submit**: the page says `✓ Uploaded for team <name>: policy.pt, pilot.py ...` and your team appears in the queue

Every file is checked when you upload (the policy's network, every `.py` compiles, `pilot.py` defines `class
Pilot`, no reading of the map). If something is wrong, the page says what (table below) and nothing is saved:
fix it and submit again.

### Or from the terminal

`scripts/submit.py` does the same as the form. It needs only `python3` (no Isaac Lab, no uv), so it also runs on a
machine that only has the files; on its own it is one file you can copy.

```bash
python scripts/submit.py --team team_tiger --pin tiger2026 --code maze8071 \
    --policy logs/rsl_rl/<experiment>/<run>/model_XXXX.pt --pilot pilot.py --helper tuning.py --wait
```

```
Uploading model_6499.pt, pilot.py, tuning.py (7.8 MB) as team team_tiger to https://maze.mcp-digitalstudio.com ...
✓ Uploaded for team team_tiger: policy.pt, pilot.py, tuning.py
It runs automatically in queue order; the result appears in the table below
team_tiger: queue #2 (uploaded 10:21:46)
10:31:02 team_tiger: running since 10:31:00 · watch https://maze.mcp-digitalstudio.com/live?team=team_tiger
...
```

- `--pilot` and `--helper` (repeat it, up to 5) are optional, like on the form. The same rules apply: a new upload
  replaces **all** your files, so send them every time.
- Leave out `--pin` and the script asks for it without showing it. `MAZE_TEAM`, `MAZE_PIN`, `MAZE_CODE` and
  `MAZE_URL` in the environment replace the options:
  `export MAZE_TEAM=team_tiger MAZE_CODE=maze8071`, then `python scripts/submit.py --policy model_XXXX.pt --pilot pilot.py`.
- `--wait` prints every change (queue → running → done / error) until the run ends; Ctrl+C stops waiting, not the run.
- `python scripts/submit.py --status --team team_tiger` shows where your team is; `--status` alone shows the
  ranking and the queue of the round.
- Exit code 0 = accepted, 1 = refused (the same `✗` message as the web page, table below).

## 4. After the upload

| Status | Meaning |
|---|---|
| `waiting` | In the queue. Teams run in upload order, up to 3 at the same time |
| `running` | On the server now: watch it on `/live` |
| `done` | Finished: the result is in the table, the replay is on `/replay` |
| `error` | The run could not finish: the message says why (table below) |

A run is slower than real time on the server: about 40 min for a full 600 s run without reaching the goal, about
8 min for a robot that reaches the goal in 120 s, plus ~1 min to start.

**Upload again** (same team name + PIN) whenever you like, except while your team is `running`:

- the new upload **replaces all your files**: send `pilot.py` and the helper files again too, not only the policy
- the team goes to the end of the queue and runs again
- the leaderboard keeps your **best** run on this round's maze, so a worse new run never lowers your place
  (`/replay` always shows the latest run)

## 5. Ranking and rules

- A run that reaches the goal beats every run that does not. Between two goals, the faster wins; then fewer falls.
- Without a goal (DNF), the team that got closest to the goal along its route wins; then fewer falls.
- Time limit: 600 s of maze time on the final level. The clock keeps running after a fall.
- A fall, a crash into a wall, touching a red spike bed or being stuck 10 s sends the robot back to the start.
- The pilot may only use what the robot senses during the run (`obs`): never the real map (`GENERATED`, `HAZARDS`,
  `HARD_CELLS`, `g1_maze_lab.terrains` are refused on upload). It may not keep data from one run for the next.
- Every team runs the same maze (same level and seed). For the real round the instructor closes uploads first.
- Your code runs in a sandbox without internet: only the Python standard library, the course's packages (torch,
  numpy, ...) and your own helper files can be imported. Nothing can be downloaded during the run.
- A run is stopped if it takes more than 65 min of real time, or if the simulation does not move for 5 min (a pilot
  that hangs). It shows as `error`, with the replay of what it did until then.

## 6. Messages and what to do

When you upload:

| Message | What to do |
|---|---|
| `wrong class code` | Type `maze8071` (no spaces). Still refused: the instructor has changed the code, ask in class |
| `team name: letters, digits, _ or -, 2-32 characters` | No spaces, no Thai letters, no `.` |
| `the PIN must be 4-12 letters or digits` | e.g. `tiger2026` |
| `team X is taken and the PIN does not match` | Wrong PIN, or another team already uses that name. Your team but you forgot the PIN: ask the instructor to reset it (your results stay) |
| `cannot open policy.pt: it must be an rsl_rl model_*.pt checkpoint` | Upload `logs/rsl_rl/.../model_XXXX.pt`, not an exported / ONNX file |
| `this policy's network does not match G1 Rough` | Train from the G1 Rough network (fine-tune `pretrained/g1_rough/model_pretrained.pt`), same observations and layer sizes |
| `pilot.py line N: ...` | A syntax error in that line |
| `pilot.py must define class Pilot` | Name the class exactly `Pilot` |
| `... line N: GENERATED is not allowed` | The pilot reads the real map: use `obs` only |
| `at most 5 helper files` / `file name X is not allowed` | Max 5 helpers; names like `my_planner.py` (letters, digits, `_`) |
| `the upload is larger than 60 MB` | Upload only the checkpoint and the `.py` files |
| `team X is running right now` | Wait until the run has finished, then upload |
| `uploads are closed` | The instructor is running the round; watch `/live` |

In the table (status `error`):

| Message | Meaning |
|---|---|
| A Python error (e.g. `AttributeError: ...`) | Your pilot crashed: fix it and test it with `scripts/challenge.py` before uploading again |
| `stopped at t=... s: over the 65 min wall-clock limit` | The run was too slow (a heavy pilot: make `act()` faster) |
| `stopped at t=... s: the simulation did not move for 300 s` | The pilot hangs (an endless loop, a `sleep`, waiting for input) |
| `this run was stopped by the system, not by the team's code` | Not your fault: the instructor re-runs it |
