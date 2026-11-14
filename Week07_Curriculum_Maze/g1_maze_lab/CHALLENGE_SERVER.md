# Challenge 7.5: hosting the server (instructor)

Two processes on one Linux machine with an NVIDIA GPU:

- `scripts/challenge_server.py`: the web pages (upload form, leaderboard, live view, replays, `/admin`). Standard
  library + torch, no simulator.
- `scripts/challenge_runner.py`: the queue. Runs every upload with `scripts/challenge.py` (Isaac Lab + Newton), each
  in a throw-away Docker container, records the replay and writes the result.

Students only need a browser: see [CHALLENGE_SUBMIT.md](CHALLENGE_SUBMIT.md) (give them that page + the class code).

## 1. One-time setup

```bash
cd Week07_Curriculum_Maze/g1_maze_lab
uv sync                                                      # the course environment (README "Setup")
CHALLENGE_TIME=20 uv run python scripts/challenge.py --level 7 --seed 1   # a short run on the host must work first
```

**GPU driver.** The course pins torch for CUDA 13.0 (driver 580 or newer). With an older driver (`cat
/proc/driver/nvidia/version`, e.g. 535 = CUDA 12.2), switch torch to the CUDA 12.6 wheels in `pyproject.toml`, then
`uv sync`:

```toml
[tool.uv.sources]
"torch" = { index = "pytorch-cu126" }        # was pytorch-cu130, same for torchaudio / torchvision
"torchaudio" = { index = "pytorch-cu126" }
"torchvision" = { index = "pytorch-cu126" }

[[tool.uv.index]]                            # new, next to the pytorch-cu130 index
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

If Isaac Lab is not at `../../../../IsaacLab` (next to the course repo's folder), change the `isaaclab*` paths in
`[tool.uv.sources]` too. Keep these edits local (do not commit them for the students).

`challenge.py` turns Newton's CUDA graphs off by itself when the driver is older than Warp's CUDA toolkit
("conditional graph nodes require CUDA driver 12.4+"): the run works, only slower. With driver 580+ the graphs stay
on.

**The final level.** `--level final` is task `G1-Maze-Final` in `src/g1_maze_lab/tasks/challenge_final/` (10 x 10,
600 s, start / goal random from the seed). It is not in the students' repo: put your own there and keep it out of
git (`echo Week07_Curriculum_Maze/g1_maze_lab/src/g1_maze_lab/tasks/challenge_final/ >> .git/info/exclude`).
Without it, rounds can use level `7` or `9`.

**The sandbox** (every student run inside it: no network, read-only system, only a copy of that team's files
writable, the GPU passed in as device files; no nvidia-container-toolkit needed):

```bash
docker build -t maze-sandbox:1 scripts/sandbox
mkdir -p /srv/maze/assets /srv/maze/warp-cache /srv/maze/submissions      # any folder; /srv/maze in this guide
cp -r /tmp/https /srv/maze/assets/      # the robot USD that Isaac Lab downloaded in the test run above ($TMPDIR/https/...)
# compile the simulator's GPU kernels once (a trusted run of the pretrained policy, ~2 min); every run starts from a copy
CHALLENGE_DIR=/srv/maze/submissions CHALLENGE_SANDBOX=docker CHALLENGE_WARP_CACHE=/srv/maze/warp-cache \
  CHALLENGE_ASSET_MIRROR=/srv/maze/assets ISAACLAB_PATH=<your IsaacLab folder> CHALLENGE_TIME=20 \
  uv run python scripts/challenge_runner.py --warm-cache --level final --seed 1
```

Docker cleanups that remove unused images (`docker image prune -a`, the nightly cleanup of Coolify / Portainer ...)
delete `maze-sandbox:1` between runs. Keep it in use with a tiny container (the runner also rebuilds the image when it
is missing, ~2 min, before the next run):

```bash
docker run -d --name maze-sandbox-keep --restart unless-stopped --network none --read-only --cap-drop ALL \
  --user 65534:65534 --memory 32m maze-sandbox:1 sleep infinity
```

Refill the kernel cache (`rm -rf /srv/maze/warp-cache/*` + the command above) after updating Isaac Lab, Newton or the
GPU driver. 3D replays need OpenGL in the container: Mesa on a non-NVIDIA render node (the CPU's integrated GPU,
`/dev/dri/renderD*`). Without one, runs still work and `/replay` plays the 2D map replay instead.

## 2. Start

Pick a class code and a long admin key (e.g. `openssl rand -hex 16`), keep them out of git.

```bash
cd Week07_Curriculum_Maze/g1_maze_lab
export CHALLENGE_DIR=/srv/maze/submissions CHALLENGE_LEVEL=final

# terminal 1: the pages
CHALLENGE_HOST=127.0.0.1 CHALLENGE_PORT=8000 \
CHALLENGE_UPLOAD_CODE=<class code> CHALLENGE_TEAM_PIN=1 CHALLENGE_ADMIN_KEY=<admin key> \
  uv run python scripts/challenge_server.py

# terminal 2: the queue (runs every upload as it comes in, the round from /admin)
CHALLENGE_SANDBOX=docker CHALLENGE_WARP_CACHE=/srv/maze/warp-cache CHALLENGE_ASSET_MIRROR=/srv/maze/assets \
ISAACLAB_PATH=<your IsaacLab folder> CHALLENGE_PARALLEL=3 MAZE_REC_EVERY=2 \
  uv run python scripts/challenge_runner.py --watch --follow-round
```

Online: put the pages behind a reverse proxy or a tunnel (e.g. a Cloudflare tunnel to `http://127.0.0.1:8000`) and
keep `CHALLENGE_HOST=127.0.0.1`. On a classroom network only, `CHALLENGE_HOST=0.0.0.0` and students open
`http://<this machine's IP>:8000`. To keep both running after logout / reboot use pm2 or systemd (stop the runner
with Ctrl+C / SIGINT: it stops its containers and puts the running teams back in the queue).

| Setting | Default | |
|---|---|---|
| `CHALLENGE_DIR` | `submissions/` | uploads, `status.json`, `leaderboard.csv`, `round.json`, `_teams.json` (PIN hashes), `_logs/`, `_replays/`, `_live/` |
| `CHALLENGE_HOST` / `CHALLENGE_PORT` | `0.0.0.0` / `8000` | where the pages listen |
| `CHALLENGE_UPLOAD_CODE` | off | class code: an upload needs it |
| `CHALLENGE_TEAM_PIN` | off | `1`: the first upload of a team sets its PIN, later uploads need it |
| `CHALLENGE_ADMIN_KEY` | off | enables `/admin?key=<key>` |
| `CHALLENGE_LEVEL` / `CHALLENGE_SEED` | `7` / none | the round until it is set on `/admin` (`round.json` wins) |
| `CHALLENGE_SANDBOX` | off | `docker`: every run in the sandbox. Off = runs on this machine (trusted code only) |
| `CHALLENGE_PARALLEL` | `1` | runs at the same time; each uses ~2 CPU cores, ~1-2 GB RAM, < 1 GB GPU memory |
| `MAZE_REC_EVERY` | `2` | replay video at 50 / N fps (`2` = 25 fps) |
| `CHALLENGE_STALL` | `300` | stop a run whose simulation has not moved for N s (a pilot that hangs) |
| `--timeout` | 6 x limit + 300 s | wall-clock limit per run (65 min for 600 s) |
| `CHALLENGE_SANDBOX_MEMORY` / `_CPUS` / `_IMAGE` | `32g` / `12` / `maze-sandbox:1` | container limits and image |

## 3. Running a round (`/admin?key=<admin key>`)

The link sets a cookie: afterwards `/admin` alone works in that browser.

1. **Practice**: level `final`, a practice seed, **Uploads open** ticked, Save. Tell the class the URL and the class
   code. Every upload runs automatically in upload order.
2. **Real round**: untick **Uploads open** and Save *first* (the live view shows the maze: nobody may upload after
   seeing it), then enter the announced seed and Save. Every team that has not run on that seed goes back in the
   queue by itself.
3. Watch `/live` (`3D camera`, `Full screen`), then play `/replay` on the projector: last place first, the winner at
   the end, each result revealed when its clip ends (`Autoplay next`).
4. Before announcing the winners, read the top teams' `pilot.py` (link in the team row): a pilot runs in the same
   process as the simulator, so the score is only as honest as the pilot.

Changing the seed or the level never deletes results; the pages show the current round only.

Per team on `/admin`: the uploaded files, the run `log`, and

| Button | |
|---|---|
| **Re-run** | queue the team again (after a run stopped by the system, or to check a result) |
| **Reset PIN** | the team forgot its PIN: its next upload sets a new one (files and results stay) |
| **Delete** | remove the team: files, PIN, results and replays |

**Re-run all teams** queues every team again.

## 4. How long it takes

On an RTX 3090 with driver 535 (CUDA graphs off): ~0.27 x real time with the replay recorded, ~0.4 x without. A full
600 s DNF takes ~40 min of wall clock, a goal in 120 s ~8 min, plus ~1 min to start. Runs are limited by kernel
launches on one CPU core, not by the GPU: 3-6 at once run at nearly the same speed each. With driver 580+ the CUDA
graphs stay on and runs should be several times faster (not measured on this setup).

A run is stopped (`error`) after `--timeout` or when it stalls for `CHALLENGE_STALL` s; it still gets a replay of
what it did ("Stopped at ... s · no result", shown before last place).

## 5. When something goes wrong

| Symptom | |
|---|---|
| `Conditional graph nodes require CUDA driver 12.4+` in a log | Harmless with an old driver: Isaac Lab falls back to normal execution |
| `this run was stopped by the system` | The container was killed (runner stopped, out of memory, `docker kill`): **Re-run** |
| `pull access denied for maze-sandbox` / `the server's sandbox is broken` | The image was removed (Docker cleanup) and could not be rebuilt: `docker build -t maze-sandbox:1 scripts/sandbox`, start the keeper container above, **Re-run** |
| Every run fails at start with a CUDA error | Driver libraries changed (driver update without reboot?): reboot, then refill the kernel cache |
| No 3D video, only the 2D replay | No usable Mesa render node for the container: see the run log (`OpenGL ...` line) |
| A team is stuck in `running` after a crash of the runner | Start the runner again: it stops leftover containers of this `CHALLENGE_DIR` and re-queues the team |

Logs: the runner prints one line per start / result; the full log of a run is `<CHALLENGE_DIR>/_logs/<team>_L<level>_seed<seed>.log`
(also on `/admin`). Only one runner per `CHALLENGE_DIR`.
