#!/usr/bin/env bash
# Train one curriculum, then measure the policy the same way for every run.
#
#   scripts/run_curriculum.sh <name> [iterations]
#
# name: Base Reward Push Velocity Full ActionRate Knobs TerrainRule Difficulty
# iterations: default 300 (about 6 min on an RTX 3060 Laptop, plus a few minutes of measurements)
#
# The result is printed and saved to logs/rsl_rl/g1_curriculum/<run>/summary.txt
# Measure with G1-Curriculum-Base so every policy gets the same test, whatever it trained on.
set -euo pipefail

NAMES="Base Reward Push Velocity Full ActionRate Knobs TerrainRule Difficulty"
NAME="${1:-}"
ITERS="${2:-300}"
if [[ " $NAMES " != *" $NAME "* ]]; then
    echo "usage: scripts/run_curriculum.sh <name> [iterations]"
    echo "name: $NAMES"
    exit 1
fi

cd "$(dirname "$0")/.."
RUN="$(echo "$NAME" | tr '[:upper:]' '[:lower:]')$ITERS"
TRAIN_LOG="$(mktemp)"

uv run isaaclab train --rl_library rsl_rl --task "G1-Curriculum-$NAME" --num_envs 1024 \
    --checkpoint pretrained/g1_rough/model_pretrained.pt \
    agent.max_iterations="$ITERS" agent.run_name="$RUN" 2>&1 | tee "$TRAIN_LOG"

DIR="$(ls -dt logs/rsl_rl/g1_curriculum/*_"$RUN" | head -1)"
MODEL="$DIR/$(ls -v "$DIR" | grep -E '^model_[0-9]+\.pt$' | tail -1)"

measure() {  # title, then the environment variables for eval_g1.py
    echo "--- $1"
    shift
    env EVAL_STEPS=450 "$@" uv run python scripts/eval_g1.py --task G1-Curriculum-Base --num_envs 64 \
        --checkpoint "$MODEL" 2>&1 | grep '^\[EVAL\]'
}

{
    echo "=== G1-Curriculum-$NAME, $ITERS iterations: $MODEL"
    echo "--- end of training"
    # the last block of the training log
    awk '/Learning iteration/ { block = "" } { block = block $0 "\n" } END { printf "%s", block }' "$TRAIN_LOG" \
        | grep -E 'Mean reward|Curriculum/|Episode_Termination/base_contact' | sed 's/^ *//'
    measure "speed and step length" EVAL_CMDS="1,0,0;2,0,0;0.8,0,0"
    measure "falls with a 1.0 m/s push every 2 s" EVAL_PUSH=1.0 EVAL_CMDS="0.5,0,0"
    measure "falls with a 2.0 m/s push every 2 s" EVAL_PUSH=2.0 EVAL_CMDS="0.5,0,0"
} | tee "$DIR/summary.txt"

rm -f "$TRAIN_LOG"
echo "Saved: $DIR/summary.txt"
