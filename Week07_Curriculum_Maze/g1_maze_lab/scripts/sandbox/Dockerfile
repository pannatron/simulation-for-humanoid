# Sandbox for challenge_runner.py (CHALLENGE_SANDBOX=docker): system libraries only.
# Python, the project's .venv and Isaac Lab are mounted read-only from the host at their own paths, and the host's
# NVIDIA driver libraries are mounted in by the runner (they must match the loaded kernel module).
#   docker build -t maze-sandbox:1 scripts/sandbox
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 libglib2.0-0t64 fonts-dejavu-core \
        libgl1 libegl1 libgles2 libglx-mesa0 libegl-mesa0 libgl1-mesa-dri libglu1-mesa \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libxxf86vm1 libxfixes3 \
        libxkbcommon0 libsm6 libice6 xvfb xauth \
    && rm -rf /var/lib/apt/lists/*
