"""Run an exported Isaac Lab policy outside Isaac Lab and check it against the TorchScript export.

Week 3 exercise. ``isaaclab play`` writes ``<run>/exported/policy.onnx`` and ``policy.pt``;
this script loads both, feeds the same random observation, and times ONNX Runtime on CPU.

    cd ~/IsaacLab
    uv run python ~/Downloads/Simulation_for_Humanoid/Week03_G1_Humanoid/code/run_onnx_policy.py logs/rsl_rl/<task>/<run>/exported
"""

import sys
import time

import numpy as np
import onnxruntime as ort
import torch

export_dir = sys.argv[1]
sess = ort.InferenceSession(f"{export_dir}/policy.onnx", providers=["CPUExecutionProvider"])
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print("input ", inp.name, inp.shape)
print("output", out.name, out.shape)

obs = np.random.default_rng(0).standard_normal((1, inp.shape[1])).astype(np.float32)
a_onnx = sess.run(None, {inp.name: obs})[0]
a_ts = torch.jit.load(f"{export_dir}/policy.pt", map_location="cpu")(torch.from_numpy(obs)).detach().numpy()
print("max |onnx - torchscript| =", float(np.abs(a_onnx - a_ts).max()))

n = 1000
t = time.perf_counter()
for _ in range(n):
    sess.run(None, {inp.name: obs})
print(f"onnxruntime CPU: {(time.perf_counter() - t) / n * 1e3:.3f} ms per call")
