"""Spread of the learned per-frame exposure gain / offset of trained models (train_gs.py --lamp: lamp_gain / lamp_off;
--exposure: expo_gain / expo_off) over the frames that were trained (the others keep the start value 0): per channel
the median gain, its p5-p95, the standard deviation of the log gain and the offset (docs/MODEL.md B4, docs/EVALUATION.md 5). WSL (torch).

usage: /opt/gsenv/bin/python gain_spread.py RUN_DIR/model.pt [RUN_DIR/model.pt ...]
"""
import sys

import numpy as np
import torch

for p in sys.argv[1:]:
    m = torch.load(p, map_location="cpu", weights_only=False)
    pre = next((k for k in ("lamp_", "expo_") if k + "gain" in m), None)
    if pre is None:
        print(f"{p}: no per-frame gain (keys {sorted(m.keys())[:12]} ...)")
        continue
    g, o = m[pre + "gain"].numpy(), m[pre + "off"].numpy()
    used = np.abs(g).sum(1) > 0
    lg = g[used]
    gain = np.exp(lg)
    print(f"{p.split('/g0/')[-1]} ({pre[:-1]}): frames {used.sum()}/{len(g)}")
    for c, name in enumerate("RGB"):
        q = np.percentile(gain[:, c], [5, 50, 95])
        print(f"  {name}: gain median {q[1]:.3f}  p5-p95 {q[0]:.3f}-{q[2]:.3f}  sd(log) {lg[:, c].std():.3f}  "
              f"offset median {np.median(o[used, c]):+.4f} p5-p95 {np.percentile(o[used, c], 5):+.4f}.."
              f"{np.percentile(o[used, c], 95):+.4f}")
