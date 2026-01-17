# add_com.py

import os, numpy as np, trimesh as tm
from pathlib import Path

obj = "acropora_cervicornis"  # your coral object name
HERE = Path(__file__).resolve().parent          # .../code
REPO = HERE.parent                              # .../ (repo root)  <-- adjust if needed
mesh_path = REPO / "models" / "ycb" / obj / "google_16k" / "nontextured.ply"

assert mesh_path.is_file(), f"Missing mesh at {mesh_path}"
m = tm.load(mesh_path, force="mesh")
if not isinstance(m, tm.Trimesh):
    # some files load as Scene; merge to single mesh
    m = tm.util.concatenate(m.dump())

# choose a COM estimate: try mass properties; fallback to vertex mean
try:
    com = m.center_mass
    if not np.all(np.isfinite(com)):
        com = m.vertices.mean(axis=0)
except Exception:
    com = m.vertices.mean(axis=0)

com = np.asarray(com, dtype=np.float32)

# write/update models/coms/coms.npz (absolute, robust)
path = REPO / "models" / "coms" / "coms.npz"
path.parent.mkdir(parents=True, exist_ok=True)

if path.exists():
    d = dict(np.load(path, allow_pickle=True)["com_dict"].item())
else:
    d = {}

d[obj] = com
np.savez(path, com_dict=d)

print(f"{obj} COM:", com, "-> wrote", path)
