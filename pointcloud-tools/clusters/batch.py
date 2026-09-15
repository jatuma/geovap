"""Cluster every tile: writes rgb_<tile>.laz (true colour + cluster_id/hag extra dims)
and objects_<tile>.laz (RGB replaced by a random colour per cluster)."""
import sys, glob, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, laspy
HERE = os.path.dirname(os.path.abspath(__file__))                     # code: pointcloud-tools/clusters
DATA = os.environ.get("CLUSTERS_DATA", os.path.normpath(f"{HERE}/../output/clusters/src"))  # rgb_t*/objects_t*.laz, logs
SRC = os.environ.get("GEOVAP_DATA", "/home/jatuma/repos/Geovap/Geovap_data/DTM_Dražkov") + "/LAZ_Dražkov_ground"
PY = sys.executable
ARGS = ["--eps", "0.3", "--hmin", "0.4", "--voxel", "0.1", "--hsplit", "2.5", "--watershed", "--classes"]

def one(f):
    tile = os.path.basename(f).split("_")[1]          # ID3432_000035_JTSK.laz -> 000035
    rgb = f"{DATA}/rgb_t{tile}.laz"; obj = f"{DATA}/objects_t{tile}.laz"
    if os.path.exists(obj):
        return tile, "skip"
    t = time.time()
    r = subprocess.run([PY, f"{HERE}/cluster_laz.py", f, rgb, *ARGS], capture_output=True, text=True)
    if r.returncode:
        return tile, "FAIL " + r.stderr[-300:]
    l = laspy.read(rgb); c = np.asarray(l.cluster_id)
    rng = np.random.default_rng(int(tile))
    pal = rng.integers(40, 255, (max(c.max(), 0) + 1, 3)).astype(np.uint16) * 257
    col = np.zeros((len(c), 3), np.uint16); m = c >= 0; col[m] = pal[c[m]]
    col[c == -1] = 140 * 257; col[c == -2] = 60 * 257
    l.red, l.green, l.blue = col[:, 0], col[:, 1], col[:, 2]
    l.write(obj)
    ncl = [ln for ln in r.stdout.splitlines() if ln.startswith("clusters")]
    return tile, f"{len(c):,} pts, {ncl[0] if ncl else ''}  {time.time()-t:.0f}s"

if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*.laz"))
    # remove the old test copies named without zero padding
    for old in ("rgb_t35.laz", "objects_t35.laz", "rgb_t13.laz", "objects_t13.laz"):
        p = f"{DATA}/{old}"
        if os.path.exists(p): os.remove(p)
    with ProcessPoolExecutor(6) as ex:
        for tile, msg in ex.map(one, files):
            print(tile, msg, flush=True)
    print("DONE")
