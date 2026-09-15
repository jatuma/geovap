"""Merge clusters across tile borders (consistent with the in-tile rules).

1. border band = object points whose 1 m cell (or an 8-neighbour) is occupied by another tile
2. voxelize band points per tile (0.1 m); link voxels of *different* tiles within EPS, but only
   - high band (hag >= HSPLIT) voxel <-> high band voxel, both clusters tall, or
   - low band voxel <-> low band voxel, both clusters low-only (no point above HSPLIT)
   (a fence touching a house wall across a border must not glue them, as in-tile it would not)
3. union-find over (tile, cluster_id) -> global cluster id, one shared random palette
4. rewrite rgb_t*.laz (cluster_id = global) and objects_t*.laz (colours by global id)
"""
import glob, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, laspy
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))                     # code: pointcloud-tools/clusters
DATA = os.environ.get("CLUSTERS_DATA", os.path.normpath(f"{HERE}/../output/clusters/src"))  # rgb_t*/objects_t*.laz
EPS, VOXEL, HSPLIT = 0.3, 0.1, 2.5
FILES = sorted(glob.glob(f"{DATA}/rgb_t*.laz"))
TILES = [os.path.basename(f)[5:11] for f in FILES]

def cell_key(x, y):
    return np.floor(x).astype(np.int64) * 10_000_000 + np.floor(y).astype(np.int64)

def occupancy(f):
    l = laspy.read(f)
    return np.unique(cell_key(np.asarray(l.x), np.asarray(l.y)))

def border_voxels(args):
    tile, f, other_cells = args
    l = laspy.read(f)
    x, y, z, c, h = (np.asarray(l.x), np.asarray(l.y), np.asarray(l.z),
                     np.asarray(l.cluster_id), np.asarray(l.hag))
    oc = np.asarray(l.obj_class) if "obj_class" in l.point_format.dimension_names else np.ones(len(c), np.uint8)
    obj = c >= 0
    x, y, z, c, h, oc = x[obj], y[obj], z[obj], c[obj], h[obj], oc[obj]
    n_cl = int(c.max()) + 1 if len(c) else 0
    tall = np.zeros(n_cl, bool); ccls = np.zeros(n_cl, np.uint8)
    if n_cl:
        tall[np.unique(c[h >= HSPLIT])] = True
        ccls[c] = oc                      # class is uniform within a cluster (clustered per class)
    near = np.zeros(len(x), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near |= np.isin(cell_key(x + dx, y + dy), other_cells)
    x, y, z, c, h = x[near], y[near], z[near], c[near], h[near]
    q = np.floor(np.c_[x, y, z] / VOXEL).astype(np.int64)
    key = np.c_[q, c, (h >= HSPLIT).astype(np.int64)]
    return tile, np.unique(key, axis=0), tall, n_cl, ccls

def rewrite(args):
    f, tile, gmap, pal = args
    l = laspy.read(f)
    c = np.asarray(l.cluster_id)
    g = c.copy(); m = c >= 0
    g[m] = gmap[c[m]]
    l.cluster_id = g.astype(np.int32)
    l.write(f)
    col = np.zeros((len(c), 3), np.uint16)
    col[m] = pal[g[m]]; col[c == -1] = 140 * 257; col[c == -2] = 60 * 257
    l.red, l.green, l.blue = col[:, 0], col[:, 1], col[:, 2]
    l.write(f.replace("rgb_t", "objects_t"))
    return tile

if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        occ = dict(zip(TILES, ex.map(occupancy, FILES)))
    all_cells = np.concatenate(list(occ.values()))
    uniq, cnt = np.unique(all_cells, return_counts=True)
    print(f"occupancy done, {(cnt > 1).sum():,} shared border cells ({time.time()-t0:.0f}s)")

    def others_for(tile):           # cells that some other tile occupies
        own_only = np.isin(uniq, occ[tile]) & (cnt == 1)
        return uniq[~own_only]
    jobs = [(tile, f, others_for(tile)) for tile, f in zip(TILES, FILES)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(border_voxels, jobs))
    bands = {t: v for t, v, _, _, _ in res}
    tall_t = {t: tl for t, _, tl, _, _ in res}
    n_cl = {t: k for t, _, _, k, _ in res}
    cls_t = {t: cc for t, _, _, _, cc in res}
    print(f"border voxels: {sum(len(v) for v in bands.values()):,} ({time.time()-t0:.0f}s)")

    offset = {}; n = 0
    for tile in TILES:
        offset[tile] = n; n += n_cl[tile]
    tall = np.concatenate([tall_t[t] for t in TILES])
    ccls = np.concatenate([cls_t[t] for t in TILES])
    print(f"{n:,} tile-local clusters, {tall.sum():,} tall ({time.time()-t0:.0f}s)")

    pts = np.concatenate([v[:, :3] for v in bands.values()]).astype(np.float64) * VOXEL
    node = np.concatenate([v[:, 3] + offset[t] for t, v in bands.items()])
    band = np.concatenate([v[:, 4] for v in bands.values()])
    tid = np.concatenate([np.full(len(v), i) for i, v in enumerate(bands.values())])
    pairs = cKDTree(pts).query_pairs(EPS, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    ok = (tid[a] != tid[b]) & (band[a] == band[b]) & (node[a] != node[b]) & (ccls[node[a]] == ccls[node[b]])
    ok &= np.where(band[a] == 1, tall[node[a]] & tall[node[b]], ~tall[node[a]] & ~tall[node[b]])
    ok |= (tid[a] != tid[b]) & (node[a] != node[b]) & (ccls[node[a]] == 3) & (ccls[node[b]] == 3)   # wires: any touch
    a, b = node[a[ok]], node[b[ok]]
    g = coo_matrix((np.ones(len(a), bool), (a, b)), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    print(f"cross-tile links: {len(a):,} voxel pairs -> merged {n-ncomp:,} clusters, {ncomp:,} global objects ({time.time()-t0:.0f}s)")
    _, cc = np.unique(lab, return_counts=True)
    print("chain sizes (tile clusters per object):", dict(zip(*[x.tolist() for x in np.unique(cc, return_counts=True)])))

    rng = np.random.default_rng(11)
    pal = rng.integers(40, 255, (ncomp, 3)).astype(np.uint16) * 257
    jobs = [(f, tile, lab[offset[tile]:offset[tile] + n_cl[tile]], pal) for tile, f in zip(TILES, FILES)]
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(rewrite, jobs))
    np.save(f"{DATA}/global_labels.npy", lab)
    print(f"rewrote {len(FILES)*2} files ({time.time()-t0:.0f}s)")
