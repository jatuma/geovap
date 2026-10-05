"""Cluster a LAZ tile into ground + separate objects.

Pipeline: ground raster from class 2 -> height above ground -> drop low points
-> voxelize -> connected components on voxel grid (26-neighbourhood, radius in
voxels) -> optional height-band split -> write cluster_id extra dim.

Usage: python -m geovap.stages.objects.cluster_laz in.laz out.laz [--voxel 0.1] [--eps 0.4] ...

This module imports NOTHING from the project -- numpy, laspy and scipy only -- and that is a
requirement, not an accident. It is what lets clustering be developed in parallel by someone with no
Geovap context, run on its own, or dropped from the pipeline entirely. Its driver (`cluster.py`)
knows about datasets; the algorithm here only knows about a LAZ file in and a LAZ file out.
"""
import argparse, time
import numpy as np
import laspy
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def ground_raster(x, y, z, gmask, cell):
    x0, y0 = x.min(), y.min()
    nx = int((x.max() - x0) / cell) + 2
    ny = int((y.max() - y0) / cell) + 2
    ix = ((x[gmask] - x0) / cell).astype(int)
    iy = ((y[gmask] - y0) / cell).astype(int)
    # per-cell minimum z of ground points
    ras = np.full((ny, nx), np.nan)
    order = np.lexsort((z[gmask], iy * nx + ix))
    key = (iy * nx + ix)[order]
    first = np.r_[True, key[1:] != key[:-1]]
    ras.flat[key[first]] = z[gmask][order][first]
    # fill holes by nearest valid cell, then smooth
    valid = ~np.isnan(ras)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    ras = ras[tuple(idx)]
    ras = ndimage.median_filter(ras, size=5)
    return ras, x0, y0, cell


def height_above_ground(x, y, z, ras, x0, y0, cell):
    ny, nx = ras.shape
    fx = (x - x0) / cell
    fy = (y - y0) / cell
    fx = np.clip(fx, 0, nx - 1.001)
    fy = np.clip(fy, 0, ny - 1.001)
    return z - ndimage.map_coordinates(ras, [fy, fx], order=1, mode="nearest")


def cluster_voxels(vox_xyz, eps_vox):
    """Connected components among voxel centres within eps (in voxel units)."""
    tree = cKDTree(vox_xyz)
    pairs = tree.query_pairs(r=eps_vox, output_type="ndarray")
    n = len(vox_xyz)
    g = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def cluster_points(xyz, voxel, eps):
    """Voxelize and connect; returns per-point component label."""
    if len(xyz) == 0:
        return np.zeros(0, np.int64)
    q = np.floor(xyz / voxel).astype(np.int64)
    q -= q.min(0)
    key = (q[:, 0] * (q[:, 1].max() + 1) + q[:, 1]) * (q[:, 2].max() + 1) + q[:, 2]
    ukey, inv = np.unique(key, return_inverse=True)
    vox = q[np.unique(inv, return_index=True)[1]].astype(np.float32)
    return cluster_voxels(vox, eps / voxel)[inv]


def band_split(xyz, hag, voxel, eps, hsplit, cell=0.5, overlap=0.5):
    """Cluster high band (>= hsplit) and low band separately, then attach a low
    cluster to the high cluster under whose XY footprint most of it lies."""
    hi = hag >= hsplit
    lab_hi = cluster_points(xyz[hi], voxel, eps)
    lab_lo = cluster_points(xyz[~hi], voxel, eps)
    n_hi = lab_hi.max() + 1 if len(lab_hi) else 0
    # footprint lookup: XY cell -> high cluster id (majority)
    x0, y0 = xyz[:, 0].min(), xyz[:, 1].min()
    nx = int((xyz[:, 0].max() - x0) / cell) + 2
    ny = int((xyz[:, 1].max() - y0) / cell) + 2
    def cells(p):
        return ((p[:, 1] - y0) / cell).astype(int) * nx + ((p[:, 0] - x0) / cell).astype(int)
    foot = np.full(nx * ny, -1, np.int64)
    if n_hi:
        c_hi = cells(xyz[hi])
        # majority per cell: sort by (cell, label), take most frequent
        uniq, cnt = np.unique(np.c_[c_hi, lab_hi], axis=0, return_counts=True)
        pc, pl = uniq[:, 0], uniq[:, 1]
        order = np.lexsort((-cnt, pc)); pc, pl = pc[order], pl[order]
        first = np.r_[True, pc[1:] != pc[:-1]]
        foot[pc[first]] = pl[first]
        # grow footprint by one cell so walls just outside the roof edge attach
        f2 = foot.reshape(ny, nx)
        grown = ndimage.grey_dilation(f2, size=3)
        f2 = np.where(f2 < 0, grown, f2); foot = f2.ravel()
    lab = np.empty(len(xyz), np.int64)
    lab[hi] = lab_hi
    if len(lab_lo):
        c_lo = cells(xyz[~hi])
        under = foot[c_lo]
        out = np.full(len(lab_lo), -1, np.int64)
        for k in np.unique(lab_lo):
            m = lab_lo == k
            u = under[m]
            ids, cnt = np.unique(u[u >= 0], return_counts=True)
            if len(ids) and cnt.max() >= overlap * m.sum():
                out[m] = ids[cnt.argmax()]
        new = out < 0
        # remaining low clusters become their own objects
        _, dense = np.unique(lab_lo[new], return_inverse=True)
        out[new] = n_hi + dense
        lab[~hi] = out
    return lab


def tree_like(xy, hag, nret, cell=0.5, min_area=40.0, min_multi=0.3, min_std=0.4):
    """Heuristic: leafless trees return several echoes per pulse and fill a column
    vertically; roofs/facades return one echo and are thin per 0.5 m column."""
    q = np.floor(xy / cell).astype(np.int64); q -= q.min(0)
    key = q[:, 0] * (q[:, 1].max() + 1) + q[:, 1]
    _, inv, cc = np.unique(key, return_inverse=True, return_counts=True)
    area = len(cc) * cell * cell
    if area < min_area:
        return False, area
    s1 = np.bincount(inv, hag); s2 = np.bincount(inv, hag ** 2)
    std = np.sqrt(np.maximum(s2 / cc - (s1 / cc) ** 2, 0))
    ok = cc >= 5
    med_std = np.median(std[ok]) if ok.any() else 0.0
    multi = np.mean(nret > 1)
    return (multi >= min_multi and med_std >= min_std), area


def watershed_crowns(xy, hag, cell=0.25, sigma=1.5, min_dist=2.0, min_height=2.0, min_seg_area=3.0):
    """Split one cluster into crowns: max-height raster -> smooth -> local maxima
    as crown tops -> marker watershed. Returns per-point label 0..k-1."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed, expand_labels
    x0, y0 = xy.min(0)
    ix = ((xy[:, 0] - x0) / cell).astype(int); iy = ((xy[:, 1] - y0) / cell).astype(int)
    nx, ny = ix.max() + 2, iy.max() + 2
    chm = np.full((ny, nx), -1.0)
    np.maximum.at(chm, (iy, ix), hag)
    occ = chm >= 0
    occ_d = ndimage.binary_closing(occ, iterations=2) | occ
    # fill unoccupied cells from neighbours so smoothing does not pull crowns down
    idx = ndimage.distance_transform_edt(~occ, return_distances=False, return_indices=True)
    filled = chm[tuple(idx)]
    sm = ndimage.gaussian_filter(filled, sigma)
    md = max(int(round(min_dist / cell)), 1)
    peaks = peak_local_max(sm, min_distance=md, threshold_abs=min_height, labels=occ_d.astype(int), exclude_border=False)
    if len(peaks) < 2:
        return np.zeros(len(xy), np.int64)
    markers = np.zeros(sm.shape, np.int32)
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)
    lab = watershed(-sm, markers, mask=occ_d)
    # drop tiny segments and let neighbours absorb them
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    small = ids[cnt * cell * cell < min_seg_area]
    if len(small):
        lab[np.isin(lab, small)] = 0
        lab = expand_labels(lab, distance=max(nx, ny)) * occ_d
    if lab.max() == 0:
        return np.zeros(len(xy), np.int64)
    plab = lab[iy, ix]
    # points whose cell has no label (should be rare): nearest labelled cell
    if (plab == 0).any():
        idx = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        plab = lab[tuple(idx)][iy, ix]
    _, dense = np.unique(plab, return_inverse=True)
    return dense


def split_tree_groups(xyz, hag, nret, lab, min_pts=2000, verbose=True):
    """Apply watershed_crowns to every large tree-like cluster; returns new labels."""
    out = lab.copy(); nxt = lab.max() + 1
    ids, cnt = np.unique(lab, return_counts=True)
    n_split = 0
    for k in ids[cnt >= min_pts]:
        m = lab == k
        is_tree, area = tree_like(xyz[m, :2], hag[m], nret[m])
        if not is_tree:
            continue
        sub = watershed_crowns(xyz[m, :2], hag[m])
        if sub.max() == 0:
            continue
        out[m] = np.where(sub == 0, k, nxt + sub - 1)
        nxt += sub.max(); n_split += 1
        if verbose:
            print(f"    cluster {k}: {m.sum():,} pts, {area:.0f} m2 -> {sub.max()+1} crowns")
    return out, n_split



def voxel_features(xyz, hag, nret, cell=0.25, nbr=1):
    """Per-cell local geometry on a sparse grid. Neighbourhood = (2*nbr+1)^3 cells.
    Returns cell keys per point (inverse) and per-cell arrays: n, multi, lin, plan, sph, l1, l2, l3, unsupported."""
    q = np.floor(xyz / cell).astype(np.int64); q -= q.min(0)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    ukey, inv, n = np.unique(key, return_inverse=True, return_counts=True)
    # per-cell sums
    def S(v): return np.bincount(inv, v, minlength=len(ukey))
    x, y, z = xyz[:, 0] - xyz[:, 0].mean(), xyz[:, 1] - xyz[:, 1].mean(), xyz[:, 2] - xyz[:, 2].mean()
    sums = np.stack([n.astype(float), S(x), S(y), S(z), S(x*x), S(y*y), S(z*z), S(x*y), S(x*z), S(y*z), S((nret > 1).astype(float))], 1)
    # aggregate over neighbour cells via sorted key lookup
    uq = np.stack(np.unravel_index(ukey, dims), 1)
    agg = np.zeros_like(sums)
    occ = np.zeros(len(ukey), int)
    rng_ = range(-nbr, nbr + 1)
    for dx in rng_:
        for dy in rng_:
            for dz in rng_:
                nq = uq + (dx, dy, dz)
                ok = np.all((nq >= 0) & (nq < dims), 1)
                nk = (nq[:, 0] * dims[1] + nq[:, 1]) * dims[2] + nq[:, 2]
                pos = np.searchsorted(ukey, nk)
                pos[pos >= len(ukey)] = 0
                hit = ok & (ukey[pos] == nk)
                agg[hit] += sums[pos[hit]]
                occ += hit
    N = agg[:, 0]; mx, my, mz = agg[:, 1]/N, agg[:, 2]/N, agg[:, 3]/N
    cov = np.empty((len(N), 3, 3))
    cov[:, 0, 0] = agg[:, 4]/N - mx*mx; cov[:, 1, 1] = agg[:, 5]/N - my*my; cov[:, 2, 2] = agg[:, 6]/N - mz*mz
    cov[:, 0, 1] = cov[:, 1, 0] = agg[:, 7]/N - mx*my
    cov[:, 0, 2] = cov[:, 2, 0] = agg[:, 8]/N - mx*mz
    cov[:, 1, 2] = cov[:, 2, 1] = agg[:, 9]/N - my*mz
    ev = np.linalg.eigvalsh(cov)[:, ::-1]           # l1 >= l2 >= l3
    ev = np.maximum(ev, 1e-9)
    l1, l2, l3 = ev[:, 0], ev[:, 1], ev[:, 2]
    f = dict(n=N, multi=agg[:, 10]/N, lin=(l1-l2)/l1, plan=(l2-l3)/l1, sph=l3/l1, l1=l1, l2=l2, l3=l3)
    f['occ'] = occ
    f['hag'] = np.bincount(inv, hag, minlength=len(ukey)) / n
    return inv, f

STRUCTURE, VEGETATION, WIRE = 1, 2, 3


def wire_mask_2d(xyz, hag, cell=0.25, hmin=3.0, gap=1.5, min_len=5.0, max_ratio=0.1):
    """Powerlines: high points with nothing beneath them that form thin elongated lines in plan.
    Roofs are also unsupported but wide (removed by a 3x3 opening); crowns are supported."""
    x, y = xyz[:, 0], xyz[:, 1]
    x0, y0 = x.min(), y.min()
    nx, ny = int((x.max() - x0) / cell) + 2, int((y.max() - y0) / cell) + 2
    ix = ((x - x0) / cell).astype(int); iy = ((y - y0) / cell).astype(int); k = iy * nx + ix
    hi = hag > hmin
    minhi = np.full(nx * ny, np.inf); np.minimum.at(minhi, k[hi], hag[hi])
    maxlo = np.full(nx * ny, -np.inf); np.maximum.at(maxlo, k[~hi], hag[~hi])
    has_hi = np.isfinite(minhi)
    unsup = (has_hi & (~np.isfinite(maxlo) | (maxlo < minhi - 1.5))).reshape(ny, nx)
    thin = unsup & ~ndimage.binary_opening(unsup, structure=np.ones((3, 3)))
    g = int(round(gap / cell)) | 1
    bridged = ndimage.binary_closing(thin, structure=np.ones((g, g))) | thin
    lab, n = ndimage.label(bridged, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        yy, xx = np.nonzero(lab[sl] == i)
        if len(xx) < 8:
            continue
        ev = np.linalg.eigvalsh(np.cov(np.c_[xx, yy].T * cell))[::-1]
        if 4 * np.sqrt(ev[0]) >= min_len and ev[1] / max(ev[0], 1e-9) < max_ratio:
            keep[i] = True
    wire_cells = keep[lab] & thin
    return wire_cells.ravel()[k] & hi


def classify_points(xyz, hag, nret):
    """Per-point class: WIRE (2D thin unsupported lines), VEGETATION (multi-return / volumetric),
    else STRUCTURE (buildings, walls, fences, poles)."""
    inv, f = voxel_features(xyz, hag, nret)
    veg = (f["multi"] >= 0.5) | (f["sph"] >= 0.25) | ((f["multi"] >= 0.3) & (f["sph"] >= 0.1))
    cls = np.where(veg[inv], VEGETATION, STRUCTURE).astype(np.int8)
    cls[wire_mask_2d(xyz, hag)] = WIRE
    return cls


def footprints(xyz, inv, n_ids, cell=0.5):
    q = np.floor(xyz[:, :2] / cell).astype(np.int64); q -= q.min(0)
    ncell = (q[:, 0].max() + 1) * (q[:, 1].max() + 1)
    key = inv.astype(np.int64) * ncell + q[:, 0] * (q[:, 1].max() + 1) + q[:, 1]
    return np.bincount(np.unique(key) // ncell, minlength=n_ids) * cell * cell


def absorb_fragments(xyz, lab, voxel, eps, small_area=4.0, big_area=16.0):
    """Small clusters (footprint < small_area) touching a big cluster (>= big_area) are merged
    into it: chimneys/antennas labelled vegetation on a roof, trunks labelled structure in a crown."""
    ids, inv = np.unique(lab, return_inverse=True)
    fp = footprints(xyz, inv, len(ids))
    small = fp < small_area; big = fp >= big_area
    if not small.any() or not big.any():
        return lab, 0
    v = np.floor(xyz / voxel).astype(np.int64); v -= v.min(0)
    vkey = (v[:, 0] * (v[:, 1].max() + 1) + v[:, 1]) * (v[:, 2].max() + 1) + v[:, 2]
    _, first = np.unique(vkey, return_index=True)
    pts = xyz[first]; pl = inv[first]
    sel = small[pl] | big[pl]
    pts, pl = pts[sel], pl[sel]
    pairs = cKDTree(pts).query_pairs(eps, output_type="ndarray")
    a, b = pl[pairs[:, 0]], pl[pairs[:, 1]]
    sb = small[a] & big[b]; bs = big[a] & small[b]
    s = np.concatenate([a[sb], b[bs]]); g = np.concatenate([b[sb], a[bs]])
    if len(s) == 0:
        return lab, 0
    pair, n = np.unique(np.c_[s, g], axis=0, return_counts=True)
    order = np.lexsort((-n, pair[:, 0])); pair = pair[order]
    firstp = np.r_[True, pair[1:, 0] != pair[:-1, 0]]
    remap = np.arange(len(ids)); remap[pair[firstp, 0]] = pair[firstp, 1]
    return ids[remap[inv]], int(firstp.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--voxel", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=0.40, help="link distance [m]")
    ap.add_argument("--hmin", type=float, default=0.30, help="drop points below this height above ground")
    ap.add_argument("--min-pts", type=int, default=50, help="clusters smaller than this -> noise")
    ap.add_argument("--cell", type=float, default=0.5, help="ground raster cell")
    ap.add_argument("--hsplit", type=float, default=0.0, help="if >0: cluster bands above/below this height separately")
    ap.add_argument("--watershed", action="store_true", help="split large tree-like clusters into crowns")
    ap.add_argument("--classes", action="store_true", help="classify structure/vegetation/wire and cluster each class separately")
    a = ap.parse_args()

    t = time.time()
    las = laspy.read(a.inp)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification)
    ground = cls == 2
    print(f"read {len(x):,} pts, {ground.sum():,} ground  ({time.time()-t:.1f}s)")

    ras, x0, y0, cell = ground_raster(x, y, z, ground, a.cell)
    hag = height_above_ground(x, y, z, ras, x0, y0, cell)
    print(f"HAG: ground pts median |hag| = {np.median(np.abs(hag[ground])):.3f} m  ({time.time()-t:.1f}s)")

    obj = (~ground) & (hag >= a.hmin)
    print(f"object candidates: {obj.sum():,}  (dropped {((~ground)&(hag<a.hmin)).sum():,} low pts)")

    xyz = np.c_[x[obj], y[obj], z[obj]]
    hobj = hag[obj]
    nret = np.asarray(las.number_of_returns)[obj]
    if a.classes:
        ocls = classify_points(xyz, hobj, nret)
        print(f"classes: structure {np.mean(ocls==STRUCTURE):.2f} vegetation {np.mean(ocls==VEGETATION):.2f} wire {np.mean(ocls==WIRE):.3f}  ({time.time()-t:.1f}s)")
        plab = np.full(len(xyz), -1, np.int64); nxt = 0
        for k, eps_k in ((STRUCTURE, a.eps), (VEGETATION, a.eps), (WIRE, 1.0)):
            m = ocls == k
            if not m.any():
                continue
            if k == WIRE or a.hsplit <= 0:
                lk = cluster_points(xyz[m], a.voxel, eps_k)
            else:
                lk = band_split(xyz[m], hobj[m], a.voxel, eps_k, a.hsplit)
            plab[m] = lk + nxt; nxt += lk.max() + 1
        plab, n_abs = absorb_fragments(xyz, plab, a.voxel, a.eps)
        print(f"raw components: {len(np.unique(plab)):,}, absorbed {n_abs} fragments  ({time.time()-t:.1f}s)")
    else:
        ocls = np.full(len(xyz), STRUCTURE, np.int8)
        if a.hsplit > 0:
            plab = band_split(xyz, hobj, a.voxel, a.eps, a.hsplit)
        else:
            plab = cluster_points(xyz, a.voxel, a.eps)
        print(f"raw components: {plab.max()+1:,}  ({time.time()-t:.1f}s)")
    if a.watershed:
        plab, n_split = split_tree_groups(xyz, hobj, nret, plab)
        print(f"watershed: split {n_split} tree groups -> {len(np.unique(plab)):,} components  ({time.time()-t:.1f}s)")
    # size filter -> noise (-1) and relabel densely, biggest first
    ids, cnt = np.unique(plab, return_counts=True)
    keep = ids[cnt >= a.min_pts]
    order = keep[np.argsort(-cnt[cnt >= a.min_pts])]
    remap = np.full(ids.max() + 1, -1, np.int64)
    remap[order] = np.arange(len(order))
    plab = remap[plab]
    print(f"clusters >= {a.min_pts} pts: {len(order):,}; noise pts: {(plab<0).sum():,}")
    top = [int(c) for c in np.sort(np.unique(plab[plab>=0], return_counts=True)[1])[::-1][:10]]
    print("largest clusters (pts):", top)

    cid = np.full(len(x), -2, np.int32)  # -2 ground/low, -1 noise, >=0 cluster
    cid[obj] = plab
    las.add_extra_dim(laspy.ExtraBytesParams(name="cluster_id", type=np.int32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="hag", type=np.float32))
    las.add_extra_dim(laspy.ExtraBytesParams(name="obj_class", type=np.uint8))
    oc = np.zeros(len(x), np.uint8); oc[obj] = ocls
    las.obj_class = oc
    las.cluster_id = cid
    las.hag = hag.astype(np.float32)
    las.write(a.out)
    print(f"wrote {a.out}  ({time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
