# Object clustering of the Dražkov tiles

Code for the per-tile object clustering and its Potree viewer page. Data (input copies with
`cluster_id` / `hag` / `obj_class`, random-colour copies, logs, `global_labels.npy`) stays under
`pointcloud-tools/output/clusters/src/` (12 GB, not versioned); override with `CLUSTERS_DATA=<dir>`.
Input tiles come from `$GEOVAP_DATA/LAZ_Dražkov_ground` (default `Geovap_data/DTM_Dražkov`).

| file | purpose |
|---|---|
| `cluster_laz.py` | one tile: ground raster → height above ground → voxel DBSCAN with height-band split, optional tree-crown watershed (`--watershed`) and structure/vegetation/wire classes (`--classes`); writes `cluster_id`, `hag`, `obj_class` extra dims |
| `batch.py` | all 38 tiles with the tuned parameters (voxel 0.1 m, eps 0.3 m, hmin 0.4 m, hsplit 2.5 m, watershed, classes); writes `rgb_t<tile>.laz` and `objects_t<tile>.laz` (RGB = random colour per cluster) |
| `merge_tiles.py` | links clusters across tile borders (same height band, same class) → global ids, rewrites both LAZ sets and saves `global_labels.npy` |
| `index.html` | viewer page; copy to `output/clusters/index.html` (served by the viewer container without rebuild) |

```
uv run python pointcloud-tools/clusters/batch.py
uv run python pointcloud-tools/clusters/merge_tiles.py
cd pointcloud-tools
docker compose run --rm potreeconverter /output/clusters/src/rgb_t*.laz     -o /output/clusters/rgb
docker compose run --rm potreeconverter /output/clusters/src/objects_t*.laz -o /output/clusters/objects
docker compose run --rm -v $PWD/../../Geovap_cache/out/tw45/tiles:/tw45:ro potreeconverter /tw45/*.laz -o /output/clusters/colored
cp clusters/index.html output/clusters/index.html
```

Viewer: `http://localhost:8080/pointclouds/clusters/index.html` — modes RGB (vendor) / Colored
(panoramas, run `tw45`) / Objects / Classification / Object class (keys 1–5). Octree directories are
root-owned; remove with `docker compose run --rm --entrypoint rm potreeconverter -rf /output/clusters/<dir>`.
