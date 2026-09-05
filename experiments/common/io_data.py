"""Loadery pro data z Geovap_data/DTM_Drazkov.

Zamerne bez pandas (v tomto prostredi je pandas rozbity kombinaci
numpy2/pyarrow) -- cisty csv + numpy + json.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

DATA_ROOT = "/home/jatuma/repos/Geovap/Geovap_data/DTM_Dražkov"
PANO_DIR = os.path.join(DATA_ROOT, "LB5, Camera Ladybug")
EXPORT_CSV = os.path.join(PANO_DIR, "export.csv")
GEOJSON_PATH = os.path.join(DATA_ROOT, "1_ZPS_GAD.geojson")
LAZ_DIR = os.path.join(DATA_ROOT, "LAZ_Dražkov_ground")


@dataclass
class Frames:
    filename: np.ndarray  # [N] str
    timestamp: np.ndarray  # [N] float64
    origin_enh: np.ndarray  # [N, 3] float64 (Easting, Northing, Height)
    roll_deg: np.ndarray  # [N] float64
    pitch_deg: np.ndarray  # [N] float64
    yaw_deg: np.ndarray  # [N] float64

    def __len__(self) -> int:
        return len(self.filename)

    def path(self, i: int) -> str:
        return os.path.join(PANO_DIR, str(self.filename[i]))


def load_frames() -> Frames:
    filenames, timestamps, origins, rolls, pitches, yaws = [], [], [], [], [], []
    with open(EXPORT_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 17, f"unexpected export.csv header shape: {len(header)} fields"
        for row in reader:
            if not row or not row[0]:
                continue
            timestamps.append(float(row[0]))
            filenames.append(row[1])
            origins.append((float(row[2]), float(row[3]), float(row[4])))
            rolls.append(float(row[11]))
            pitches.append(float(row[12]))
            yaws.append(float(row[13]))
    frames = Frames(
        filename=np.array(filenames, dtype=object),
        timestamp=np.array(timestamps, dtype=np.float64),
        origin_enh=np.array(origins, dtype=np.float64),
        roll_deg=np.array(rolls, dtype=np.float64),
        pitch_deg=np.array(pitches, dtype=np.float64),
        yaw_deg=np.array(yaws, dtype=np.float64),
    )
    order = np.argsort(frames.timestamp)
    return Frames(
        filename=frames.filename[order],
        timestamp=frames.timestamp[order],
        origin_enh=frames.origin_enh[order],
        roll_deg=frames.roll_deg[order],
        pitch_deg=frames.pitch_deg[order],
        yaw_deg=frames.yaw_deg[order],
    )


@dataclass
class JvfObject:
    jvfcode: str  # base code, e.g. "0100000162"
    jvfcode_full: str  # e.g. "0100000162-01"
    geom_type: str  # "LineString" | "Point" | "Polygon"
    coords: np.ndarray  # [K, 3] float64 (E, N, H); K=1 for Point
    name: str  # properties["rc"] or properties["name"], human-readable hint
    uid: str


def load_jvf_objects(path: str = GEOJSON_PATH) -> list[JvfObject]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    objects = []
    for feat in data["features"]:
        props = feat.get("properties", {}) or {}
        code = props.get("jvfcode")
        if not code:
            continue
        geom = feat.get("geometry")
        if geom is None:
            continue
        gtype = geom["type"]
        raw = geom["coordinates"]
        if gtype == "Point":
            coords = np.array([raw], dtype=np.float64)
        elif gtype == "LineString":
            coords = np.array(raw, dtype=np.float64)
        elif gtype == "Polygon":
            coords = np.array(raw[0], dtype=np.float64)
        else:
            continue
        if coords.shape[1] == 2:
            # chybejici Z - nelze projektovat, ale ponechat pro pocty v E0
            coords = np.concatenate([coords, np.full((coords.shape[0], 1), np.nan)], axis=1)
        base = code.split("-")[0]
        objects.append(
            JvfObject(
                jvfcode=base,
                jvfcode_full=code,
                geom_type=gtype,
                coords=coords,
                name=props.get("rc") or props.get("name") or "",
                uid=props.get("uniqueID", ""),
            )
        )
    return objects


def list_laz_tiles() -> list[str]:
    return sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))
