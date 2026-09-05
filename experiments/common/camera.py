"""Projekcni model panoramaticke kamery Ladybug -> equirektangularni pixel.

Vzorec prevzaty z 02_obarveni_pointcloudu.md SS2.1 (overeno: median dE00 6.08 proti
TerraScan RGB na dlazdici 037, 49897 bodu, 44 panoramat). Implementace je vektorizovana
v torch tak, aby jedno volani spocitalo projekci pro (N bodu) x (M snimku) najednou na GPU.

Konvence (uz rozhodnute, nemenit bez opakovani ablace v 02_obarveni_pointcloudu.md):
  - yaw je matematicky azimut, CCW od osy +Easting
  - roll i pitch se aplikuji se ZAPORNYM znamenkem
  - v = 0 je ZENIT (ne nadir)
  - sev panoramatu (u=0) je v azimutu == yaw
"""
from __future__ import annotations

import torch

PANO_W = 8000
PANO_H = 4000


def _deg2rad(x: torch.Tensor) -> torch.Tensor:
    return x * (torch.pi / 180.0)


def world_to_panorama_px(
    point_enh: torch.Tensor,
    frame_origin_enh: torch.Tensor,
    frame_roll_deg: torch.Tensor,
    frame_pitch_deg: torch.Tensor,
    frame_yaw_deg: torch.Tensor,
    pano_w: int = PANO_W,
    pano_h: int = PANO_H,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projekce bodu sveta do panoramatickeho pixelu.

    Tvary (broadcastovatelne mezi body a snimky, typicky point_enh je [N,3] a
    frame_* je [M] nebo [N,M] uz vybrane pro kazdy bod jeho kandidatni snimky):
      point_enh:        [..., 3]  (Easting, Northing, Height) bodu, S-JTSK
      frame_origin_enh: [..., 3]  stred kamery snimku, stejne CRS
      frame_roll_deg, frame_pitch_deg, frame_yaw_deg: [...]  ve stupnich

    Vraci (u, v, el_deg): pixelove souradnice a elevaci (pro filtrovani bodu
    za kamerou / mimo rozumny rozsah elevace, viz SS3.2 03_semanticka_segmentace.md).
    """
    d = point_enh - frame_origin_enh
    dE, dN, dH = d[..., 0], d[..., 1], d[..., 2]

    yaw = _deg2rad(frame_yaw_deg)
    roll = _deg2rad(-frame_roll_deg)
    pitch = _deg2rad(-frame_pitch_deg)

    # 1) srovnani do smeru jizdy
    x = dE * torch.cos(yaw) + dN * torch.sin(yaw)
    y = -dE * torch.sin(yaw) + dN * torch.cos(yaw)
    z = dH

    # 2) roll (kolem osy x, dopredu), pak pitch (kolem prumetu y)
    y2 = y * torch.cos(roll) - z * torch.sin(roll)
    z2 = y * torch.sin(roll) + z * torch.cos(roll)
    x3 = x * torch.cos(pitch) + z2 * torch.sin(pitch)
    z3 = -x * torch.sin(pitch) + z2 * torch.cos(pitch)
    y3 = y2

    az = torch.rad2deg(torch.atan2(y3, x3))
    el = torch.rad2deg(torch.atan2(z3, torch.hypot(x3, y3)))

    u = torch.remainder(az, 360.0) / 360.0 * pano_w
    v = (90.0 - el) / 180.0 * pano_h
    return u, v, el


def horizontal_distance(point_enh: torch.Tensor, frame_origin_enh: torch.Tensor) -> torch.Tensor:
    d = point_enh - frame_origin_enh
    return torch.hypot(d[..., 0], d[..., 1])
