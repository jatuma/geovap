"""Gnomonicka reprojekce ERP <-> perspektivni vysec, cela na GPU pres torch.nn.functional.grid_sample.

Podle SS4.3 03_semanticka_segmentace.md: prstenec vyseci se sklonem 0 stupnu (8x, FOV 90x90,
krok 45 stupnu = prekryv 50 %) + spodni prstenec se sklonem -45 stupnu na vozovku.

Pristup: grid_sample umi jen "gather" (pro kazdy vystupni pixel dohledej vstupni), ne "scatter".
Proto se fuze pocita z pohledu ERP: pro kazdy ERP pixel se spocita, kam by padl v souradnicich
dane vysece (inverzni rotace), a pokud lezi uvnitr FOV vysece, nabere se odtud pravdepodobnost
s kosinovou vahou k okraji. Tim padem neni potreba scatter vubec.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def erp_pixel_rays(pano_w: int, pano_h: int, device: torch.device) -> torch.Tensor:
    """Smerove vektory (dopredu=+z v yaw=0, vpravo=+x, nahoru=+y) pro kazdy ERP pixel. [H,W,3]."""
    us = torch.arange(pano_w, device=device, dtype=torch.float64) + 0.5
    vs = torch.arange(pano_h, device=device, dtype=torch.float64) + 0.5
    az = us / pano_w * 360.0  # stupne, sev u=0 je az=0 relativne k yaw snimku
    el = 90.0 - vs / pano_h * 180.0
    az_r = torch.deg2rad(az)[None, :].expand(pano_h, pano_w)
    el_r = torch.deg2rad(el)[:, None].expand(pano_h, pano_w)
    fwd = torch.cos(el_r) * torch.cos(az_r)
    right = torch.cos(el_r) * torch.sin(az_r)
    up = torch.sin(el_r)
    return torch.stack([right, up, fwd], dim=-1)  # [H,W,3]


def view_sample_grid(
    rays: torch.Tensor,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pro dany pohled (yaw, pitch, fov) spocita, kam kazdy ERP paprsek padne v lokalnich
    souradnicich vysece. Vraci (grid [1,H,W,2] pro grid_sample, weight [1,1,H,W] kosinovy
    dobeh k okraji, 0 mimo FOV / za kamerou).
    """
    yaw = torch.deg2rad(torch.tensor(yaw_deg, device=device, dtype=rays.dtype))
    pitch = torch.deg2rad(torch.tensor(pitch_deg, device=device, dtype=rays.dtype))
    right2, up2, fwd2 = rays[..., 0], rays[..., 1], rays[..., 2]

    # presna inverze rotaci z make_extract_grid (nejdriv zrusit yaw, pak pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    right1 = right2 * cy - fwd2 * sy
    fwd1 = right2 * sy + fwd2 * cy
    up1 = up2

    cp, sp = torch.cos(pitch), torch.sin(pitch)
    up = up1 * cp - fwd1 * sp
    fwd = up1 * sp + fwd1 * cp
    right = right1

    f = 0.5 / torch.tan(torch.deg2rad(torch.tensor(fov_deg / 2.0, device=device, dtype=rays.dtype)))
    valid_front = fwd > 1e-6
    fwd_safe = torch.where(valid_front, fwd, torch.ones_like(fwd))
    x = right / fwd_safe * f
    # pozor: "up" je smer nahoru ve svete, ale vysec (make_extract_grid) je indexovana radkem
    # obrazu gy, kde gy = -up (radek 0 = nahore = up kladne). Bez tehle negace vychazi vertikalne
    # zrcadleny vysledek (proto ho grid_sample cetl "vzhuru nohama" pro pitch != 0 i pro pitch == 0).
    y = -up / fwd_safe * f

    inside = valid_front & (x.abs() <= 0.5) & (y.abs() <= 0.5)

    grid = torch.stack([x * 2, y * 2], dim=-1).unsqueeze(0).float()  # [-1,1] pro grid_sample

    d = torch.maximum(x.abs(), y.abs()) / 0.5  # 0 ve stredu, 1 na okraji
    edge = torch.cos(d.clamp(max=1.0) * (torch.pi / 2)) ** 2  # kosinovy dobeh
    weight = torch.where(inside, edge, torch.zeros_like(edge)).float().unsqueeze(0).unsqueeze(0)
    return grid, weight


def pad_pano_horizontal(pano: torch.Tensor, pad: int) -> torch.Tensor:
    """Periodicky dolepi `pad` sloupcu na obe strany ERP obrazu, aby slo bezpecne vzorkovat
    vysece i pres sev u=0/pano_w (napr. pohled primo dopredu, yaw_offset blizko 0)."""
    left = pano[..., -pad:]
    right = pano[..., :pad]
    return torch.cat([left, pano, right], dim=-1)


def make_extract_grid(
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_size: int,
    pano_w: int,
    pano_h: int,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    """Grid pro F.grid_sample, ktery z (periodicky dolepeneho) ERP panoramatu vytahne
    perspektivni vysec se stredem v (yaw_deg, pitch_deg) a danym FOV. Vraci [1,S,S,2]."""
    lin = torch.linspace(-0.5, 0.5, out_size, device=device, dtype=torch.float64)
    gx, gy = torch.meshgrid(lin, lin, indexing="xy")
    f = 0.5 / torch.tan(torch.deg2rad(torch.tensor(fov_deg / 2.0, device=device, dtype=torch.float64)))
    ray = torch.stack([gx, -gy, torch.full_like(gx, f)], dim=-1)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    right0, up0, fwd0 = ray[..., 0], ray[..., 1], ray[..., 2]

    pitch = torch.deg2rad(torch.tensor(pitch_deg, device=device, dtype=torch.float64))
    up1 = up0 * torch.cos(pitch) + fwd0 * torch.sin(pitch)
    fwd1 = -up0 * torch.sin(pitch) + fwd0 * torch.cos(pitch)
    right1 = right0

    yaw = torch.deg2rad(torch.tensor(yaw_deg, device=device, dtype=torch.float64))
    right2 = right1 * torch.cos(yaw) + fwd1 * torch.sin(yaw)
    fwd2 = -right1 * torch.sin(yaw) + fwd1 * torch.cos(yaw)
    up2 = up1

    az = torch.rad2deg(torch.atan2(right2, fwd2))
    el = torch.rad2deg(torch.atan2(up2, torch.hypot(right2, fwd2)))

    u = torch.remainder(az, 360.0) / 360.0 * pano_w + pad
    v = (90.0 - el) / 180.0 * pano_h

    padded_w = pano_w + 2 * pad
    gx_norm = u / padded_w * 2 - 1
    gy_norm = v / pano_h * 2 - 1
    return torch.stack([gx_norm, gy_norm], dim=-1).unsqueeze(0).float()


def extract_view(pano_padded: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """pano_padded: [1,C,H,W_padded]. Vraci [1,C,S,S]."""
    return F.grid_sample(pano_padded, grid, mode="bilinear", padding_mode="border", align_corners=False)


def fuse_views_to_erp(
    view_probs: list[torch.Tensor],
    view_params: list[tuple[float, float, float]],
    pano_w: int,
    pano_h: int,
    device: torch.device,
    out_w: int | None = None,
    out_h: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """view_probs[i]: [1,C,S,S] softmax vystup pro pohled s parametry view_params[i] = (yaw,pitch,fov).
    Vraci fuzovane pravdepodobnosti [1,C,out_h,out_w] v ERP prostoru (bez argmaxu).

    `pano_w`/`pano_h` popisuji puvodni rozliseni panoramatu (pro spravny vypocet azimutu/elevace
    kazdeho vysledneho pixelu), zatimco `out_w`/`out_h` je rozliseni, ve kterem se fuze skutecne
    pocita (nizsi, aby se buffer [1,C=65,H,W] vesel do pameti GPU -- viz SS4.3+4.4
    03_semanticka_segmentace.md, doporucen float16 akumulator).
    """
    out_w = out_w or pano_w
    out_h = out_h or pano_h
    rays = erp_pixel_rays(out_w, out_h, device)
    C = view_probs[0].shape[1]
    accum = torch.zeros(1, C, out_h, out_w, device=device, dtype=dtype)
    wsum = torch.zeros(1, 1, out_h, out_w, device=device, dtype=dtype)
    for probs, (yaw, pitch, fov) in zip(view_probs, view_params):
        grid, weight = view_sample_grid(rays, yaw, pitch, fov, device)
        sampled = F.grid_sample(probs.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        accum += (sampled * weight).to(dtype)
        wsum += weight.to(dtype)
    coverage = wsum[:, 0] > 1e-3  # [1,H,W] -- kde nemame zadny pohled (zenit/nadir), argmax by byl nesmyslny
    wsum = wsum.clamp(min=1e-3)
    return accum / wsum, coverage
