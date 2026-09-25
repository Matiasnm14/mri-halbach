"""Primer modelo diferenciable de un resonador Halbach cilíndrico.

Las longitudes están en metros, Br en teslas y el campo resultante en teslas.
Cada bloque se aproxima por una cuadratura de dipolos puntuales en 3D.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


DTYPE = torch.float64


@dataclass(frozen=True)
class Geometry:
    magnets: int = 16
    ring_radius: float = 0.10
    radial_width: float = 0.020
    tangential_width: float = 0.020
    length: float = 0.10
    remanence: float = 1.2
    roi_radius: float = 0.040
    roi_half_length: float = 0.025
    grid_xy: int = 9
    grid_z: int = 5
    quadrature: int = 3

    def validate(self) -> None:
        if self.magnets < 4 or self.grid_xy < 3 or self.grid_z < 2 or self.quadrature < 1:
            raise ValueError("Se requieren >=4 imanes, malla >=3x3x2 y cuadratura >=1.")
        values = (self.ring_radius, self.radial_width, self.tangential_width,
                  self.length, self.remanence, self.roi_radius, self.roi_half_length)
        if any(value <= 0 for value in values):
            raise ValueError("Las dimensiones y Br deben ser positivas.")
        if self.roi_radius >= self.ring_radius - self.radial_width / 2:
            raise ValueError("La ROI debe quedar dentro del diámetro interior.")
        if self.roi_half_length > self.length / 2:
            raise ValueError("La ROI debe quedar dentro de la longitud de los imanes.")
        if self.tangential_width >= 2 * self.ring_radius * math.sin(math.pi / self.magnets):
            raise ValueError("Los imanes vecinos se superponen aproximadamente.")


def sample_points(g: Geometry) -> torch.Tensor:
    x = torch.linspace(-g.roi_radius, g.roi_radius, g.grid_xy, dtype=DTYPE)
    z = torch.linspace(-g.roi_half_length, g.roi_half_length, g.grid_z, dtype=DTYPE)
    xx, yy, zz = torch.meshgrid(x, x, z, indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return points[points[:, 0].square() + points[:, 1].square() <= g.roi_radius**2]


def magnet_sources(g: Geometry) -> tuple[torch.Tensor, torch.Tensor]:
    """Centros de integración [N,Q,3] y ángulos azimutales [N]."""
    phi = 2 * math.pi * torch.arange(g.magnets, dtype=DTYPE) / g.magnets
    radial = torch.stack((phi.cos(), phi.sin(), torch.zeros_like(phi)), dim=-1)
    tangent = torch.stack((-phi.sin(), phi.cos(), torch.zeros_like(phi)), dim=-1)
    axial = torch.tensor([0.0, 0.0, 1.0], dtype=DTYPE)
    q = (torch.arange(g.quadrature, dtype=DTYPE) + 0.5) / g.quadrature - 0.5
    a, b, c = torch.meshgrid(q, q, q, indexing="ij")
    local = torch.stack((a, b, c), dim=-1).reshape(-1, 3)
    centers = (g.ring_radius * radial[:, None, :]
               + g.radial_width * local[None, :, 0, None] * radial[:, None, :]
               + g.tangential_width * local[None, :, 1, None] * tangent[:, None, :]
               + g.length * local[None, :, 2, None] * axial)
    return centers, phi


def field(points: torch.Tensor, sources: torch.Tensor, angles: torch.Tensor,
          g: Geometry) -> torch.Tensor:
    """Campo B [P,3] de bloques uniformemente magnetizados.

    m = Br * dV / mu0; al sustituir en la fórmula del dipolo queda Br*dV/(4*pi).
    """
    displacement = points[:, None, None, :] - sources[None, :, :, :]
    r2 = displacement.square().sum(dim=-1)
    direction = torch.stack((angles.cos(), angles.sin(), torch.zeros_like(angles)), dim=-1)
    projection = (displacement * direction[None, :, None, :]).sum(dim=-1)
    inv_r3 = r2.pow(-1.5)
    inv_r5 = r2.pow(-2.5)
    contribution = (3 * projection[..., None] * displacement * inv_r5[..., None]
                    - direction[None, :, None, :] * inv_r3[..., None])
    cell_volume = g.radial_width * g.tangential_width * g.length / g.quadrature**3
    return (g.remanence * cell_volume / (4 * math.pi)) * contribution.sum(dim=(1, 2))


def metrics(b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Homogeneidad de Bx, componente del campo principal del Halbach ideal."""
    bx = b[:, 0]
    mean_bx = bx.mean()
    rms = ((bx - mean_bx).square().mean()).sqrt()
    transverse = b[:, 1:].square().sum(dim=-1).mean().sqrt()
    scale = mean_bx.abs().clamp_min(1e-12)
    return {
        "mean_bx_t": mean_bx,
        "rms_ppm": 1e6 * rms / scale,
        "peak_to_peak_ppm": 1e6 * (bx.max() - bx.min()) / scale,
        "transverse_rms_ppm": 1e6 * transverse / scale,
    }


def optimize(g: Geometry, steps: int, max_correction_deg: float,
             error_deg: float, seed: int, learning_rate: float,
             device_name: str = "auto") -> dict:
    g.validate()
    if steps < 0 or max_correction_deg <= 0 or error_deg < 0 or learning_rate <= 0:
        raise ValueError("Parámetros de optimización inválidos.")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible; comprueba el controlador y nvidia-smi.")
    device = torch.device("cuda" if device_name == "cuda" or
                          (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    points = sample_points(g).to(device)
    sources, phi = magnet_sources(g)
    sources, phi = sources.to(device), phi.to(device)
    generator = torch.Generator().manual_seed(seed)
    error = torch.randn(g.magnets, generator=generator, dtype=DTYPE).to(device) * math.radians(error_deg)
    ideal_angles = 2 * phi
    initial_angles = ideal_angles + error

    with torch.no_grad():
        ideal = metrics(field(points, sources, ideal_angles, g))
        before = metrics(field(points, sources, initial_angles, g))

    raw = torch.nn.Parameter(torch.zeros(g.magnets, dtype=DTYPE, device=device))
    optimizer = torch.optim.Adam([raw], lr=learning_rate)
    max_rad = math.radians(max_correction_deg)
    for _ in range(steps):
        optimizer.zero_grad()
        correction = max_rad * raw.tanh()
        b = field(points, sources, initial_angles + correction, g)
        score = metrics(b)
        # Priorizar la uniformidad de Bx; limitar el campo transversal y la pérdida de intensidad.
        homogeneity = (score["rms_ppm"] / 1e6).square()
        transverse = (score["transverse_rms_ppm"] / 1e6).square()
        strength_floor = 0.95 * ideal["mean_bx_t"].abs()
        strength_penalty = (torch.relu(strength_floor - score["mean_bx_t"].abs())
                            / strength_floor.clamp_min(1e-12)).square()
        regularization = 1e-7 * (correction / max_rad).square().mean()
        loss = homogeneity + 0.05 * transverse + 10 * strength_penalty + regularization
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        correction = max_rad * raw.tanh()
        after = metrics(field(points, sources, initial_angles + correction, g))

    def values(items: dict[str, torch.Tensor]) -> dict[str, float]:
        return {key: float(value) for key, value in items.items()}

    return {
        "model": "3D point-dipole quadrature; approximate, no demagnetization",
        "device": str(device),
        "torch_version": torch.__version__,
        "geometry": vars(g),
        "roi_points": len(points),
        "seed": seed,
        "orientation_error_deg": error_deg,
        "max_correction_deg": max_correction_deg,
        "ideal": values(ideal),
        "before": values(before),
        "after": values(after),
        "corrections_deg": [float(v) for v in torch.rad2deg(correction)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magnets", type=int, default=16)
    parser.add_argument("--ring-radius", type=float, default=0.10, help="metros")
    parser.add_argument("--radial-width", type=float, default=0.020, help="metros")
    parser.add_argument("--tangential-width", type=float, default=0.020, help="metros")
    parser.add_argument("--length", type=float, default=0.10, help="metros")
    parser.add_argument("--remanence", type=float, default=1.2, help="teslas")
    parser.add_argument("--roi-radius", type=float, default=0.040, help="metros")
    parser.add_argument("--roi-half-length", type=float, default=0.025, help="metros")
    parser.add_argument("--grid-xy", type=int, default=9)
    parser.add_argument("--grid-z", type=int, default=5)
    parser.add_argument("--quadrature", type=int, default=3)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--error-deg", type=float, default=3.0)
    parser.add_argument("--max-correction-deg", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("resultado.json"))
    args = parser.parse_args()
    g = Geometry(args.magnets, args.ring_radius, args.radial_width,
                 args.tangential_width, args.length, args.remanence,
                 args.roi_radius, args.roi_half_length, args.grid_xy,
                 args.grid_z, args.quadrature)
    result = optimize(g, args.steps, args.max_correction_deg, args.error_deg,
                      args.seed, args.learning_rate, args.device)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Dispositivo: {result['device']}")
    for label in ("ideal", "before", "after"):
        item = result[label]
        print(f"{label:>6}: Bx={item['mean_bx_t']*1e3:.3f} mT, "
              f"RMS={item['rms_ppm']:.0f} ppm, pico-pico={item['peak_to_peak_ppm']:.0f} ppm")
    print(f"Resultado: {args.output}")


if __name__ == "__main__":
    main()
