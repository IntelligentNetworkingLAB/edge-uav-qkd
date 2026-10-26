"""Free-space optical (FSO) channel + decoy-state BB84 secret-key-rate model.

The channel transmittance combines beam-spreading geometric loss, atmospheric
attenuation (Beer-Lambert), and detector efficiency. The secret key rate uses the
standard GLLP / decoy-state asymptotic formula (Ma et al., PRA 72, 012326 (2005)),
which with decoy states lets the single-photon contribution be estimated; here we
use the well-known closed-form single-photon yield/error expressions.

All distances in metres, rates per pulse unless multiplied by the pulse rate.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelParams:
    # --- optics ---
    wavelength_nm: float = 1550.0        # telecom band
    w0_m: float = 0.05                   # transmitter beam waist (5 cm)
    divergence_rad: float = 5.0e-4       # sub-mrad far-field half-angle divergence
    rx_aperture_radius_m: float = 0.15   # receiver telescope radius (30 cm dia.)
    # --- atmosphere ---
    atten_db_per_km: float = 0.43        # clear-air extinction at 1550 nm
    # --- detector / system ---
    det_efficiency: float = 0.5          # detector + coupling efficiency
    # --- decoy-BB84 protocol parameters ---
    mu: float = 0.5                      # signal-state mean photon number
    dark_count: float = 1.0e-6           # background/dark yield Y0 per pulse
    misalignment: float = 0.015          # optical/system error e_d
    err_corr_ineff: float = 1.16         # error-correction inefficiency f(E)
    pulse_rate_hz: float = 1.0e9         # source repetition rate (1 GHz)
    sift_factor: float = 0.5             # BB84 basis-sift fraction q


E0 = 0.5  # error rate of background counts


def _h2(x: np.ndarray | float) -> np.ndarray | float:
    """Binary Shannon entropy, safe at the {0,1} boundaries."""
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def geometric_transmittance(distance_m, p: ChannelParams):
    """Fraction of beam power captured by the receiver aperture (Gaussian beam)."""
    L = np.asarray(distance_m, dtype=float)
    w = np.sqrt(p.w0_m**2 + (p.divergence_rad * L) ** 2)   # beam radius at range L
    a = p.rx_aperture_radius_m
    return 1.0 - np.exp(-2.0 * a**2 / w**2)


def atmospheric_transmittance(distance_m, p: ChannelParams):
    """Beer-Lambert atmospheric transmittance over the slant path."""
    L_km = np.asarray(distance_m, dtype=float) / 1000.0
    return 10.0 ** (-p.atten_db_per_km * L_km / 10.0)


def transmittance(distance_m, p: ChannelParams):
    """Total channel transmittance eta = geometric * atmospheric * detector."""
    return (
        geometric_transmittance(distance_m, p)
        * atmospheric_transmittance(distance_m, p)
        * p.det_efficiency
    )


def secret_key_rate_per_pulse(eta, p: ChannelParams):
    """Decoy-state BB84 asymptotic secret key rate per pulse, given transmittance.

    Returns a non-negative array (rate clipped at 0 below the secure threshold).
    """
    eta = np.asarray(eta, dtype=float)
    Y0, ed, mu = p.dark_count, p.misalignment, p.mu

    # Overall gain and QBER of the signal state.
    Q_mu = Y0 + 1.0 - np.exp(-eta * mu)
    E_mu = (E0 * Y0 + ed * (1.0 - np.exp(-eta * mu))) / Q_mu

    # Single-photon yield / error (decoy-state estimates).
    Y1 = Y0 + eta - Y0 * eta
    Q1 = Y1 * mu * np.exp(-mu)
    e1 = (E0 * Y0 + ed * eta) / Y1

    R = p.sift_factor * (
        Q1 * (1.0 - _h2(e1)) - Q_mu * p.err_corr_ineff * _h2(E_mu)
    )
    return np.maximum(R, 0.0)


def secret_key_rate(distance_m, p: ChannelParams):
    """End-to-end secret key rate in bits/s for a UAV-to-ground link at `distance_m`."""
    eta = transmittance(distance_m, p)
    return secret_key_rate_per_pulse(eta, p) * p.pulse_rate_hz


if __name__ == "__main__":
    p = ChannelParams()
    for L in [200, 500, 1000, 2000, 4000, 8000]:
        eta = transmittance(L, p)
        R = secret_key_rate(L, p)
        print(f"L={L:5d} m  eta={eta:.3e}  R={R/1e3:9.2f} kbit/s")
