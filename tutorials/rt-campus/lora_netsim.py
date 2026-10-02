"""Packet-level LoRa network simulation on the ray-traced campus.

Three layers that can be used separately:

1. Link gains   -- Sionna RT antenna-to-antenna channel gain between positions
                   (``load_campus_scene``, ``gain_matrix_db``, ``pair_gains_db``).
2. Calibration  -- fit offsets, shadowing/fading spread, noise floor and a
                   packet-error curve to measured RSSI/SNR logs
                   (``load_measurements``, ``fit_calibration``, ``fit_per_curve``).
3. Simulation   -- discrete-event packet simulation with airtime, queueing,
                   co-/cross-SF collisions with capture, half-duplex radios,
                   optional ACKs/retries and duty cycle (``simulate``, ``summarize``).

Only layer 1 imports Sionna RT / Mitsuba; layers 2 and 3 need NumPy only.

All propagation, PER and interference defaults are explicit assumptions, not
measurements. Replace them with calibrated values before drawing conclusions.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import itertools
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
THERMAL_NOISE_DBM_HZ = -174.0  # kT at 290 K
HW_DIPOLE_PEAK_DBI = 10 * math.log10(1.643)  # Peak gain of Sionna's "hw_dipole" pattern

# Demodulator SNR limits of SX126x/SX127x LoRa modems (Semtech datasheets).
SNR_LIMIT_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}

# Minimum signal-to-interference ratio [dB] for the desired packet (row, SF7..SF12)
# to survive an interferer (column, SF7..SF12). Diagonal: co-SF capture threshold.
# Literature values (Goursaud & Gorce 2015, as tabulated in the ns-3 lorawan
# module); replace with receiver-specific collision measurements where available.
SIR_THRESHOLD_DB = np.array([
    [6, -16, -18, -19, -19, -20],
    [-24, 6, -20, -22, -22, -22],
    [-27, -27, 6, -23, -25, -25],
    [-30, -30, -30, 6, -26, -28],
    [-33, -33, -33, -33, 6, -29],
    [-36, -36, -36, -36, -36, 6],
], dtype=float)

# Campus geometry audited in 1_peer_to_peer_lora_923mhz.ipynb. The F2 RF slab is
# modeled at its mid-plane, below the finished walking surface.
CAMPUS_FLOORS = {
    "F1": {"surface_z": -1.4370243549, "support_z": -1.4370243549},
    "F2": {"surface_z": 2.5087316036, "support_z": 2.0540938},
}
# Public regions (xmin, xmax, ymin, ymax) where nodes may be placed.
CAMPUS_REGIONS = [
    {"name": "South corridor", "floor": "F1", "bounds": (-57, -28, 13, 15.5)},
    {"name": "Library", "floor": "F1", "bounds": (-19, -3, 35, 44)},
    {"name": "Atrium ground floor", "floor": "F1", "bounds": (-34, -24, 39, 44)},
    {"name": "Public computing suite", "floor": "F2", "bounds": (-58, -38, 25.5, 28.5)},
    {"name": "Atrium east walkway", "floor": "F2", "bounds": (-16, -14, 25, 43)},
    {"name": "South junction", "floor": "F2", "bounds": (-32, -25, 14, 20)},
]


# ---------------------------------------------------------------------------
# LoRa PHY configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoRaConfig:
    """Radio settings of one transmitter/receiver (SF7-SF12)."""

    sf: int = 7
    bw_hz: float = 125e3
    cr: int = 1                 # 1..4 means coding rate 4/5..4/8
    payload_bytes: int = 16
    preamble_symbols: int = 8
    explicit_header: bool = True
    crc: bool = True
    tx_power_dbm: float = 14.0
    channel: int = 0            # Packets on different channels do not interfere
    ldro: bool | None = None    # Low-data-rate optimization; None = automatic

    def __post_init__(self):
        if self.sf not in SNR_LIMIT_DB:
            raise ValueError("sf must be 7..12")
        if self.cr not in (1, 2, 3, 4):
            raise ValueError("cr must be 1..4 (4/5..4/8)")
        if self.bw_hz <= 0 or not 0 <= self.payload_bytes <= 255:
            raise ValueError("invalid bandwidth or payload length")

    @property
    def symbol_time_s(self) -> float:
        return (1 << self.sf) / self.bw_hz

    @property
    def time_on_air_s(self) -> float:
        """Semtech LoRa time-on-air formula for SF7-SF12."""
        de = int(self.symbol_time_s >= 16e-3) if self.ldro is None else int(self.ldro)
        ih = 0 if self.explicit_header else 1
        numerator = 8 * self.payload_bytes - 4 * self.sf + 28 + 16 * int(self.crc) - 20 * ih
        payload_symbols = 8 + max(math.ceil(numerator / (4 * (self.sf - 2 * de))) * (self.cr + 4), 0)
        return (self.preamble_symbols + 4.25 + payload_symbols) * self.symbol_time_s

    @property
    def snr_limit_db(self) -> float:
        return SNR_LIMIT_DB[self.sf]


@dataclass(frozen=True, eq=False)
class RadioModel:
    """Link-budget, noise, packet-error and energy assumptions.

    Received power [dBm] = tx_power - 2*cable_loss + gain_db[i, j]
                           + static shadowing (per link) + fading (per packet).
    The gain matrix already contains antenna patterns and any calibration offset.
    """

    noise_figure_db: float = 6.0
    cable_loss_db: float = 1.0          # Per endpoint
    shadowing_std_db: float = 0.0       # Static per-link model error (symmetric)
    fading_std_db: float = 2.0          # Per packet and receiver; assumption until measured
    per_at_limit: float = 0.01          # Packet error rate at the SNR limit ...
    per_width_db: float = 0.5           # ... and logistic width of the PER curve
    per_snr50_offset_db: float | None = None  # Fitted 50%-PER point relative to the SNR limit
    detect_margin_db: float = 3.0       # Preamble detection down to limit - margin
    sir_threshold_db: np.ndarray = field(default_factory=lambda: SIR_THRESHOLD_DB.copy())
    rssi_report_std_db: float = 0.0     # Reporting noise added to simulated RSSI
    rssi_report_range_dbm: tuple = (-127.5, 0.0)  # SX126x RssiPkt register range
    snr_report_range_db: tuple = (-32.0, 31.75)   # SX126x SnrPkt register range; real
                                                  # estimators often saturate near +10 dB
    supply_v: float = 3.3               # Energy bookkeeping (editable assumptions;
    tx_current_ma: float = 45.0         #   check the radio datasheet for your
    rx_current_ma: float = 5.3          #   PA setting and receiver mode)

    def noise_dbm(self, bw_hz: float) -> float:
        return THERMAL_NOISE_DBM_HZ + 10 * math.log10(bw_hz) + self.noise_figure_db

    def snr50_db(self, sf: int) -> float:
        if self.per_snr50_offset_db is not None:
            return SNR_LIMIT_DB[sf] + self.per_snr50_offset_db
        return SNR_LIMIT_DB[sf] - self.per_width_db * math.log(1 / self.per_at_limit - 1)

    def per(self, snr_db, sf: int):
        """Packet error probability versus SNR (logistic waterfall)."""
        x = (np.asarray(snr_db, dtype=float) - self.snr50_db(sf)) / self.per_width_db
        return 1 / (1 + np.exp(np.clip(x, -60, 60)))

    def sensitivity_dbm(self, sf: int, bw_hz: float = 125e3) -> float:
        return self.noise_dbm(bw_hz) + SNR_LIMIT_DB[sf]


# ---------------------------------------------------------------------------
# Layer 1: Sionna RT link gains
# ---------------------------------------------------------------------------

def load_campus_scene(scene_path, frequency_hz=923e6, pattern="hw_dipole", polarization="V"):
    """Load the campus XML and set single-antenna arrays. Set the Mitsuba variant first."""
    from sionna.rt import PlanarArray, load_scene

    scene = load_scene(str(scene_path), merge_shapes=True)
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern, polarization=polarization)
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern=pattern, polarization=polarization)
    return scene


def _local_offsets(radius_m, points):
    """Centre plus a horizontal ring, for local averaging of small-scale fading."""
    if points <= 1 or radius_m <= 0:
        return np.zeros((1, 3))
    angles = 2 * np.pi * np.arange(points - 1) / (points - 1)
    ring = np.column_stack([radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(points - 1)])
    return np.vstack([np.zeros((1, 3)), ring])


def trace_gain_db(scene, tx_position, rx_positions, *, bandwidth_hz=125e3, antenna_correction_db=0.0,
                  max_depth=8, samples_per_src=1_000_000, diffraction=True, refraction=True,
                  seed=42, num_freq=17, local_avg_radius_m=0.0, local_avg_points=1, batch_size=None):
    """Channel power gain [dB] from one transmitter to many receivers.

    Linear power is averaged over the channel bandwidth (coherent multipath per
    frequency sample) and over optional local points around each receiver.
    ``antenna_correction_db`` is added once per endpoint. Returns -inf where no
    path was found and NaN where the solver produced non-finite coefficients.

    Solver memory grows with receivers x rays x depth (about 0.3 GB per receiver at
    1M rays and depth 8 on the CPU backend). ``batch_size`` receivers are solved at
    once; the default keeps one solve near 5 GB.
    """
    import mitsuba as mi
    from sionna.rt import PathSolver, Receiver, Transmitter

    tx = np.asarray(tx_position, dtype=float).reshape(3)
    rx = np.asarray(rx_positions, dtype=float).reshape(-1, 3)
    if scene.transmitters or scene.receivers:
        raise ValueError("Remove existing transmitters/receivers from the scene first.")
    if np.any(np.linalg.norm(rx - tx, axis=1) < 0.5):
        raise ValueError("Receivers must be at least 0.5 m from the transmitter.")
    offsets = _local_offsets(local_avg_radius_m, local_avg_points)
    probes = (rx[:, None, :] + offsets[None]).reshape(-1, 3)
    freqs = np.linspace(-bandwidth_hz / 2, bandwidth_hz / 2, num_freq)
    if batch_size is None:
        batch_size = max(1, int(16 * 8e6 / (samples_per_src * max(max_depth, 1))))
    solver = PathSolver(deterministic=True)
    power = np.empty(len(probes))
    for start in range(0, len(probes), batch_size):
        chunk = probes[start:start + batch_size]
        try:
            scene.add(Transmitter(name="probe_tx", position=mi.Point3f(*map(float, tx))))
            for k, p in enumerate(chunk):
                scene.add(Receiver(name=f"probe_rx_{k}", position=mi.Point3f(*map(float, p))))
            paths = solver(scene=scene, max_depth=max_depth, samples_per_src=samples_per_src,
                           los=True, specular_reflection=True, refraction=refraction,
                           diffuse_reflection=False, diffraction=diffraction,
                           synthetic_array=True, seed=seed)
            h = paths.cfr(frequencies=mi.Float(freqs), normalize=False, out_type="numpy")
            h = h.reshape(len(chunk), -1, len(freqs))  # single antennas, single time step
            power[start:start + len(chunk)] = np.mean(np.abs(h) ** 2, axis=(1, 2))
        finally:
            for name in list(scene.transmitters) + list(scene.receivers):
                scene.remove(name)
    gain = power.reshape(len(rx), len(offsets)).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(gain) + 2 * antenna_correction_db


def _cache_file(cache_dir, tag, positions, kwargs):
    key = json.dumps({"tag": tag, "kw": kwargs, "pos": np.round(positions, 4).tolist()}, sort_keys=True)
    return Path(cache_dir) / f"gains_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def gain_matrix_db(scene, positions, *, symmetrize=True, cache_dir=None, cache_tag="", progress=print, **trace_kw):
    """N x N gain matrix [dB] (row = transmitter, column = receiver) for node positions.

    Each node is traced as a transmitter to all others. Physical reciprocity means
    forward/reverse gains should agree; their difference is returned as a
    convergence diagnostic and, with ``symmetrize``, the two directions are
    averaged in linear power. Results are cached when ``cache_dir`` is given.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(positions)
    cache = _cache_file(cache_dir, cache_tag, positions, trace_kw) if cache_dir else None
    if cache is not None and cache.is_file():
        raw = np.load(cache)["raw_gain_db"]
        if progress:
            progress(f"Loaded cached gains: {cache}")
    else:
        raw = np.full((n, n), np.nan)
        for i in range(n):
            others = np.arange(n) != i
            raw[i, others] = trace_gain_db(scene, positions[i], positions[others], **trace_kw)
            if progress:
                progress(f"traced node {i + 1}/{n}")
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, raw_gain_db=raw, positions=positions)
    with np.errstate(invalid="ignore"):
        reciprocity = np.abs(raw - raw.T)
        reciprocity[np.isneginf(raw) ^ np.isneginf(raw.T)] = np.inf
    gain = raw
    if symmetrize:
        # Average both directions in linear power; if the solver returned NaN in
        # one direction only, fall back to the other direction.
        forward, reverse = 10 ** (raw / 10), 10 ** (raw.T / 10)
        with np.errstate(divide="ignore", invalid="ignore"):
            linear = np.where(np.isnan(forward), reverse,
                              np.where(np.isnan(reverse), forward, (forward + reverse) / 2))
            gain = 10 * np.log10(linear)
    np.fill_diagonal(gain, np.nan)
    return {"gain_db": gain, "raw_gain_db": raw, "reciprocity_error_db": reciprocity}


def pair_gains_db(scene, tx_positions, rx_positions, progress=print, **trace_kw):
    """Gain [dB] for arbitrary (tx, rx) position pairs, grouped by unique transmitter."""
    tx = np.asarray(tx_positions, dtype=float).reshape(-1, 3)
    rx = np.asarray(rx_positions, dtype=float).reshape(-1, 3)
    out = np.full(len(tx), np.nan)
    unique_tx, inverse = np.unique(np.round(tx, 3), axis=0, return_inverse=True)
    for k, position in enumerate(unique_tx):
        rows = np.flatnonzero(inverse.ravel() == k)
        unique_rx, rx_inverse = np.unique(np.round(rx[rows], 3), axis=0, return_inverse=True)
        gains = trace_gain_db(scene, position, unique_rx, **trace_kw)
        out[rows] = gains[rx_inverse.ravel()]
        if progress:
            progress(f"measurement transmitter position {k + 1}/{len(unique_tx)}")
    return out


# ---------------------------------------------------------------------------
# Node placement on the campus
# ---------------------------------------------------------------------------

def geometry_clear(scene, points, floor, *, floors=CAMPUS_FLOORS, clearance_m=0.35):
    """True where a point stands above the given floor sheet with horizontal clearance."""
    import mitsuba as mi

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.zeros(0, dtype=bool)
    hit = scene.mi_scene.ray_intersect(mi.Ray3f(mi.Point3f(points.T), mi.Vector3f(0, 0, -1)))
    valid = np.asarray(hit.is_valid()) & np.isclose(np.asarray(hit.p.z), floors[floor]["support_z"], atol=0.03)
    surface = floors[floor]["surface_z"]
    # Sparse rays are a screen, not a volumetric collision certificate.
    for fraction in (0.1, 0.5, 1.0):
        origins = points.copy()
        origins[:, 2] = surface + fraction * (points[:, 2] - surface)
        for angle in np.arange(8) * np.pi / 4:
            ray = mi.Ray3f(mi.Point3f(origins.T), mi.Vector3f(float(np.cos(angle)), float(np.sin(angle)), 0))
            ray.maxt = clearance_m
            valid &= ~np.asarray(scene.mi_scene.ray_test(ray))
    return valid


def sample_positions(scene, counts, *, regions=CAMPUS_REGIONS, floors=CAMPUS_FLOORS, seed=0,
                     antenna_height_m=1.3, min_spacing_m=5.0, clearance_m=0.35, max_attempts=2000,
                     existing=(), prefix="N"):
    """Seeded random placement: ``counts[k]`` nodes in ``regions[k]``.

    ``existing`` nodes (dicts with floor and position, e.g. a gateway) are kept
    clear by the same spacing rule. Returns a list of dicts with name, floor,
    region and position.
    """
    if len(counts) != len(regions):
        raise ValueError("Provide one count per region.")
    rng = np.random.default_rng(seed)
    nodes = []
    placed = list(existing)
    for region, count in zip(regions, counts):
        xmin, xmax, ymin, ymax = region["bounds"]
        z = floors[region["floor"]]["surface_z"] + antenna_height_m
        for _ in range(count):
            for _ in range(max_attempts):
                p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax), z])
                spaced = all(n["floor"] != region["floor"] or
                             np.linalg.norm(p[:2] - np.asarray(n["position"])[:2]) >= min_spacing_m for n in placed)
                if spaced and geometry_clear(scene, [p], region["floor"], floors=floors, clearance_m=clearance_m)[0]:
                    node = {"name": f"{prefix}{len(nodes) + 1}", "floor": region["floor"],
                            "region": region["name"], "position": p}
                    nodes.append(node)
                    placed.append(node)
                    break
            else:
                raise ValueError(f"Could not place a node in {region['name']}; reduce count or spacing.")
    return nodes


def round_robin_counts(n, num_regions):
    """Spread n nodes over regions as evenly as possible."""
    return [n // num_regions + (k < n % num_regions) for k in range(num_regions)]


def floor_outline_segments(scene, z):
    """2D wall outline: intersect every scene triangle with the plane at height z."""
    import mitsuba as mi

    segments = []
    for mesh in scene.mi_scene.shapes():
        params = mi.traverse(mesh)
        vertices = np.asarray(params["vertex_positions"]).reshape(-1, 3)
        triangles = vertices[np.asarray(params["faces"]).reshape(-1, 3)]
        triangles = triangles[(triangles[:, :, 2].min(1) < z) & (triangles[:, :, 2].max(1) > z)]
        crossings = np.full((len(triangles), 3, 2), np.nan)
        for e, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
            p, q = triangles[:, a], triangles[:, b]
            crosses = (p[:, 2] < z) != (q[:, 2] < z)
            t = (z - p[crosses, 2]) / (q[crosses, 2] - p[crosses, 2])
            crossings[crosses, e] = (p[crosses] + t[:, None] * (q[crosses] - p[crosses]))[:, :2]
        segments.extend(crossings[np.isfinite(crossings).all(-1)].reshape(-1, 2, 2))
    return np.asarray(segments)


def floor_class_matrix(floors):
    """'same_floor' / 'cross_floor' label for each node pair."""
    floors = np.asarray(floors)
    return np.where(floors[:, None] == floors[None, :], "same_floor", "cross_floor")


# ---------------------------------------------------------------------------
# Layer 2: calibration from measured RSSI/SNR
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ("tx_x", "tx_y", "tx_z", "rx_x", "rx_y", "rx_z", "rssi_dbm", "snr_db")


def load_measurements(path):
    """Read a measurement CSV (see measurements_template.csv).

    Required columns: tx_x, tx_y, tx_z, rx_x, rx_y, rx_z (scene metres),
    rssi_dbm, snr_db. Optional: received (1/0; lost packets may leave RSSI/SNR
    empty), tx_power_dbm, sf, bw_hz, link_class, and any identifiers.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty measurement file.")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    def numeric(name, default=np.nan):
        return np.array([float(r[name]) if r.get(name, "") not in ("", None) else default for r in rows])

    data = {name: numeric(name) for name in REQUIRED_COLUMNS}
    data["received"] = (numeric("received", 1.0) > 0) if "received" in rows[0] else np.isfinite(data["rssi_dbm"])
    for name in ("tx_power_dbm", "sf", "bw_hz"):
        if name in rows[0]:
            data[name] = numeric(name)
    data["link_class"] = np.array([r.get("link_class") or "all" for r in rows])
    data["tx_position"] = np.column_stack([data["tx_x"], data["tx_y"], data["tx_z"]])
    data["rx_position"] = np.column_stack([data["rx_x"], data["rx_y"], data["rx_z"]])
    return data


def signal_dbm_from_report(rssi_dbm, snr_db, rssi_kind="total"):
    """Desired-signal power from a reported RSSI/SNR pair.

    ``rssi_kind="total"``: RSSI is in-band power of signal + noise (SX126x
    packet RSSI, SX127x packet RSSI). ``"signal"``: RSSI already estimates the
    despread signal (e.g. SX126x SignalRssiPkt).
    """
    rssi_dbm, snr_db = np.asarray(rssi_dbm, float), np.asarray(snr_db, float)
    if rssi_kind == "signal":
        return rssi_dbm
    return rssi_dbm + snr_db - 10 * np.log10(1 + 10 ** (snr_db / 10))


def noise_dbm_from_report(rssi_dbm, snr_db, rssi_kind="total"):
    """Noise-plus-interference power implied by a reported RSSI/SNR pair."""
    return signal_dbm_from_report(rssi_dbm, snr_db, rssi_kind) - np.asarray(snr_db, float)


@dataclass
class Calibration:
    offset_db: dict            # Per link class: measured minus predicted signal power
    shadowing_std_db: float    # Location-to-location model error after offsets
    fading_std_db: float | None  # Packet-to-packet spread at a fixed location pair
    noise_figure_db: float | None
    rmse_uncalibrated_db: float
    rmse_calibrated_loo_db: float  # Leave-one-location-pair-out
    pairs: list                # Per location pair: class, n, mean residual, std

    def apply(self, gain_db, link_class=None):
        """Add the class offsets to a gain matrix (``link_class`` matrix or None = 'all')."""
        gain_db = np.asarray(gain_db, float)
        if link_class is None:
            link_class = np.full(gain_db.shape, "all")
        fallback = float(np.mean(list(self.offset_db.values())))
        offsets = np.vectorize(lambda c: self.offset_db.get(c, fallback), otypes=[float])(link_class)
        return gain_db + offsets

    def radio_model(self, base: RadioModel) -> RadioModel:
        """Return ``base`` with the fitted variance and noise terms."""
        updates = {"shadowing_std_db": self.shadowing_std_db}
        if self.fading_std_db is not None:
            updates["fading_std_db"] = self.fading_std_db
        if self.noise_figure_db is not None:
            updates["noise_figure_db"] = self.noise_figure_db
        return replace(base, **updates)


def fit_calibration(meas, predicted_gain_db, *, tx_power_dbm=14.0, cable_loss_db=1.0,
                    bandwidth_hz=125e3, rssi_kind="total", noise_snr_max_db=5.0):
    """Fit offsets and spreads from received packets with known positions.

    residual = measured signal power - (tx power - 2*cable loss + predicted gain).
    Packets are grouped per (tx position, rx position, class); each location pair
    counts once for offsets and shadowing, so long logs at one spot do not dominate.

    The noise floor uses only packets with SNR <= ``noise_snr_max_db``: modem SNR
    estimates saturate on strong links, where RSSI - SNR would overstate the noise.

    Lost packets carry no RSSI, so weak links are under-represented (censoring):
    offsets fitted only on received packets are biased optimistic near the
    sensitivity limit. Use ``fit_per_curve`` with the received flags for those links.
    """
    received = meas["received"] & np.isfinite(meas["rssi_dbm"]) & np.isfinite(meas["snr_db"])
    predicted = np.asarray(predicted_gain_db, float)
    usable = received & np.isfinite(predicted)
    if usable.sum() < 2:
        raise ValueError("Need at least two received packets with finite predictions.")
    tx_power = meas.get("tx_power_dbm", np.full(len(predicted), tx_power_dbm))
    tx_power = np.where(np.isfinite(tx_power), tx_power, tx_power_dbm)
    signal = signal_dbm_from_report(meas["rssi_dbm"], meas["snr_db"], rssi_kind)
    residual = signal - (tx_power - 2 * cable_loss_db + predicted)

    keys = [(tuple(np.round(t, 2)), tuple(np.round(r, 2)), c) for t, r, c in
            zip(meas["tx_position"], meas["rx_position"], meas["link_class"])]
    groups = {}
    for k in np.flatnonzero(usable):
        groups.setdefault(keys[k], []).append(residual[k])
    pairs = [{"tx": key[0], "rx": key[1], "link_class": key[2], "n": len(v),
              "mean_db": float(np.mean(v)), "std_db": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan}
             for key, v in groups.items()]
    classes = sorted({p["link_class"] for p in pairs})
    offset = {c: float(np.mean([p["mean_db"] for p in pairs if p["link_class"] == c])) for c in classes}

    deviations = np.array([p["mean_db"] - offset[p["link_class"]] for p in pairs])
    dof = len(pairs) - len(classes)
    shadowing = 0.0
    within = [(p["n"] - 1, p["std_db"] ** 2) for p in pairs if p["n"] > 1]
    fading = float(np.sqrt(sum(d * v for d, v in within) / sum(d for d, _ in within))) if within else None
    if dof > 0:
        between = float(np.sum(deviations ** 2) / dof)
        # Remove the part of the between-pair spread explained by averaging fading.
        if fading is not None:
            between -= float(np.mean([fading ** 2 / p["n"] for p in pairs]))
        shadowing = float(np.sqrt(max(between, 0.0)))

    loo = []
    for k, p in enumerate(pairs):
        same = [q["mean_db"] for j, q in enumerate(pairs) if j != k and q["link_class"] == p["link_class"]]
        rest = same or [q["mean_db"] for j, q in enumerate(pairs) if j != k]
        if rest:
            loo.append(p["mean_db"] - np.mean(rest))
    low_snr = received & (meas["snr_db"] <= noise_snr_max_db)
    noise = noise_dbm_from_report(meas["rssi_dbm"][low_snr], meas["snr_db"][low_snr], rssi_kind)
    noise_figure = float(np.median(noise) - THERMAL_NOISE_DBM_HZ - 10 * np.log10(bandwidth_hz)) if len(noise) else None
    return Calibration(
        offset_db=offset, shadowing_std_db=shadowing, fading_std_db=fading, noise_figure_db=noise_figure,
        rmse_uncalibrated_db=float(np.sqrt(np.mean([p["mean_db"] ** 2 for p in pairs]))),
        rmse_calibrated_loo_db=float(np.sqrt(np.mean(np.square(loo)))) if loo else np.nan,
        pairs=pairs)


def fit_per_curve(snr_db, received, sf, iterations=50):
    """Fit P(received) = 1 - PER with PER = 1/(1+exp((snr - snr50)/w)).

    ``snr_db`` is the predicted (calibrated, mean) SNR of every transmitted packet,
    received or not. SNRs are referenced to each packet's SF limit, so mixed SFs
    share one curve. Returns (snr50 offset from the SF limit [dB], width [dB]).
    Because the fit uses mean SNR per location, its width also absorbs fading:
    simulate with ``fading_std_db=0`` when using it.
    """
    x = np.asarray(snr_db, float) - np.vectorize(SNR_LIMIT_DB.get)(np.asarray(sf, int))
    y = np.asarray(received, float)
    if y.min() == y.max():
        raise ValueError("Need both received and lost packets to fit a PER curve.")
    if x[y == 0].max() < x[y == 1].min():
        raise ValueError("Received and lost packets do not overlap in predicted SNR (perfect separation): "
                         "the curve width is unidentifiable. Log more packets near the sensitivity limit.")
    centre, scale = x.mean(), x.std() or 1.0
    X = np.column_stack([np.ones_like(x), (x - centre) / scale])

    def log_likelihood(b):
        z = X @ b
        return float(np.sum(y * z - np.logaddexp(0, z)))

    beta = np.zeros(2)
    ll = log_likelihood(beta)
    for _ in range(iterations):  # Newton/IRLS with step halving on standardized SNR
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -60, 60)))
        w = np.maximum(p * (1 - p), 1e-12)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(2), X.T @ (y - p))
        for _ in range(30):
            candidate = beta + step
            new_ll = log_likelihood(candidate)
            if new_ll >= ll - 1e-12:
                break
            step /= 2
        beta, ll = candidate, new_ll
        if np.max(np.abs(step)) < 1e-9:
            break
    beta = np.array([beta[0] - beta[1] * centre / scale, beta[1] / scale])
    if beta[1] <= 0:
        raise ValueError("Reception does not increase with SNR in these data; check inputs.")
    return float(-beta[0] / beta[1]), float(1 / beta[1])


# ---------------------------------------------------------------------------
# Layer 3: discrete-event network simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    """One radio. Gateways only receive (and send ACKs); end nodes generate traffic."""

    name: str
    lora: LoRaConfig = LoRaConfig()
    gateway: bool = False
    dest: tuple | None = None       # Receiver indices; None = all gateways (delivered if any decodes)
    period_s: float = 60.0          # Mean (poisson) or exact (periodic) generation interval
    traffic: str = "poisson"        # "poisson" | "periodic"
    start_offset_s: float | None = None  # First generation time; None = uniform in [0, period)
    rx_always_on: bool = False      # End nodes otherwise listen only while awaiting an ACK
    demod_paths: int | None = None  # Concurrent receptions; default 8 for gateways, 1 otherwise
    extra_noise_db: float = 0.0     # Local noise rise above thermal + noise figure


@dataclass(frozen=True)
class SimConfig:
    duration_s: float = 3600.0      # Traffic generation stops here; pending retries drain
    confirmed: bool = False         # ACK + retransmissions
    max_retransmissions: int = 3
    ack_delay_s: float = 1.0        # After the end of the data packet (LoRaWAN RX1-like)
    ack_payload_bytes: int = 4
    backoff_s: tuple = (1.0, 3.0)   # Uniform random backoff before a retransmission
    duty_cycle: float | None = None  # e.g. 0.01 => off-time = 99 x airtime after each TX
    queue_limit: int = 8
    seed: int = 0


@dataclass
class SimResult:
    nodes: list
    messages: list        # One per generated application message
    receptions: list      # One per (transmission, intended receiver)
    transmissions: list
    sim: SimConfig
    radio: RadioModel
    energy_j: np.ndarray


def simulate(nodes, gain_db, radio: RadioModel = RadioModel(), sim: SimConfig = SimConfig()):
    """Run the packet-level simulation over a (calibrated) gain matrix [dB].

    Reception model per (packet, receiver):
      * the receiver must be listening, not transmitting (half-duplex), configured
        for the packet's channel/bandwidth (and SF unless it is a gateway), have a
        free demodulator, and detect the preamble (SNR >= limit - detect margin);
      * it fails if a node at the receiver starts transmitting during the packet,
        if any SF group of overlapping same-channel packets violates the SIR
        threshold (interference energy weighted by time overlap), or with
        probability PER(SNR).
    """
    nodes = list(nodes)
    n = len(nodes)
    gain = np.asarray(gain_db, dtype=float)
    if gain.shape != (n, n):
        raise ValueError("gain_db must be N x N for N nodes.")
    if np.isnan(gain[~np.eye(n, dtype=bool)]).any():
        raise ValueError("gain_db has NaN links (invalid solver results); fix or set them to -inf explicitly.")
    rng = np.random.default_rng(sim.seed)

    gateways = [i for i, node in enumerate(nodes) if node.gateway]
    dests = []
    for i, node in enumerate(nodes):
        if node.gateway:
            dests.append(())
            continue
        d = tuple(gateways) if node.dest is None else tuple(node.dest)
        if not d or i in d or any(not 0 <= j < n for j in d):
            raise ValueError(f"{node.name}: invalid destination {d}.")
        dests.append(d)
    always_on = [node.gateway or node.rx_always_on for node in nodes]
    for d in dests:
        for j in d:
            always_on[j] = True
    demod = [node.demod_paths or (8 if node.gateway else 1) for node in nodes]

    shadow = np.triu(rng.normal(0, radio.shadowing_std_db, (n, n)), 1)
    link_db = gain + shadow + shadow.T
    sir = radio.sir_threshold_db

    events, counter = [], itertools.count()

    def push(t, event, **data):
        heapq.heappush(events, (t, next(counter), event, data))

    state = [{"queue": deque(), "busy": False, "msg": None, "dc_free": 0.0, "awaiting": False,
              "token": 0, "rx_window_start": None} for _ in nodes]
    transmitting = [None] * n
    locks = [dict() for _ in nodes]           # receiver -> {tx_id: None}
    messages, receptions, transmissions = [], [], []
    recent = []                               # Transmissions that may still overlap
    max_airtime = [0.0]
    tx_time = np.zeros(n)
    rx_time = np.zeros(n)

    def try_start(i, t):
        st = state[i]
        if st["queue"] and not st["busy"]:
            st["busy"], st["msg"] = True, st["queue"][0]
            push(max(t, st["dc_free"]), "tx_start", src=i, kind="data", msg=st["msg"])

    def finish(i, t):
        st = state[i]
        st["msg"]["done_s"] = t
        st["queue"].popleft()
        st["busy"], st["msg"] = False, None
        try_start(i, t)

    def record(tx, r, outcome, **extra):
        receptions.append({"tx_id": tx["id"], "kind": tx["kind"], "msg_id": tx["msg_id"], "src": tx["src"],
                           "dst": r, "attempt": tx["attempt"], "start_s": tx["start"], "end_s": tx["end"],
                           "sf": tx["sf"], "channel": tx["channel"], "prx_dbm": float(tx["prx"][r]),
                           "outcome": outcome, **extra})

    def is_intended(tx, r):
        return r in dests[tx["src"]] if tx["kind"] == "data" else r == tx["target"]

    def ack_config(data_src, responder):
        return replace(nodes[data_src].lora, payload_bytes=sim.ack_payload_bytes,
                       tx_power_dbm=nodes[responder].lora.tx_power_dbm)

    def start_tx(t, src, kind, msg, target=None):
        cfg = nodes[src].lora if kind == "data" else ack_config(target, src)
        airtime = cfg.time_on_air_s
        for tx_id in list(locks[src]):            # Half-duplex: transmitting aborts receptions
            other = transmissions[tx_id]
            if is_intended(other, src):
                record(other, src, "half_duplex")
            del locks[src][tx_id]
        fading = rng.normal(0, radio.fading_std_db, n) if radio.fading_std_db > 0 else np.zeros(n)
        prx = cfg.tx_power_dbm - 2 * radio.cable_loss_db + link_db[src] + fading
        prx[src] = np.nan
        tx = {"id": len(transmissions), "src": src, "kind": kind, "msg_id": msg["id"], "target": target,
              "attempt": msg["attempts"] + (kind == "data"), "start": t, "end": t + airtime,
              "sf": cfg.sf, "bw": cfg.bw_hz, "channel": cfg.channel, "power_dbm": cfg.tx_power_dbm,
              "prx": prx}
        transmissions.append(tx)
        recent.append(tx)
        max_airtime[0] = max(max_airtime[0], airtime)
        transmitting[src] = tx["id"]
        tx_time[src] += airtime
        if sim.duty_cycle:
            state[src]["dc_free"] = tx["end"] + airtime * (1 / sim.duty_cycle - 1)
        if kind == "data":
            msg["attempts"] += 1
            if msg["first_tx_s"] is None:
                msg["first_tx_s"] = t
        for r in range(n):
            if r == src:
                continue
            reason = None
            rx_cfg = nodes[r].lora
            listening = always_on[r] or (state[r]["awaiting"] and kind == "ack")
            if transmitting[r] is not None:
                reason = "half_duplex"
            elif not listening:
                reason = "not_listening"
            elif (rx_cfg.channel != cfg.channel or rx_cfg.bw_hz != cfg.bw_hz
                  or (not nodes[r].gateway and rx_cfg.sf != cfg.sf)):
                reason = "config_mismatch"
            elif not np.isfinite(prx[r]) or prx[r] - (radio.noise_dbm(cfg.bw_hz) + nodes[r].extra_noise_db) \
                    < SNR_LIMIT_DB[cfg.sf] - radio.detect_margin_db:
                reason = "not_detected"
            elif len(locks[r]) >= demod[r]:
                reason = "receiver_busy"
            if reason is None:
                locks[r][tx["id"]] = None
            elif is_intended(tx, r):
                record(tx, r, reason)
        push(tx["end"], "tx_end", tx=tx["id"])
        return tx

    def evaluate(tx, r):
        p_mw = 10 ** (tx["prx"][r] / 10)
        noise_dbm = radio.noise_dbm(tx["bw"]) + nodes[r].extra_noise_db
        interference = Counter()
        n_interferers = 0
        duration = tx["end"] - tx["start"]
        for other in recent:
            if other is tx or other["channel"] != tx["channel"] or other["src"] == r:
                continue
            overlap = min(other["end"], tx["end"]) - max(other["start"], tx["start"])
            if overlap > 0:
                interference[other["sf"]] += 10 ** (other["prx"][r] / 10) * overlap / duration
                n_interferers += 1
        collided = any(10 * np.log10(p_mw / i_mw) < sir[tx["sf"] - 7, s - 7]
                       for s, i_mw in interference.items() if i_mw > 0)
        i_total = sum(interference.values())
        snr = tx["prx"][r] - noise_dbm
        sinr = 10 * np.log10(p_mw / (10 ** (noise_dbm / 10) + i_total))
        if collided:
            outcome = "collision"
        elif rng.random() < radio.per(snr, tx["sf"]):
            outcome = "per_loss"
        else:
            outcome = "ok"
        rssi = 10 * np.log10(p_mw + 10 ** (noise_dbm / 10) + i_total) + rng.normal(0, radio.rssi_report_std_db)
        rssi = np.clip(np.round(rssi * 2) / 2, *radio.rssi_report_range_dbm)
        snr_report = np.clip(np.round(sinr * 4) / 4, *radio.snr_report_range_db)
        return outcome, {"snr_db": float(snr), "sinr_db": float(sinr), "rssi_report_dbm": float(rssi),
                         "snr_report_db": float(snr_report), "n_interferers": n_interferers}

    for i, node in enumerate(nodes):
        if not node.gateway:
            first = rng.uniform(0, node.period_s) if node.start_offset_s is None else node.start_offset_s
            push(first, "gen", node=i)

    while events:
        t, _, event, data = heapq.heappop(events)
        if event == "gen":
            i = data["node"]
            node = nodes[i]
            if t > sim.duration_s:
                continue
            gap = node.period_s if node.traffic == "periodic" else rng.exponential(node.period_s)
            push(t + gap, "gen", node=i)
            msg = {"id": len(messages), "node": i, "gen_s": t, "attempts": 0, "first_tx_s": None,
                   "delivered": False, "delivered_s": None, "delivered_by": None, "acked": False,
                   "dropped": False, "done_s": None, "rssi_report_dbm": None, "snr_report_db": None}
            messages.append(msg)
            if len(state[i]["queue"]) >= sim.queue_limit:
                msg["dropped"] = True
            else:
                state[i]["queue"].append(msg)
                try_start(i, t)

        elif event == "tx_start":
            src = data["src"]
            if transmitting[src] is not None:
                if data["kind"] == "ack":
                    continue  # Radio busy: this ACK is never sent
                # Data waits until the radio finishes its current (ACK) transmission.
                push(transmissions[transmitting[src]]["end"] + 1e-6, "tx_start", **data)
                continue
            start_tx(t, src, data["kind"], data["msg"], data.get("target"))

        elif event == "tx_end":
            tx = transmissions[data["tx"]]
            src = tx["src"]
            transmitting[src] = None
            recent[:] = [o for o in recent if o["end"] > t - 2 * max_airtime[0]]
            outcomes = {}
            for r in range(n):
                if tx["id"] in locks[r]:
                    del locks[r][tx["id"]]
                    outcome, extra = evaluate(tx, r)
                    outcomes[r] = (outcome, extra)
                    if is_intended(tx, r):
                        record(tx, r, outcome, **extra)
            msg = messages[tx["msg_id"]]
            if tx["kind"] == "data":
                decoded = [r for r in dests[src] if outcomes.get(r, ("",))[0] == "ok"]
                if decoded and not msg["delivered"]:
                    best = max(decoded, key=lambda r: tx["prx"][r])
                    msg.update(delivered=True, delivered_s=tx["end"], delivered_by=best,
                               rssi_report_dbm=outcomes[best][1]["rssi_report_dbm"],
                               snr_report_db=outcomes[best][1]["snr_report_db"])
                if sim.confirmed:
                    ack_airtime = ack_config(src, src).time_on_air_s
                    if decoded:
                        responder = max(decoded, key=lambda r: tx["prx"][r])
                        push(tx["end"] + sim.ack_delay_s, "tx_start", src=responder, kind="ack", msg=msg,
                             target=src)
                    st = state[src]
                    st["awaiting"], st["token"], st["rx_window_start"] = True, st["token"] + 1, tx["end"]
                    push(tx["end"] + sim.ack_delay_s + ack_airtime + 0.05, "ack_timeout", node=src, token=st["token"])
                else:
                    finish(src, t)
            else:
                st = state[tx["target"]]
                if (outcomes.get(tx["target"], ("",))[0] == "ok" and st["awaiting"]
                        and st["msg"] is not None and st["msg"]["id"] == tx["msg_id"]):
                    msg["acked"] = True
                    rx_time[tx["target"]] += t - st["rx_window_start"]
                    st["awaiting"], st["token"] = False, st["token"] + 1
                    finish(tx["target"], t)

        elif event == "ack_timeout":
            i = data["node"]
            st = state[i]
            if data["token"] != st["token"] or not st["awaiting"]:
                continue
            st["awaiting"] = False
            rx_time[i] += t - st["rx_window_start"]
            msg = st["msg"]
            if msg["attempts"] <= sim.max_retransmissions:
                push(max(t + rng.uniform(*sim.backoff_s), st["dc_free"]), "tx_start", src=i, kind="data", msg=msg)
            else:
                finish(i, t)

    span = max([sim.duration_s] + [m["done_s"] or 0 for m in messages])
    for i in range(n):
        if always_on[i]:
            rx_time[i] = span - tx_time[i]
    energy = radio.supply_v * 1e-3 * (radio.tx_current_ma * tx_time + radio.rx_current_ma * rx_time)
    for tx in transmissions:
        tx["prx"] = tx["prx"].tolist()
    return SimResult(nodes, messages, receptions, transmissions, sim, radio, energy)


def summarize(result: SimResult):
    """Application- and link-level metrics, overall and per node."""
    nodes, msgs, recs = result.nodes, result.messages, result.receptions
    span = max([result.sim.duration_s] + [m["done_s"] or 0 for m in msgs])

    def latency_stats(ms):
        lat = np.array([m["delivered_s"] - m["gen_s"] for m in ms if m["delivered"]])
        if not len(lat):
            return {"latency_mean_s": np.nan, "latency_p50_s": np.nan, "latency_p95_s": np.nan}
        return {"latency_mean_s": float(lat.mean()), "latency_p50_s": float(np.median(lat)),
                "latency_p95_s": float(np.percentile(lat, 95))}

    def link_stats(rs):
        data = [r for r in rs if r["kind"] == "data"]
        ok = [r for r in data if r["outcome"] == "ok"]
        out = {"attempt_prr": len(ok) / len(data) if data else np.nan}
        for key in ("rssi_report_dbm", "snr_report_db", "sinr_db"):
            values = [r[key] for r in ok]
            out[f"{key}_mean"] = float(np.mean(values)) if values else np.nan
        return out

    per_node = []
    for i, node in enumerate(nodes):
        if node.gateway:
            continue
        ms = [m for m in msgs if m["node"] == i]
        rs = [r for r in recs if r["src"] == i]
        delivered = sum(m["delivered"] for m in ms)
        per_node.append({
            "node": node.name, "sf": node.lora.sf, "tx_power_dbm": node.lora.tx_power_dbm,
            "generated": len(ms), "delivered": delivered, "pdr": delivered / len(ms) if ms else np.nan,
            "dropped_queue": sum(m["dropped"] for m in ms),
            "attempts_per_msg": float(np.mean([m["attempts"] for m in ms if not m["dropped"]] or [np.nan])),
            **latency_stats(ms), **link_stats(rs),
            "airtime_s": sum(t["end"] - t["start"] for t in result.transmissions if t["src"] == i),
            "energy_per_delivered_mj": 1e3 * result.energy_j[i] / delivered if delivered else np.nan,
        })

    data_recs = [r for r in recs if r["kind"] == "data"]
    outcome_counts = Counter(r["outcome"] for r in data_recs)
    airtime_by_channel = Counter()
    for t in result.transmissions:
        airtime_by_channel[t["channel"]] += t["end"] - t["start"]
    delivered = sum(m["delivered"] for m in msgs)
    overall = {
        "generated": len(msgs), "delivered": delivered, "pdr": delivered / len(msgs) if msgs else np.nan,
        **latency_stats(msgs), **link_stats(recs),
        "throughput_bps": 8 * sum(nodes[m["node"]].lora.payload_bytes for m in msgs if m["delivered"]) / span,
        "channel_utilization": {ch: a / span for ch, a in sorted(airtime_by_channel.items())},
        "outcomes": dict(outcome_counts),
        "collision_fraction": outcome_counts["collision"] / len(data_recs) if data_recs else np.nan,
    }
    return {"overall": overall, "per_node": per_node}


def write_csv(rows, path):
    """Write a list of flat dicts (e.g. result.messages or result.receptions) to CSV."""
    rows = list(rows)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
