"""NumPy-only checks of lora_netsim (no Sionna RT needed).

Run from this directory: python -m pytest test_lora_netsim.py
"""

import numpy as np
import pytest

from lora_netsim import (LoRaConfig, Node, RadioModel, SimConfig, fit_calibration, fit_per_curve,
                         noise_dbm_from_report, signal_dbm_from_report, simulate, summarize)

QUIET = RadioModel(fading_std_db=0.0)  # Deterministic link budget


def star(gains_db, sfs=None, period_s=3600.0, **node_kw):
    """Gateway at index 0 and one end node per entry of gains_db (dB, symmetric)."""
    n = len(gains_db) + 1
    g = np.full((n, n), -200.0)
    for k, value in enumerate(gains_db, start=1):
        g[0, k] = g[k, 0] = value
    np.fill_diagonal(g, np.nan)
    nodes = [Node("GW", gateway=True)]
    for k in range(1, n):
        lora = LoRaConfig(sf=(sfs or [7] * n)[k - 1])
        nodes.append(Node(f"N{k}", lora=lora, period_s=period_s, **node_kw))
    return nodes, g


def test_time_on_air_matches_semtech_calculator():
    assert LoRaConfig(sf=7).time_on_air_s == pytest.approx(0.051456)
    assert LoRaConfig(sf=12).time_on_air_s == pytest.approx(1.318912)  # LDRO on
    assert LoRaConfig(sf=9, cr=4, payload_bytes=51).time_on_air_s > LoRaConfig(sf=9, payload_bytes=51).time_on_air_s


def test_strong_single_link_delivers_everything_with_airtime_latency():
    nodes, g = star([-60.0], period_s=10.0)
    result = simulate(nodes, g, QUIET, SimConfig(duration_s=600, seed=1))
    s = summarize(result)["overall"]
    assert s["pdr"] == 1.0 and s["generated"] > 30
    assert s["latency_p50_s"] == pytest.approx(LoRaConfig().time_on_air_s, abs=1e-9)
    assert s["outcomes"] == {"ok": s["generated"]}


def test_link_far_below_sensitivity_fails():
    nodes, g = star([-160.0], period_s=10.0)
    s = summarize(simulate(nodes, g, QUIET, SimConfig(duration_s=300)))["overall"]
    assert s["pdr"] == 0.0 and set(s["outcomes"]) == {"not_detected"}


def collide(gain_a, gain_b, sf_a=7, sf_b=7):
    """Two end nodes whose single packets start together at the gateway."""
    nodes, g = star([gain_a, gain_b], sfs=[sf_a, sf_b])
    nodes = [nodes[0]] + [Node(n.name, lora=n.lora, period_s=1e6, traffic="periodic", start_offset_s=0.5)
                          for n in nodes[1:]]
    result = simulate(nodes, g, QUIET, SimConfig(duration_s=1.0))
    return {r["src"]: r["outcome"] for r in result.receptions}


def test_equal_power_same_sf_collision_loses_both():
    assert collide(-80.0, -80.0) == {1: "collision", 2: "collision"}


def test_capture_of_stronger_same_sf_packet():
    assert collide(-70.0, -80.0) == {1: "ok", 2: "collision"}


def test_different_sf_packets_survive_equal_power():
    assert collide(-80.0, -80.0, sf_a=7, sf_b=9) == {1: "ok", 2: "ok"}


def test_confirmed_mode_retries_on_weak_link_and_reports_attempts():
    radio = RadioModel(fading_std_db=0.0)
    edge = radio.sensitivity_dbm(7) - (14 - 2)  # gain putting mean SNR at the limit
    nodes, g = star([edge - 2.0], period_s=30.0)
    unconfirmed = summarize(simulate(nodes, g, radio, SimConfig(duration_s=3000, seed=2)))["overall"]
    confirmed_result = simulate(nodes, g, radio, SimConfig(duration_s=3000, seed=2, confirmed=True,
                                                           max_retransmissions=4))
    confirmed = summarize(confirmed_result)["overall"]
    assert 0.05 < unconfirmed["pdr"] < 0.95
    assert confirmed["pdr"] > unconfirmed["pdr"]
    assert summarize(confirmed_result)["per_node"][0]["attempts_per_msg"] > 1.2


def test_peer_to_peer_destination_and_half_duplex_listener():
    g = np.array([[np.nan, -70.0], [-70.0, np.nan]])
    nodes = [Node("A", dest=(1,), period_s=20.0), Node("B", dest=(0,), period_s=20.0)]
    s = summarize(simulate(nodes, g, QUIET, SimConfig(duration_s=2000, seed=4)))
    assert {p["node"] for p in s["per_node"]} == {"A", "B"}
    assert s["overall"]["pdr"] > 0.95  # rare overlaps lose packets to half-duplex


def test_confirmed_peers_never_transmit_twice_at_once():
    g = np.full((4, 4), -70.0)
    np.fill_diagonal(g, np.nan)
    nodes = [Node(f"P{k}", dest=((k + 1) % 4,), period_s=2.0, rx_always_on=True) for k in range(4)]
    result = simulate(nodes, g, QUIET, SimConfig(duration_s=600, confirmed=True, seed=5))
    for k in range(4):
        spans = sorted((t["start"], t["end"]) for t in result.transmissions if t["src"] == k)
        assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))
    assert any(t["kind"] == "ack" for t in result.transmissions)


def test_report_conversion_round_trip():
    signal, noise = -110.0, -117.0
    rssi_total = 10 * np.log10(10 ** (signal / 10) + 10 ** (noise / 10))
    snr = signal - noise
    assert signal_dbm_from_report(rssi_total, snr) == pytest.approx(signal)
    assert noise_dbm_from_report(rssi_total, snr) == pytest.approx(noise)


def test_calibration_recovers_offset_shadowing_and_fading():
    rng = np.random.default_rng(0)
    pairs, per_pair = 80, 25
    predicted = rng.uniform(-110, -70, pairs)
    true_offset, true_shadow, true_fade, noise = -7.0, 4.0, 2.0, -117.0
    pair_mean = predicted + true_offset + rng.normal(0, true_shadow, pairs)
    rows = np.repeat(np.arange(pairs), per_pair)
    signal = 14 - 2 + pair_mean[rows] + rng.normal(0, true_fade, rows.size)
    snr = signal - noise
    rssi = 10 * np.log10(10 ** (signal / 10) + 10 ** (noise / 10))
    tx = np.column_stack([np.zeros(pairs), np.zeros(pairs), np.zeros(pairs)])[rows]
    rx = np.column_stack([np.arange(pairs) * 3.0, np.ones(pairs), np.ones(pairs)])[rows]
    meas = {"tx_position": tx, "rx_position": rx, "rssi_dbm": rssi, "snr_db": snr,
            "received": np.ones(rows.size, bool), "link_class": np.full(rows.size, "all")}
    cal = fit_calibration(meas, predicted[rows], tx_power_dbm=14, cable_loss_db=1)
    assert cal.offset_db["all"] == pytest.approx(true_offset, abs=1.2)
    assert cal.shadowing_std_db == pytest.approx(true_shadow, abs=1.0)
    assert cal.fading_std_db == pytest.approx(true_fade, abs=0.2)
    assert cal.noise_figure_db == pytest.approx(noise + 174 - 10 * np.log10(125e3), abs=0.1)
    assert cal.rmse_calibrated_loo_db < cal.rmse_uncalibrated_db


def test_noise_floor_ignores_saturated_snr_reports():
    rng = np.random.default_rng(2)
    signal = rng.uniform(-125, -60, 400)
    noise = -117.0
    rssi = 10 * np.log10(10 ** (signal / 10) + 10 ** (noise / 10))
    snr_reported = np.minimum(signal - noise, 10.0)  # Modem saturates at +10 dB
    pos = np.column_stack([np.arange(400.0), np.zeros(400), np.zeros(400)])
    meas = {"tx_position": np.zeros((400, 3)), "rx_position": pos, "rssi_dbm": rssi, "snr_db": snr_reported,
            "received": np.ones(400, bool), "link_class": np.full(400, "all")}
    cal = fit_calibration(meas, signal - 12, tx_power_dbm=14, cable_loss_db=1)
    assert cal.noise_figure_db == pytest.approx(noise + 174 - 10 * np.log10(125e3), abs=0.1)


def test_simulated_reports_respect_register_ranges():
    nodes, g = star([-20.0], period_s=10.0)
    result = simulate(nodes, g, QUIET, SimConfig(duration_s=100))
    assert max(r["snr_report_db"] for r in result.receptions) == 31.75


def test_per_curve_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    sf = np.full(20000, 9)
    snr = rng.uniform(-20, -5, sf.size)
    true = RadioModel(per_snr50_offset_db=-2.0, per_width_db=0.8)
    received = rng.random(sf.size) > true.per(snr, 9)
    offset, width = fit_per_curve(snr, received, sf)
    assert offset == pytest.approx(-2.0, abs=0.1) and width == pytest.approx(0.8, abs=0.08)


def test_per_curve_fit_is_stable_over_wide_snr_range_and_rejects_separation():
    rng = np.random.default_rng(3)
    snr = rng.uniform(-80, 90, 5000)  # Link budgets spanning very weak to very strong links
    true = RadioModel(per_snr50_offset_db=-1.0, per_width_db=3.0)
    received = rng.random(snr.size) > true.per(snr, 7)
    offset, width = fit_per_curve(snr, received, np.full(snr.size, 7))
    assert offset == pytest.approx(-1.0, abs=0.5) and width == pytest.approx(3.0, abs=0.4)
    with pytest.raises(ValueError, match="separation"):
        fit_per_curve(np.array([-20.0, -15, 5, 10]), np.array([0, 0, 1, 1]), np.full(4, 7))


def test_nan_gain_is_rejected():
    nodes, g = star([-80.0])
    g[0, 1] = np.nan
    with pytest.raises(ValueError):
        simulate(nodes, g, QUIET, SimConfig(duration_s=10))
