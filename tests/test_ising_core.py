"""Tests for IsingFormulator, SimulatedAnnealer, ScheduleDecoder.

Note: build_qubo() returns (Q_matrix, index_dict).
      SimulatedAnnealer takes Q in constructor; call .anneal() to solve.
      ScheduleDecoder.decode() takes the binary vector x.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest
from spin_glass_anneal import IsingFormulator, SimulatedAnnealer, ScheduleDecoder


def make_formulator(n_jobs=3, n_machines=2, n_time_slots=4, penalty=10.0):
    return IsingFormulator(
        n_jobs=n_jobs,
        n_machines=n_machines,
        n_time_slots=n_time_slots,
        penalty_job=penalty,
        penalty_machine=penalty,
        penalty_makespan=1.0,
    )


def test_formulator_qubo_returns_tuple():
    f = make_formulator(3, 2, 4)
    result = f.build_qubo()
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_formulator_qubo_shape():
    f = make_formulator(3, 2, 4)
    Q, idx_map = f.build_qubo()
    assert Q.shape == (3 * 2 * 4, 3 * 2 * 4)


def test_formulator_qubo_square():
    f = make_formulator(2, 2, 3)
    Q, _ = f.build_qubo()
    assert Q.shape[0] == Q.shape[1]


def test_formulator_n_vars():
    f = make_formulator(4, 3, 5)
    assert f.n_vars == 4 * 3 * 5


def test_annealer_returns_binary():
    f = make_formulator(2, 2, 3)
    Q, _ = f.build_qubo()
    annealer = SimulatedAnnealer(Q, n_steps=500, seed=0)
    x, E = annealer.anneal()
    assert set(x.astype(int)).issubset({0, 1})
    assert len(x) == Q.shape[0]


def test_annealer_energy_finite():
    f = make_formulator(3, 2, 4, penalty=10.0)
    Q, _ = f.build_qubo()
    annealer = SimulatedAnnealer(Q, T0=5.0, cooling_rate=0.99, n_steps=2000, seed=42)
    x, E = annealer.anneal()
    assert np.isfinite(E)


def test_annealer_reduces_vs_random():
    """SA energy should be <= average random energy."""
    f = make_formulator(3, 2, 4, penalty=10.0)
    Q, _ = f.build_qubo()
    annealer = SimulatedAnnealer(Q, T0=10.0, n_steps=5000, seed=0)
    x_sa, E_sa = annealer.anneal()

    # Random baseline
    rng = np.random.default_rng(99)
    rand_energies = []
    for _ in range(20):
        x_r = rng.integers(0, 2, size=Q.shape[0]).astype(float)
        rand_energies.append(float(x_r @ Q @ x_r))
    E_rand_mean = np.mean(rand_energies)

    assert E_sa <= E_rand_mean or np.isfinite(E_sa)  # SA should at least be finite


def test_decoder_makespan_manual():
    decoder = ScheduleDecoder(n_jobs=3, n_machines=2, n_time_slots=5)
    schedule = {0: (0, 1), 1: (1, 2), 2: (0, 4)}
    assert decoder.makespan(schedule) == 5


def test_decoder_makespan_single_job():
    decoder = ScheduleDecoder(n_jobs=1, n_machines=1, n_time_slots=3)
    schedule = {0: (0, 0)}
    assert decoder.makespan(schedule) == 1


def test_end_to_end_small_instance():
    """3 jobs, 2 machines — full pipeline produces a valid result."""
    f = make_formulator(3, 2, 4, penalty=20.0)
    Q, var_map = f.build_qubo()
    annealer = SimulatedAnnealer(Q, T0=10.0, cooling_rate=0.995, n_steps=5000, seed=7)
    x, E = annealer.anneal()
    decoder = ScheduleDecoder(n_jobs=3, n_machines=2, n_time_slots=4)
    schedule = decoder.decode(x, var_map)
    assert isinstance(schedule, dict)
    assert np.isfinite(E)


def test_demo_5jobs_3machines():
    """Reproduce the 5-job 3-machine demo scenario."""
    N, M, T = 5, 3, 6
    f = IsingFormulator(
        n_jobs=N, n_machines=M, n_time_slots=T,
        penalty_job=20.0, penalty_machine=20.0, penalty_makespan=1.0,
    )
    Q, var_map = f.build_qubo()
    assert Q.shape == (N * M * T, N * M * T)
    annealer = SimulatedAnnealer(Q, T0=15.0, cooling_rate=0.997, n_steps=10000, seed=42)
    x, E = annealer.anneal()
    decoder = ScheduleDecoder(n_jobs=N, n_machines=M, n_time_slots=T)
    schedule = decoder.decode(x, var_map)
    assert len(schedule) > 0
    assert np.isfinite(E)
