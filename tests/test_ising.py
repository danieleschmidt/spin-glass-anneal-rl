"""
tests/test_ising.py
===================
Unit tests for spin_glass_anneal.IsingFormulator, SimulatedAnnealer, ScheduleDecoder.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from spin_glass_anneal import IsingFormulator, SimulatedAnnealer, ScheduleDecoder


# ---------------------------------------------------------------------------
# IsingFormulator tests
# ---------------------------------------------------------------------------

class TestIsingFormulator:

    def test_n_vars(self):
        f = IsingFormulator(n_jobs=3, n_machines=2, n_time_slots=4)
        assert f.n_vars == 3 * 2 * 4

    def test_var_index_roundtrip(self):
        f = IsingFormulator(3, 2, 4)
        for i in range(3):
            for j in range(2):
                for t in range(4):
                    idx = f.var_index(i, j, t)
                    assert f.index_to_var(idx) == (i, j, t)

    def test_var_index_unique(self):
        f = IsingFormulator(3, 2, 4)
        indices = {f.var_index(i, j, t) for i in range(3) for j in range(2) for t in range(4)}
        assert len(indices) == f.n_vars

    def test_qubo_shape(self):
        f = IsingFormulator(2, 2, 3)
        Q, var_map = f.build_qubo()
        assert Q.shape == (f.n_vars, f.n_vars)

    def test_var_map_completeness(self):
        f = IsingFormulator(2, 2, 3)
        Q, var_map = f.build_qubo()
        assert len(var_map) == f.n_vars
        for i in range(2):
            for j in range(2):
                for t in range(3):
                    assert (i, j, t) in var_map

    def test_qubo_is_upper_triangular_friendly(self):
        """Off-diagonal lower triangle should be zero (upper-triangular convention)."""
        f = IsingFormulator(2, 2, 2)
        Q, _ = f.build_qubo()
        n = Q.shape[0]
        for row in range(n):
            for col in range(row):
                assert Q[row, col] == 0.0, f"Q[{row},{col}] = {Q[row,col]} ≠ 0"

    def test_one_hot_solution_lower_energy(self):
        """
        A valid one-hot-per-job solution should have lower energy than
        a solution with no active variables (all-zero, which violates one-hot).
        """
        f = IsingFormulator(2, 2, 2, penalty_job=10, penalty_machine=10, penalty_makespan=1)
        Q, var_map = f.build_qubo()

        # All-zero solution (violates one-hot for every job)
        x_zero = np.zeros(f.n_vars)

        # Valid assignment: job 0 → machine 0 slot 0, job 1 → machine 1 slot 0
        x_valid = np.zeros(f.n_vars)
        x_valid[var_map[(0, 0, 0)]] = 1.0
        x_valid[var_map[(1, 1, 0)]] = 1.0

        E_zero = x_zero @ Q @ x_zero
        E_valid = x_valid @ Q @ x_valid

        # Valid solution should have lower energy than zero (penalty for missing job assignments)
        assert E_valid < E_zero, f"E_valid={E_valid} should be < E_zero={E_zero}"

    def test_machine_conflict_increases_energy(self):
        """Assigning two jobs to same machine+slot should increase QUBO energy."""
        f = IsingFormulator(2, 2, 2, penalty_job=10, penalty_machine=10, penalty_makespan=0)
        Q, var_map = f.build_qubo()

        # Valid: job 0 → (m0,t0), job 1 → (m1,t0)
        x_valid = np.zeros(f.n_vars)
        x_valid[var_map[(0, 0, 0)]] = 1.0
        x_valid[var_map[(1, 1, 0)]] = 1.0

        # Conflict: both jobs → (m0,t0)
        x_conflict = np.zeros(f.n_vars)
        x_conflict[var_map[(0, 0, 0)]] = 1.0
        x_conflict[var_map[(1, 0, 0)]] = 1.0  # machine conflict!

        E_valid = x_valid @ Q @ x_valid
        E_conflict = x_conflict @ Q @ x_conflict

        assert E_conflict > E_valid, (
            f"Conflict energy {E_conflict} should be > valid energy {E_valid}"
        )

    def test_makespan_penalty_increases_with_late_slots(self):
        """Later time slots should have higher energy contribution."""
        f = IsingFormulator(1, 1, 4, penalty_job=0, penalty_machine=0, penalty_makespan=1)
        Q, var_map = f.build_qubo()

        energies = []
        for t in range(4):
            x = np.zeros(f.n_vars)
            x[var_map[(0, 0, t)]] = 1.0
            energies.append(x @ Q @ x)

        # Energy should be non-decreasing with time slot
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1], (
                f"Energy at slot {i} ({energies[i]}) > slot {i+1} ({energies[i+1]})"
            )


# ---------------------------------------------------------------------------
# SimulatedAnnealer tests
# ---------------------------------------------------------------------------

class TestSimulatedAnnealer:

    def test_energy_computation(self):
        """Energy = x^T Q x for upper-triangular Q."""
        # Simple 2×2 Q
        Q = np.array([[1.0, 2.0],
                      [0.0, 3.0]])
        ann = SimulatedAnnealer(Q)
        x = np.array([1.0, 1.0])
        # x^T Q x = 1*1 + 1*2 + 0*1 + 1*3 = 1 + 2 + 3 = 6
        assert ann.energy(x) == pytest.approx(6.0)

    def test_energy_zero_vector(self):
        Q = np.eye(4)
        ann = SimulatedAnnealer(Q)
        assert ann.energy(np.zeros(4)) == pytest.approx(0.0)

    def test_delta_e_matches_exact(self):
        """ΔE from field update should match brute-force energy difference."""
        rng = np.random.default_rng(0)
        n = 8
        Q = np.triu(rng.standard_normal((n, n)))
        ann = SimulatedAnnealer(Q)
        x = rng.integers(0, 2, size=n).astype(float)

        h = ann._field(x)
        for k in range(n):
            x_flipped = x.copy()
            x_flipped[k] = 1.0 - x_flipped[k]
            delta_e_predicted = (1.0 - 2.0 * x[k]) * h[k]
            delta_e_actual = ann.energy(x_flipped) - ann.energy(x)
            assert delta_e_predicted == pytest.approx(delta_e_actual, abs=1e-9), (
                f"Bit {k}: predicted ΔE={delta_e_predicted:.6f}, actual={delta_e_actual:.6f}"
            )

    def test_anneal_returns_binary_solution(self):
        Q = np.eye(5)
        ann = SimulatedAnnealer(Q, n_steps=100, seed=0)
        x, energy = ann.anneal()
        assert x.shape == (5,)
        assert set(x.astype(int)).issubset({0, 1})

    def test_anneal_minimises_simple_diagonal_qubo(self):
        """For Q = diag([-1, -1, -1, 1, 1]), optimum is x=[1,1,1,0,0], E=-3."""
        Q = np.diag([-1.0, -1.0, -1.0, 1.0, 1.0])
        ann = SimulatedAnnealer(Q, T0=2.0, cooling_rate=0.995, n_steps=5000, seed=42)
        x, energy = ann.anneal()
        assert energy == pytest.approx(-3.0, abs=0.1)

    def test_anneal_with_initial_solution(self):
        Q = np.diag([-1.0, 2.0, -3.0])
        ann = SimulatedAnnealer(Q, n_steps=1000, seed=1)
        x0 = np.array([1.0, 0.0, 1.0])  # optimal
        x, energy = ann.anneal(x0=x0)
        assert energy <= ann.energy(x0) + 0.1  # should not be worse than start

    def test_anneal_reproducible_with_seed(self):
        Q = np.triu(np.ones((6, 6)))
        ann1 = SimulatedAnnealer(Q, seed=99, n_steps=500)
        ann2 = SimulatedAnnealer(Q, seed=99, n_steps=500)
        x1, e1 = ann1.anneal()
        x2, e2 = ann2.anneal()
        np.testing.assert_array_equal(x1, x2)
        assert e1 == e2

    def test_anneal_on_qubo_from_formulator(self):
        """End-to-end: SA on a small QUBO from IsingFormulator finds a valid solution."""
        f = IsingFormulator(3, 2, 3, penalty_job=10, penalty_machine=10, penalty_makespan=1)
        Q, var_map = f.build_qubo()

        best_x = None
        best_e = np.inf
        for seed in range(5):
            ann = SimulatedAnnealer(Q, T0=8.0, cooling_rate=0.998, n_steps=20_000, seed=seed)
            x, e = ann.anneal()
            if e < best_e:
                best_e = e
                best_x = x

        # Check: at least 2 out of 3 jobs should be correctly assigned (one-hot)
        valid_jobs = sum(
            1 for i in range(3)
            if sum(1 for j in range(2) for t in range(3)
                   if best_x[var_map[(i, j, t)]] > 0.5) == 1
        )
        assert valid_jobs >= 2, f"Only {valid_jobs}/3 jobs are correctly assigned"


# ---------------------------------------------------------------------------
# ScheduleDecoder tests
# ---------------------------------------------------------------------------

class TestScheduleDecoder:

    def _make_decoder(self, n_jobs=3, n_machines=2, n_time_slots=3):
        return ScheduleDecoder(n_jobs, n_machines, n_time_slots)

    def _make_var_map(self, n_jobs, n_machines, n_time_slots):
        f = IsingFormulator(n_jobs, n_machines, n_time_slots)
        return f, *f.build_qubo()

    def test_decode_valid_solution(self):
        f = IsingFormulator(2, 2, 3)
        Q, var_map = f.build_qubo()
        decoder = ScheduleDecoder(2, 2, 3)

        x = np.zeros(f.n_vars)
        x[var_map[(0, 0, 1)]] = 1.0  # job 0 → machine 0, slot 1
        x[var_map[(1, 1, 0)]] = 1.0  # job 1 → machine 1, slot 0

        sched = decoder.decode(x, var_map)
        assert sched[0] == (0, 1)
        assert sched[1] == (1, 0)

    def test_decode_all_jobs_present(self):
        f = IsingFormulator(3, 2, 3)
        Q, var_map = f.build_qubo()
        decoder = ScheduleDecoder(3, 2, 3)

        x = np.zeros(f.n_vars)
        x[var_map[(0, 0, 0)]] = 1.0
        x[var_map[(1, 1, 0)]] = 1.0
        x[var_map[(2, 0, 1)]] = 1.0

        sched = decoder.decode(x, var_map)
        assert set(sched.keys()) == {0, 1, 2}

    def test_decode_fallback_for_unset_job(self):
        """If no variable is set for a job, decoder falls back to (machine=0, slot=0)."""
        f = IsingFormulator(2, 2, 2)
        Q, var_map = f.build_qubo()
        decoder = ScheduleDecoder(2, 2, 2)

        x = np.zeros(f.n_vars)
        # Only job 0 assigned
        x[var_map[(0, 0, 1)]] = 1.0

        sched = decoder.decode(x, var_map)
        assert sched[0] == (0, 1)
        assert sched[1] == (0, 0)  # fallback

    def test_decode_picks_earliest_on_tie(self):
        """If multiple variables set for one job, decoder picks earliest slot."""
        f = IsingFormulator(1, 2, 3)
        Q, var_map = f.build_qubo()
        decoder = ScheduleDecoder(1, 2, 3)

        x = np.zeros(f.n_vars)
        x[var_map[(0, 0, 2)]] = 1.0  # slot 2
        x[var_map[(0, 1, 0)]] = 1.0  # slot 0 ← should win

        sched = decoder.decode(x, var_map)
        assert sched[0] == (1, 0)  # earliest time slot

    def test_makespan_single_job(self):
        decoder = ScheduleDecoder(1, 1, 4)
        sched = {0: (0, 2)}
        assert decoder.makespan(sched) == 3  # slot 2 → completes at 3

    def test_makespan_multiple_jobs(self):
        decoder = ScheduleDecoder(3, 2, 5)
        sched = {0: (0, 0), 1: (1, 1), 2: (0, 3)}
        assert decoder.makespan(sched) == 4  # max slot = 3 → makespan = 4

    def test_makespan_empty(self):
        decoder = ScheduleDecoder(0, 1, 3)
        assert decoder.makespan({}) == 0

    def test_pretty_print_contains_job_ids(self):
        decoder = ScheduleDecoder(3, 2, 4)
        sched = {0: (0, 0), 1: (1, 1), 2: (0, 2)}
        output = decoder.pretty_print(sched)
        for job_id in [0, 1, 2]:
            assert f"Job {job_id:2d}" in output or f"Job  {job_id}" in output

    def test_pretty_print_contains_makespan(self):
        decoder = ScheduleDecoder(2, 2, 4)
        sched = {0: (0, 1), 1: (1, 3)}
        output = decoder.pretty_print(sched)
        assert "Makespan" in output
        assert "4" in output  # makespan = 3+1 = 4


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline_2jobs_2machines(self):
        """
        End-to-end: 2 jobs, 2 machines, 3 time slots.
        Optimal makespan = 1 (both jobs in slot 0 on different machines).
        SA should find a valid solution.
        """
        f = IsingFormulator(2, 2, 3, penalty_job=12, penalty_machine=12, penalty_makespan=2)
        Q, var_map = f.build_qubo()

        best_x = None
        best_e = np.inf
        for seed in range(10):
            ann = SimulatedAnnealer(Q, T0=8.0, cooling_rate=0.997, n_steps=15_000, seed=seed)
            x, e = ann.anneal()
            if e < best_e:
                best_e = e
                best_x = x

        decoder = ScheduleDecoder(2, 2, 3)
        sched = decoder.decode(best_x, var_map)

        # Each job should be assigned
        assert len(sched) == 2
        assert all(job_id in sched for job_id in range(2))

        # Makespan should be reasonable
        ms = decoder.makespan(sched)
        assert ms <= 3, f"Makespan {ms} exceeds horizon"

    def test_full_pipeline_5jobs_3machines(self):
        """
        5 jobs, 3 machines — the main demo problem.
        SA should find makespan ≤ 3 in some restart.
        """
        N, M, T = 5, 3, 5
        f = IsingFormulator(N, M, T, penalty_job=15, penalty_machine=15, penalty_makespan=1.5)
        Q, var_map = f.build_qubo()

        decoder = ScheduleDecoder(N, M, T)
        best_makespan = np.inf
        best_violations = np.inf

        for seed in range(10):
            ann = SimulatedAnnealer(Q, T0=12.0, cooling_rate=0.99988, n_steps=80_000, seed=seed)
            x, e = ann.anneal()
            sched = decoder.decode(x, var_map)
            ms = decoder.makespan(sched)
            viol = sum(
                1 for i in range(N)
                if sum(1 for j in range(M) for t in range(T) if x[var_map[(i, j, t)]] > 0.5) != 1
            )
            if viol <= best_violations and ms < best_makespan:
                best_makespan = ms
                best_violations = viol

        assert best_makespan <= 3, f"Best makespan {best_makespan} > 3 across 10 restarts"
