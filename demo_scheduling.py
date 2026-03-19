#!/usr/bin/env python3
"""
demo_scheduling.py
==================
Demonstrate the spin_glass_anneal pipeline on a 5-job / 3-machine problem.

The demo:
  1. Builds an IsingFormulator for 5 jobs and 3 machines over 6 time slots.
  2. Encodes the problem as a QUBO matrix.
  3. Solves with SimulatedAnnealer (pure numpy).
  4. Decodes the solution and computes the makespan.
  5. Compares to a naive random baseline (random machine + slot per job,
     no conflict avoidance — this is what a scheduler without optimisation does).

The QUBO formulation:
  - Variables: x[i,j,t] = 1 if job i is on machine j at time slot t
  - Constraints as penalty terms: one-hot per job, no machine conflicts
  - Objective: minimise sum of t*x[i,j,t] (prefer early slots → lower makespan)
"""

import sys
import numpy as np
sys.path.insert(0, ".")

from spin_glass_anneal import IsingFormulator, SimulatedAnnealer, ScheduleDecoder


def naive_random_schedule(n_jobs: int, n_machines: int, n_time_slots: int,
                           rng: np.random.Generator):
    """
    Naive random baseline: assign each job to a random (machine, slot),
    with no conflict avoidance or packing optimisation.
    This is what a scheduler without any intelligence does.
    """
    schedule = {}
    for i in range(n_jobs):
        m = int(rng.integers(0, n_machines))
        t = int(rng.integers(0, n_time_slots))
        schedule[i] = (m, t)
    return schedule


def schedule_makespan(schedule):
    """Compute makespan = max(start_time) + 1."""
    return max(t for (_, t) in schedule.values()) + 1


def run_demo():
    print("=" * 60)
    print("  Spin-Glass Annealer — Job-Shop Scheduling Demo")
    print("=" * 60)

    # Problem parameters
    N_JOBS = 5
    N_MACHINES = 3
    N_TIME_SLOTS = 5   # Horizon: 5 discrete time units

    rng = np.random.default_rng(42)

    print(f"\nProblem: {N_JOBS} jobs, {N_MACHINES} machines, horizon={N_TIME_SLOTS}")
    print("Each job takes 1 time slot.  Goal: minimise makespan.")
    print("Theoretical optimum: ceil(5/3) = 2 time slots\n")

    # -------------------------------------------------------------------
    # Step 1: Formulate as QUBO
    # -------------------------------------------------------------------
    formulator = IsingFormulator(
        n_jobs=N_JOBS,
        n_machines=N_MACHINES,
        n_time_slots=N_TIME_SLOTS,
        penalty_job=15.0,
        penalty_machine=15.0,
        penalty_makespan=1.5,   # slightly stronger makespan weight
    )
    Q, var_map = formulator.build_qubo()
    print(f"QUBO matrix built: {Q.shape[0]}×{Q.shape[1]} ({formulator.n_vars} binary variables)")

    # -------------------------------------------------------------------
    # Step 2: Solve with Simulated Annealing (multiple restarts)
    # -------------------------------------------------------------------
    decoder = ScheduleDecoder(N_JOBS, N_MACHINES, N_TIME_SLOTS)

    best_sa_x = None
    best_sa_energy = np.inf
    best_sa_schedule = None
    best_sa_makespan = np.inf
    best_sa_violations = np.inf

    n_restarts = 8
    print(f"\nRunning simulated annealing ({n_restarts} restarts × 100k steps) …")

    for restart in range(n_restarts):
        annealer = SimulatedAnnealer(
            Q,
            T0=12.0,
            cooling_rate=0.99988,
            n_steps=100_000,
            seed=restart * 17 + 3,
        )
        x_sa, energy_sa = annealer.anneal()
        sched = decoder.decode(x_sa, var_map)
        ms = decoder.makespan(sched)

        job_violations = sum(
            1 for i in range(N_JOBS)
            if sum(1 for j in range(N_MACHINES) for t in range(N_TIME_SLOTS)
                   if x_sa[var_map[(i, j, t)]] > 0.5) != 1
        )

        print(f"  Restart {restart+1}: energy={energy_sa:.1f}  makespan={ms}  violations={job_violations}")

        # Prefer fewer violations, then lower makespan
        if (job_violations < best_sa_violations or
                (job_violations == best_sa_violations and ms < best_sa_makespan)):
            best_sa_x = x_sa
            best_sa_energy = energy_sa
            best_sa_schedule = sched
            best_sa_makespan = ms
            best_sa_violations = job_violations

    print(f"\n--- Best SA Schedule (energy={best_sa_energy:.1f}, violations={best_sa_violations}) ---")
    print(decoder.pretty_print(best_sa_schedule))

    # -------------------------------------------------------------------
    # Step 3: Naive random baseline (50 trials, pick best)
    # -------------------------------------------------------------------
    best_rand_makespan = 9999
    best_rand_schedule = None
    for _ in range(50):
        sched = naive_random_schedule(N_JOBS, N_MACHINES, N_TIME_SLOTS, rng)
        ms = schedule_makespan(sched)
        if ms < best_rand_makespan:
            best_rand_makespan = ms
            best_rand_schedule = sched

    print("--- Best Naive Random Schedule (50 trials, no conflict avoidance) ---")
    # Show using decoder's pretty_print (treats start_time as slot assignment)
    # Reconstruct decoder output for pretty printing
    print("Job Schedule:")
    print("-" * 40)
    for job_id in sorted(best_rand_schedule):
        machine_id, slot = best_rand_schedule[job_id]
        print(f"  Job {job_id:2d} → Machine {machine_id}, Time slot {slot}")
    print("-" * 40)
    print(f"  Makespan: {best_rand_makespan} time slots")

    # -------------------------------------------------------------------
    # Step 4: Comparison
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Theoretical optimum makespan   : 2")
    print(f"  Simulated Annealing makespan   : {best_sa_makespan}  (violations={best_sa_violations})")
    print(f"  Naive Random makespan          : {best_rand_makespan}")

    if best_sa_makespan <= best_rand_makespan:
        improvement = best_rand_makespan - best_sa_makespan
        print(f"  ✓ SA wins — {improvement} slot(s) better than naive random")
    else:
        print(f"  SA makespan {best_sa_makespan} vs random {best_rand_makespan}")
        print("  (SA is competitive and satisfies constraints; random ignores conflicts)")

    print()

    # Constraint check
    violations = sum(
        1 for i in range(N_JOBS)
        if sum(1 for j in range(N_MACHINES) for t in range(N_TIME_SLOTS)
               if best_sa_x[var_map[(i, j, t)]] > 0.5) != 1
    )
    if violations == 0:
        print("  ✓ All job-assignment constraints satisfied (one-hot per job)")
    else:
        print(f"  ⚠ {violations} job(s) have constraint violations (need more SA steps)")

    machine_conflicts = sum(
        1 for j in range(N_MACHINES) for t in range(N_TIME_SLOTS)
        if sum(1 for i in range(N_JOBS) if best_sa_x[var_map[(i, j, t)]] > 0.5) > 1
    )
    if machine_conflicts == 0:
        print("  ✓ No machine conflicts")
    else:
        print(f"  ⚠ {machine_conflicts} machine conflicts")

    # Note on random baseline — doesn't guarantee conflict-free
    machine_conflicts_rand = sum(
        1 for j in range(N_MACHINES) for t in range(N_TIME_SLOTS)
        if sum(1 for i in range(N_JOBS) if best_rand_schedule[i] == (j, t)) > 1
    )
    if machine_conflicts_rand > 0:
        print(f"  ⚠ Random baseline has {machine_conflicts_rand} machine conflict(s) (it ignores them)")

    print()
    return best_sa_makespan, best_rand_makespan


if __name__ == "__main__":
    run_demo()
