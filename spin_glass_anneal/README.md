# spin_glass_anneal

**Job-shop scheduling via QUBO / Ising formulation + simulated annealing.**

Pure-numpy implementation. No GPU required.

---

## Overview

This package solves the **job-shop scheduling problem** by reformulating it as a
**Quadratic Unconstrained Binary Optimization (QUBO)** problem and solving it with
**simulated annealing** — the classical analogue of quantum annealing on an Ising spin system.

```
Scheduling problem
       ↓  IsingFormulator
   QUBO matrix Q  (x^T Q x)
       ↓  SimulatedAnnealer
   Binary solution x* ∈ {0,1}^n
       ↓  ScheduleDecoder
   Schedule {job → (machine, time)}
```

---

## Quick Start

```python
from spin_glass_anneal import IsingFormulator, SimulatedAnnealer, ScheduleDecoder

# 5 jobs, 3 machines, 5 discrete time slots
formulator = IsingFormulator(n_jobs=5, n_machines=3, n_time_slots=5,
                             penalty_job=15.0, penalty_machine=15.0, penalty_makespan=1.5)
Q, var_map = formulator.build_qubo()

# Solve with simulated annealing
annealer = SimulatedAnnealer(Q, T0=12.0, cooling_rate=0.99988, n_steps=100_000, seed=42)
x_best, energy = annealer.anneal()

# Decode solution
decoder = ScheduleDecoder(5, 3, 5)
schedule = decoder.decode(x_best, var_map)
print(decoder.pretty_print(schedule))
```

Run the full demo:

```bash
python demo_scheduling.py
```

---

## Mathematical Formulation

### 1. Scheduling → QUBO Encoding

**Variables:**

```
x[i, j, t] ∈ {0, 1}
```

where `x[i, j, t] = 1` iff job `i` is assigned to machine `j` at time slot `t`.

With `N` jobs, `M` machines, and `T` time slots, there are `N × M × T` binary variables.
The flat index is `idx(i, j, t) = i·M·T + j·T + t`.

**QUBO energy:**

```
E(x) = x^T Q x
```

where `Q` encodes three terms:

#### Term 1 — One-hot per job (each job assigned exactly once)

For each job `i`:

```
penalty_job · ( Σ_{j,t} x[i,j,t] − 1 )²
```

Expanding: diagonal terms get `−penalty_job`, off-diagonal pairs (same job) get `+2·penalty_job`.

#### Term 2 — Machine conflict (at most one job per machine per slot)

For each `(machine j, time t)`:

```
penalty_machine · Σ_{i < i'} x[i,j,t] · x[i',j,t]
```

Off-diagonal pairs (same machine + slot, different jobs) get `+2·penalty_machine`.

#### Term 3 — Makespan minimisation

```
penalty_makespan · Σ_{i,j,t} t · x[i,j,t]
```

This adds `penalty_makespan · t` to the diagonal of variable `(i, j, t)`,
penalising late slot assignments and encouraging early packing.

---

### 2. QUBO → Ising Hamiltonian

The QUBO and Ising formalisms are equivalent via the substitution `x_k = (1 − s_k) / 2`
where `s_k ∈ {−1, +1}` are Ising spins.

For a QUBO `E(x) = x^T Q x`:

```
H_Ising = Σ_{k} h_k · s_k  +  Σ_{k < l} J_{kl} · s_k · s_l  +  const
```

where:
- `J_{kl} = Q[k,l] / 4`  (coupling between spins k and l)
- `h_k = − (Q[k,k] + Σ_{l≠k} Q[k,l]) / 2`  (local field on spin k)

This is the **spin-glass** Hamiltonian whose ground state corresponds to the optimal schedule.
Real quantum annealers (D-Wave) and digital annealers (Fujitsu) operate natively on this form.

---

### 3. Simulated Annealing as Classical Quantum Annealing Approximation

**Quantum annealing** exploits quantum tunnelling to escape local minima by slowly reducing
a transverse magnetic field. The system ideally follows the ground state of a time-evolving
Hamiltonian `H(t) = Γ(t) H_transverse + H_Ising`.

**Simulated annealing (SA)** approximates this classically via thermal fluctuations:

```
P(accept worse move) = exp(−ΔE / T)
```

with a cooling schedule `T(s) = T₀ · α^s` (geometric cooling, `α < 1`).

The SA implementation here:

1. Initialise with a random binary solution `x ∈ {0,1}^n`
2. Precompute the effective field `h[k] = Q[k,k] + Σ_{l≠k} Q_upper[k,l] · x_l`
3. At each step: pick a random bit `k`, compute `ΔE = (1 − 2x_k) · h[k]`
4. Accept if `ΔE < 0` or with probability `exp(−ΔE / T)`
5. After accepting, update field in O(n): `h[l] += Q_upper[l,k] · Δx_k`
6. Cool temperature: `T ← T · α`
7. Track best solution found

**Complexity:** O(n_steps · n) per run, where `n = N·M·T`.

---

### 4. Complexity and Approximation Guarantees

| Property | Value |
|----------|-------|
| Problem class | NP-hard (job-shop scheduling is NP-hard in general) |
| QUBO variables | N × M × T |
| SA runtime | O(n_steps × N × M × T) |
| SA solution quality | Heuristic; no worst-case guarantee |
| Optimality | Finds exact optimum for small instances with enough steps |
| Approximation ratio | Empirically 1–1.2× optimum for moderate instances |

**Parameter guidance:**
- `penalty_job`, `penalty_machine`: should be ≥ maximum possible objective value (typically 10–20× `penalty_makespan · T`)
- `T0`: start high enough to accept ~50% of uphill moves initially
- `cooling_rate`: 0.999–0.9999 (slower cooling = better quality, more time)
- `n_steps`: 50k–500k for production use; 10k for quick exploration

---

## API Reference

### `IsingFormulator`

```python
IsingFormulator(n_jobs, n_machines, n_time_slots,
                penalty_job=10.0, penalty_machine=10.0, penalty_makespan=1.0)
```

- `build_qubo()` → `(Q: np.ndarray, var_map: dict)`
- `var_index(job, machine, time_slot)` → `int`
- `index_to_var(idx)` → `(job, machine, time_slot)`

### `SimulatedAnnealer`

```python
SimulatedAnnealer(Q, T0=5.0, cooling_rate=0.9995, n_steps=20_000, seed=None)
```

- `anneal(x0=None)` → `(best_x: np.ndarray, best_energy: float)`
- `energy(x)` → `float`

### `ScheduleDecoder`

```python
ScheduleDecoder(n_jobs, n_machines, n_time_slots)
```

- `decode(x, var_map)` → `{job_id: (machine_id, start_time)}`
- `makespan(schedule)` → `int`
- `pretty_print(schedule)` → `str`

---

## Running Tests

```bash
pytest tests/test_ising.py -v --no-cov
```

28 tests cover all three classes including edge cases and end-to-end integration.
