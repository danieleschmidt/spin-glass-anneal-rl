"""
IsingFormulator: encode a job-shop scheduling problem as a QUBO matrix.

Variable encoding
-----------------
We discretise time into T slots (0 .. T-1).
For N jobs and M machines the binary variables are:

    x[i, j, t]  =  1  iff  job i is assigned to machine j starting at time slot t

The flat index of variable (i, j, t) is:

    idx(i, j, t) = i*M*T + j*T + t

QUBO energy  E(x) = x^T Q x
------------------------------
Three penalty terms are summed into Q:

1. One-hot per job (each job assigned to exactly one (machine, slot)):
   penalty_A * (sum_{j,t} x[i,j,t] - 1)^2  for each job i

2. Machine conflict (at most one job per machine per slot):
   penalty_B * (sum_{i} x[i,j,t])*(sum_{i'} x[i',j,t]) cross terms for each j,t

3. Makespan minimisation objective:
   penalty_C * sum_{i,j,t} t * x[i,j,t]   (linear → diagonal of Q)
"""

from __future__ import annotations
from typing import Tuple, Dict
import numpy as np


class IsingFormulator:
    """
    Encode a job-shop scheduling problem as a QUBO matrix.

    Parameters
    ----------
    n_jobs : int
        Number of jobs.
    n_machines : int
        Number of machines.
    n_time_slots : int
        Discrete time horizon.  Each slot represents one unit of time.
    penalty_job : float
        Strength of the one-hot-per-job constraint.
    penalty_machine : float
        Strength of the machine-conflict constraint.
    penalty_makespan : float
        Weight of the makespan minimisation objective.
    """

    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        n_time_slots: int,
        penalty_job: float = 10.0,
        penalty_machine: float = 10.0,
        penalty_makespan: float = 1.0,
    ) -> None:
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_time_slots = n_time_slots
        self.penalty_job = penalty_job
        self.penalty_machine = penalty_machine
        self.penalty_makespan = penalty_makespan

        # Total number of binary variables
        self.n_vars = n_jobs * n_machines * n_time_slots

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def var_index(self, job: int, machine: int, time_slot: int) -> int:
        """Return the flat QUBO index for variable x[job, machine, time_slot]."""
        return job * self.n_machines * self.n_time_slots + machine * self.n_time_slots + time_slot

    def index_to_var(self, idx: int) -> Tuple[int, int, int]:
        """Inverse of var_index: return (job, machine, time_slot) for a flat index."""
        t = idx % self.n_time_slots
        rest = idx // self.n_time_slots
        j = rest % self.n_machines
        i = rest // self.n_machines
        return (i, j, t)

    # ------------------------------------------------------------------
    # QUBO construction
    # ------------------------------------------------------------------

    def build_qubo(self) -> Tuple[np.ndarray, Dict[Tuple[int, int, int], int]]:
        """
        Build and return the QUBO matrix Q and a variable index mapping.

        Returns
        -------
        Q : np.ndarray, shape (n_vars, n_vars)
            Upper-triangular QUBO matrix (standard convention).
            Energy = x @ Q @ x  where x is a binary column vector.
        var_map : dict
            Mapping (job, machine, time_slot) → flat index.
        """
        N, M, T = self.n_jobs, self.n_machines, self.n_time_slots
        n = self.n_vars
        Q = np.zeros((n, n), dtype=np.float64)

        # Build the var_map
        var_map: Dict[Tuple[int, int, int], int] = {}
        for i in range(N):
            for j in range(M):
                for t in range(T):
                    var_map[(i, j, t)] = self.var_index(i, j, t)

        # ----------------------------------------------------------
        # Term 1: One-hot per job
        # (sum_{j,t} x[i,j,t] - 1)^2
        # = sum_{j,t} x[i,j,t]^2 - 2*sum_{j,t} x[i,j,t]
        #   + 2*sum_{(j,t)<(j',t')} x[i,j,t]*x[i,j',t']
        #   + 1   (constant, ignored)
        # In QUBO: x_k^2 = x_k (binary), so diagonal gets -1 coeff (from -2+1),
        # off-diag pairs get +2.
        # ----------------------------------------------------------
        A = self.penalty_job
        for i in range(N):
            indices = [self.var_index(i, j, t) for j in range(M) for t in range(T)]
            for k in indices:
                Q[k, k] += -A  # (x^2 - 2x) → -A per variable
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    ka, kb = indices[a], indices[b]
                    Q[ka, kb] += 2 * A  # cross-product penalty

        # ----------------------------------------------------------
        # Term 2: Machine conflict — at most one job per (machine, slot)
        # penalty_B * sum_{j,t} (sum_i x[i,j,t])*(sum_{i'>i} x[i',j,t])
        # = penalty_B * 2 * sum_{j,t} sum_{i<i'} x[i,j,t]*x[i',j,t]
        # (No diagonal change needed since same-variable term is 0 for i≠i')
        # ----------------------------------------------------------
        B = self.penalty_machine
        for j in range(M):
            for t in range(T):
                vars_jt = [self.var_index(i, j, t) for i in range(N)]
                for a in range(len(vars_jt)):
                    for b in range(a + 1, len(vars_jt)):
                        Q[vars_jt[a], vars_jt[b]] += 2 * B

        # ----------------------------------------------------------
        # Term 3: Makespan objective — penalise late assignments
        # penalty_C * sum_{i,j,t} t * x[i,j,t]  (linear term → diagonal)
        # ----------------------------------------------------------
        C = self.penalty_makespan
        for i in range(N):
            for j in range(M):
                for t in range(T):
                    k = self.var_index(i, j, t)
                    Q[k, k] += C * t

        return Q, var_map
