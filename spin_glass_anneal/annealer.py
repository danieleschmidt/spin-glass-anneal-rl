"""
SimulatedAnnealer: solve a QUBO problem via simulated annealing (pure numpy).

QUBO energy (upper-triangular convention)
------------------------------------------
For upper-triangular Q (Q[l,k]=0 for l>k):

  E(x) = x^T Q x = Σ_k Q[k,k]*x_k + Σ_{k<l} Q[k,l]*x_k*x_l

Note: the off-diagonal terms appear ONCE (not twice), because Q is upper-tri.

Delta-E for flipping bit k
---------------------------
  ΔE = (1 - 2*x_k) * h[k]

where the effective field is:
  h[k] = Q[k,k] + Σ_{l≠k} Q_upper[k,l] * x_l

and Q_upper[k,l] = Q[k,l] if k<l, else Q[l,k].
= (Q + Q^T)[k,l]  for k≠l (since Q is upper-triangular).

So h = Q_diag + Q_off @ x, where Q_off = (Q+Q^T) with zeroed diagonal.

Field update when bit k is flipped (Δx_k = flip = 1-2*x_k_old):
  h[l] += Q_upper[l,k] * flip  for l ≠ k
        = Q_off[l, k] * flip   (= Q_off column k, excluding row k)
  h[k] is unchanged  (h[k] depends only on x_l for l≠k)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class SimulatedAnnealer:
    """
    Solve a QUBO E(x) = x^T Q x via simulated annealing.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        QUBO matrix — upper-triangular convention as produced by IsingFormulator.
    T0 : float
        Initial temperature.
    cooling_rate : float
        Multiplicative factor per step: T(s) = T0 * cooling_rate^s.
    n_steps : int
        Number of Monte-Carlo steps.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        Q: np.ndarray,
        T0: float = 5.0,
        cooling_rate: float = 0.9995,
        n_steps: int = 20_000,
        seed: Optional[int] = None,
    ) -> None:
        self.Q = Q.astype(np.float64)
        self.n = Q.shape[0]
        self.T0 = T0
        self.cooling_rate = cooling_rate
        self.n_steps = n_steps
        self.rng = np.random.default_rng(seed)

        # Build the symmetric interaction matrix (off-diagonal only for field updates)
        # Q_eff[k,l] = Q[k,l] + Q[l,k] for k≠l, = Q[k,k] for k=l
        Q_eff = Q + Q.T
        Q_eff[np.arange(self.n), np.arange(self.n)] = np.diag(Q)
        self._Q_eff = Q_eff  # shape (n,n)

        # Off-diagonal part: zero the diagonal so h = Q_diag + 2*Q_off@x
        self._Q_off = Q_eff.copy()
        np.fill_diagonal(self._Q_off, 0.0)
        self._Q_diag = np.diag(Q)  # linear coefficients

    # ------------------------------------------------------------------

    def energy(self, x: np.ndarray) -> float:
        """QUBO energy E(x) = x^T Q x (upper-triangular convention)."""
        return float(x @ self.Q @ x)

    def _field(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the effective field h[k] = Q_diag[k] + sum_{l≠k} Q_upper[k,l]*x_l.

        For upper-triangular Q, Q_upper[k,l] = Q[k,l] if k<l, else Q[l,k].
        (Q + Q^T)[k,l] = Q_upper[k,l] for k≠l when Q is upper-triangular.
        ΔE when flipping bit k = (1 - 2*x[k]) * h[k].
        """
        return self._Q_diag + self._Q_off @ x

    def anneal(self, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
        """
        Run simulated annealing.

        Parameters
        ----------
        x0 : array of {0,1}, shape (n,), or None
            Initial solution.  Random initialisation if None.

        Returns
        -------
        best_x : np.ndarray, dtype float64
            Best binary solution found.
        best_energy : float
            Corresponding QUBO energy.
        """
        n = self.n
        x = (self.rng.integers(0, 2, size=n).astype(np.float64)
             if x0 is None else np.array(x0, dtype=np.float64))

        # Precompute initial field: O(n^2) once
        h = self._field(x)

        cur_energy = self.energy(x)
        best_x = x.copy()
        best_energy = cur_energy

        T = self.T0
        ln_cooling = np.log(self.cooling_rate)

        for step in range(self.n_steps):
            k = int(self.rng.integers(0, n))
            delta_e = (1.0 - 2.0 * x[k]) * h[k]

            if delta_e < 0.0 or self.rng.random() < np.exp(-delta_e / (T + 1e-12)):
                # Accept flip: x[k] ← 1 - x[k]
                flip = 1.0 - 2.0 * x[k]   # +1 if 0→1, -1 if 1→0
                x[k] = 1.0 - x[k]
                cur_energy += delta_e

                # Update field: h[l] += Q_upper[l,k] * flip = Q_off[l,k] * flip  for l≠k
                # h[k] is unchanged (it depends only on x_l for l≠k)
                h_k_saved = h[k]
                h += self._Q_off[:, k] * flip
                h[k] = h_k_saved   # restore

                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_x = x.copy()

            T *= self.cooling_rate

        return best_x, best_energy
