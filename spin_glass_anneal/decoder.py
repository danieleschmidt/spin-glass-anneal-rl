"""
ScheduleDecoder: decode a binary QUBO solution into a human-readable job schedule.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np


class ScheduleDecoder:
    """
    Decode a binary solution vector from SimulatedAnnealer into a job schedule.

    Parameters
    ----------
    n_jobs : int
    n_machines : int
    n_time_slots : int
    """

    def __init__(self, n_jobs: int, n_machines: int, n_time_slots: int) -> None:
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_time_slots = n_time_slots

    def decode(
        self,
        x: np.ndarray,
        var_map: Dict[Tuple[int, int, int], int],
    ) -> Dict[int, Tuple[int, int]]:
        """
        Decode a binary solution into a schedule.

        For each job, find the (machine, time_slot) with x[idx] == 1.
        If multiple bits are set for a job (constraint violation), pick the
        earliest time slot to remain valid.  If no bit is set, assign to
        slot 0 on machine 0 as a fallback.

        Parameters
        ----------
        x : np.ndarray, shape (n_vars,)
            Binary solution vector (values ∈ {0, 1}).
        var_map : dict
            Mapping (job, machine, time_slot) → flat index.

        Returns
        -------
        schedule : dict
            {job_id: (machine_id, start_time)}
        """
        schedule: Dict[int, Tuple[int, int]] = {}

        for i in range(self.n_jobs):
            # Collect all active variables for this job, prefer earliest time
            candidates = []
            for j in range(self.n_machines):
                for t in range(self.n_time_slots):
                    idx = var_map.get((i, j, t))
                    if idx is not None and x[idx] > 0.5:
                        candidates.append((t, j))  # (time, machine)

            if candidates:
                # Pick earliest assignment
                t, j = min(candidates, key=lambda c: c[0])
                schedule[i] = (j, t)
            else:
                # Fallback: no variable set — assign to machine 0, slot 0
                schedule[i] = (0, 0)

        return schedule

    def makespan(self, schedule: Dict[int, Tuple[int, int]]) -> int:
        """
        Compute the makespan (= max completion time) of a decoded schedule.

        Assuming each job takes exactly 1 time slot, makespan = max(start_time) + 1.
        """
        if not schedule:
            return 0
        return max(start_time for (_, start_time) in schedule.values()) + 1

    def pretty_print(self, schedule: Dict[int, Tuple[int, int]]) -> str:
        """Return a human-readable schedule string."""
        lines = ["Job Schedule:", "-" * 40]
        for job_id in sorted(schedule):
            machine_id, start_time = schedule[job_id]
            lines.append(
                f"  Job {job_id:2d} → Machine {machine_id}, Time slot {start_time}"
            )
        lines.append("-" * 40)
        lines.append(f"  Makespan: {self.makespan(schedule)} time slots")
        return "\n".join(lines)
