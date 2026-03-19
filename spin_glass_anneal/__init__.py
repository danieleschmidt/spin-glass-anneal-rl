"""
spin_glass_anneal - Job-shop scheduling via QUBO/Ising formulation + simulated annealing.

Classes:
    IsingFormulator  - Encode a scheduling problem as a QUBO matrix
    SimulatedAnnealer - Solve QUBO via simulated annealing (pure numpy)
    ScheduleDecoder  - Decode a binary solution back to a job schedule
"""

from .formulator import IsingFormulator
from .annealer import SimulatedAnnealer
from .decoder import ScheduleDecoder

__all__ = ["IsingFormulator", "SimulatedAnnealer", "ScheduleDecoder"]
__version__ = "0.1.0"
