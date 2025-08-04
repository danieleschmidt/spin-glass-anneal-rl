"""Problem domain implementations for common optimization problems."""

from spin_glass_rl.problems.base import ProblemTemplate
from spin_glass_rl.problems.scheduling import SchedulingProblem, JobShopScheduling
from spin_glass_rl.problems.routing import RoutingProblem, TSPProblem, VRPProblem
from spin_glass_rl.problems.resource_allocation import ResourceAllocationProblem
from spin_glass_rl.problems.coordination import CoordinationProblem

__all__ = [
    "ProblemTemplate",
    "SchedulingProblem", 
    "JobShopScheduling",
    "RoutingProblem",
    "TSPProblem",
    "VRPProblem", 
    "ResourceAllocationProblem",
    "CoordinationProblem",
]