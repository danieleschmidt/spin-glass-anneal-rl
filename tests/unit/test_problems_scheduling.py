"""Unit tests for scheduling problem implementation."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from spin_glass_rl.problems.scheduling import (
    SchedulingProblem, Task, Agent, JobShopScheduling
)
from spin_glass_rl.problems.base import ProblemSolution
from spin_glass_rl.utils.exceptions import ValidationError, ProblemEncodingError


class TestTask:
    """Test Task dataclass."""
    
    def test_basic_task(self):
        """Test basic task creation."""
        task = Task(id=0, duration=5.0)
        
        assert task.id == 0
        assert task.duration == 5.0
        assert task.release_time == 0.0
        assert task.due_date is None
        assert task.priority == 1.0
        assert task.resource_requirements == {}
    
    def test_task_with_all_parameters(self):
        """Test task with all parameters."""
        resource_req = {"cpu": 2.0, "memory": 4.0}
        task = Task(
            id=1,
            duration=10.0,
            release_time=2.0,
            due_date=50.0,
            priority=2.5,
            resource_requirements=resource_req
        )
        
        assert task.id == 1
        assert task.duration == 10.0
        assert task.release_time == 2.0
        assert task.due_date == 50.0
        assert task.priority == 2.5
        assert task.resource_requirements == resource_req
    
    def test_task_post_init(self):
        """Test task __post_init__ method."""
        task = Task(id=0, duration=1.0, resource_requirements=None)
        assert task.resource_requirements == {}


class TestAgent:
    """Test Agent dataclass."""
    
    def test_basic_agent(self):
        """Test basic agent creation."""
        agent = Agent(id=0, name="Worker1")
        
        assert agent.id == 0
        assert agent.name == "Worker1"
        assert agent.capacity == {}
        assert agent.availability_windows == [(0.0, float('inf'))]
        assert agent.cost_per_hour == 1.0
    
    def test_agent_with_all_parameters(self):
        """Test agent with all parameters."""
        capacity = {"cpu": 4.0, "memory": 8.0}
        windows = [(0.0, 40.0), (50.0, 100.0)]
        
        agent = Agent(
            id=1,
            name="Server1",
            capacity=capacity,
            availability_windows=windows,
            cost_per_hour=25.0
        )
        
        assert agent.id == 1
        assert agent.name == "Server1"
        assert agent.capacity == capacity
        assert agent.availability_windows == windows
        assert agent.cost_per_hour == 25.0
    
    def test_agent_post_init(self):
        """Test agent __post_init__ method."""
        agent = Agent(id=0, name="Test", capacity=None, availability_windows=None)
        assert agent.capacity == {}
        assert agent.availability_windows == [(0.0, float('inf'))]


class TestSchedulingProblem:
    """Test SchedulingProblem class."""
    
    @pytest.fixture
    def problem(self):
        """Create test scheduling problem."""
        return SchedulingProblem()
    
    @pytest.fixture
    def simple_problem(self, problem):
        """Create simple scheduling problem with tasks and agents."""
        # Add tasks
        tasks = [
            Task(id=0, duration=3.0, priority=1.0),
            Task(id=1, duration=2.0, priority=2.0),
            Task(id=2, duration=4.0, priority=1.5)
        ]
        for task in tasks:
            problem.add_task(task)
        
        # Add agents
        agents = [
            Agent(id=0, name="Agent0"),
            Agent(id=1, name="Agent1")
        ]
        for agent in agents:
            problem.add_agent(agent)
        
        problem.time_horizon = 20.0
        problem.time_discretization = 20
        
        return problem
    
    def test_initialization(self, problem):
        """Test problem initialization."""
        assert problem.name == "Multi-Agent Scheduling"
        assert len(problem.tasks) == 0
        assert len(problem.agents) == 0
        assert problem.time_horizon == 100.0
        assert problem.time_discretization == 100
    
    def test_add_task(self, problem):
        """Test adding tasks."""
        task = Task(id=0, duration=5.0)
        problem.add_task(task)
        
        assert len(problem.tasks) == 1
        assert problem.tasks[0] == task
    
    def test_add_agent(self, problem):
        """Test adding agents."""
        agent = Agent(id=0, name="Worker")
        problem.add_agent(agent)
        
        assert len(problem.agents) == 1
        assert problem.agents[0] == agent
    
    def test_encode_to_ising_basic(self, simple_problem):
        """Test basic Ising encoding."""
        ising_model = simple_problem.encode_to_ising()
        
        assert ising_model is not None
        assert ising_model.n_spins > 0
        
        # Should have spins for each (task, agent, time) combination
        expected_spins = len(simple_problem.tasks) * len(simple_problem.agents) * simple_problem.time_discretization
        assert ising_model.n_spins == expected_spins
    
    def test_encode_to_ising_with_objectives(self, simple_problem):
        """Test Ising encoding with different objectives."""
        objectives = ["makespan", "total_time", "weighted_completion"]
        
        for objective in objectives:
            ising_model = simple_problem.encode_to_ising(objective=objective)
            assert ising_model.n_spins > 0
    
    def test_encode_to_ising_with_penalties(self, simple_problem):
        """Test Ising encoding with custom penalty weights."""
        penalty_weights = {
            "assignment": 150.0,
            "capacity": 100.0,
            "precedence": 75.0,
            "time_window": 50.0
        }
        
        ising_model = simple_problem.encode_to_ising(penalty_weights=penalty_weights)
        assert ising_model.n_spins > 0
    
    def test_variable_mapping(self, simple_problem):
        """Test variable mapping creation."""
        simple_problem.encode_to_ising()
        
        mapping = simple_problem.get_variable_mapping()
        assert len(mapping) > 0
        
        # Check that mapping includes expected keys
        n_tasks = len(simple_problem.tasks)
        n_agents = len(simple_problem.agents)
        n_times = simple_problem.time_discretization
        
        expected_keys = n_tasks * n_agents * n_times
        assert len(mapping) == expected_keys
    
    def test_decode_solution(self, simple_problem):
        """Test solution decoding."""
        ising_model = simple_problem.encode_to_ising()
        
        # Create a test spin configuration
        spins = torch.randint(0, 2, (ising_model.n_spins,)) * 2 - 1
        
        solution = simple_problem.decode_solution(spins)
        
        assert isinstance(solution, ProblemSolution)
        assert "assignments" in solution.variables
        assert "schedule" in solution.variables
        assert isinstance(solution.objective_value, float)
        assert isinstance(solution.is_feasible, bool)
        assert isinstance(solution.constraint_violations, dict)
    
    def test_solution_validation(self, simple_problem):
        """Test solution validation."""
        ising_model = simple_problem.encode_to_ising()
        spins = torch.randint(0, 2, (ising_model.n_spins,)) * 2 - 1
        solution = simple_problem.decode_solution(spins)
        
        is_valid = simple_problem.validate_solution(solution)
        assert isinstance(is_valid, bool)
    
    def test_generate_random_instance(self, problem):
        """Test random instance generation."""
        instance_params = problem.generate_random_instance(
            n_tasks=5,
            n_agents=3,
            time_horizon=50.0
        )
        
        assert instance_params["n_tasks"] == 5
        assert instance_params["n_agents"] == 3
        assert instance_params["time_horizon"] == 50.0
        
        assert len(problem.tasks) == 5
        assert len(problem.agents) == 3
        assert problem.time_horizon == 50.0
        
        # Check task properties
        for task in problem.tasks:
            assert task.duration > 0
            assert task.priority > 0
        
        # Check agent properties
        for agent in problem.agents:
            assert agent.cost_per_hour > 0
    
    def test_makespan_calculation(self, simple_problem):
        """Test makespan calculation."""
        assignments = {
            0: {"agent_id": 0, "start_time": 0.0, "end_time": 3.0},
            1: {"agent_id": 1, "start_time": 0.0, "end_time": 2.0},
            2: {"agent_id": 0, "start_time": 3.0, "end_time": 7.0}
        }
        
        makespan = simple_problem._calculate_makespan(assignments)
        assert makespan == 7.0
    
    def test_total_completion_time_calculation(self, simple_problem):
        """Test total completion time calculation."""
        assignments = {
            0: {"agent_id": 0, "start_time": 0.0, "end_time": 3.0},
            1: {"agent_id": 1, "start_time": 0.0, "end_time": 2.0},
            2: {"agent_id": 0, "start_time": 3.0, "end_time": 7.0}
        }
        
        total_time = simple_problem._calculate_total_completion_time(assignments)
        assert total_time == 12.0  # 3 + 2 + 7
    
    def test_constraint_checking(self, simple_problem):
        """Test constraint violation checking."""
        # Create valid assignments
        valid_assignments = {
            0: {"agent_id": 0, "start_time": 0.0, "end_time": 3.0},
            1: {"agent_id": 1, "start_time": 0.0, "end_time": 2.0},
            2: {"agent_id": 0, "start_time": 3.0, "end_time": 7.0}
        }
        
        violations = simple_problem._check_constraints(valid_assignments)
        assert violations["unassigned_tasks"] == 0.0
        
        # Create invalid assignments (unassigned task)
        invalid_assignments = {
            0: {"agent_id": 0, "start_time": 0.0, "end_time": 3.0},
            1: None,  # Unassigned
            2: {"agent_id": 0, "start_time": 3.0, "end_time": 7.0}
        }
        
        violations = simple_problem._check_constraints(invalid_assignments)
        assert violations["unassigned_tasks"] == 1.0
    
    def test_empty_problem_handling(self, problem):
        """Test handling of empty problem."""
        with pytest.raises((ValidationError, ValueError)):
            problem.encode_to_ising()
    
    def test_single_task_single_agent(self):
        """Test minimal problem with one task and one agent."""
        problem = SchedulingProblem()
        
        task = Task(id=0, duration=5.0)
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        problem.time_horizon = 10.0
        problem.time_discretization = 10
        
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins == 10  # 1 task * 1 agent * 10 time slots
    
    def test_large_problem(self):
        """Test larger problem instance."""
        problem = SchedulingProblem()
        
        # Add many tasks and agents
        n_tasks = 20
        n_agents = 5
        
        for i in range(n_tasks):
            task = Task(id=i, duration=np.random.uniform(1, 5))
            problem.add_task(task)
        
        for i in range(n_agents):
            agent = Agent(id=i, name=f"Agent{i}")
            problem.add_agent(agent)
        
        problem.time_discretization = 50
        
        ising_model = problem.encode_to_ising()
        expected_spins = n_tasks * n_agents * 50
        assert ising_model.n_spins == expected_spins
    
    def test_time_window_constraints(self, simple_problem):
        """Test time window constraint handling."""
        # Add due dates to tasks
        simple_problem.tasks[0].due_date = 5.0
        simple_problem.tasks[1].due_date = 10.0
        
        ising_model = simple_problem.encode_to_ising(
            penalty_weights={"time_window": 100.0}
        )
        
        # Should still encode successfully
        assert ising_model.n_spins > 0
    
    def test_problem_info(self, simple_problem):
        """Test problem information summary."""
        simple_problem.encode_to_ising()
        
        info = simple_problem.get_problem_info()
        
        assert info["name"] == "Multi-Agent Scheduling"
        assert info["n_variables"] == 0  # No variables explicitly defined
        assert info["n_constraints"] == 0  # No constraints explicitly defined
        assert info["ising_spins"] > 0


class TestJobShopScheduling:
    """Test JobShopScheduling class."""
    
    @pytest.fixture
    def job_shop(self):
        """Create test job shop problem."""
        return JobShopScheduling()
    
    def test_initialization(self, job_shop):
        """Test job shop initialization."""
        assert job_shop.name == "Job Shop Scheduling"
        assert len(job_shop.jobs) == 0
        assert len(job_shop.machines) == 0
    
    def test_add_job(self, job_shop):
        """Test adding job with operations."""
        operations = [(0, 3.0), (1, 2.0), (0, 1.0)]  # (machine_id, duration)
        job_shop.add_job(operations)
        
        assert len(job_shop.jobs) == 1
        assert job_shop.jobs[0] == operations
        
        # Should have created tasks for each operation
        assert len(job_shop.tasks) == 3
        
        # Check task properties
        for i, (machine_id, duration) in enumerate(operations):
            task = job_shop.tasks[i]
            assert task.duration == duration
            assert hasattr(task, 'metadata')
            assert task.metadata["machine_id"] == machine_id
            assert task.metadata["operation_id"] == i
    
    def test_add_machine(self, job_shop):
        """Test adding machine."""
        job_shop.add_machine(0, "Machine_A")
        
        assert len(job_shop.machines) == 1
        assert 0 in job_shop.machines
        assert len(job_shop.agents) == 1
        assert job_shop.agents[0].name == "Machine_A"
    
    def test_job_precedence_constraints(self, job_shop):
        """Test job-specific precedence constraints."""
        # Add job and machines
        operations = [(0, 2.0), (1, 3.0)]
        job_shop.add_job(operations)
        job_shop.add_machine(0)
        job_shop.add_machine(1)
        
        # Encode with precedence constraints
        ising_model = job_shop.encode_to_ising()
        assert ising_model.n_spins > 0
    
    def test_multiple_jobs(self, job_shop):
        """Test multiple jobs."""
        # Job 1: Machine 0 -> Machine 1
        job_shop.add_job([(0, 2.0), (1, 3.0)])
        
        # Job 2: Machine 1 -> Machine 0
        job_shop.add_job([(1, 1.0), (0, 2.0)])
        
        job_shop.add_machine(0)
        job_shop.add_machine(1)
        
        assert len(job_shop.jobs) == 2
        assert len(job_shop.tasks) == 4  # 2 operations per job
        
        ising_model = job_shop.encode_to_ising()
        assert ising_model.n_spins > 0


class TestSchedulingProblemEdgeCases:
    """Test edge cases for scheduling problems."""
    
    def test_zero_duration_task(self):
        """Test handling of zero-duration task."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=0.0)
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        
        # Should handle gracefully
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins > 0
    
    def test_very_long_duration_task(self):
        """Test task with duration longer than time horizon."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=200.0)  # Longer than default horizon
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins > 0
    
    def test_negative_priority_task(self):
        """Test task with negative priority."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=5.0, priority=-1.0)
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins > 0
    
    def test_very_fine_time_discretization(self):
        """Test with very fine time discretization."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=1.0)
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        problem.time_horizon = 10.0
        problem.time_discretization = 1000  # Very fine
        
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins == 1000  # 1 task * 1 agent * 1000 time slots
    
    def test_coarse_time_discretization(self):
        """Test with very coarse time discretization."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=1.0)
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        problem.time_horizon = 100.0
        problem.time_discretization = 2  # Very coarse
        
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins == 2  # 1 task * 1 agent * 2 time slots
    
    def test_impossible_due_date(self):
        """Test task with impossible due date."""
        problem = SchedulingProblem()
        task = Task(id=0, duration=10.0, release_time=0.0, due_date=5.0)  # Impossible
        agent = Agent(id=0, name="Worker")
        
        problem.add_task(task)
        problem.add_agent(agent)
        
        # Should still encode but likely lead to infeasible solution
        ising_model = problem.encode_to_ising()
        assert ising_model.n_spins > 0