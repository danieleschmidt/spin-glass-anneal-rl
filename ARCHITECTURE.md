# Architecture Documentation

## System Overview

Spin-Glass-Anneal-RL is a hybrid optimization framework that combines reinforcement learning with physics-inspired spin-glass models to solve complex multi-agent scheduling and coordination problems. The system leverages GPU-accelerated digital annealing to find near-optimal solutions for computationally intractable problems.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Problem Domain APIs  │  RL Integration  │  Visualization      │
│  - Scheduling         │  - Policy Nets   │  - Energy Plots     │
│  - Routing           │  - Value Funcs   │  - Solution Analysis │
│  - Resource Alloc.   │  - Exploration   │  - 3D Landscapes    │
├─────────────────────────────────────────────────────────────────┤
│                    Core Engine Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  Ising Model Core     │  Annealing Engine │  Constraint System │
│  - Spin Dynamics      │  - GPU Kernels    │  - Penalty Methods │
│  - Coupling Matrix    │  - Temp Schedules │  - Soft Constraints│
│  - Energy Compute     │  - Parallel Temp  │  - Learned Rules   │
├─────────────────────────────────────────────────────────────────┤
│                    Hardware Layer                              │
├─────────────────────────────────────────────────────────────────┤
│  CUDA Runtime         │  Quantum Hardware │  Multi-GPU Support │
│  - Memory Management  │  - D-Wave API     │  - Graph Partition │
│  - Kernel Execution   │  - Fujitsu DAU    │  - Load Balancing  │
│  - Stream Processing  │  - QUBO Interface │  - Communication   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Ising Model Core (`spin_glass_rl/core/`)

**Purpose**: Fundamental spin-glass model representation and operations

**Key Classes**:
- `IsingModel`: Main model representation with spin variables and couplings
- `SpinDynamics`: Handles spin update rules and Monte Carlo moves
- `CouplingMatrix`: Manages interaction graphs between spins
- `EnergyComputer`: Efficient energy calculation for spin configurations

**Data Flow**:
```
Problem Instance → Ising Encoder → Ising Model → Energy Computation
```

### 2. GPU Annealing Engine (`spin_glass_rl/annealing/`)

**Purpose**: High-performance GPU-accelerated optimization

**Key Components**:
- `CUDAAnnealer`: Main CUDA-based annealing engine
- `TemperatureScheduler`: Various cooling schedules (geometric, linear, adaptive)
- `ParallelTempering`: Replica exchange for better exploration
- `QuantumInspired`: Quantum fluctuation simulation

**Architecture**:
```
Host Problem → GPU Memory → CUDA Kernels → Host Solution
     ↓              ↓           ↓              ↑
   Upload     Spin Updates  Energy Calc    Download
```

### 3. RL Integration (`spin_glass_rl/rl_integration/`)

**Purpose**: Hybrid reinforcement learning and annealing

**Components**:
- `MDPToIsing`: Converts MDP/POMDP to spin-glass formulation
- `HybridAgent`: RL agents that guide annealing process
- `RewardDesigner`: Energy-based reward functions
- `ExplorationStrategy`: Spin-based exploration methods

### 4. Problem Domains (`spin_glass_rl/problems/`)

**Purpose**: Specific problem formulations and benchmarks

**Domains**:
- **Scheduling**: Job shop, flow shop, resource-constrained scheduling
- **Routing**: TSP, VRP, multi-depot routing
- **Resource Allocation**: Facility location, assignment problems
- **Coordination**: Multi-agent pathfinding, task allocation

## Data Flow Architecture

### 1. Problem Formulation Flow
```
Real Problem → Feature Extraction → Ising Encoding → Constraint Addition → Optimization
```

### 2. Optimization Flow
```
Initial Config → GPU Upload → Annealing Loop → Energy Monitoring → Solution Download
```

### 3. RL-Guided Flow
```
Problem Features → Policy Network → Initial Guess → Guided Annealing → Reward Feedback
```

## Memory Architecture

### CPU Memory Layout
```
┌─────────────────┐
│ Problem Data    │ ← Original problem representation
├─────────────────┤
│ Ising Model     │ ← Couplings, fields, constraints
├─────────────────┤
│ Solution Buffer │ ← Best configurations found
├─────────────────┤
│ Metrics Data    │ ← Energy trajectories, statistics
└─────────────────┘
```

### GPU Memory Layout
```
┌─────────────────┐
│ Spin Arrays     │ ← Current configurations (multiple replicas)
├─────────────────┤
│ Coupling Matrix │ ← Interaction strengths (sparse format)
├─────────────────┤
│ Local Fields    │ ← External field values
├─────────────────┤
│ Random States   │ ← CURAND generator states
├─────────────────┤
│ Energy Buffers  │ ← Intermediate energy calculations
└─────────────────┘
```

## Concurrency Model

### Multi-GPU Scaling
```
Problem Decomposition → Graph Partitioning → GPU Assignment → Result Aggregation
```

### Thread Safety
- **GPU Kernels**: Lock-free algorithms with atomic operations
- **CPU Threads**: Reader-writer locks for shared state
- **MPI Communication**: Non-blocking collective operations

## Performance Characteristics

### Computational Complexity
- **Energy Evaluation**: O(E) where E is number of edges
- **Spin Update**: O(N) where N is number of spins
- **Temperature Schedule**: O(1) per step
- **Overall Annealing**: O(S × N × E) where S is sweeps

### Memory Requirements
- **Sparse Coupling**: O(E) storage for E non-zero couplings
- **Dense Coupling**: O(N²) for fully connected problems
- **Parallel Replicas**: Linear scaling with replica count

### Scaling Properties
- **Problem Size**: Near-linear GPU scaling up to memory limits
- **Replica Count**: Perfect parallelization across GPU cores
- **Multi-GPU**: Sub-linear scaling due to communication overhead

## Extension Points

### Custom Problem Integration
1. Implement `ProblemTemplate` interface
2. Define `to_ising()` method for problem conversion
3. Add problem-specific constraints and objectives
4. Register with problem factory for automatic discovery

### Hardware Backend Integration
1. Implement `HardwareInterface` for new annealing hardware
2. Provide format conversion (QUBO, Ising, etc.)
3. Handle hardware-specific constraints and limitations
4. Add performance profiling and benchmarking

### Algorithm Extensions
1. Extend `AnnealingAlgorithm` base class
2. Implement custom update rules and schedules
3. Add algorithm-specific parameters and tuning
4. Provide theoretical analysis and complexity bounds

## Quality Attributes

### Performance
- Sub-second solutions for problems with 10k+ variables
- Linear scaling with GPU compute units
- Memory-efficient sparse matrix representations

### Reliability
- Graceful degradation when hardware limits exceeded
- Automatic fallback to CPU computation
- Comprehensive error handling and recovery

### Maintainability
- Modular architecture with clear interfaces
- Extensive unit testing and benchmarking
- Comprehensive documentation and examples

### Extensibility
- Plugin architecture for new problem domains
- Abstract interfaces for hardware backends
- Configurable algorithm parameters and schedules