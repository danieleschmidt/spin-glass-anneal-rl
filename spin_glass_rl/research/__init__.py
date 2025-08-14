"""
Research module for novel optimization algorithms.

This module contains cutting-edge algorithmic contributions to the
spin-glass optimization field, including:

- Adaptive Quantum-Inspired Annealing (AQIA)
- Multi-Scale Hierarchical Optimization (MSHO)
- Learning-Enhanced Spin Dynamics (LESD)
- Comprehensive experimental validation framework
- Advanced performance analysis and scaling studies

All algorithms are research-grade implementations with statistical
validation and reproducible experimental protocols.

Novel Research Contributions:
============================

1. Adaptive Quantum-Inspired Annealing (AQIA)
   - Quantum fluctuation modeling with adaptive transverse fields
   - Energy barrier detection and adaptive tunneling
   - Real-time parameter optimization based on exploration efficiency

2. Multi-Scale Hierarchical Optimization (MSHO)
   - Hierarchical decomposition with adaptive scale selection
   - Cross-scale information transfer using renormalization group
   - Adaptive resolution refinement based on solution quality

3. Learning-Enhanced Spin Dynamics (LESD)
   - Neural network-guided spin updates with adaptive learning
   - Meta-learning adapts to different problem classes
   - Experience replay for improved sample efficiency

4. Experimental Validation Framework
   - Comprehensive statistical validation with multiple benchmark problems
   - Statistical significance testing with multiple comparison correction
   - Publication-ready experimental reporting
   - Reproducible experimental protocols with confidence intervals

5. Performance Analysis and Scaling Studies
   - Theoretical complexity analysis with empirical validation
   - Multi-dimensional scaling studies (problem size, algorithm parameters)
   - Performance prediction models using machine learning
   - Real-time performance profiling with bottleneck identification

Research Quality Assurance:
==========================
- All algorithms implement robust error handling
- Comprehensive statistical validation
- Reproducible experimental protocols
- Publication-ready code and documentation
- Academic-grade peer review standards
"""

from .novel_algorithms import (
    NovelAlgorithm,
    AlgorithmConfig,
    AdaptiveQuantumInspiredAnnealing,
    MultiScaleHierarchicalOptimization,
    LearningEnhancedSpinDynamics,
    NovelAlgorithmFactory,
    run_algorithm_comparison
)

from .experimental_validation import (
    ExperimentalValidation,
    ExperimentConfig,
    ProblemGenerator,
    StatisticalAnalyzer,
    run_quick_validation
)

from .performance_analysis import (
    ScalingAnalyzer,
    ScalingConfig,
    PerformanceMetrics,
    ComplexityAnalyzer,
    PerformanceProfiler,
    run_quick_scaling_analysis
)

__all__ = [
    # Core algorithm classes
    "NovelAlgorithm",
    "AlgorithmConfig", 
    "AdaptiveQuantumInspiredAnnealing",
    "MultiScaleHierarchicalOptimization",
    "LearningEnhancedSpinDynamics",
    "NovelAlgorithmFactory",
    "run_algorithm_comparison",
    
    # Experimental validation
    "ExperimentalValidation",
    "ExperimentConfig",
    "ProblemGenerator", 
    "StatisticalAnalyzer",
    "run_quick_validation",
    
    # Performance analysis and scaling
    "ScalingAnalyzer",
    "ScalingConfig",
    "PerformanceMetrics",
    "ComplexityAnalyzer", 
    "PerformanceProfiler",
    "run_quick_scaling_analysis"
]