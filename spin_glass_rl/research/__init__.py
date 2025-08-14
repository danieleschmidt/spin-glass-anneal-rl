"""
Research module for novel optimization algorithms.

This module contains cutting-edge algorithmic contributions to the
spin-glass optimization field, including:

- Adaptive Quantum-Inspired Annealing (AQIA)
- Multi-Scale Hierarchical Optimization (MSHO)
- Learning-Enhanced Spin Dynamics (LESD)
- Comprehensive experimental validation framework

All algorithms are research-grade implementations with statistical
validation and reproducible experimental protocols.
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
    "run_quick_validation"
]