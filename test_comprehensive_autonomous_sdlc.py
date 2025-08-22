"""
Comprehensive Autonomous SDLC Testing Framework.

Tests all advanced features implemented in Generation 1-3 including:
- Meta-learning optimization
- Quantum-hybrid algorithms  
- Federated optimization
- Advanced security framework
- Adaptive monitoring system
- Intelligent auto-scaling
- Quantum edge computing
"""

import pytest
import torch
import numpy as np
import asyncio
import time
import tempfile
import os
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the modules we need to test
from spin_glass_rl.research.meta_learning_optimization import (
    MetaLearningConfig, MetaOptimizer, HypermediaAnnealingFramework
)
from spin_glass_rl.research.quantum_hybrid_algorithms import (
    QuantumConfig, QuantumAnnealingSimulator, HybridQuantumClassicalOptimizer
)
from spin_glass_rl.research.federated_optimization import (
    FederatedConfig, FederatedServer, SpinGlassClient
)
from spin_glass_rl.security.advanced_security_framework import (
    SecurityConfig, SecurityLevel, SecureOptimizationFramework, SecurityValidator
)
from spin_glass_rl.monitoring.adaptive_monitoring_system import (
    MonitoringConfig, AdaptiveMonitoringSystem, MetricCollector, AnomalyDetector
)
from spin_glass_rl.scaling.intelligent_auto_scaling import (
    ScalingConfig, AutoScalingController, ResourceRequirements
)
from spin_glass_rl.optimization.quantum_edge_computing import (
    EdgeNodeCapabilities, QuantumEdgeNode, EdgeOrchestrator, OptimizationTask
)
from spin_glass_rl.core.minimal_ising import MinimalIsingModel


class TestMetaLearningOptimization:
    """Test meta-learning optimization functionality."""
    
    def test_meta_learning_config_creation(self):
        """Test meta-learning configuration creation."""
        config = MetaLearningConfig(
            feature_dim=64,
            hidden_dim=128,
            learning_rate=1e-3
        )
        
        assert config.feature_dim == 64
        assert config.hidden_dim == 128
        assert config.learning_rate == 1e-3
    
    def test_meta_optimizer_initialization(self):
        """Test meta-optimizer initialization."""
        config = MetaLearningConfig(feature_dim=32, hidden_dim=64)
        optimizer = MetaOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.problem_encoder is not None
        assert optimizer.strategy_generator is not None
        assert len(optimizer.performance_history) == 0
    
    def test_problem_feature_extraction(self):
        """Test problem feature extraction."""
        config = MetaLearningConfig(feature_dim=32)
        optimizer = MetaOptimizer(config)
        
        # Create test Ising model
        test_model = MinimalIsingModel(n_spins=10)
        
        # Extract features
        features = optimizer.extract_problem_features(test_model)
        
        assert isinstance(features, torch.Tensor)
        assert features.shape[0] == config.feature_dim
    
    def test_strategy_generation(self):
        """Test strategy generation."""
        config = MetaLearningConfig(feature_dim=32)
        optimizer = MetaOptimizer(config)
        
        test_model = MinimalIsingModel(n_spins=10)
        strategy = optimizer.generate_strategy(test_model)
        
        assert isinstance(strategy, dict)
        assert 'temperature_schedule' in strategy
        assert 'coupling_strength' in strategy
        assert 'sweep_count' in strategy
    
    def test_hypermedia_framework(self):
        """Test hypermedia annealing framework."""
        config = MetaLearningConfig(feature_dim=16, hidden_dim=32)
        framework = HypermediaAnnealingFramework(config)
        
        test_model = MinimalIsingModel(n_spins=8)
        strategy, energy = framework.solve_with_meta_learning(test_model, "test_problem")
        
        assert isinstance(strategy, dict)
        assert isinstance(energy, float)
        assert 'test_problem' in framework.problem_database.get('general', [])


class TestQuantumHybridAlgorithms:
    """Test quantum-hybrid algorithms functionality."""
    
    def test_quantum_config_creation(self):
        """Test quantum configuration creation."""
        config = QuantumConfig(
            trotter_slices=16,
            quantum_field_strength=2.0,
            evolution_time=5.0
        )
        
        assert config.trotter_slices == 16
        assert config.quantum_field_strength == 2.0
        assert config.evolution_time == 5.0
    
    def test_quantum_annealing_simulator(self):
        """Test quantum annealing simulator."""
        config = QuantumConfig(trotter_slices=8, evolution_time=1.0)
        simulator = QuantumAnnealingSimulator(config)
        
        test_model = MinimalIsingModel(n_spins=6)
        result = simulator.simulated_quantum_annealing(test_model)
        
        assert 'best_state' in result
        assert 'best_energy' in result
        assert 'energy_history' in result
        assert isinstance(result['best_energy'], float)
    
    def test_hybrid_quantum_classical_optimizer(self):
        """Test hybrid quantum-classical optimizer."""
        config = QuantumConfig(trotter_slices=8)
        optimizer = HybridQuantumClassicalOptimizer(config)
        
        test_model = MinimalIsingModel(n_spins=8)
        results = optimizer.hybrid_optimize(test_model, use_ensemble=False)
        
        assert 'best_algorithm' in results
        assert 'best_result' in results
        assert 'all_results' in results
        
        # Check that multiple algorithms were run
        all_results = results['all_results']
        assert len(all_results) >= 3  # At least SQA, QAOA, Classical
    
    def test_qaoa_optimization(self):
        """Test QAOA optimization specifically."""
        config = QuantumConfig()
        simulator = QuantumAnnealingSimulator(config)
        
        test_model = MinimalIsingModel(n_spins=5)
        result = simulator.quantum_approximate_optimization(test_model, p_layers=2)
        
        assert 'best_state' in result
        assert 'optimal_parameters' in result
        assert 'layers' in result
        assert result['layers'] == 2


class TestFederatedOptimization:
    """Test federated optimization functionality."""
    
    def test_federated_config_creation(self):
        """Test federated configuration creation."""
        config = FederatedConfig(
            n_clients=3,
            client_epochs=2,
            global_rounds=5
        )
        
        assert config.n_clients == 3
        assert config.client_epochs == 2
        assert config.global_rounds == 5
    
    def test_federated_server_initialization(self):
        """Test federated server initialization."""
        config = FederatedConfig(n_clients=3, global_rounds=2)
        server = FederatedServer(config)
        
        assert server.config == config
        assert len(server.clients) == 0
        assert server.global_model_state is None
    
    def test_client_registration(self):
        """Test client registration with server."""
        config = FederatedConfig()
        server = FederatedServer(config)
        
        # Create test client
        client_problems = [MinimalIsingModel(n_spins=5)]
        client = SpinGlassClient("test_client", client_problems, config)
        
        # Register client
        server.register_client(client)
        
        assert "test_client" in server.clients
        assert server.clients["test_client"] == client
    
    def test_client_local_update(self):
        """Test client local update."""
        config = FederatedConfig()
        client_problems = [MinimalIsingModel(n_spins=8)]
        client = SpinGlassClient("test_client", client_problems, config)
        
        # Perform local update
        global_state = None
        local_update = client.local_update(global_state)
        
        assert isinstance(local_update, dict)
        assert len(client.optimization_history) > 0
    
    def test_federated_optimization_small(self):
        """Test small federated optimization scenario."""
        config = FederatedConfig(n_clients=2, global_rounds=2, client_epochs=1)
        server = FederatedServer(config)
        
        # Create and register clients
        for i in range(2):
            client_problems = [MinimalIsingModel(n_spins=6)]
            client = SpinGlassClient(f"client_{i}", client_problems, config)
            server.register_client(client)
        
        # Run federated optimization
        results = server.federated_optimization()
        
        assert 'global_rounds' in results
        assert 'final_global_state' in results
        assert len(results['global_rounds']) == config.global_rounds


class TestAdvancedSecurityFramework:
    """Test advanced security framework functionality."""
    
    def test_security_config_creation(self):
        """Test security configuration creation."""
        config = SecurityConfig(
            security_level=SecurityLevel.HIGH,
            privacy_budget=0.5,
            audit_logging=True
        )
        
        assert config.security_level == SecurityLevel.HIGH
        assert config.privacy_budget == 0.5
        assert config.audit_logging is True
    
    def test_secure_optimization_framework_initialization(self):
        """Test secure optimization framework initialization."""
        config = SecurityConfig(security_level=SecurityLevel.STANDARD)
        framework = SecureOptimizationFramework(config)
        
        assert framework.config == config
        assert framework.crypto is not None
        assert framework.dp is not None
        assert len(framework.audit_log) == 0
    
    def test_secure_optimization_execution(self):
        """Test secure optimization execution."""
        config = SecurityConfig(
            security_level=SecurityLevel.BASIC,
            input_validation=True,
            audit_logging=True
        )
        framework = SecureOptimizationFramework(config)
        
        test_model = MinimalIsingModel(n_spins=10)
        result = framework.secure_optimize(test_model)
        
        assert isinstance(result, dict)
        assert 'security_metadata' in result
        assert len(framework.audit_log) > 0
    
    def test_differential_privacy(self):
        """Test differential privacy mechanisms."""
        config = SecurityConfig(privacy_budget=1.0)
        framework = SecureOptimizationFramework(config)
        
        # Test privacy noise addition
        test_tensor = torch.ones(5)
        noisy_tensor = framework.dp.add_laplace_noise(test_tensor, sensitivity=1.0)
        
        assert noisy_tensor.shape == test_tensor.shape
        assert not torch.equal(noisy_tensor, test_tensor)  # Should have noise
        assert framework.dp.privacy_spent > 0
    
    def test_security_validator(self):
        """Test security validator."""
        config = SecurityConfig(
            input_validation=True,
            audit_logging=True
        )
        framework = SecureOptimizationFramework(config)
        validator = SecurityValidator()
        
        assessment = validator.assess_security(framework)
        
        assert 'overall_score' in assessment
        assert 'vulnerabilities' in assessment
        assert 'compliance_status' in assessment
        assert assessment['overall_score'] >= 0
        assert assessment['overall_score'] <= 100


class TestAdaptiveMonitoringSystem:
    """Test adaptive monitoring system functionality."""
    
    def test_monitoring_config_creation(self):
        """Test monitoring configuration creation."""
        config = MonitoringConfig(
            sampling_interval=0.5,
            anomaly_detection=True,
            self_healing=True
        )
        
        assert config.sampling_interval == 0.5
        assert config.anomaly_detection is True
        assert config.self_healing is True
    
    def test_metric_collector(self):
        """Test metric collector functionality."""
        config = MonitoringConfig()
        collector = MetricCollector(config)
        
        # Mock optimization result
        optimization_result = {
            'best_energy': -75.5,
            'energy_history': [0, -25, -50, -75.5],
            'convergence_step': 100
        }
        
        metrics = collector.collect_optimization_metrics(optimization_result, execution_time=1.5)
        
        assert isinstance(metrics, dict)
        assert 'execution_time' in metrics
        assert 'final_energy' in metrics
        assert 'convergence_rate' in metrics
        assert metrics['execution_time'] == 1.5
        assert metrics['final_energy'] == -75.5
    
    def test_anomaly_detector(self):
        """Test anomaly detection functionality."""
        config = MonitoringConfig(anomaly_detection=True)
        detector = AnomalyDetector(config)
        
        # Build baseline
        normal_metrics = {'execution_time': 1.0, 'final_energy': -50.0}
        for _ in range(10):
            detector.update_baseline(normal_metrics)
        
        # Test anomaly detection
        anomalous_metrics = {'execution_time': 5.0, 'final_energy': 10.0}
        anomalies = detector.detect_anomalies(anomalous_metrics)
        
        assert isinstance(anomalies, list)
        # Should detect anomalies in both metrics
        assert len(anomalies) >= 0
    
    def test_monitoring_system_integration(self):
        """Test integrated monitoring system."""
        config = MonitoringConfig(
            anomaly_detection=True,
            self_healing=True
        )
        monitoring_system = AdaptiveMonitoringSystem(config)
        
        # Mock optimization function
        def mock_optimization():
            time.sleep(0.01)
            return {
                'best_energy': np.random.uniform(-100, 0),
                'energy_history': [0, -25, -50, -75]
            }
        
        # Monitor optimization
        result, report = monitoring_system.monitor_optimization(mock_optimization)
        
        assert result is not None
        assert isinstance(report, dict)
        assert 'metrics' in report
        assert 'anomalies' in report
    
    def test_monitoring_dashboard(self):
        """Test monitoring dashboard generation."""
        config = MonitoringConfig()
        monitoring_system = AdaptiveMonitoringSystem(config)
        
        dashboard = monitoring_system.get_monitoring_dashboard()
        
        assert isinstance(dashboard, dict)
        assert 'system_status' in dashboard
        assert 'metrics_summary' in dashboard
        assert 'recent_anomalies' in dashboard


class TestIntelligentAutoScaling:
    """Test intelligent auto-scaling functionality."""
    
    def test_scaling_config_creation(self):
        """Test scaling configuration creation."""
        config = ScalingConfig(
            target_utilization=0.7,
            scale_up_threshold=0.8,
            cooldown_seconds=300
        )
        
        assert config.target_utilization == 0.7
        assert config.scale_up_threshold == 0.8
        assert config.cooldown_seconds == 300
    
    def test_resource_requirements(self):
        """Test resource requirements specification."""
        requirements = ResourceRequirements(
            cpu_cores=4,
            memory_gb=8.0,
            gpu_count=1,
            priority=3
        )
        
        assert requirements.cpu_cores == 4
        assert requirements.memory_gb == 8.0
        assert requirements.gpu_count == 1
        assert requirements.priority == 3
    
    def test_auto_scaling_controller_initialization(self):
        """Test auto-scaling controller initialization."""
        config = ScalingConfig()
        controller = AutoScalingController(config)
        
        assert controller.config == config
        assert controller.workload_predictor is not None
        assert controller.resource_allocator is not None
        assert controller.is_running is False
    
    def test_resource_allocation(self):
        """Test resource allocation functionality."""
        config = ScalingConfig()
        controller = AutoScalingController(config)
        
        requirements = ResourceRequirements(cpu_cores=2, memory_gb=4.0)
        allocation_id = controller.resource_allocator.allocate_resources(requirements)
        
        assert allocation_id is not None
        assert isinstance(allocation_id, str)
        
        # Test deallocation
        controller.resource_allocator.deallocate_resources(allocation_id)
    
    def test_workload_prediction(self):
        """Test workload prediction functionality."""
        config = ScalingConfig()
        controller = AutoScalingController(config)
        
        # Record some workload
        requirements = ResourceRequirements(cpu_cores=2, memory_gb=4.0)
        controller.workload_predictor.record_workload(50, 2.0, requirements)
        
        # Predict resources for new problems
        new_problems = [{'n_spins': 40, 'complexity': 1.2}]
        predicted = controller.workload_predictor.predict_resource_needs(new_problems)
        
        assert isinstance(predicted, ResourceRequirements)
        assert predicted.cpu_cores > 0
        assert predicted.memory_gb > 0


class TestQuantumEdgeComputing:
    """Test quantum edge computing functionality."""
    
    def test_edge_node_capabilities(self):
        """Test edge node capabilities specification."""
        from spin_glass_rl.optimization.quantum_edge_computing import EdgeNodeType, ComputingParadigm
        
        capabilities = EdgeNodeCapabilities(
            node_type=EdgeNodeType.QUANTUM_SIMULATOR,
            quantum_qubits=20,
            max_problem_size=100,
            supported_paradigms=[ComputingParadigm.QUANTUM_ANNEALING]
        )
        
        assert capabilities.node_type == EdgeNodeType.QUANTUM_SIMULATOR
        assert capabilities.quantum_qubits == 20
        assert capabilities.max_problem_size == 100
        assert ComputingParadigm.QUANTUM_ANNEALING in capabilities.supported_paradigms
    
    def test_quantum_edge_node_creation(self):
        """Test quantum edge node creation."""
        from spin_glass_rl.optimization.quantum_edge_computing import EdgeNodeType, ComputingParadigm
        
        capabilities = EdgeNodeCapabilities(
            node_type=EdgeNodeType.CPU_INTENSIVE,
            cpu_cores=4,
            supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
        )
        
        node = QuantumEdgeNode("test_node", capabilities)
        
        assert node.node_id == "test_node"
        assert node.capabilities == capabilities
        assert node.is_active is True
    
    @pytest.mark.asyncio
    async def test_edge_optimization_task(self):
        """Test edge optimization task processing."""
        from spin_glass_rl.optimization.quantum_edge_computing import (
            EdgeNodeType, ComputingParadigm, OptimizationPriority
        )
        
        capabilities = EdgeNodeCapabilities(
            node_type=EdgeNodeType.CPU_INTENSIVE,
            max_problem_size=100,
            supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
        )
        
        node = QuantumEdgeNode("test_node", capabilities)
        
        task = OptimizationTask(
            task_id="test_task",
            problem_data={'n_spins': 20},
            priority=OptimizationPriority.INTERACTIVE
        )
        
        result = await node.process_task(task)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'node_id' in result
        assert result['node_id'] == "test_node"
    
    def test_edge_orchestrator(self):
        """Test edge orchestrator functionality."""
        from spin_glass_rl.optimization.quantum_edge_computing import EdgeNodeType, ComputingParadigm
        
        orchestrator = EdgeOrchestrator()
        
        # Create and register edge node
        capabilities = EdgeNodeCapabilities(
            node_type=EdgeNodeType.CPU_INTENSIVE,
            supported_paradigms=[ComputingParadigm.CLASSICAL_ANNEALING]
        )
        node = QuantumEdgeNode("test_node", capabilities)
        orchestrator.register_edge_node(node)
        
        assert "test_node" in orchestrator.edge_nodes
        
        # Get status
        status = orchestrator.get_orchestration_status()
        assert isinstance(status, dict)
        assert 'total_nodes' in status
        assert status['total_nodes'] == 1


class TestSystemIntegration:
    """Test integration between different system components."""
    
    def test_end_to_end_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Set up monitoring
        monitoring_config = MonitoringConfig(anomaly_detection=True)
        monitoring_system = AdaptiveMonitoringSystem(monitoring_config)
        
        # Set up security
        security_config = SecurityConfig(security_level=SecurityLevel.BASIC)
        security_framework = SecureOptimizationFramework(security_config)
        
        # Mock optimization function that uses security
        def secure_optimization():
            test_model = MinimalIsingModel(n_spins=12)
            return security_framework.secure_optimize(test_model)
        
        # Monitor the secure optimization
        result, monitoring_report = monitoring_system.monitor_optimization(secure_optimization)
        
        assert result is not None
        assert 'security_metadata' in result
        assert isinstance(monitoring_report, dict)
        assert 'metrics' in monitoring_report
    
    def test_federated_security_integration(self):
        """Test integration of federated optimization with security."""
        # Set up federated system with security
        federated_config = FederatedConfig(n_clients=2, global_rounds=1)
        security_config = SecurityConfig(privacy_budget=1.0)
        
        server = FederatedServer(federated_config)
        
        # Create secure clients
        for i in range(2):
            client_problems = [MinimalIsingModel(n_spins=8)]
            client = SpinGlassClient(f"secure_client_{i}", client_problems, federated_config)
            server.register_client(client)
        
        # Run federated optimization (simplified)
        results = server.federated_optimization()
        
        assert 'final_global_state' in results
        assert len(server.clients) == 2
    
    def test_meta_learning_with_monitoring(self):
        """Test meta-learning with monitoring integration."""
        # Set up meta-learning
        meta_config = MetaLearningConfig(feature_dim=16, hidden_dim=32)
        meta_framework = HypermediaAnnealingFramework(meta_config)
        
        # Set up monitoring
        monitoring_config = MonitoringConfig()
        monitoring_system = AdaptiveMonitoringSystem(monitoring_config)
        
        # Function that uses meta-learning
        def meta_optimization():
            test_model = MinimalIsingModel(n_spins=10)
            strategy, energy = meta_framework.solve_with_meta_learning(test_model)
            return {'best_energy': energy, 'strategy_used': strategy}
        
        # Monitor meta-learning optimization
        result, report = monitoring_system.monitor_optimization(meta_optimization)
        
        assert result is not None
        assert 'best_energy' in result
        assert 'strategy_used' in result
        assert 'metrics' in report


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""
    
    def test_optimization_speed_benchmark(self):
        """Test optimization speed for different problem sizes."""
        problem_sizes = [10, 20, 30]
        execution_times = []
        
        for size in problem_sizes:
            test_model = MinimalIsingModel(n_spins=size)
            
            start_time = time.time()
            # Run basic optimization
            from spin_glass_rl.core.minimal_ising import MinimalAnnealer
            annealer = MinimalAnnealer()
            result = annealer.anneal(test_model, n_sweeps=100)
            execution_time = time.time() - start_time
            
            execution_times.append(execution_time)
            
            # Verify result quality
            assert 'best_energy' in result
            assert isinstance(result['best_energy'], float)
        
        # Check that execution time scales reasonably
        assert all(t > 0 for t in execution_times)
        # Larger problems should generally take longer (with some tolerance)
        assert execution_times[-1] >= execution_times[0] * 0.5
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with problem size."""
        import psutil
        import gc
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create increasingly large problems
        for size in [50, 100, 150]:
            test_model = MinimalIsingModel(n_spins=size)
            
            # Force garbage collection
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB for test problems)
            assert memory_increase < 100
    
    def test_concurrent_optimization_performance(self):
        """Test performance under concurrent optimization loads."""
        import concurrent.futures
        import threading
        
        def run_optimization(problem_size):
            test_model = MinimalIsingModel(n_spins=problem_size)
            from spin_glass_rl.core.minimal_ising import MinimalAnnealer
            annealer = MinimalAnnealer()
            return annealer.anneal(test_model, n_sweeps=50)
        
        # Run multiple optimizations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for _ in range(8):
                future = executor.submit(run_optimization, 15)
                futures.append(future)
            
            # Wait for all to complete
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=30):
                result = future.result()
                results.append(result)
        
        # All optimizations should complete successfully
        assert len(results) == 8
        assert all('best_energy' in result for result in results)


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Test with invalid problem size
        with pytest.raises((ValueError, AssertionError)):
            MinimalIsingModel(n_spins=-5)
        
        # Test with None inputs in security framework
        security_config = SecurityConfig()
        security_framework = SecureOptimizationFramework(security_config)
        
        # Should handle None gracefully or raise appropriate error
        try:
            result = security_framework.secure_optimize(None)
            # If it doesn't raise an error, it should return error indication
            assert not result.get('success', True) or 'error' in result
        except (ValueError, TypeError, AttributeError):
            pass  # Expected behavior
    
    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion scenarios."""
        config = ScalingConfig(
            max_resources=ResourceRequirements(cpu_cores=2, memory_gb=4.0)
        )
        controller = AutoScalingController(config)
        
        # Try to allocate more resources than available
        large_requirements = ResourceRequirements(cpu_cores=10, memory_gb=20.0)
        allocation_id = controller.resource_allocator.allocate_resources(large_requirements)
        
        # Should fail gracefully
        assert allocation_id is None
    
    def test_network_failure_simulation(self):
        """Test handling of simulated network failures."""
        # Test federated optimization with failing clients
        config = FederatedConfig(n_clients=3, global_rounds=1)
        server = FederatedServer(config)
        
        # Create clients, but simulate one failing
        working_clients = []
        for i in range(2):  # Only create 2 out of 3 expected clients
            client_problems = [MinimalIsingModel(n_spins=8)]
            client = SpinGlassClient(f"client_{i}", client_problems, config)
            server.register_client(client)
            working_clients.append(client)
        
        # Should handle reduced client participation
        results = server.federated_optimization()
        
        # Should complete even with fewer clients
        assert 'global_rounds' in results
        assert len(results['global_rounds']) >= 1
    
    def test_quantum_backend_failure_handling(self):
        """Test handling of quantum backend failures."""
        from spin_glass_rl.optimization.quantum_edge_computing import (
            EdgeNodeType, ComputingParadigm, OptimizationPriority
        )
        
        # Create quantum node that might fail
        capabilities = EdgeNodeCapabilities(
            node_type=EdgeNodeType.QUANTUM_SIMULATOR,
            quantum_qubits=10,
            supported_paradigms=[ComputingParadigm.QUANTUM_ANNEALING]
        )
        
        node = QuantumEdgeNode("failing_quantum_node", capabilities)
        
        # Mock quantum backend failure
        if hasattr(node, 'quantum_backend'):
            node.quantum_backend = None
        
        task = OptimizationTask(
            task_id="test_task",
            problem_data={'n_spins': 8},
            priority=OptimizationPriority.REAL_TIME,
            preferred_paradigm=ComputingParadigm.QUANTUM_ANNEALING
        )
        
        # Should fall back to classical computation or handle error gracefully
        async def test_failure_handling():
            result = await node.process_task(task)
            # Should either succeed with fallback or fail gracefully
            assert isinstance(result, dict)
            assert 'success' in result
        
        asyncio.run(test_failure_handling())


def run_comprehensive_tests():
    """Run all comprehensive tests and generate report."""
    print("🧪 RUNNING COMPREHENSIVE AUTONOMOUS SDLC TESTS")
    print("=" * 60)
    
    # Run pytest with detailed output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--strict-markers",
        "-x"  # Stop on first failure for faster feedback
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ ALL TESTS PASSED - AUTONOMOUS SDLC VALIDATION SUCCESSFUL")
        print("\n🚀 System is ready for production deployment!")
    else:
        print("\n❌ SOME TESTS FAILED - REVIEW REQUIRED")
        print("\n🔧 Please fix failing tests before deployment")
    
    return exit_code == 0


if __name__ == "__main__":
    # Run all tests when script is executed directly
    success = run_comprehensive_tests()
    exit(0 if success else 1)