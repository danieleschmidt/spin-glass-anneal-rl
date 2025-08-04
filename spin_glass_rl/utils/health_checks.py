"""Health checks and system diagnostics."""

import time
import torch
import numpy as np
import psutil
import platform
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from pathlib import Path

from spin_glass_rl.utils.exceptions import DeviceError, ResourceError
from spin_glass_rl.core.ising_model import IsingModel, IsingModelConfig
from spin_glass_rl.annealing.gpu_annealer import GPUAnnealer, GPUAnnealerConfig


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    duration: float


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        self.logger = logging.getLogger(f"spin_glass_rl.health.{name}")
    
    def run(self) -> HealthCheckResult:
        """Run the health check."""
        start_time = time.time()
        
        try:
            result = self._run_check()
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name=self.name,
                status=result.get('status', HealthStatus.UNKNOWN),
                message=result.get('message', ''),
                details=result.get('details', {}),
                timestamp=start_time,
                duration=duration
            )
        
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Health check failed: {e}")
            
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}",
                details={"error": str(e), "error_type": type(e).__name__},
                timestamp=start_time,
                duration=duration
            )
    
    def _run_check(self) -> Dict[str, Any]:
        """Override this method to implement the actual check."""
        raise NotImplementedError


class SystemResourceCheck(BaseHealthCheck):
    """Check system resource usage."""
    
    def __init__(self, memory_threshold: float = 0.9, cpu_threshold: float = 0.9, disk_threshold: float = 0.9):
        super().__init__("system_resources")
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
    
    def _run_check(self) -> Dict[str, Any]:
        """Check system resources."""
        # Memory check
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100.0
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1) / 100.0
        
        # Disk check
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent / 100.0
        
        # Determine status
        status = HealthStatus.HEALTHY
        issues = []
        
        if memory_percent > self.memory_threshold:
            status = HealthStatus.CRITICAL
            issues.append(f"Memory usage high: {memory_percent:.1%}")
        elif memory_percent > self.memory_threshold * 0.8:
            status = HealthStatus.WARNING
            issues.append(f"Memory usage elevated: {memory_percent:.1%}")
        
        if cpu_percent > self.cpu_threshold:
            status = HealthStatus.CRITICAL
            issues.append(f"CPU usage high: {cpu_percent:.1%}")
        elif cpu_percent > self.cpu_threshold * 0.8:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            issues.append(f"CPU usage elevated: {cpu_percent:.1%}")
        
        if disk_percent > self.disk_threshold:
            status = HealthStatus.CRITICAL
            issues.append(f"Disk usage high: {disk_percent:.1%}")
        elif disk_percent > self.disk_threshold * 0.8:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            issues.append(f"Disk usage elevated: {disk_percent:.1%}")
        
        message = "System resources OK" if not issues else "; ".join(issues)
        
        return {
            "status": status,
            "message": message,
            "details": {
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "disk_percent": disk_percent,
                "memory_available": memory.available,
                "memory_total": memory.total,
                "cpu_count": psutil.cpu_count(),
                "disk_free": disk.free,
                "disk_total": disk.total
            }
        }


class CUDAHealthCheck(BaseHealthCheck):
    """Check CUDA availability and status."""
    
    def __init__(self):
        super().__init__("cuda")
    
    def _run_check(self) -> Dict[str, Any]:
        """Check CUDA status."""
        if not torch.cuda.is_available():
            return {
                "status": HealthStatus.WARNING,
                "message": "CUDA not available",
                "details": {"cuda_available": False}
            }
        
        try:
            device_count = torch.cuda.device_count()
            details = {
                "cuda_available": True,
                "device_count": device_count,
                "devices": []
            }
            
            for i in range(device_count):
                device = torch.device(f"cuda:{i}")
                props = torch.cuda.get_device_properties(device)
                
                # Get memory info
                torch.cuda.set_device(device)
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                memory_total = props.total_memory
                
                device_info = {
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_total": memory_total,
                    "memory_allocated": memory_allocated,
                    "memory_reserved": memory_reserved,
                    "memory_free": memory_total - memory_reserved,
                    "multiprocessor_count": props.multiprocessor_count
                }
                
                details["devices"].append(device_info)
            
            # Test basic CUDA operation
            test_tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(test_tensor, test_tensor.t())
            torch.cuda.synchronize()
            
            # Check for any issues
            issues = []
            for device_info in details["devices"]:
                memory_usage = device_info["memory_allocated"] / device_info["memory_total"]
                if memory_usage > 0.9:
                    issues.append(f"GPU {device_info['index']} memory high: {memory_usage:.1%}")
            
            status = HealthStatus.CRITICAL if issues else HealthStatus.HEALTHY
            message = "CUDA OK" if not issues else "; ".join(issues)
            
            return {
                "status": status,
                "message": message,
                "details": details
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"CUDA test failed: {e}",
                "details": {"error": str(e), "cuda_available": True}
            }


class SpinGlassModelCheck(BaseHealthCheck):
    """Check that core spin glass functionality works."""
    
    def __init__(self):
        super().__init__("spin_glass_model")
    
    def _run_check(self) -> Dict[str, Any]:
        """Test basic spin glass model functionality."""
        try:
            # Create test model
            config = IsingModelConfig(
                n_spins=10,
                coupling_strength=1.0,
                external_field_strength=0.5,
                use_sparse=False,
                device="cpu"
            )
            
            model = IsingModel(config)
            
            # Add some couplings
            for i in range(9):
                model.set_coupling(i, i + 1, -1.0)
            
            # Set external fields
            fields = torch.randn(10) * 0.5
            model.set_external_fields(fields)
            
            # Compute energy
            initial_energy = model.compute_energy()
            
            # Flip a spin and check energy change
            old_spin = model.spins[0].item()
            delta_energy = model.flip_spin(0)
            new_energy = model.compute_energy()
            
            # Verify energy calculation
            expected_new_energy = initial_energy + delta_energy
            energy_diff = abs(new_energy - expected_new_energy)
            
            if energy_diff > 1e-6:
                return {
                    "status": HealthStatus.CRITICAL,
                    "message": f"Energy calculation inconsistent: diff={energy_diff}",
                    "details": {
                        "initial_energy": initial_energy,
                        "delta_energy": delta_energy,
                        "new_energy": new_energy,
                        "expected_energy": expected_new_energy,
                        "difference": energy_diff
                    }
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Spin glass model working correctly",
                "details": {
                    "model_size": model.n_spins,
                    "initial_energy": initial_energy,
                    "energy_change": delta_energy,
                    "final_energy": new_energy
                }
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Spin glass model test failed: {e}",
                "details": {"error": str(e)}
            }


class AnnealerHealthCheck(BaseHealthCheck):
    """Check that annealing functionality works."""
    
    def __init__(self):
        super().__init__("annealer")
    
    def _run_check(self) -> Dict[str, Any]:
        """Test basic annealing functionality."""
        try:
            # Create test model
            config = IsingModelConfig(n_spins=20, use_sparse=False, device="cpu")
            model = IsingModel(config)
            
            # Create frustrated system
            np.random.seed(42)
            for i in range(model.n_spins):
                for j in range(i + 1, model.n_spins):
                    if np.random.random() < 0.1:  # 10% connectivity
                        coupling = np.random.choice([-1.0, 1.0])
                        model.set_coupling(i, j, coupling)
            
            # Create annealer
            annealer_config = GPUAnnealerConfig(
                n_sweeps=100,
                initial_temp=2.0,
                final_temp=0.1,
                random_seed=42
            )
            
            annealer = GPUAnnealer(annealer_config)
            annealer.device = torch.device("cpu")  # Force CPU for reliability
            
            # Run annealing
            initial_energy = model.compute_energy()
            result = annealer.anneal(model)
            
            # Check result
            if result.best_energy > initial_energy + 1e-6:
                return {
                    "status": HealthStatus.WARNING,
                    "message": f"Annealing did not improve energy: {initial_energy} -> {result.best_energy}",
                    "details": {
                        "initial_energy": initial_energy,
                        "final_energy": result.best_energy,
                        "n_sweeps": result.n_sweeps,
                        "final_acceptance_rate": result.final_acceptance_rate
                    }
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Annealer working correctly",
                "details": {
                    "initial_energy": initial_energy,
                    "final_energy": result.best_energy,
                    "energy_improvement": initial_energy - result.best_energy,
                    "n_sweeps": result.n_sweeps,
                    "total_time": result.total_time,
                    "final_acceptance_rate": result.final_acceptance_rate
                }
            }
        
        except Exception as e:
            return {
                "status": HealthStatus.CRITICAL,
                "message": f"Annealer test failed: {e}",
                "details": {"error": str(e)}
            }


class DependencyCheck(BaseHealthCheck):
    """Check that required dependencies are available."""
    
    def __init__(self):
        super().__init__("dependencies")
    
    def _run_check(self) -> Dict[str, Any]:
        """Check required dependencies."""
        dependencies = {
            "torch": self._check_torch,
            "numpy": self._check_numpy,
            "scipy": self._check_scipy,
            "matplotlib": self._check_matplotlib,
            "networkx": self._check_networkx,
            "psutil": self._check_psutil
        }
        
        results = {}
        issues = []
        
        for name, check_func in dependencies.items():
            try:
                version = check_func()
                results[name] = {"available": True, "version": version}
            except ImportError:
                results[name] = {"available": False, "version": None}
                issues.append(f"{name} not available")
            except Exception as e:
                results[name] = {"available": False, "version": None, "error": str(e)}
                issues.append(f"{name} error: {e}")
        
        status = HealthStatus.CRITICAL if issues else HealthStatus.HEALTHY
        message = "All dependencies OK" if not issues else "; ".join(issues)
        
        return {
            "status": status,
            "message": message,
            "details": {"dependencies": results}
        }
    
    def _check_torch(self) -> str:
        import torch
        return torch.__version__
    
    def _check_numpy(self) -> str:
        import numpy
        return numpy.__version__
    
    def _check_scipy(self) -> str:
        import scipy
        return scipy.__version__
    
    def _check_matplotlib(self) -> str:
        import matplotlib
        return matplotlib.__version__
    
    def _check_networkx(self) -> str:
        import networkx
        return networkx.__version__
    
    def _check_psutil(self) -> str:
        import psutil
        return psutil.__version__


class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(self):
        self.checks: List[BaseHealthCheck] = []
        self.logger = logging.getLogger("spin_glass_rl.health")
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.add_check(SystemResourceCheck())
        self.add_check(CUDAHealthCheck())
        self.add_check(SpinGlassModelCheck())
        self.add_check(AnnealerHealthCheck())
        self.add_check(DependencyCheck())
    
    def add_check(self, check: BaseHealthCheck):
        """Add a health check."""
        self.checks.append(check)
    
    def run_all_checks(self, parallel: bool = True) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        if parallel:
            return self._run_checks_parallel()
        else:
            return self._run_checks_sequential()
    
    def _run_checks_sequential(self) -> Dict[str, HealthCheckResult]:
        """Run checks sequentially."""
        results = {}
        
        for check in self.checks:
            self.logger.info(f"Running health check: {check.name}")
            result = check.run()
            results[check.name] = result
            
            self.logger.info(
                f"Health check {check.name}: {result.status.value} - {result.message}",
                extra={
                    "health_check": check.name,
                    "status": result.status.value,
                    "duration": result.duration
                }
            )
        
        return results
    
    def _run_checks_parallel(self) -> Dict[str, HealthCheckResult]:
        """Run checks in parallel."""
        import concurrent.futures
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.checks)) as executor:
            # Submit all checks
            future_to_check = {executor.submit(check.run): check for check in self.checks}
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_check):
                check = future_to_check[future]
                try:
                    result = future.result()
                    results[check.name] = result
                    
                    self.logger.info(
                        f"Health check {check.name}: {result.status.value} - {result.message}",
                        extra={
                            "health_check": check.name,
                            "status": result.status.value,
                            "duration": result.duration
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Health check {check.name} failed with exception: {e}")
                    results[check.name] = HealthCheckResult(
                        name=check.name,
                        status=HealthStatus.CRITICAL,
                        message=f"Check failed with exception: {e}",
                        details={"error": str(e)},
                        timestamp=time.time(),
                        duration=0.0
                    )
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Get overall system health status."""
        if any(result.status == HealthStatus.CRITICAL for result in results.values()):
            return HealthStatus.CRITICAL
        elif any(result.status == HealthStatus.WARNING for result in results.values()):
            return HealthStatus.WARNING
        elif any(result.status == HealthStatus.UNKNOWN for result in results.values()):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def generate_report(self, results: Dict[str, HealthCheckResult]) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        overall_status = self.get_overall_status(results)
        
        report = {
            "timestamp": time.time(),
            "overall_status": overall_status.value,
            "checks": {},
            "summary": {
                "total_checks": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "warning": sum(1 for r in results.values() if r.status == HealthStatus.WARNING),
                "critical": sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL),
                "unknown": sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN)
            },
            "system_info": self._get_system_info()
        }
        
        for name, result in results.items():
            report["checks"][name] = {
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "timestamp": result.timestamp,
                "details": result.details
            }
        
        return report
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "hostname": platform.node()
        }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
        
        return info


class ContinuousHealthMonitor:
    """Continuous health monitoring with alerts."""
    
    def __init__(self, monitor: HealthMonitor, check_interval: float = 300.0):
        self.monitor = monitor
        self.check_interval = check_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("spin_glass_rl.health.continuous")
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def start(self):
        """Start continuous monitoring."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        self.logger.info(f"Started continuous health monitoring (interval: {self.check_interval}s)")
    
    def stop(self):
        """Stop continuous monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10.0)
        
        self.logger.info("Stopped continuous health monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                results = self.monitor.run_all_checks(parallel=True)
                report = self.monitor.generate_report(results)
                
                # Check for alerts
                if report["overall_status"] in ["critical", "warning"]:
                    for callback in self.alert_callbacks:
                        try:
                            callback(report)
                        except Exception as e:
                            self.logger.error(f"Alert callback failed: {e}")
                
                # Log summary
                self.logger.info(
                    f"Health check summary: {report['summary']['healthy']} healthy, "
                    f"{report['summary']['warning']} warning, "
                    f"{report['summary']['critical']} critical",
                    extra={"health_summary": report["summary"]}
                )
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
            
            # Wait for next check
            time.sleep(self.check_interval)


# Global health monitor instance
health_monitor = HealthMonitor()