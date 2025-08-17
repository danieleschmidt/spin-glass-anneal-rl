"""High-performance optimization accelerator with caching and auto-scaling."""

import time
import threading
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import json


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    computation_time: float = 0.0
    size_bytes: int = 0


class IntelligentCache:
    """Intelligent caching system with automatic eviction and optimization."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0,
        ttl_seconds: float = 3600.0
    ):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_stats = {}
        self.lock = threading.RLock()
        
        # Cache performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _compute_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Compute stable cache key."""
        # Create a deterministic representation
        key_data = {
            'func': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        
        # Handle non-serializable objects by converting to string
        try:
            serialized = json.dumps(key_data, sort_keys=True, default=str)
        except Exception:
            # Fallback to string representation
            serialized = str(key_data)
        
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimate based on type
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:10])  # Sample first 10
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(obj.items())[:10])  # Sample first 10
            else:
                return 1024  # Default estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        return time.time() - entry.timestamp > self.ttl_seconds
    
    def _evict_entries(self) -> None:
        """Evict entries based on LRU and memory pressure."""
        current_memory = sum(entry.size_bytes for entry in self.cache.values())
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self.cache.items() 
            if self._is_expired(entry)
        ]
        for key in expired_keys:
            del self.cache[key]
            self.evictions += 1
        
        # If still over limits, use intelligent eviction
        while (len(self.cache) > self.max_size or 
               current_memory > self.max_memory_bytes):
            if not self.cache:
                break
                
            # Score entries based on access frequency and recency
            scored_entries = []
            now = time.time()
            
            for key, entry in self.cache.items():
                age_score = (now - entry.timestamp) / 3600.0  # Age in hours
                access_score = 1.0 / max(entry.access_count, 1)  # Inverse access frequency
                size_score = entry.size_bytes / (1024 * 1024)  # Size in MB
                
                # Higher score = more likely to evict
                total_score = age_score + access_score + size_score * 0.1
                scored_entries.append((total_score, key))
            
            # Evict highest scoring (least valuable) entry
            scored_entries.sort(reverse=True)
            key_to_evict = scored_entries[0][1]
            entry = self.cache[key_to_evict]
            current_memory -= entry.size_bytes
            del self.cache[key_to_evict]
            self.evictions += 1
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if not self._is_expired(entry):
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    entry.access_count += 1
                    self.hits += 1
                    return entry
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.evictions += 1
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put entry in cache."""
        with self.lock:
            size = self._estimate_size(value)
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                access_count=1,
                computation_time=computation_time,
                size_bytes=size
            )
            
            self.cache[key] = entry
            self._evict_entries()
    
    def invalidate(self, pattern: str = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self.lock:
            if pattern is None:
                # Clear all
                count = len(self.cache)
                self.cache.clear()
                return count
            
            # Pattern matching (simple substring match)
            keys_to_remove = [
                key for key in self.cache.keys() 
                if pattern in key
            ]
            
            for key in keys_to_remove:
                del self.cache[key]
            
            return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(total_requests, 1)
            
            total_memory = sum(entry.size_bytes for entry in self.cache.values())
            avg_computation_time = 0.0
            if self.cache:
                avg_computation_time = sum(
                    entry.computation_time for entry in self.cache.values()
                ) / len(self.cache)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": total_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "avg_computation_time": avg_computation_time
            }


class PerformanceOptimizer:
    """Performance optimization with automatic parameter tuning."""
    
    def __init__(self):
        self.execution_history = []
        self.optimal_params = {}
        self.performance_cache = IntelligentCache(max_size=500, max_memory_mb=50.0)
    
    def memoize(
        self,
        ttl_seconds: float = 3600.0,
        cache_key_func: Optional[Callable] = None
    ):
        """Advanced memoization decorator with intelligent caching."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Compute cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = self.performance_cache._compute_cache_key(
                        func.__name__, args, kwargs
                    )
                
                # Try to get from cache
                entry = self.performance_cache.get(cache_key)
                if entry is not None:
                    return entry.value
                
                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time
                
                # Store in cache
                self.performance_cache.put(cache_key, result, computation_time)
                
                # Record performance data
                self._record_execution(func.__name__, computation_time, args, kwargs)
                
                return result
            
            return wrapper
        return decorator
    
    def parallel_execute(
        self,
        func: Callable,
        arg_list: List[tuple],
        max_workers: Optional[int] = None,
        use_processes: bool = False
    ) -> List[Any]:
        """Execute function in parallel with automatic load balancing."""
        if not arg_list:
            return []
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(arg_list), self._get_optimal_workers(func.__name__))
        
        # Choose execution strategy
        if use_processes and len(arg_list) > 2:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        
        start_time = time.time()
        
        try:
            with executor_class(max_workers=max_workers) as executor:
                if isinstance(arg_list[0], tuple):
                    futures = [executor.submit(func, *args) for args in arg_list]
                else:
                    futures = [executor.submit(func, args) for args in arg_list]
                
                results = [future.result() for future in futures]
            
            execution_time = time.time() - start_time
            
            # Record parallel execution performance
            self._record_parallel_execution(
                func.__name__, len(arg_list), max_workers, execution_time, use_processes
            )
            
            return results
            
        except Exception as e:
            print(f"Parallel execution failed: {e}")
            # Fallback to sequential execution
            if isinstance(arg_list[0], tuple):
                return [func(*args) for args in arg_list]
            else:
                return [func(args) for args in arg_list]
    
    def adaptive_batch_size(
        self,
        func: Callable,
        data: List[Any],
        initial_batch_size: int = 10,
        target_time_per_batch: float = 1.0
    ) -> List[Any]:
        """Execute with adaptive batch sizing for optimal performance."""
        if not data:
            return []
        
        results = []
        batch_size = initial_batch_size
        i = 0
        
        while i < len(data):
            batch = data[i:i + batch_size]
            start_time = time.time()
            
            # Process batch
            batch_results = [func(item) for item in batch]
            results.extend(batch_results)
            
            batch_time = time.time() - start_time
            
            # Adjust batch size based on performance
            if batch_time > 0:
                items_per_second = len(batch) / batch_time
                target_batch_size = int(items_per_second * target_time_per_batch)
                
                # Smooth the adjustment
                batch_size = int(0.7 * batch_size + 0.3 * max(target_batch_size, 1))
                batch_size = max(1, min(batch_size, len(data) - i - len(batch)))
            
            i += len(batch)
        
        return results
    
    def _record_execution(self, func_name: str, duration: float, args: tuple, kwargs: dict):
        """Record execution for performance analysis."""
        self.execution_history.append({
            "function": func_name,
            "duration": duration,
            "timestamp": time.time(),
            "args_count": len(args),
            "kwargs_count": len(kwargs)
        })
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]
    
    def _record_parallel_execution(
        self, func_name: str, task_count: int, workers: int, 
        total_time: float, used_processes: bool
    ):
        """Record parallel execution metrics."""
        throughput = task_count / total_time if total_time > 0 else 0
        
        key = f"{func_name}_parallel"
        if key not in self.optimal_params:
            self.optimal_params[key] = []
        
        self.optimal_params[key].append({
            "workers": workers,
            "task_count": task_count,
            "throughput": throughput,
            "total_time": total_time,
            "used_processes": used_processes,
            "timestamp": time.time()
        })
        
        # Keep only recent measurements
        if len(self.optimal_params[key]) > 50:
            self.optimal_params[key] = self.optimal_params[key][-25:]
    
    def _get_optimal_workers(self, func_name: str) -> int:
        """Get optimal number of workers based on historical performance."""
        key = f"{func_name}_parallel"
        if key not in self.optimal_params or not self.optimal_params[key]:
            # Default based on system
            try:
                import os
                return min(8, os.cpu_count() or 4)
            except Exception:
                return 4
        
        # Find configuration with best throughput
        best_config = max(
            self.optimal_params[key],
            key=lambda x: x["throughput"]
        )
        
        return best_config["workers"]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance optimization report."""
        cache_stats = self.performance_cache.get_stats()
        
        # Analyze execution history
        if self.execution_history:
            recent_executions = self.execution_history[-100:]
            avg_duration = sum(e["duration"] for e in recent_executions) / len(recent_executions)
            
            # Group by function
            func_stats = {}
            for execution in recent_executions:
                func = execution["function"]
                if func not in func_stats:
                    func_stats[func] = []
                func_stats[func].append(execution["duration"])
            
            # Compute function-level statistics
            for func, durations in func_stats.items():
                func_stats[func] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations)
                }
        else:
            avg_duration = 0.0
            func_stats = {}
        
        return {
            "cache_performance": cache_stats,
            "execution_analysis": {
                "total_executions": len(self.execution_history),
                "avg_duration": avg_duration,
                "function_stats": func_stats
            },
            "parallel_optimizations": len(self.optimal_params),
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        cache_stats = self.performance_cache.get_stats()
        
        # Cache recommendations
        if cache_stats["hit_rate"] < 0.5:
            recommendations.append("Consider increasing cache TTL - low hit rate detected")
        
        if cache_stats["memory_usage_mb"] > cache_stats["max_memory_mb"] * 0.9:
            recommendations.append("Consider increasing cache memory limit")
        
        # Function performance recommendations
        if self.execution_history:
            recent = self.execution_history[-50:]
            slow_functions = [
                e["function"] for e in recent 
                if e["duration"] > 1.0
            ]
            
            if slow_functions:
                recommendations.append(
                    f"Consider optimizing slow functions: {set(slow_functions)}"
                )
        
        return recommendations


class AutoScaler:
    """Automatic scaling based on load and performance metrics."""
    
    def __init__(self):
        self.load_history = []
        self.scaling_decisions = []
        self.current_scale = 1
        self.min_scale = 1
        self.max_scale = 10
    
    def record_load_metric(self, metric_name: str, value: float):
        """Record load metric for scaling decisions."""
        self.load_history.append({
            "metric": metric_name,
            "value": value,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        if len(self.load_history) > 1000:
            self.load_history = self.load_history[-500:]
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed."""
        if not self.load_history:
            return False
        
        # Analyze recent load
        recent_load = [
            entry for entry in self.load_history 
            if time.time() - entry["timestamp"] < 300  # Last 5 minutes
        ]
        
        if not recent_load:
            return False
        
        # Check for sustained high load
        high_load_count = sum(
            1 for entry in recent_load 
            if entry["value"] > 0.8  # 80% threshold
        )
        
        return (high_load_count / len(recent_load)) > 0.7  # 70% of time
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is possible."""
        if not self.load_history or self.current_scale <= self.min_scale:
            return False
        
        # Check for sustained low load
        recent_load = [
            entry for entry in self.load_history 
            if time.time() - entry["timestamp"] < 600  # Last 10 minutes
        ]
        
        if not recent_load:
            return False
        
        low_load_count = sum(
            1 for entry in recent_load 
            if entry["value"] < 0.3  # 30% threshold
        )
        
        return (low_load_count / len(recent_load)) > 0.8  # 80% of time
    
    def auto_scale(self) -> Optional[int]:
        """Perform automatic scaling decision."""
        old_scale = self.current_scale
        
        if self.should_scale_up() and self.current_scale < self.max_scale:
            self.current_scale = min(self.current_scale * 2, self.max_scale)
        elif self.should_scale_down():
            self.current_scale = max(self.current_scale // 2, self.min_scale)
        
        if self.current_scale != old_scale:
            decision = {
                "timestamp": time.time(),
                "old_scale": old_scale,
                "new_scale": self.current_scale,
                "reason": "scale_up" if self.current_scale > old_scale else "scale_down"
            }
            self.scaling_decisions.append(decision)
            return self.current_scale
        
        return None


# Global performance optimizer
global_optimizer = PerformanceOptimizer()


def fast_memoize(ttl_seconds: float = 3600.0):
    """Quick memoization decorator."""
    return global_optimizer.memoize(ttl_seconds=ttl_seconds)


def parallel_map(func: Callable, arg_list: List[Any], max_workers: Optional[int] = None):
    """Parallel map with automatic optimization."""
    return global_optimizer.parallel_execute(func, arg_list, max_workers)


# Test the performance optimization system
def test_performance_optimization():
    """Test performance optimization framework."""
    print("⚡ Testing Performance Optimization Framework...")
    
    # Test memoization
    @fast_memoize(ttl_seconds=60)
    def expensive_function(n):
        time.sleep(0.01)  # Simulate computation
        return n ** 2
    
    # First call (cache miss)
    start_time = time.time()
    result1 = expensive_function(10)
    time1 = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = expensive_function(10)
    time2 = time.time() - start_time
    
    print(f"✅ Memoization: {result1} == {result2}, speedup: {time1/max(time2, 0.001):.1f}x")
    
    # Test parallel execution
    def square(x):
        time.sleep(0.001)
        return x ** 2
    
    numbers = list(range(20))
    
    # Sequential
    start_time = time.time()
    seq_results = [square(x) for x in numbers]
    seq_time = time.time() - start_time
    
    # Parallel
    start_time = time.time()
    par_results = parallel_map(square, numbers, max_workers=4)
    par_time = time.time() - start_time
    
    print(f"✅ Parallel execution: speedup {seq_time/max(par_time, 0.001):.1f}x")
    print(f"   Results match: {seq_results == par_results}")
    
    # Test adaptive batching
    def process_item(x):
        time.sleep(0.001)
        return x * 2
    
    data = list(range(50))
    batch_results = global_optimizer.adaptive_batch_size(
        process_item, data, initial_batch_size=5, target_time_per_batch=0.1
    )
    
    print(f"✅ Adaptive batching: processed {len(batch_results)} items")
    
    # Test auto-scaling
    scaler = AutoScaler()
    
    # Simulate load
    for i in range(10):
        scaler.record_load_metric("cpu_usage", 0.9)  # High load
    
    scale_decision = scaler.auto_scale()
    print(f"✅ Auto-scaling: {scale_decision}")
    
    # Get performance report
    report = global_optimizer.get_performance_report()
    cache_stats = report["cache_performance"]
    print(f"✅ Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"✅ Cache size: {cache_stats['size']} entries")
    
    print("⚡ Performance Optimization Framework test completed!")


if __name__ == "__main__":
    test_performance_optimization()