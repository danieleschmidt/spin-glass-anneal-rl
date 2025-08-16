"""Advanced performance optimization and caching utilities."""

import time
import functools
import threading
import weakref
import hashlib
import json
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import logging
from pathlib import Path
import psutil
import gc

from spin_glass_rl.utils.exceptions import ResourceError, ValidationError


class LRUCache:
    """Thread-safe LRU cache with size and TTL limits."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live in seconds (0 for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self.ttl > 0:
                age = time.time() - self.access_times[key]
                if age > self.ttl:
                    del self.cache[key]
                    del self.access_times[key]
                    self.misses += 1
                    return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.hits += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
            
            # Add new item
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            # Evict if over size limit
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "ttl": self.ttl
            }


class TensorCache:
    """Specialized cache for PyTorch tensors with memory management."""
    
    def __init__(self, max_memory_mb: int = 1000, device: Optional[torch.device] = None):
        """Initialize tensor cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            device: Device for cached tensors
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.device = device or torch.device('cpu')
        self.cache = OrderedDict()
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def _tensor_memory_size(self, tensor: torch.Tensor) -> int:
        """Calculate tensor memory size in bytes."""
        return tensor.element_size() * tensor.numel()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get tensor from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            tensor = self.cache.pop(key)
            self.cache[key] = tensor
            self.hits += 1
            return tensor.clone()  # Return copy to avoid modification
    
    def put(self, key: str, tensor: torch.Tensor) -> None:
        """Put tensor in cache."""
        with self.lock:
            # Move tensor to cache device
            if tensor.device != self.device:
                tensor = tensor.to(self.device)
            
            tensor_size = self._tensor_memory_size(tensor)
            
            # Remove if already exists
            if key in self.cache:
                old_tensor = self.cache[key]
                self.memory_usage -= self._tensor_memory_size(old_tensor)
                del self.cache[key]
            
            # Evict until we have space
            while self.memory_usage + tensor_size > self.max_memory_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                oldest_tensor = self.cache.pop(oldest_key)
                self.memory_usage -= self._tensor_memory_size(oldest_tensor)
            
            # Add new tensor if it fits
            if tensor_size <= self.max_memory_bytes:
                self.cache[key] = tensor.clone()  # Store copy
                self.memory_usage += tensor_size
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.memory_usage = 0
            self.hits = 0
            self.misses = 0
            
            # Force garbage collection
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "memory_usage_mb": self.memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "device": str(self.device)
            }


class PersistentCache:
    """Persistent cache that survives process restarts."""
    
    def __init__(self, cache_dir: Union[str, Path], max_size_mb: int = 500):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "cache_index.json"
        self.lock = threading.RLock()
        
        # Load existing index
        self.index = self._load_index()
        self.logger = logging.getLogger(__name__)
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save cache index: {e}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.index:
                return None
            
            file_path = self._get_file_path(key)
            if not file_path.exists():
                # Remove stale index entry
                del self.index[key]
                self._save_index()
                return None
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    value = self._deserialize_value(data)
                
                # Update access time
                self.index[key]['last_access'] = time.time()
                self._save_index()
                
                return value
            
            except Exception as e:
                self.logger.error(f"Failed to load cache entry {key}: {e}")
                # Remove corrupted entry
                if file_path.exists():
                    file_path.unlink()
                if key in self.index:
                    del self.index[key]
                    self._save_index()
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            file_path = self._get_file_path(key)
            
            try:
                # Serialize value safely
                serialized_data = self._serialize_value(value)
                with open(file_path, 'w') as f:
                    json.dump(serialized_data, f)
                
                # Update index
                file_size = file_path.stat().st_size
                self.index[key] = {
                    'size': file_size,
                    'created': time.time(),
                    'last_access': time.time()
                }
                
                # Clean up if over size limit
                self._cleanup_if_needed()
                self._save_index()
                
            except Exception as e:
                self.logger.error(f"Failed to save cache entry {key}: {e}")
                if file_path.exists():
                    file_path.unlink()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if over size limit."""
        total_size = sum(entry['size'] for entry in self.index.values())
        
        if total_size <= self.max_size_bytes:
            return
        
        # Sort by last access time (oldest first)
        sorted_keys = sorted(
            self.index.keys(),
            key=lambda k: self.index[k]['last_access']
        )
        
        # Remove oldest entries until under limit
        for key in sorted_keys:
            if total_size <= self.max_size_bytes:
                break
            
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
            
            total_size -= self.index[key]['size']
            del self.index[key]
    
    def _serialize_value(self, value: Any) -> Dict:
        """Safely serialize values for JSON storage."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return {"type": "primitive", "data": value}
        elif isinstance(value, list):
            return {"type": "list", "data": [self._serialize_value(item)["data"] for item in value]}
        elif isinstance(value, dict):
            return {"type": "dict", "data": {k: self._serialize_value(v)["data"] for k, v in value.items()}}
        elif isinstance(value, np.ndarray):
            return {"type": "numpy", "data": value.tolist(), "shape": value.shape, "dtype": str(value.dtype)}
        elif hasattr(value, '__dict__'):
            return {"type": "object", "data": value.__dict__, "class": value.__class__.__name__}
        else:
            return {"type": "string", "data": str(value)}
    
    def _deserialize_value(self, data: Dict) -> Any:
        """Safely deserialize values from JSON storage."""
        value_type = data.get("type", "primitive")
        value_data = data.get("data")
        
        if value_type == "primitive":
            return value_data
        elif value_type == "list":
            return value_data
        elif value_type == "dict":
            return value_data
        elif value_type == "numpy":
            shape = data.get("shape", [])
            dtype = data.get("dtype", "float64")
            array = np.array(value_data, dtype=dtype)
            return array.reshape(shape) if shape else array
        elif value_type == "object":
            return value_data  # Return as dict for safety
        else:
            return value_data

    def clear(self) -> None:
        """Clear entire cache."""
        with self.lock:
            # Remove all cache files
            for key in list(self.index.keys()):
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
            
            # Clear index
            self.index.clear()
            self._save_index()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry['size'] for entry in self.index.values())
            
            return {
                "entries": len(self.index),
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "cache_dir": str(self.cache_dir)
            }


class PerformanceProfiler:
    """Performance profiling and optimization recommendations."""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function execution."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    profile_data = {
                        'name': name,
                        'duration': end_time - start_time,
                        'memory_delta': end_memory - start_memory,
                        'start_memory': start_memory,
                        'end_memory': end_memory,
                        'success': success,
                        'error': error,
                        'timestamp': start_time,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                    
                    with self.lock:
                        self.profiles[name].append(profile_data)
                        
                        # Keep only recent profiles
                        if len(self.profiles[name]) > 1000:
                            self.profiles[name] = self.profiles[name][-500:]
                
                return result
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def get_stats(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            if func_name:
                if func_name not in self.profiles:
                    return {}
                data = self.profiles[func_name]
            else:
                # Aggregate all functions
                data = []
                for profiles in self.profiles.values():
                    data.extend(profiles)
            
            if not data:
                return {}
            
            # Calculate statistics
            durations = [p['duration'] for p in data if p['success']]
            memory_deltas = [p['memory_delta'] for p in data if p['success']]
            error_count = sum(1 for p in data if not p['success'])
            
            stats = {
                'call_count': len(data),
                'success_count': len(durations),
                'error_count': error_count,
                'success_rate': len(durations) / len(data) if data else 0.0
            }
            
            if durations:
                stats.update({
                    'avg_duration': np.mean(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'p50_duration': np.percentile(durations, 50),
                    'p95_duration': np.percentile(durations, 95),
                    'p99_duration': np.percentile(durations, 99)
                })
            
            if memory_deltas:
                stats.update({
                    'avg_memory_delta': np.mean(memory_deltas),
                    'max_memory_delta': np.max(memory_deltas),
                    'total_memory_leaked': sum(d for d in memory_deltas if d > 0)
                })
            
            return stats
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        with self.lock:
            for func_name, profiles in self.profiles.items():
                if not profiles:
                    continue
                
                stats = self.get_stats(func_name)
                
                # Slow function recommendation
                if stats.get('avg_duration', 0) > 1.0:
                    recommendations.append({
                        'type': 'slow_function',
                        'function': func_name,
                        'avg_duration': stats['avg_duration'],
                        'recommendation': f"Function {func_name} is slow (avg: {stats['avg_duration']:.3f}s). Consider optimization or caching."
                    })
                
                # Memory leak recommendation
                if stats.get('total_memory_leaked', 0) > 100 * 1024 * 1024:  # 100MB
                    recommendations.append({
                        'type': 'memory_leak',
                        'function': func_name,
                        'memory_leaked': stats['total_memory_leaked'],
                        'recommendation': f"Function {func_name} may have memory leaks (total: {stats['total_memory_leaked'] / (1024*1024):.1f}MB)."
                    })
                
                # High error rate recommendation
                if stats.get('success_rate', 1.0) < 0.9:
                    recommendations.append({
                        'type': 'high_error_rate',
                        'function': func_name,
                        'success_rate': stats['success_rate'],
                        'recommendation': f"Function {func_name} has high error rate ({stats['success_rate']:.1%}). Check error handling."
                    })
        
        return recommendations
    
    def clear(self) -> None:
        """Clear all profiling data."""
        with self.lock:
            self.profiles.clear()


class AdaptiveOptimizer:
    """Adaptive optimization based on runtime performance."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.optimizations = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def optimize_batch_size(self, func_name: str, current_batch_size: int, 
                           target_duration: float = 1.0) -> int:
        """Optimize batch size based on performance data."""
        stats = self.profiler.get_stats(func_name)
        
        if not stats or 'avg_duration' not in stats:
            return current_batch_size
        
        avg_duration = stats['avg_duration']
        
        # Adjust batch size to target duration
        if avg_duration > target_duration * 1.2:
            # Too slow, reduce batch size
            new_batch_size = max(1, int(current_batch_size * 0.8))
            self.logger.info(f"Reducing batch size for {func_name}: {current_batch_size} -> {new_batch_size}")
        elif avg_duration < target_duration * 0.5:
            # Too fast, increase batch size
            new_batch_size = int(current_batch_size * 1.2)
            self.logger.info(f"Increasing batch size for {func_name}: {current_batch_size} -> {new_batch_size}")
        else:
            new_batch_size = current_batch_size
        
        with self.lock:
            self.optimizations[func_name] = {
                'type': 'batch_size',
                'old_value': current_batch_size,
                'new_value': new_batch_size,
                'reason': f"Duration {avg_duration:.3f}s vs target {target_duration}s"
            }
        
        return new_batch_size
    
    def should_use_cache(self, func_name: str, call_frequency_threshold: float = 10.0) -> bool:
        """Determine if function should use caching."""
        stats = self.profiler.get_stats(func_name)
        
        if not stats:
            return False
        
        # Calculate call frequency (calls per minute)
        recent_profiles = []
        current_time = time.time()
        
        with self.profiler.lock:
            if func_name in self.profiler.profiles:
                recent_profiles = [
                    p for p in self.profiler.profiles[func_name]
                    if current_time - p['timestamp'] < 60  # Last minute
                ]
        
        call_frequency = len(recent_profiles)
        avg_duration = stats.get('avg_duration', 0)
        
        # Recommend caching if frequently called and not too fast
        should_cache = call_frequency >= call_frequency_threshold and avg_duration > 0.1
        
        with self.lock:
            self.optimizations[func_name] = {
                'type': 'caching',
                'should_cache': should_cache,
                'call_frequency': call_frequency,
                'avg_duration': avg_duration,
                'reason': f"Frequency: {call_frequency}/min, Duration: {avg_duration:.3f}s"
            }
        
        return should_cache
    
    def get_optimizations(self) -> Dict[str, Any]:
        """Get current optimizations."""
        with self.lock:
            return self.optimizations.copy()


# Global instances
global_profiler = PerformanceProfiler()
global_optimizer = AdaptiveOptimizer(global_profiler)

# Caches
memory_cache = LRUCache(max_size=10000, ttl=3600)
tensor_cache = TensorCache(max_memory_mb=500)


def cached(cache_type: str = "memory", key_func: Optional[Callable] = None, ttl: float = 3600):
    """Caching decorator with multiple cache backends."""
    def decorator(func):
        cache = {
            "memory": memory_cache,
            "tensor": tensor_cache
        }.get(cache_type, memory_cache)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def profile(func_name: str = None):
    """Convenience profiling decorator."""
    return global_profiler.profile_function(func_name)


def optimize_memory():
    """Optimize memory usage."""
    # Clear caches
    memory_cache.clear()
    tensor_cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logging.getLogger(__name__).info("Memory optimization completed")