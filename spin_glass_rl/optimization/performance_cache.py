"""Advanced caching and performance optimization system."""

import time
import hashlib
import threading
from typing import Any, Dict, Optional, Tuple, Callable, List
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import json
import torch
import numpy as np
from pathlib import Path

from spin_glass_rl.utils.robust_logging import get_logger, LoggingContext
from spin_glass_rl.utils.monitoring import PerformanceMonitor

logger = get_logger("performance_cache")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    computation_time: float = 0.0
    last_access: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    average_access_time_ms: float = 0.0
    memory_efficiency: float = 0.0


class LRUCache:
    """Thread-safe LRU cache with size limits and performance monitoring."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: int = 1000,
                 ttl_seconds: Optional[float] = None):
        
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        
        # Thread-safe storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = CacheStats()
        self._access_times = []
        
        logger.info(f"LRU cache initialized: max_size={max_size}, max_memory_mb={max_memory_mb}")
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.numel()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            else:
                return len(json.dumps(obj.__dict__ if hasattr(obj, '__dict__') else str(obj)).encode())
        except Exception:
            return 1024  # Default estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - entry.timestamp) > self.ttl_seconds
    
    def _evict_expired(self) -> None:
        """Remove expired entries."""
        if self.ttl_seconds is None:
            return
        
        expired_keys = []
        current_time = time.time()
        
        for key, entry in self._cache.items():
            if (current_time - entry.timestamp) > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self._stats.evictions += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries to maintain size limits."""
        # Remove expired entries first
        self._evict_expired()
        
        # Evict by count
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats.evictions += 1
        
        # Evict by memory usage
        while self._stats.total_size_bytes > self.max_memory_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            entry = self._cache[oldest_key]
            self._stats.total_size_bytes -= entry.size_bytes
            del self._cache[oldest_key]
            self._stats.evictions += 1
    
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache. Returns (value, hit)."""
        with self._lock:
            start_time = time.time()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._cache[key]
                    self._stats.misses += 1
                    return None, False
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                
                # Update access statistics
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Record access time
                access_time_ms = (time.time() - start_time) * 1000
                self._access_times.append(access_time_ms)
                if len(self._access_times) > 1000:
                    self._access_times = self._access_times[-1000:]  # Keep recent
                
                self._stats.hits += 1
                return entry.value, True
            else:
                self._stats.misses += 1
                return None, False
    
    def put(self, key: str, value: Any, computation_time: float = 0.0) -> None:
        """Put value in cache."""
        with self._lock:
            size_bytes = self._calculate_size(value)
            
            # Check if single item is too large
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Cache entry too large: {size_bytes} bytes")
                return
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size_bytes -= old_entry.size_bytes
                del self._cache[key]
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                computation_time=computation_time
            )
            
            # Add to cache
            self._cache[key] = entry
            self._stats.total_size_bytes += size_bytes
            
            # Evict if necessary
            self._evict_lru()
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.total_size_bytes -= entry.size_bytes
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.total_size_bytes = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_size_bytes=self._stats.total_size_bytes,
                memory_efficiency=self._stats.total_size_bytes / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
            )
            
            if self._access_times:
                stats.average_access_time_ms = np.mean(self._access_times)
            
            return stats
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total_requests = self._stats.hits + self._stats.misses
        return self._stats.hits / total_requests if total_requests > 0 else 0.0
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._stats.total_size_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hit_rate': self.get_hit_rate(),
                'stats': self.get_stats().__dict__,
                'ttl_seconds': self.ttl_seconds
            }


class ComputationCache:
    """High-level cache for expensive computations with automatic key generation."""
    
    def __init__(self, 
                 name: str = "computation_cache",
                 max_size: int = 1000,
                 max_memory_mb: int = 1000,
                 ttl_seconds: Optional[float] = None,
                 enable_disk_cache: bool = True,
                 disk_cache_dir: Optional[str] = None):
        
        self.name = name
        self.enable_disk_cache = enable_disk_cache
        
        # Memory cache
        self._memory_cache = LRUCache(max_size, max_memory_mb, ttl_seconds)
        
        # Disk cache setup
        if enable_disk_cache:
            cache_dir = Path(disk_cache_dir) if disk_cache_dir else Path.home() / ".spin_glass_rl_cache"
            self.disk_cache_dir = cache_dir / name
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Disk cache enabled: {self.disk_cache_dir}")
        else:
            self.disk_cache_dir = None
        
        # Performance monitoring
        self._computation_times = defaultdict(list)
        self._cache_effectiveness = {}
        
        logger.info(f"Computation cache '{name}' initialized")
    
    def _generate_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key from function and arguments."""
        # Create a deterministic key from function name and arguments
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Handle special argument types
        key_parts = [func_name]
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Use tensor shape, dtype, and hash of values for small tensors
                if arg.numel() < 1000:
                    key_parts.append(f"tensor_{arg.shape}_{arg.dtype}_{hash(arg.data.tobytes())}")
                else:
                    key_parts.append(f"tensor_{arg.shape}_{arg.dtype}_{arg.sum().item()}")
            elif isinstance(arg, np.ndarray):
                if arg.size < 1000:
                    key_parts.append(f"array_{arg.shape}_{arg.dtype}_{hash(arg.tobytes())}")
                else:
                    key_parts.append(f"array_{arg.shape}_{arg.dtype}_{arg.sum()}")
            else:
                key_parts.append(str(hash(arg)))
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={hash(v)}")
        
        # Create final key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load cached result from disk."""
        if not self.enable_disk_cache:
            return None
        
        cache_file = self.disk_cache_dir / f"{key}.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    data = json.load(f)
                
                # Check TTL if specified
                if self._memory_cache.ttl_seconds is not None:
                    age = time.time() - cache_file.stat().st_mtime
                    if age > self._memory_cache.ttl_seconds:
                        cache_file.unlink()  # Remove expired file
                        return None
                
                logger.debug(f"Loaded from disk cache: {key}")
                return data
                
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            
        return None
    
    def _save_to_disk(self, key: str, value: Any) -> None:
        """Save result to disk cache."""
        if not self.enable_disk_cache:
            return
        
        cache_file = self.disk_cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                json.dump(self._make_serializable(value), f)
            logger.debug(f"Saved to disk cache: {key}")
            
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def cached_call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with caching."""
        # Generate cache key
        cache_key = self._generate_key(func, args, kwargs)
        
        # Check memory cache first
        start_time = time.time()
        result, hit = self._memory_cache.get(cache_key)
        
        if hit:
            access_time = (time.time() - start_time) * 1000
            logger.debug(f"Memory cache hit: {func.__name__} ({access_time:.2f}ms)")
            return result
        
        # Check disk cache
        if self.enable_disk_cache:
            result = self._load_from_disk(cache_key)
            if result is not None:
                # Add to memory cache
                self._memory_cache.put(cache_key, result)
                access_time = (time.time() - start_time) * 1000
                logger.debug(f"Disk cache hit: {func.__name__} ({access_time:.2f}ms)")
                return result
        
        # Cache miss - compute result
        logger.debug(f"Cache miss: {func.__name__}")
        computation_start = time.time()
        
        try:
            result = func(*args, **kwargs)
            computation_time = time.time() - computation_start
            
            # Store in both caches
            self._memory_cache.put(cache_key, result, computation_time)
            if self.enable_disk_cache:
                self._save_to_disk(cache_key, result)
            
            # Track performance
            self._computation_times[func.__name__].append(computation_time)
            
            logger.debug(f"Computed and cached: {func.__name__} ({computation_time*1000:.2f}ms)")
            return result
            
        except Exception as e:
            logger.error(f"Computation failed: {func.__name__}", exception=e)
            raise
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {"type": "numpy", "data": obj.tolist(), "shape": obj.shape, "dtype": str(obj.dtype)}
        elif isinstance(obj, torch.Tensor):
            return {"type": "tensor", "data": obj.cpu().numpy().tolist(), "shape": list(obj.shape)}
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            return {"type": "object", "data": obj.__dict__, "class": obj.__class__.__name__}
        else:
            return {"type": "string", "data": str(obj)}

    def invalidate_function(self, func: Callable) -> int:
        """Invalidate all cache entries for a specific function."""
        func_prefix = f"{func.__module__}.{func.__name__}"
        
        # Memory cache
        invalidated_count = 0
        keys_to_remove = []
        
        with self._memory_cache._lock:
            for key in self._memory_cache._cache.keys():
                # Regenerate key to check if it matches function
                if key.startswith(hashlib.md5(func_prefix.encode()).hexdigest()[:8]):
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if self._memory_cache.invalidate(key):
                invalidated_count += 1
        
        # Disk cache
        if self.enable_disk_cache:
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove disk cache file: {e}")
        
        logger.info(f"Invalidated {invalidated_count} cache entries for {func.__name__}")
        return invalidated_count
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        if self.enable_disk_cache and self.disk_cache_dir.exists():
            for cache_file in self.disk_cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove disk cache file: {e}")
        
        logger.info(f"Cleared all cache entries for {self.name}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        memory_stats = self._memory_cache.get_stats()
        
        report = {
            'cache_name': self.name,
            'memory_cache': self._memory_cache.get_cache_info(),
            'disk_cache_enabled': self.enable_disk_cache,
            'computation_stats': {}
        }
        
        # Add computation time statistics
        for func_name, times in self._computation_times.items():
            if times:
                report['computation_stats'][func_name] = {
                    'call_count': len(times),
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'total_time_s': np.sum(times)
                }
        
        # Disk cache statistics
        if self.enable_disk_cache and self.disk_cache_dir.exists():
            disk_files = list(self.disk_cache_dir.glob("*.pkl"))
            total_disk_size = sum(f.stat().st_size for f in disk_files)
            report['disk_cache'] = {
                'file_count': len(disk_files),
                'total_size_mb': total_disk_size / (1024 * 1024),
                'directory': str(self.disk_cache_dir)
            }
        
        return report


# Global cache instances
_global_caches: Dict[str, ComputationCache] = {}


def get_computation_cache(name: str = "default", **kwargs) -> ComputationCache:
    """Get or create a global computation cache."""
    if name not in _global_caches:
        _global_caches[name] = ComputationCache(name, **kwargs)
    return _global_caches[name]


def cached_computation(cache_name: str = "default", ttl_seconds: Optional[float] = None):
    """Decorator for automatic caching of expensive computations."""
    def decorator(func: Callable) -> Callable:
        cache = get_computation_cache(cache_name, ttl_seconds=ttl_seconds)
        
        def wrapper(*args, **kwargs):
            return cache.cached_call(func, *args, **kwargs)
        
        wrapper._cache = cache
        wrapper._original_func = func
        return wrapper
    
    return decorator


def clear_all_caches() -> None:
    """Clear all global caches."""
    for cache in _global_caches.values():
        cache.clear_all()
    logger.info("Cleared all global caches")


def get_cache_performance_summary() -> Dict[str, Any]:
    """Get performance summary for all caches."""
    summary = {
        'total_caches': len(_global_caches),
        'caches': {}
    }
    
    for name, cache in _global_caches.items():
        summary['caches'][name] = cache.get_performance_report()
    
    return summary