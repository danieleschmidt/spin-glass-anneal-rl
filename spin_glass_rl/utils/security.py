"""Security utilities and input sanitization."""

import os
import hashlib
import secrets
import base64
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re
import logging

from spin_glass_rl.utils.exceptions import ValidationError, SpinGlassError


class SecurityError(SpinGlassError):
    """Security-related errors."""
    pass


class InputSanitizer:
    """Input sanitization and validation."""
    
    # Dangerous patterns to detect
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
        r'globals\s*\(',
        r'locals\s*\(',
        r'vars\s*\(',
        r'dir\s*\(',
        r'getattr\s*\(',
        r'setattr\s*\(',
        r'delattr\s*\(',
        r'hasattr\s*\(',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'\.system\s*\(',
        r'subprocess',
        r'os\.system',
        r'os\.popen',
        r'os\.spawn',
        r'pickle\.loads',
        r'pickle\.load',
        r'marshal\.loads',
        r'marshal\.load',
    ]
    
    # Safe file extensions
    SAFE_EXTENSIONS = {'.json', '.yaml', '.yml', '.txt', '.csv', '.npy', '.npz', '.pt', '.pth'}
    
    # Maximum file size (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    @classmethod
    def sanitize_string(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValidationError("Input must be a string")
        
        # Check length
        if len(value) > max_length:
            raise ValidationError(f"String too long: {len(value)} > {max_length}")
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityError(f"Dangerous pattern detected: {pattern}")
        
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    @classmethod
    def sanitize_path(cls, path: Union[str, Path]) -> Path:
        """Sanitize file path."""
        path_obj = Path(path)
        
        # Resolve path to prevent directory traversal
        try:
            resolved_path = path_obj.resolve()
        except Exception as e:
            raise SecurityError(f"Invalid path: {path}") from e
        
        # Check for suspicious patterns
        path_str = str(resolved_path)
        if '..' in path_str or path_str.startswith('/etc') or path_str.startswith('/sys'):
            raise SecurityError(f"Potentially dangerous path: {path}")
        
        # Check extension
        if resolved_path.suffix.lower() not in cls.SAFE_EXTENSIONS:
            raise SecurityError(f"Unsafe file extension: {resolved_path.suffix}")
        
        return resolved_path
    
    @classmethod
    def sanitize_json(cls, json_str: str, max_size: int = 10000) -> Dict[str, Any]:
        """Sanitize and parse JSON input."""
        if len(json_str) > max_size:
            raise ValidationError(f"JSON too large: {len(json_str)} > {max_size}")
        
        # Check for dangerous patterns
        sanitized_json = cls.sanitize_string(json_str, max_length=max_size)
        
        try:
            data = json.loads(sanitized_json)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")
        
        # Validate parsed data structure
        cls._validate_json_structure(data)
        
        return data
    
    @classmethod
    def _validate_json_structure(cls, data: Any, depth: int = 0, max_depth: int = 10):
        """Validate JSON data structure for safety."""
        if depth > max_depth:
            raise SecurityError("JSON structure too deep")
        
        if isinstance(data, dict):
            if len(data) > 1000:
                raise SecurityError("Too many keys in JSON object")
            
            for key, value in data.items():
                if not isinstance(key, str):
                    raise SecurityError("Non-string keys not allowed")
                
                if len(key) > 100:
                    raise SecurityError("Key too long")
                
                cls._validate_json_structure(value, depth + 1, max_depth)
        
        elif isinstance(data, list):
            if len(data) > 10000:
                raise SecurityError("Array too large")
            
            for item in data:
                cls._validate_json_structure(item, depth + 1, max_depth)
        
        elif isinstance(data, str):
            if len(data) > 10000:
                raise SecurityError("String value too long")
    
    @classmethod
    def validate_file_upload(cls, file_path: Union[str, Path]) -> None:
        """Validate uploaded file for safety."""
        path_obj = cls.sanitize_path(file_path)
        
        if not path_obj.exists():
            raise ValidationError(f"File does not exist: {path_obj}")
        
        # Check file size
        file_size = path_obj.stat().st_size
        if file_size > cls.MAX_FILE_SIZE:
            raise SecurityError(f"File too large: {file_size} > {cls.MAX_FILE_SIZE}")
        
        # Check if it's actually a file
        if not path_obj.is_file():
            raise SecurityError(f"Not a regular file: {path_obj}")


class SecureConfigManager:
    """Secure configuration management."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize secure config manager."""
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.spin_glass_rl'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Set secure permissions
        self.config_dir.chmod(0o700)
        
        self.logger = logging.getLogger(__name__)
    
    def save_config(self, config: Dict[str, Any], name: str) -> None:
        """Save configuration securely."""
        # Sanitize config name
        safe_name = InputSanitizer.sanitize_string(name, max_length=50)
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_name)
        
        config_file = self.config_dir / f"{safe_name}.json"
        
        # Validate configuration
        self._validate_config(config)
        
        # Save with secure permissions
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, default=str)
        
        config_file.chmod(0o600)
        
        self.logger.info(f"Configuration saved: {config_file}")
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load configuration securely."""
        safe_name = InputSanitizer.sanitize_string(name, max_length=50)
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', safe_name)
        
        config_file = self.config_dir / f"{safe_name}.json"
        
        if not config_file.exists():
            raise ValidationError(f"Configuration not found: {name}")
        
        # Validate file permissions
        file_stat = config_file.stat()
        if file_stat.st_mode & 0o077:
            self.logger.warning(f"Configuration file has loose permissions: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config = InputSanitizer.sanitize_json(content, max_size=100000)
        self._validate_config(config)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        if not isinstance(config, dict):
            raise ValidationError("Configuration must be a dictionary")
        
        # Check for sensitive data
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        for key in config.keys():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                self.logger.warning(f"Potentially sensitive key in config: {key}")


class MemorySecurityMonitor:
    """Monitor for memory-based security issues."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_memory_threshold = 0.9  # 90% of available memory
        self.suspicious_allocations = []
    
    def check_memory_usage(self, operation: str) -> None:
        """Check memory usage for security issues."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > self.max_memory_threshold * 100:
                self.logger.warning(
                    f"High memory usage during {operation}: {memory.percent}%",
                    extra={"operation": operation, "memory_percent": memory.percent}
                )
                
                # Could be a DoS attempt
                if memory.percent > 95:
                    raise SecurityError(f"Memory usage too high during {operation}: {memory.percent}%")
        
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")
    
    def check_allocation_size(self, size: int, operation: str) -> None:
        """Check if allocation size is suspicious."""
        max_reasonable_size = 10 * 1024 * 1024 * 1024  # 10GB
        
        if size > max_reasonable_size:
            self.logger.error(
                f"Suspicious large allocation in {operation}: {size} bytes",
                extra={"operation": operation, "allocation_size": size}
            )
            raise SecurityError(f"Allocation too large: {size} bytes")


class SecureSerializer:
    """Secure serialization that avoids pickle vulnerabilities."""
    
    ALLOWED_TYPES = {
        'dict', 'list', 'str', 'int', 'float', 'bool', 'NoneType',
        'tuple', 'set', 'frozenset'
    }
    
    @classmethod
    def serialize(cls, data: Any) -> bytes:
        """Serialize data safely to JSON."""
        try:
            # Convert to JSON-serializable format
            serializable_data = cls._make_serializable(data)
            json_str = json.dumps(serializable_data, separators=(',', ':'))
            return json_str.encode('utf-8')
        except Exception as e:
            raise SecurityError(f"Serialization failed: {e}")
    
    @classmethod
    def deserialize(cls, data: bytes) -> Any:
        """Deserialize data safely from JSON."""
        try:
            json_str = data.decode('utf-8')
            return InputSanitizer.sanitize_json(json_str, max_size=1000000)
        except Exception as e:
            raise SecurityError(f"Deserialization failed: {e}")
    
    @classmethod
    def _make_serializable(cls, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        obj_type = type(obj).__name__
        
        if obj_type not in cls.ALLOWED_TYPES:
            # For torch tensors
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            # For numpy arrays
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                raise SecurityError(f"Cannot serialize type: {obj_type}")
        
        if isinstance(obj, dict):
            return {str(k): cls._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return [cls._make_serializable(item) for item in obj]
        else:
            return obj


def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure random token."""
    return secrets.token_urlsafe(length)


def hash_data(data: str, salt: Optional[str] = None) -> str:
    """Hash data with optional salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Use SHA-256 for hashing
    hasher = hashlib.sha256()
    hasher.update(salt.encode('utf-8'))
    hasher.update(data.encode('utf-8'))
    
    return f"{salt}:{hasher.hexdigest()}"


def verify_hash(data: str, hashed: str) -> bool:
    """Verify hashed data."""
    try:
        salt, hash_value = hashed.split(':', 1)
        expected_hash = hash_data(data, salt)
        return secrets.compare_digest(expected_hash, hashed)
    except ValueError:
        return False


class SecurityAuditLog:
    """Security audit logging."""
    
    def __init__(self):
        self.logger = logging.getLogger('spin_glass_rl.security')
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event."""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                "event_type": event_type,
                "security_event": True,
                **details
            }
        )
    
    def log_failed_validation(self, validation_type: str, details: Dict[str, Any]) -> None:
        """Log failed validation attempt."""
        self.log_security_event(
            "validation_failure",
            {"validation_type": validation_type, **details}
        )
    
    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]) -> None:
        """Log suspicious activity."""
        self.log_security_event(
            "suspicious_activity", 
            {"activity": activity, **details}
        )


# Global security components
security_audit = SecurityAuditLog()
memory_monitor = MemorySecurityMonitor()