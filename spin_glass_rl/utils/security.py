"""Security utilities for input sanitization and safe operations."""

import os
import re
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised for security-related violations."""
    pass


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename for safe filesystem operations.
    
    Args:
        filename: Input filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
        
    Raises:
        SecurityError: If filename is unsafe
    """
    if not isinstance(filename, str):
        raise SecurityError("Filename must be string")
    
    if len(filename) > max_length:
        raise SecurityError(f"Filename too long: {len(filename)} > {max_length}")
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    if not sanitized:
        raise SecurityError("Filename cannot be empty after sanitization")
    
    logger.debug(f"Sanitized filename: '{filename}' -> '{sanitized}'")
    return sanitized


def audit_log(action: str, 
              details: Dict[str, Any], 
              user_id: Optional[str] = None,
              severity: str = 'INFO') -> None:
    """
    Log security-relevant actions for audit trail.
    
    Args:
        action: Action being performed
        details: Additional details about the action
        user_id: User identifier
        severity: Log severity level
    """
    import time
    
    audit_entry = {
        'timestamp': time.time(),
        'action': action,
        'details': details,
        'user_id': user_id,
        'severity': severity
    }
    
    # In production, this would go to a secure audit log
    logger.info(f"AUDIT: {json.dumps(audit_entry)}")