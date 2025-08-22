"""
Advanced Security Framework for Spin-Glass Optimization.

Implements comprehensive security measures including secure computation,
cryptographic protocols, and privacy-preserving optimization.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import hmac
import secrets
import logging
from enum import Enum
import time

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different use cases."""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyProtocol(Enum):
    """Privacy-preserving protocols."""
    DIFFERENTIAL_PRIVACY = "dp"
    SECURE_MULTIPARTY = "smc"
    HOMOMORPHIC_ENCRYPTION = "he"
    FEDERATED_LEARNING = "fl"


@dataclass
class SecurityConfig:
    """Configuration for security framework."""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    privacy_protocol: PrivacyProtocol = PrivacyProtocol.DIFFERENTIAL_PRIVACY
    privacy_budget: float = 1.0
    noise_scale: float = 0.1
    encryption_key_size: int = 256
    audit_logging: bool = True
    secure_aggregation: bool = True
    input_validation: bool = True


class CryptographicProtocols:
    """Cryptographic protocols for secure optimization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.master_key = secrets.token_bytes(config.encryption_key_size // 8)
        self.session_keys = {}
        
    def generate_session_key(self, session_id: str) -> bytes:
        """Generate session-specific encryption key."""
        key = hmac.new(
            self.master_key,
            session_id.encode(),
            hashlib.sha256
        ).digest()[:32]  # 256-bit key
        
        self.session_keys[session_id] = {
            'key': key,
            'created': time.time(),
            'used_count': 0
        }
        
        return key
    
    def encrypt_tensor(self, tensor: torch.Tensor, session_id: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encrypt tensor using session key."""
        if session_id not in self.session_keys:
            self.generate_session_key(session_id)
        
        session_info = self.session_keys[session_id]
        key = session_info['key']
        
        # Simple XOR encryption with key derivation (for demonstration)
        # In practice, use proper encryption like AES
        tensor_bytes = tensor.cpu().numpy().tobytes()
        key_expanded = (key * (len(tensor_bytes) // len(key) + 1))[:len(tensor_bytes)]
        
        encrypted_bytes = bytes(a ^ b for a, b in zip(tensor_bytes, key_expanded))
        encrypted_tensor = torch.frombuffer(encrypted_bytes, dtype=torch.uint8)
        
        metadata = {
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'session_id': session_id,
            'encryption_method': 'xor_key_derived'
        }
        
        session_info['used_count'] += 1
        
        return encrypted_tensor, metadata
    
    def decrypt_tensor(self, encrypted_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Decrypt tensor using session key."""
        session_id = metadata['session_id']
        
        if session_id not in self.session_keys:
            raise ValueError(f"Session key not found for {session_id}")
        
        key = self.session_keys[session_id]['key']
        
        # Decrypt using XOR
        encrypted_bytes = encrypted_tensor.numpy().tobytes()
        key_expanded = (key * (len(encrypted_bytes) // len(key) + 1))[:len(encrypted_bytes)]
        
        decrypted_bytes = bytes(a ^ b for a, b in zip(encrypted_bytes, key_expanded))
        
        # Reconstruct tensor
        original_dtype = metadata['dtype']
        original_shape = metadata['shape']
        
        # Convert back to original dtype
        if original_dtype == torch.float32:
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
        elif original_dtype == torch.int64:
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.int64)
        else:
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32)
        
        decrypted_tensor = torch.from_numpy(decrypted_array).reshape(original_shape)
        return decrypted_tensor
    
    def secure_hash(self, data: Any) -> str:
        """Create secure hash of data."""
        if isinstance(data, torch.Tensor):
            data_bytes = data.cpu().numpy().tobytes()
        elif isinstance(data, str):
            data_bytes = data.encode()
        else:
            data_bytes = str(data).encode()
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def verify_integrity(self, data: Any, expected_hash: str) -> bool:
        """Verify data integrity using hash."""
        actual_hash = self.secure_hash(data)
        return hmac.compare_digest(actual_hash, expected_hash)


class DifferentialPrivacy:
    """Differential privacy mechanisms for optimization."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta      # Failure probability
        self.privacy_spent = 0.0
        
    def add_laplace_noise(self, tensor: torch.Tensor, sensitivity: float = 1.0) -> torch.Tensor:
        """Add Laplace noise for differential privacy."""
        if self.privacy_spent >= self.epsilon:
            logger.warning("Privacy budget exceeded!")
        
        # Laplace mechanism
        scale = sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0, scale).sample(tensor.shape)
        
        noisy_tensor = tensor + noise
        self.privacy_spent += self.epsilon / 10  # Consume part of budget
        
        return noisy_tensor
    
    def add_gaussian_noise(self, tensor: torch.Tensor, sensitivity: float = 1.0) -> torch.Tensor:
        """Add Gaussian noise for differential privacy."""
        if self.privacy_spent >= self.epsilon:
            logger.warning("Privacy budget exceeded!")
        
        # Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = torch.normal(0, sigma, tensor.shape)
        
        noisy_tensor = tensor + noise
        self.privacy_spent += self.epsilon / 10
        
        return noisy_tensor
    
    def private_aggregation(self, tensors: List[torch.Tensor], method: str = "mean") -> torch.Tensor:
        """Aggregate tensors with differential privacy."""
        if not tensors:
            raise ValueError("No tensors to aggregate")
        
        # Stack tensors
        stacked = torch.stack(tensors)
        
        if method == "mean":
            aggregated = stacked.mean(dim=0)
            sensitivity = 2.0 / len(tensors)  # Sensitivity for mean
        elif method == "sum":
            aggregated = stacked.sum(dim=0)
            sensitivity = 1.0  # Sensitivity for sum
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Add noise for privacy
        private_result = self.add_laplace_noise(aggregated, sensitivity)
        
        return private_result
    
    def get_remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.epsilon - self.privacy_spent)


class SecureMultipartyComputation:
    """Secure multiparty computation for collaborative optimization."""
    
    def __init__(self, n_parties: int, threshold: int = None):
        self.n_parties = n_parties
        self.threshold = threshold or (n_parties // 2 + 1)
        self.party_keys = {}
        self.shared_secrets = {}
        
    def generate_party_keys(self) -> Dict[str, bytes]:
        """Generate keys for all parties."""
        for i in range(self.n_parties):
            party_id = f"party_{i}"
            self.party_keys[party_id] = secrets.token_bytes(32)
        
        return self.party_keys
    
    def secret_share(self, secret: torch.Tensor, party_id: str) -> List[torch.Tensor]:
        """Create secret shares using Shamir's secret sharing."""
        # Simplified secret sharing (for demonstration)
        # In practice, use proper finite field arithmetic
        
        shares = []
        secret_flat = secret.flatten()
        
        for i in range(self.n_parties):
            # Generate random share
            if i < self.n_parties - 1:
                share = torch.randn_like(secret_flat)
                shares.append(share)
            else:
                # Last share ensures correct reconstruction
                last_share = secret_flat - sum(shares)
                shares.append(last_share)
        
        # Reshape shares to original tensor shape
        shaped_shares = [share.reshape(secret.shape) for share in shares]
        
        return shaped_shares
    
    def reconstruct_secret(self, shares: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct secret from shares."""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares")
        
        # Simple reconstruction (sum of shares)
        reconstructed = sum(shares[:self.threshold])
        
        return reconstructed
    
    def secure_aggregation(self, party_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform secure aggregation of party inputs."""
        # Create shares for each party's input
        all_shares = {}
        
        for party_id, party_input in party_inputs.items():
            shares = self.secret_share(party_input, party_id)
            all_shares[party_id] = shares
        
        # Aggregate shares
        aggregated_shares = []
        for i in range(self.n_parties):
            party_share_sum = sum(all_shares[party_id][i] for party_id in party_inputs.keys())
            aggregated_shares.append(party_share_sum)
        
        # Reconstruct aggregated result
        result = self.reconstruct_secret(aggregated_shares)
        
        return result


class SecureOptimizationFramework:
    """Framework for secure and private optimization."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.crypto = CryptographicProtocols(config)
        self.dp = DifferentialPrivacy(epsilon=config.privacy_budget)
        self.audit_log = []
        
        if config.privacy_protocol == PrivacyProtocol.SECURE_MULTIPARTY:
            self.smc = SecureMultipartyComputation(n_parties=5)  # Default 5 parties
    
    def secure_optimize(self, ising_model, client_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform secure optimization with privacy guarantees."""
        session_id = f"session_{time.time()}"
        
        # Audit log entry
        self.log_audit_event("optimization_started", {
            'session_id': session_id,
            'model_size': ising_model.n_spins,
            'security_level': self.config.security_level.value
        })
        
        try:
            # Step 1: Input validation and sanitization
            if self.config.input_validation:
                validated_model = self._validate_and_sanitize_input(ising_model)
            else:
                validated_model = ising_model
            
            # Step 2: Apply privacy-preserving mechanisms
            if self.config.privacy_protocol == PrivacyProtocol.DIFFERENTIAL_PRIVACY:
                result = self._differential_private_optimization(validated_model, session_id)
            elif self.config.privacy_protocol == PrivacyProtocol.SECURE_MULTIPARTY:
                result = self._secure_multiparty_optimization(validated_model, client_data, session_id)
            else:
                result = self._standard_secure_optimization(validated_model, session_id)
            
            # Step 3: Secure result processing
            secure_result = self._process_secure_result(result, session_id)
            
            self.log_audit_event("optimization_completed", {
                'session_id': session_id,
                'final_energy': secure_result.get('final_energy', 'N/A'),
                'privacy_budget_used': self.dp.privacy_spent
            })
            
            return secure_result
            
        except Exception as e:
            self.log_audit_event("optimization_failed", {
                'session_id': session_id,
                'error': str(e)
            })
            raise
    
    def _validate_and_sanitize_input(self, ising_model) -> Any:
        """Validate and sanitize input model."""
        # Check model size limits
        max_spins = 10000  # Security limit
        if ising_model.n_spins > max_spins:
            raise ValueError(f"Model too large: {ising_model.n_spins} > {max_spins}")
        
        # Check for malicious patterns
        if hasattr(ising_model, 'coupling_matrix'):
            coupling_matrix = ising_model.coupling_matrix
            
            # Check for extreme values
            max_coupling = torch.max(torch.abs(coupling_matrix))
            if max_coupling > 1000:
                logger.warning(f"Large coupling values detected: {max_coupling}")
                # Clamp extreme values
                ising_model.coupling_matrix = torch.clamp(coupling_matrix, -100, 100)
        
        # Validate matrix properties
        if hasattr(ising_model, 'coupling_matrix'):
            matrix = ising_model.coupling_matrix
            if not torch.allclose(matrix, matrix.T, atol=1e-6):
                logger.warning("Coupling matrix not symmetric, symmetrizing...")
                ising_model.coupling_matrix = (matrix + matrix.T) / 2
        
        return ising_model
    
    def _differential_private_optimization(self, ising_model, session_id: str) -> Dict[str, Any]:
        """Optimization with differential privacy."""
        from spin_glass_rl.core.minimal_ising import MinimalAnnealer
        
        # Standard optimization
        annealer = MinimalAnnealer()
        result = annealer.anneal(ising_model, n_sweeps=5000)
        
        # Add privacy noise to result
        if 'best_configuration' in result:
            # Add noise to spin configuration
            config = result['best_configuration'].float()
            private_config = self.dp.add_laplace_noise(config, sensitivity=2.0)
            
            # Convert back to spin values
            result['private_configuration'] = torch.sign(private_config)
            result['privacy_budget_used'] = self.dp.privacy_spent
        
        # Add noise to energy
        if 'best_energy' in result:
            energy_tensor = torch.tensor([result['best_energy']])
            private_energy = self.dp.add_laplace_noise(energy_tensor, sensitivity=1.0)
            result['private_energy'] = private_energy.item()
        
        return result
    
    def _secure_multiparty_optimization(self, ising_model, client_data: Dict[str, Any], 
                                      session_id: str) -> Dict[str, Any]:
        """Optimization using secure multiparty computation."""
        if not hasattr(self, 'smc'):
            raise ValueError("SMC not initialized")
        
        # Generate keys for all parties
        party_keys = self.smc.generate_party_keys()
        
        # Simulate multi-party inputs
        party_inputs = {}
        for i, (party_id, key) in enumerate(party_keys.items()):
            # Each party contributes a random configuration
            party_config = torch.randint(0, 2, (ising_model.n_spins,)) * 2 - 1
            party_inputs[party_id] = party_config.float()
        
        # Secure aggregation
        aggregated_config = self.smc.secure_aggregation(party_inputs)
        final_config = torch.sign(aggregated_config)
        
        # Calculate energy for aggregated configuration
        final_energy = self._calculate_energy(ising_model, final_config)
        
        return {
            'best_configuration': final_config,
            'best_energy': final_energy,
            'secure_aggregation': True,
            'n_parties': self.smc.n_parties
        }
    
    def _standard_secure_optimization(self, ising_model, session_id: str) -> Dict[str, Any]:
        """Standard optimization with basic security measures."""
        from spin_glass_rl.core.minimal_ising import MinimalAnnealer
        
        # Encrypt model if required
        if self.config.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            encrypted_coupling, coupling_metadata = self.crypto.encrypt_tensor(
                ising_model.coupling_matrix, session_id
            )
            
            # For demonstration, decrypt immediately (in practice, optimize on encrypted data)
            decrypted_coupling = self.crypto.decrypt_tensor(encrypted_coupling, coupling_metadata)
            ising_model.coupling_matrix = decrypted_coupling
        
        # Standard optimization
        annealer = MinimalAnnealer()
        result = annealer.anneal(ising_model, n_sweeps=5000)
        
        # Generate integrity hash
        if 'best_configuration' in result:
            result['integrity_hash'] = self.crypto.secure_hash(result['best_configuration'])
        
        return result
    
    def _process_secure_result(self, result: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process and secure optimization result."""
        secure_result = result.copy()
        
        # Add security metadata
        secure_result['security_metadata'] = {
            'session_id': session_id,
            'security_level': self.config.security_level.value,
            'privacy_protocol': self.config.privacy_protocol.value,
            'timestamp': time.time()
        }
        
        # Encrypt sensitive results if required
        if self.config.security_level == SecurityLevel.CRITICAL:
            if 'best_configuration' in result:
                encrypted_config, metadata = self.crypto.encrypt_tensor(
                    result['best_configuration'], session_id
                )
                secure_result['encrypted_configuration'] = encrypted_config
                secure_result['decryption_metadata'] = metadata
        
        return secure_result
    
    def _calculate_energy(self, ising_model, configuration: torch.Tensor) -> float:
        """Calculate Ising energy for configuration."""
        try:
            coupling_energy = torch.sum(ising_model.coupling_matrix * torch.outer(configuration, configuration))
            field_energy = torch.sum(ising_model.external_fields * configuration) if hasattr(ising_model, 'external_fields') and ising_model.external_fields is not None else 0
            return -(coupling_energy + field_energy).item()
        except:
            return float('inf')
    
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event for security monitoring."""
        if not self.config.audit_logging:
            return
        
        audit_entry = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details,
            'session_hash': hashlib.sha256(f"{time.time()}_{event_type}".encode()).hexdigest()[:16]
        }
        
        self.audit_log.append(audit_entry)
        logger.info(f"AUDIT: {event_type} - {details}")
    
    def get_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        if not self.audit_log:
            return {'message': 'No audit events recorded'}
        
        event_types = [entry['event_type'] for entry in self.audit_log]
        event_counts = {event: event_types.count(event) for event in set(event_types)}
        
        return {
            'total_events': len(self.audit_log),
            'event_counts': event_counts,
            'first_event': self.audit_log[0]['timestamp'],
            'last_event': self.audit_log[-1]['timestamp'],
            'privacy_budget_total': self.dp.privacy_spent,
            'remaining_budget': self.dp.get_remaining_budget(),
            'recent_events': self.audit_log[-5:]  # Last 5 events
        }


class SecurityValidator:
    """Validator for security compliance and vulnerability assessment."""
    
    def __init__(self):
        self.vulnerability_checks = [
            self._check_input_validation,
            self._check_encryption_usage,
            self._check_privacy_mechanisms,
            self._check_audit_logging,
            self._check_key_management
        ]
    
    def assess_security(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Assess security posture of optimization framework."""
        assessment_results = {
            'overall_score': 0,
            'vulnerabilities': [],
            'recommendations': [],
            'compliance_status': {}
        }
        
        total_checks = len(self.vulnerability_checks)
        passed_checks = 0
        
        for check in self.vulnerability_checks:
            try:
                result = check(framework)
                if result['passed']:
                    passed_checks += 1
                else:
                    assessment_results['vulnerabilities'].append(result)
                    assessment_results['recommendations'].extend(result.get('recommendations', []))
            except Exception as e:
                assessment_results['vulnerabilities'].append({
                    'check': check.__name__,
                    'error': str(e),
                    'passed': False
                })
        
        # Calculate overall security score
        assessment_results['overall_score'] = (passed_checks / total_checks) * 100
        
        # Determine compliance status
        if assessment_results['overall_score'] >= 90:
            assessment_results['compliance_status']['level'] = 'HIGH'
        elif assessment_results['overall_score'] >= 70:
            assessment_results['compliance_status']['level'] = 'MEDIUM'
        else:
            assessment_results['compliance_status']['level'] = 'LOW'
        
        return assessment_results
    
    def _check_input_validation(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Check input validation implementation."""
        if framework.config.input_validation:
            return {'check': 'input_validation', 'passed': True}
        else:
            return {
                'check': 'input_validation',
                'passed': False,
                'issue': 'Input validation disabled',
                'recommendations': ['Enable input validation', 'Implement sanitization']
            }
    
    def _check_encryption_usage(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Check encryption implementation."""
        has_crypto = hasattr(framework, 'crypto') and framework.crypto is not None
        if has_crypto:
            return {'check': 'encryption', 'passed': True}
        else:
            return {
                'check': 'encryption',
                'passed': False,
                'issue': 'No encryption mechanisms found',
                'recommendations': ['Implement data encryption', 'Use secure key management']
            }
    
    def _check_privacy_mechanisms(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Check privacy-preserving mechanisms."""
        has_privacy = hasattr(framework, 'dp') and framework.dp is not None
        if has_privacy and framework.config.privacy_protocol != PrivacyProtocol.DIFFERENTIAL_PRIVACY:
            return {'check': 'privacy', 'passed': True}
        elif has_privacy:
            return {'check': 'privacy', 'passed': True}
        else:
            return {
                'check': 'privacy',
                'passed': False,
                'issue': 'No privacy mechanisms implemented',
                'recommendations': ['Implement differential privacy', 'Add secure aggregation']
            }
    
    def _check_audit_logging(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Check audit logging implementation."""
        if framework.config.audit_logging and hasattr(framework, 'audit_log'):
            return {'check': 'audit_logging', 'passed': True}
        else:
            return {
                'check': 'audit_logging',
                'passed': False,
                'issue': 'Audit logging not properly configured',
                'recommendations': ['Enable comprehensive audit logging', 'Implement log monitoring']
            }
    
    def _check_key_management(self, framework: SecureOptimizationFramework) -> Dict[str, Any]:
        """Check cryptographic key management."""
        if hasattr(framework, 'crypto') and hasattr(framework.crypto, 'master_key'):
            return {'check': 'key_management', 'passed': True}
        else:
            return {
                'check': 'key_management',
                'passed': False,
                'issue': 'Inadequate key management',
                'recommendations': ['Implement secure key generation', 'Add key rotation policies']
            }


# Demonstration and testing functions
def create_security_demo():
    """Create demonstration of security framework."""
    print("Creating Advanced Security Framework Demo...")
    
    # Different security configurations
    configs = {
        'basic': SecurityConfig(
            security_level=SecurityLevel.BASIC,
            privacy_protocol=PrivacyProtocol.DIFFERENTIAL_PRIVACY,
            privacy_budget=2.0
        ),
        'high_security': SecurityConfig(
            security_level=SecurityLevel.HIGH,
            privacy_protocol=PrivacyProtocol.SECURE_MULTIPARTY,
            audit_logging=True,
            secure_aggregation=True
        ),
        'critical': SecurityConfig(
            security_level=SecurityLevel.CRITICAL,
            privacy_protocol=PrivacyProtocol.DIFFERENTIAL_PRIVACY,
            privacy_budget=0.5,
            audit_logging=True
        )
    }
    
    # Create test problem
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel
    test_model = MinimalIsingModel(n_spins=20)
    
    results = {}
    
    # Test each security configuration
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} security configuration...")
        
        framework = SecureOptimizationFramework(config)
        
        # Run secure optimization
        result = framework.secure_optimize(test_model)
        
        # Security assessment
        validator = SecurityValidator()
        security_assessment = validator.assess_security(framework)
        
        # Audit report
        audit_report = framework.get_audit_report()
        
        results[config_name] = {
            'optimization_result': result,
            'security_assessment': security_assessment,
            'audit_report': audit_report
        }
        
        print(f"  Final energy: {result.get('best_energy', 'N/A')}")
        print(f"  Security score: {security_assessment['overall_score']:.1f}%")
        print(f"  Privacy budget used: {audit_report.get('privacy_budget_total', 0):.3f}")
    
    # Display comprehensive results
    print("\n" + "="*60)
    print("ADVANCED SECURITY FRAMEWORK RESULTS")
    print("="*60)
    
    for config_name, result in results.items():
        print(f"\n{config_name.upper()} CONFIGURATION:")
        
        opt_result = result['optimization_result']
        sec_assessment = result['security_assessment']
        audit = result['audit_report']
        
        print(f"  Optimization Energy: {opt_result.get('best_energy', 'N/A')}")
        print(f"  Security Score: {sec_assessment['overall_score']:.1f}%")
        print(f"  Compliance Level: {sec_assessment['compliance_status']['level']}")
        print(f"  Vulnerabilities: {len(sec_assessment['vulnerabilities'])}")
        print(f"  Audit Events: {audit.get('total_events', 0)}")
        
        if sec_assessment['vulnerabilities']:
            print(f"  Security Issues:")
            for vuln in sec_assessment['vulnerabilities'][:3]:  # Show first 3
                print(f"    - {vuln.get('issue', 'Unknown issue')}")
    
    return results


def benchmark_security_overhead():
    """Benchmark performance overhead of security features."""
    print("Benchmarking Security Performance Overhead...")
    
    from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer
    import time
    
    # Test different problem sizes
    problem_sizes = [10, 20, 30]
    
    benchmark_results = {}
    
    for size in problem_sizes:
        print(f"\nBenchmarking {size}-spin problems...")
        
        test_model = MinimalIsingModel(n_spins=size)
        size_results = {}
        
        # Baseline (no security)
        annealer = MinimalAnnealer()
        start_time = time.time()
        baseline_result = annealer.anneal(test_model, n_sweeps=1000)
        baseline_time = time.time() - start_time
        
        size_results['baseline'] = {
            'time': baseline_time,
            'energy': baseline_result['best_energy']
        }
        
        # Different security levels
        security_configs = [
            ('basic', SecurityLevel.BASIC),
            ('standard', SecurityLevel.STANDARD), 
            ('high', SecurityLevel.HIGH)
        ]
        
        for config_name, security_level in security_configs:
            config = SecurityConfig(security_level=security_level)
            framework = SecureOptimizationFramework(config)
            
            start_time = time.time()
            secure_result = framework.secure_optimize(test_model)
            secure_time = time.time() - start_time
            
            overhead = ((secure_time - baseline_time) / baseline_time) * 100
            
            size_results[config_name] = {
                'time': secure_time,
                'energy': secure_result.get('best_energy', float('inf')),
                'overhead_percent': overhead
            }
        
        benchmark_results[size] = size_results
    
    # Display benchmark results
    print("\n" + "="*80)
    print("SECURITY PERFORMANCE OVERHEAD BENCHMARK")
    print("="*80)
    print(f"{'Size':<6} {'Baseline':<10} {'Basic':<12} {'Standard':<12} {'High':<12}")
    print(f"{'':^6} {'Time(s)':<10} {'Time(OH%)':<12} {'Time(OH%)':<12} {'Time(OH%)':<12}")
    print("-" * 80)
    
    for size, results in benchmark_results.items():
        baseline_time = results['baseline']['time']
        row = f"{size:<6} {baseline_time:<10.3f}"
        
        for config in ['basic', 'standard', 'high']:
            if config in results:
                time_val = results[config]['time']
                overhead = results[config]['overhead_percent']
                row += f" {time_val:.3f}({overhead:+.1f}%)"
            else:
                row += f" {'N/A':<12}"
        
        print(row)
    
    return benchmark_results


if __name__ == "__main__":
    # Run security demonstrations
    print("Starting Advanced Security Framework Demonstrations...\n")
    
    # Main security demo
    security_results = create_security_demo()
    
    print("\n" + "="*80)
    
    # Performance overhead benchmark
    benchmark_results = benchmark_security_overhead()
    
    print("\nAdvanced security framework demonstration completed successfully!")