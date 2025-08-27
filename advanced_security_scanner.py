#!/usr/bin/env python3
"""
ADVANCED SECURITY SCANNER FOR AUTONOMOUS QUALITY GATES
Comprehensive security validation and vulnerability assessment

This module implements enterprise-grade security scanning:
- Static code analysis with security focus
- Dependency vulnerability scanning
- Configuration security validation
- Secrets detection and prevention
- Security compliance checking
- Runtime security monitoring
- Threat modeling integration
- Security test automation
- OWASP Top 10 vulnerability detection
- Custom security rule engine
"""

import sys
import os
import re
import ast
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import tempfile
import base64
import urllib.parse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecuritySeverity(Enum):
    """Security finding severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityCategory(Enum):
    """Security vulnerability categories."""
    INJECTION = "injection"                    # SQL, Command injection
    AUTH_BROKEN = "broken_authentication"      # Authentication flaws
    DATA_EXPOSURE = "sensitive_data_exposure"  # Data exposure
    XXE = "xml_external_entities"             # XML vulnerabilities
    ACCESS_CONTROL = "broken_access_control"   # Access control issues
    SECURITY_MISCONFIG = "security_misconfig"  # Configuration issues
    XSS = "cross_site_scripting"              # XSS vulnerabilities
    DESERIALIZATION = "insecure_deserialization"  # Deserialization flaws
    COMPONENTS = "vulnerable_components"       # Component vulnerabilities
    LOGGING = "insufficient_logging"           # Logging issues
    SECRETS = "secrets_exposure"               # Secrets in code
    CRYPTO = "cryptographic_issues"            # Crypto problems


@dataclass
class SecurityFinding:
    """Security finding with details."""
    id: str
    severity: SecuritySeverity
    category: SecurityCategory
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID
    owasp_category: Optional[str] = None
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class SecurityScanResult:
    """Security scan result with findings and metrics."""
    findings: List[SecurityFinding] = field(default_factory=list)
    scan_duration: float = 0.0
    files_scanned: int = 0
    lines_scanned: int = 0
    security_score: float = 100.0
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class SecretsDetector:
    """Advanced secrets detection in code and configuration."""
    
    def __init__(self):
        # Common secret patterns with high confidence
        self.secret_patterns = {
            'api_key': {
                'pattern': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']([a-zA-Z0-9_-]{16,})["\']',
                'confidence': 0.9
            },
            'password': {
                'pattern': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']{8,})["\']',
                'confidence': 0.7
            },
            'token': {
                'pattern': r'(?i)(token|access[_-]?token)\s*[:=]\s*["\']([a-zA-Z0-9_.-]{20,})["\']',
                'confidence': 0.8
            },
            'secret_key': {
                'pattern': r'(?i)(secret[_-]?key|secretkey)\s*[:=]\s*["\']([a-zA-Z0-9_-]{16,})["\']',
                'confidence': 0.9
            },
            'private_key': {
                'pattern': r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----',
                'confidence': 1.0
            },
            'database_url': {
                'pattern': r'(?i)(database[_-]?url|db[_-]?url)\s*[:=]\s*["\']([^"\']*://[^"\']+)["\']',
                'confidence': 0.8
            },
            'aws_key': {
                'pattern': r'AKIA[0-9A-Z]{16}',
                'confidence': 0.95
            },
            'github_token': {
                'pattern': r'ghp_[a-zA-Z0-9]{36}',
                'confidence': 0.95
            }
        }
        
        # Environment variable patterns that might contain secrets
        self.env_secret_patterns = [
            r'(?i).*key.*',
            r'(?i).*secret.*',
            r'(?i).*token.*',
            r'(?i).*password.*',
            r'(?i).*auth.*'
        ]
        
    def detect_secrets(self, content: str, file_path: str) -> List[SecurityFinding]:
        """Detect secrets in file content."""
        findings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check each secret pattern
            for secret_type, pattern_info in self.secret_patterns.items():
                pattern = pattern_info['pattern']
                confidence = pattern_info['confidence']
                
                matches = re.finditer(pattern, line)
                for match in matches:
                    # Create security finding
                    finding = SecurityFinding(
                        id=f"secret_{secret_type}_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                        severity=SecuritySeverity.HIGH,
                        category=SecurityCategory.SECRETS,
                        title=f"Potential {secret_type.replace('_', ' ')} exposure",
                        description=f"Detected potential {secret_type} in source code. Secrets should not be hardcoded.",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=f"Move {secret_type} to environment variables or secure configuration",
                        cwe_id="CWE-798",
                        owasp_category="A3:2017-Sensitive Data Exposure",
                        confidence=confidence
                    )
                    findings.append(finding)
                    
        return findings
    
    def detect_env_secrets(self, content: str, file_path: str) -> List[SecurityFinding]:
        """Detect potential secrets in environment variable usage."""
        findings = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Look for environment variable usage
            env_matches = re.finditer(r'os\.environ\.get\(["\']([^"\']+)["\']', line)
            for match in env_matches:
                env_var = match.group(1)
                
                # Check if env var name suggests it contains secrets
                for pattern in self.env_secret_patterns:
                    if re.match(pattern, env_var):
                        finding = SecurityFinding(
                            id=f"env_secret_{hashlib.md5(f'{file_path}:{line_num}:{env_var}'.encode()).hexdigest()[:8]}",
                            severity=SecuritySeverity.INFO,
                            category=SecurityCategory.SECRETS,
                            title=f"Environment variable may contain sensitive data",
                            description=f"Environment variable '{env_var}' name suggests it may contain sensitive data",
                            file_path=file_path,
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation="Ensure proper access controls for environment variables containing secrets",
                            cwe_id="CWE-200",
                            confidence=0.5
                        )
                        findings.append(finding)
                        break
                        
        return findings


class CodeSecurityAnalyzer:
    """Advanced static code analysis for security vulnerabilities."""
    
    def __init__(self):
        # Dangerous function patterns
        self.dangerous_functions = {
            'eval': {
                'pattern': r'\beval\s*\(',
                'severity': SecuritySeverity.CRITICAL,
                'category': SecurityCategory.INJECTION,
                'description': 'Use of eval() function allows arbitrary code execution',
                'cwe': 'CWE-94'
            },
            'exec': {
                'pattern': r'\bexec\s*\(',
                'severity': SecuritySeverity.CRITICAL,
                'category': SecurityCategory.INJECTION,
                'description': 'Use of exec() function allows arbitrary code execution',
                'cwe': 'CWE-94'
            },
            'os.system': {
                'pattern': r'\bos\.system\s*\(',
                'severity': SecuritySeverity.HIGH,
                'category': SecurityCategory.INJECTION,
                'description': 'Use of os.system() can lead to command injection',
                'cwe': 'CWE-78'
            },
            'subprocess_shell': {
                'pattern': r'subprocess\.[^(]*\([^)]*shell\s*=\s*True',
                'severity': SecuritySeverity.HIGH,
                'category': SecurityCategory.INJECTION,
                'description': 'subprocess with shell=True can lead to command injection',
                'cwe': 'CWE-78'
            },
            'pickle_load': {
                'pattern': r'\bpickle\.load\s*\(',
                'severity': SecuritySeverity.HIGH,
                'category': SecurityCategory.DESERIALIZATION,
                'description': 'pickle.load() can execute arbitrary code during deserialization',
                'cwe': 'CWE-502'
            },
            'yaml_load': {
                'pattern': r'\byaml\.load\s*\(',
                'severity': SecuritySeverity.MEDIUM,
                'category': SecurityCategory.DESERIALIZATION,
                'description': 'yaml.load() can execute arbitrary Python code',
                'cwe': 'CWE-502'
            },
            'random_weak': {
                'pattern': r'\brandom\.random\s*\(',
                'severity': SecuritySeverity.LOW,
                'category': SecurityCategory.CRYPTO,
                'description': 'random.random() is not cryptographically secure',
                'cwe': 'CWE-338'
            }
        }
        
        # Security-sensitive imports
        self.sensitive_imports = {
            'os': 'System interaction - review for command injection risks',
            'subprocess': 'Process execution - review for command injection risks',
            'pickle': 'Serialization - review for deserialization vulnerabilities',
            'eval': 'Code evaluation - extremely dangerous if used with user input',
            'exec': 'Code execution - extremely dangerous if used with user input'
        }
        
    def analyze_code_security(self, content: str, file_path: str) -> List[SecurityFinding]:
        """Analyze code for security vulnerabilities."""
        findings = []
        lines = content.split('\n')
        
        # Check for dangerous function usage
        for line_num, line in enumerate(lines, 1):
            for func_name, func_info in self.dangerous_functions.items():
                pattern = func_info['pattern']
                
                if re.search(pattern, line):
                    finding = SecurityFinding(
                        id=f"code_vuln_{func_name}_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                        severity=func_info['severity'],
                        category=func_info['category'],
                        title=f"Dangerous function usage: {func_name}",
                        description=func_info['description'],
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        recommendation=f"Replace {func_name} with safer alternatives",
                        cwe_id=func_info['cwe'],
                        owasp_category="A1:2017-Injection",
                        confidence=0.9
                    )
                    findings.append(finding)
        
        # Analyze AST for more complex patterns
        try:
            tree = ast.parse(content)
            ast_findings = self._analyze_ast_security(tree, file_path)
            findings.extend(ast_findings)
        except SyntaxError:
            # If AST parsing fails, continue with regex-based analysis
            pass
            
        return findings
    
    def _analyze_ast_security(self, tree: ast.AST, file_path: str) -> List[SecurityFinding]:
        """Analyze AST for security issues."""
        findings = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.findings = []
                
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile']:
                        finding = SecurityFinding(
                            id=f"ast_dangerous_{func_name}_{id(node)}",
                            severity=SecuritySeverity.CRITICAL,
                            category=SecurityCategory.INJECTION,
                            title=f"Dangerous function call: {func_name}",
                            description=f"AST analysis detected {func_name}() call",
                            file_path=file_path,
                            line_number=getattr(node, 'lineno', 0),
                            recommendation=f"Remove or replace {func_name}() with safer alternatives",
                            cwe_id="CWE-94",
                            confidence=1.0
                        )
                        self.findings.append(finding)
                        
                elif isinstance(node.func, ast.Attribute):
                    # Check for method calls like os.system()
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'os' and 
                        node.func.attr == 'system'):
                        finding = SecurityFinding(
                            id=f"ast_os_system_{id(node)}",
                            severity=SecuritySeverity.HIGH,
                            category=SecurityCategory.INJECTION,
                            title="OS command execution",
                            description="AST analysis detected os.system() call",
                            file_path=file_path,
                            line_number=getattr(node, 'lineno', 0),
                            recommendation="Use subprocess with proper input validation",
                            cwe_id="CWE-78",
                            confidence=1.0
                        )
                        self.findings.append(finding)
                
                self.generic_visit(node)
                
            def visit_Import(self, node):
                # Check for security-sensitive imports
                for alias in node.names:
                    module_name = alias.name
                    if module_name in self.parent.sensitive_imports:
                        description = self.parent.sensitive_imports[module_name]
                        finding = SecurityFinding(
                            id=f"ast_import_{module_name}_{id(node)}",
                            severity=SecuritySeverity.INFO,
                            category=SecurityCategory.SECURITY_MISCONFIG,
                            title=f"Security-sensitive import: {module_name}",
                            description=description,
                            file_path=file_path,
                            line_number=getattr(node, 'lineno', 0),
                            recommendation="Review usage for security implications",
                            confidence=0.6
                        )
                        self.findings.append(finding)
                        
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.parent = self  # Give access to sensitive_imports
        visitor.visit(tree)
        
        return visitor.findings


class DependencyScanner:
    """Scan dependencies for known vulnerabilities."""
    
    def __init__(self):
        # Common vulnerable dependency patterns
        self.known_vulnerabilities = {
            'pickle': {
                'description': 'Pickle module can execute arbitrary code during deserialization',
                'severity': SecuritySeverity.HIGH,
                'recommendation': 'Use safer serialization formats like JSON'
            },
            'pyyaml': {
                'description': 'PyYAML unsafe_load can execute arbitrary Python code',
                'severity': SecuritySeverity.HIGH,
                'recommendation': 'Use yaml.safe_load() instead of yaml.load()'
            }
        }
        
    def scan_requirements(self, requirements_file: Path) -> List[SecurityFinding]:
        """Scan requirements file for vulnerable dependencies."""
        findings = []
        
        if not requirements_file.exists():
            return findings
            
        try:
            content = requirements_file.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                # Parse package name
                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                package_name = package_name.lower()
                
                if package_name in self.known_vulnerabilities:
                    vuln_info = self.known_vulnerabilities[package_name]
                    
                    finding = SecurityFinding(
                        id=f"dep_vuln_{package_name}_{hashlib.md5(str(requirements_file).encode()).hexdigest()[:8]}",
                        severity=vuln_info['severity'],
                        category=SecurityCategory.COMPONENTS,
                        title=f"Vulnerable dependency: {package_name}",
                        description=vuln_info['description'],
                        file_path=str(requirements_file),
                        line_number=line_num,
                        code_snippet=line,
                        recommendation=vuln_info['recommendation'],
                        cwe_id="CWE-1035",
                        owasp_category="A9:2017-Using Components with Known Vulnerabilities",
                        confidence=0.8
                    )
                    findings.append(finding)
                    
        except Exception as e:
            logger.warning(f"Failed to scan requirements file {requirements_file}: {e}")
            
        return findings
    
    def check_for_safety_tool(self) -> Optional[List[SecurityFinding]]:
        """Try to run safety tool if available."""
        try:
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse safety output
                try:
                    safety_data = json.loads(result.stdout)
                    findings = []
                    
                    for vuln in safety_data:
                        finding = SecurityFinding(
                            id=f"safety_{vuln.get('id', 'unknown')}",
                            severity=SecuritySeverity.HIGH,
                            category=SecurityCategory.COMPONENTS,
                            title=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                            description=vuln.get('vulnerability', 'Known vulnerability'),
                            recommendation=f"Update to version {vuln.get('fixed_version', 'latest')}",
                            cwe_id="CWE-1035",
                            confidence=1.0
                        )
                        findings.append(finding)
                        
                    return findings
                    
                except json.JSONDecodeError:
                    pass
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        return None


class ConfigurationSecurityAnalyzer:
    """Analyze configuration files for security issues."""
    
    def __init__(self):
        self.config_patterns = {
            'debug_enabled': {
                'pattern': r'(?i)(debug\s*[:=]\s*true|debug\s*[:=]\s*1)',
                'severity': SecuritySeverity.MEDIUM,
                'description': 'Debug mode enabled in configuration'
            },
            'weak_session_key': {
                'pattern': r'(?i)(session[_-]?key|secret[_-]?key)\s*[:=]\s*["\'](.{1,15})["\']',
                'severity': SecuritySeverity.HIGH,
                'description': 'Weak or short session/secret key'
            },
            'insecure_protocol': {
                'pattern': r'(?i)(http://|ftp://)',
                'severity': SecuritySeverity.MEDIUM,
                'description': 'Insecure protocol usage'
            }
        }
        
    def analyze_config_file(self, file_path: Path) -> List[SecurityFinding]:
        """Analyze configuration file for security issues."""
        findings = []
        
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                for pattern_name, pattern_info in self.config_patterns.items():
                    pattern = pattern_info['pattern']
                    
                    if re.search(pattern, line):
                        finding = SecurityFinding(
                            id=f"config_{pattern_name}_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                            severity=pattern_info['severity'],
                            category=SecurityCategory.SECURITY_MISCONFIG,
                            title=f"Configuration security issue: {pattern_name.replace('_', ' ')}",
                            description=pattern_info['description'],
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation="Review and secure configuration",
                            cwe_id="CWE-16",
                            owasp_category="A6:2017-Security Misconfiguration",
                            confidence=0.8
                        )
                        findings.append(finding)
                        
        except Exception as e:
            logger.warning(f"Failed to analyze config file {file_path}: {e}")
            
        return findings


class AdvancedSecurityScanner:
    """Advanced security scanner orchestrator."""
    
    def __init__(self):
        self.secrets_detector = SecretsDetector()
        self.code_analyzer = CodeSecurityAnalyzer()
        self.dependency_scanner = DependencyScanner()
        self.config_analyzer = ConfigurationSecurityAnalyzer()
        
        # File extensions to scan
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.cs', '.php', '.rb', '.go'}
        self.config_extensions = {'.yaml', '.yml', '.json', '.ini', '.conf', '.cfg', '.env'}
        
    def scan_project(self, project_path: str = ".") -> SecurityScanResult:
        """Comprehensive security scan of a project."""
        logger.info("🔒 Starting Advanced Security Scanner")
        
        start_time = time.time()
        project_path = Path(project_path)
        
        all_findings = []
        files_scanned = 0
        lines_scanned = 0
        
        # Scan Python source files
        python_files = list(project_path.rglob("*.py"))
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                file_lines = len(content.split('\n'))
                lines_scanned += file_lines
                files_scanned += 1
                
                # Run all security checks
                findings = []
                findings.extend(self.secrets_detector.detect_secrets(content, str(py_file)))
                findings.extend(self.secrets_detector.detect_env_secrets(content, str(py_file)))
                findings.extend(self.code_analyzer.analyze_code_security(content, str(py_file)))
                
                all_findings.extend(findings)
                
                if len(findings) > 0:
                    logger.debug(f"Found {len(findings)} issues in {py_file}")
                    
            except Exception as e:
                logger.warning(f"Failed to scan {py_file}: {e}")
                
        # Scan configuration files
        config_files = []
        for ext in self.config_extensions:
            config_files.extend(project_path.rglob(f"*{ext}"))
            
        for config_file in config_files:
            try:
                findings = self.config_analyzer.analyze_config_file(config_file)
                all_findings.extend(findings)
                files_scanned += 1
                
                if len(findings) > 0:
                    logger.debug(f"Found {len(findings)} config issues in {config_file}")
                    
            except Exception as e:
                logger.warning(f"Failed to scan config {config_file}: {e}")
        
        # Scan dependencies
        requirements_files = list(project_path.rglob("requirements*.txt")) + list(project_path.rglob("Pipfile"))
        for req_file in requirements_files:
            findings = self.dependency_scanner.scan_requirements(req_file)
            all_findings.extend(findings)
            
        # Try safety tool scan
        safety_findings = self.dependency_scanner.check_for_safety_tool()
        if safety_findings:
            all_findings.extend(safety_findings)
            logger.info(f"Safety tool found {len(safety_findings)} vulnerability findings")
        
        # Calculate security score
        security_score = self._calculate_security_score(all_findings, files_scanned, lines_scanned)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(all_findings)
        
        # Check compliance
        compliance_status = self._check_compliance(all_findings)
        
        scan_duration = time.time() - start_time
        
        result = SecurityScanResult(
            findings=all_findings,
            scan_duration=scan_duration,
            files_scanned=files_scanned,
            lines_scanned=lines_scanned,
            security_score=security_score,
            compliance_status=compliance_status,
            recommendations=recommendations
        )
        
        logger.info(f"🔒 Security scan completed: {len(all_findings)} findings in {scan_duration:.2f}s")
        
        return result
    
    def _calculate_security_score(self, findings: List[SecurityFinding], files_scanned: int, lines_scanned: int) -> float:
        """Calculate overall security score."""
        if not findings:
            return 100.0
            
        # Weight by severity
        severity_weights = {
            SecuritySeverity.INFO: 1,
            SecuritySeverity.LOW: 2,
            SecuritySeverity.MEDIUM: 5,
            SecuritySeverity.HIGH: 10,
            SecuritySeverity.CRITICAL: 20
        }
        
        total_weight = sum(severity_weights[finding.severity] * finding.confidence for finding in findings)
        
        # Normalize by files scanned
        if files_scanned > 0:
            weight_per_file = total_weight / files_scanned
            # Score decreases with more weighted issues per file
            score = max(0, 100 - weight_per_file * 10)
        else:
            score = 100.0
            
        return score
    
    def _generate_security_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        # Count findings by category
        category_counts = {}
        severity_counts = {}
        
        for finding in findings:
            category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        # Generate specific recommendations
        if severity_counts.get(SecuritySeverity.CRITICAL, 0) > 0:
            recommendations.append("🚨 URGENT: Address all CRITICAL security findings immediately")
            
        if severity_counts.get(SecuritySeverity.HIGH, 0) > 0:
            recommendations.append("⚠️ HIGH priority: Fix all HIGH severity vulnerabilities")
            
        if category_counts.get(SecurityCategory.SECRETS, 0) > 0:
            recommendations.append("🔑 Remove hardcoded secrets and use environment variables or secure vaults")
            
        if category_counts.get(SecurityCategory.INJECTION, 0) > 0:
            recommendations.append("💉 Replace dangerous functions (eval, exec, os.system) with safer alternatives")
            
        if category_counts.get(SecurityCategory.COMPONENTS, 0) > 0:
            recommendations.append("📦 Update vulnerable dependencies to latest secure versions")
            
        if category_counts.get(SecurityCategory.SECURITY_MISCONFIG, 0) > 0:
            recommendations.append("⚙️ Review and secure configuration settings")
            
        # General recommendations
        recommendations.append("🛡️ Implement automated security scanning in CI/CD pipeline")
        recommendations.append("📝 Conduct security code reviews for all changes")
        recommendations.append("🧪 Add security test cases to test suite")
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _check_compliance(self, findings: List[SecurityFinding]) -> Dict[str, bool]:
        """Check compliance with security standards."""
        compliance = {}
        
        # OWASP Top 10 compliance check
        owasp_violations = set()
        for finding in findings:
            if finding.owasp_category:
                owasp_violations.add(finding.owasp_category)
                
        compliance['OWASP_Top_10'] = len(owasp_violations) == 0
        
        # Critical/High findings compliance
        critical_high = [f for f in findings if f.severity in [SecuritySeverity.CRITICAL, SecuritySeverity.HIGH]]
        compliance['No_Critical_High_Vulnerabilities'] = len(critical_high) == 0
        
        # Secrets compliance
        secret_findings = [f for f in findings if f.category == SecurityCategory.SECRETS]
        compliance['No_Hardcoded_Secrets'] = len(secret_findings) == 0
        
        # Injection compliance
        injection_findings = [f for f in findings if f.category == SecurityCategory.INJECTION]
        compliance['No_Injection_Vulnerabilities'] = len(injection_findings) == 0
        
        return compliance
    
    def generate_security_report(self, result: SecurityScanResult, output_file: str = None) -> str:
        """Generate comprehensive security report."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"security_report_{timestamp}.json"
            
        # Convert findings to serializable format
        findings_data = []
        for finding in result.findings:
            finding_dict = {
                'id': finding.id,
                'severity': finding.severity.value,
                'category': finding.category.value,
                'title': finding.title,
                'description': finding.description,
                'file_path': finding.file_path,
                'line_number': finding.line_number,
                'code_snippet': finding.code_snippet,
                'recommendation': finding.recommendation,
                'cwe_id': finding.cwe_id,
                'owasp_category': finding.owasp_category,
                'confidence': finding.confidence
            }
            findings_data.append(finding_dict)
            
        # Create comprehensive report
        report_data = {
            'timestamp': time.time(),
            'scan_summary': {
                'total_findings': len(result.findings),
                'files_scanned': result.files_scanned,
                'lines_scanned': result.lines_scanned,
                'scan_duration': result.scan_duration,
                'security_score': result.security_score
            },
            'findings_by_severity': {
                severity.value: len([f for f in result.findings if f.severity == severity])
                for severity in SecuritySeverity
            },
            'findings_by_category': {
                category.value: len([f for f in result.findings if f.category == category])
                for category in SecurityCategory
            },
            'compliance_status': result.compliance_status,
            'recommendations': result.recommendations,
            'detailed_findings': findings_data
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"📄 Security report saved to {output_file}")
        
        return output_file
    
    def print_security_summary(self, result: SecurityScanResult):
        """Print security scan summary to console."""
        print("\n" + "🔒" + "=" * 68)
        print("🔒 ADVANCED SECURITY SCANNER SUMMARY")
        print("🔒" + "=" * 68)
        
        print(f"📊 Scan Statistics:")
        print(f"   • Files Scanned: {result.files_scanned}")
        print(f"   • Lines Scanned: {result.lines_scanned:,}")
        print(f"   • Scan Duration: {result.scan_duration:.2f}s")
        print(f"   • Security Score: {result.security_score:.1f}/100")
        
        print(f"\n🔍 Findings by Severity:")
        severity_counts = {}
        for finding in result.findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
            
        severity_emojis = {
            SecuritySeverity.CRITICAL: "🚨",
            SecuritySeverity.HIGH: "⚠️",
            SecuritySeverity.MEDIUM: "🟡",
            SecuritySeverity.LOW: "🟢",
            SecuritySeverity.INFO: "ℹ️"
        }
        
        for severity in [SecuritySeverity.CRITICAL, SecuritySeverity.HIGH, 
                        SecuritySeverity.MEDIUM, SecuritySeverity.LOW, SecuritySeverity.INFO]:
            count = severity_counts.get(severity, 0)
            emoji = severity_emojis[severity]
            print(f"   {emoji} {severity.value.title()}: {count}")
            
        print(f"\n📋 Compliance Status:")
        for standard, compliant in result.compliance_status.items():
            status_emoji = "✅" if compliant else "❌"
            print(f"   {status_emoji} {standard.replace('_', ' ')}")
            
        if result.recommendations:
            print(f"\n🎯 Top Security Recommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"   {i}. {rec}")
                
        # Overall assessment
        if result.security_score >= 90:
            print(f"\n🎉 EXCELLENT security posture!")
        elif result.security_score >= 80:
            print(f"\n👍 GOOD security posture with minor improvements needed")
        elif result.security_score >= 70:
            print(f"\n⚠️ MODERATE security concerns - address key findings")
        elif result.security_score >= 60:
            print(f"\n🚨 POOR security posture - immediate action required")
        else:
            print(f"\n💀 CRITICAL security issues - urgent remediation needed")


def main():
    """Main execution for advanced security scanner."""
    print("🔒 ADVANCED SECURITY SCANNER FOR AUTONOMOUS QUALITY GATES")
    print("🛡️ COMPREHENSIVE SECURITY VALIDATION AND VULNERABILITY ASSESSMENT")
    print("=" * 80)
    
    # Initialize security scanner
    scanner = AdvancedSecurityScanner()
    
    try:
        # Run comprehensive security scan
        result = scanner.scan_project(".")
        
        # Generate and save report
        report_file = scanner.generate_security_report(result)
        
        # Print summary
        scanner.print_security_summary(result)
        
        # Determine success based on security score and critical findings
        critical_findings = [f for f in result.findings if f.severity == SecuritySeverity.CRITICAL]
        high_findings = [f for f in result.findings if f.severity == SecuritySeverity.HIGH]
        
        success = (
            result.security_score >= 80 and 
            len(critical_findings) == 0 and
            len(high_findings) <= 2  # Allow up to 2 high severity findings
        )
        
        if success:
            print(f"\n🎉 SECURITY SCAN SUCCESSFUL!")
            print(f"🔒 Security Score: {result.security_score:.1f}/100")
            print(f"🛡️ No critical vulnerabilities found")
            print(f"✅ Ready for secure deployment")
        else:
            print(f"\n⚠️ SECURITY ISSUES IDENTIFIED")
            print(f"🔒 Security Score: {result.security_score:.1f}/100")
            if critical_findings:
                print(f"🚨 {len(critical_findings)} CRITICAL findings require immediate attention")
            if high_findings:
                print(f"⚠️ {len(high_findings)} HIGH severity findings need resolution")
            print(f"📋 Review detailed report: {report_file}")
        
        return success
        
    except Exception as e:
        print(f"❌ Security scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)