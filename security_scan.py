#!/usr/bin/env python3
"""Basic security scanner for the Spin-Glass-Anneal-RL codebase."""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Set


class SecurityScanner:
    """Basic security scanner for Python code."""
    
    def __init__(self):
        self.dangerous_imports = {
            'pickle', 'marshal', 'subprocess', 'os.system', 'eval', 'exec',
            'input', 'raw_input', '__import__', 'compile'
        }
        
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'compile\s*\(',
            r'marshal\.load[s]?\s*\(',
            r'subprocess\.',
            r'os\.system\s*\(',
            r'os\.popen\s*\(',
            r'shell\s*=\s*True',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
        
        # Patterns that need context review (not automatically dangerous)
        self.warning_patterns = [
            r'pickle\.load[s]?\s*\(',
        ]
        
        self.issues = []
    
    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan a single Python file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dangerous patterns
            for i, line in enumerate(content.split('\n'), 1):
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            'type': 'dangerous_pattern',
                            'file': str(file_path),
                            'line': i,
                            'pattern': pattern,
                            'code': line.strip(),
                            'severity': 'high'
                        })
                
                # Check for warning patterns
                for pattern in self.warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            'type': 'warning_pattern',
                            'file': str(file_path),
                            'line': i,
                            'pattern': pattern,
                            'code': line.strip(),
                            'severity': 'medium'
                        })
            
            # Parse AST for more detailed analysis
            try:
                tree = ast.parse(content)
                ast_issues = self._analyze_ast(tree, file_path)
                issues.extend(ast_issues)
            except SyntaxError as e:
                issues.append({
                    'type': 'syntax_error',
                    'file': str(file_path),
                    'line': e.lineno,
                    'message': str(e),
                    'severity': 'medium'
                })
        
        except Exception as e:
            issues.append({
                'type': 'scan_error',
                'file': str(file_path),
                'error': str(e),
                'severity': 'low'
            })
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, dangerous_imports):
                self.dangerous_imports = dangerous_imports
            
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.name in self.dangerous_imports:
                        issues.append({
                            'type': 'dangerous_import',
                            'file': str(file_path),
                            'line': node.lineno,
                            'import': alias.name,
                            'severity': 'medium'
                        })
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                if node.module in self.dangerous_imports:
                    issues.append({
                        'type': 'dangerous_import',
                        'file': str(file_path),
                        'line': node.lineno,
                        'import': node.module,
                        'severity': 'medium'
                    })
                self.generic_visit(node)
            
            def visit_Call(self, node):
                # Check for eval, exec, etc.
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        issues.append({
                            'type': 'dangerous_call',
                            'file': str(file_path),
                            'line': node.lineno,
                            'function': node.func.id,
                            'severity': 'high'
                        })
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(self.dangerous_imports)
        visitor.visit(tree)
        
        return issues
    
    def scan_directory(self, directory: Path, exclude_patterns: List[str] = None) -> List[Dict[str, Any]]:
        """Scan all Python files in a directory."""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', '.pytest_cache', 'venv', 'env']
        
        all_issues = []
        
        for file_path in directory.rglob('*.py'):
            # Skip excluded directories
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
            
            issues = self.scan_file(file_path)
            all_issues.extend(issues)
        
        return all_issues
    
    def generate_report(self, issues: List[Dict[str, Any]]) -> str:
        """Generate security report."""
        if not issues:
            return "✅ No security issues found!"
        
        report = ["🔍 Security Scan Report", "=" * 50, ""]
        
        # Group by severity
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium']
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        if high_issues:
            report.append("🚨 HIGH SEVERITY ISSUES:")
            for issue in high_issues:
                report.append(f"  - {issue['type']} in {issue['file']}:{issue.get('line', '?')}")
                if 'pattern' in issue:
                    report.append(f"    Pattern: {issue['pattern']}")
                if 'code' in issue:
                    report.append(f"    Code: {issue['code']}")
                report.append("")
        
        if medium_issues:
            report.append("⚠️  MEDIUM SEVERITY ISSUES:")
            for issue in medium_issues:
                report.append(f"  - {issue['type']} in {issue['file']}:{issue.get('line', '?')}")
                if 'import' in issue:
                    report.append(f"    Import: {issue['import']}")
                report.append("")
        
        if low_issues:
            report.append("ℹ️  LOW SEVERITY ISSUES:")
            for issue in low_issues:
                report.append(f"  - {issue['type']} in {issue['file']}")
                if 'error' in issue:
                    report.append(f"    Error: {issue['error']}")
                report.append("")
        
        report.append(f"Summary: {len(high_issues)} high, {len(medium_issues)} medium, {len(low_issues)} low severity issues")
        
        return "\n".join(report)


def validate_project_structure():
    """Validate basic project structure."""
    required_files = [
        'README.md',
        'pyproject.toml',
        'spin_glass_rl/__init__.py',
        'tests/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def check_import_structure():
    """Check that imports work correctly."""
    print("🔍 Checking import structure...")
    
    try:
        # Try to import main package
        sys.path.insert(0, str(Path.cwd()))
        
        # Basic import test
        import spin_glass_rl
        print(f"✅ Main package imports successfully (version: {getattr(spin_glass_rl, '__version__', 'unknown')})")
        
        # Check core modules
        from spin_glass_rl.core import IsingModel
        print("✅ Core modules import successfully")
        
        # Check utilities
        from spin_glass_rl.utils import exceptions, validation, logging
        print("✅ Utility modules import successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False


def run_basic_code_quality_checks():
    """Run basic code quality checks."""
    print("🔍 Running code quality checks...")
    
    issues = []
    
    # Check for very long lines
    for py_file in Path('.').rglob('*.py'):
        if '__pycache__' in str(py_file) or '.git' in str(py_file):
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if len(line) > 120:
                        issues.append(f"Long line in {py_file}:{i} ({len(line)} chars)")
        except:
            continue
    
    if issues:
        print(f"⚠️  Found {len(issues)} code quality issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print("✅ No major code quality issues found")
    
    return len(issues) == 0


def main():
    """Run security scan and basic validation."""
    print("Spin-Glass-Anneal-RL Security Scan & Validation")
    print("=" * 60)
    
    # Basic project structure validation
    structure_ok = validate_project_structure()
    
    # Import structure check
    imports_ok = check_import_structure()
    
    # Code quality checks
    quality_ok = run_basic_code_quality_checks()
    
    # Security scan
    print("\n🔒 Running security scan...")
    scanner = SecurityScanner()
    
    # Scan the main source directory
    issues = scanner.scan_directory(Path('spin_glass_rl'))
    
    # Also scan tests for security issues
    if Path('tests').exists():
        test_issues = scanner.scan_directory(Path('tests'))
        issues.extend(test_issues)
    
    # Generate and print report
    report = scanner.generate_report(issues)
    print(report)
    
    # Overall summary
    print("\n" + "=" * 60)
    print("📊 OVERALL VALIDATION RESULTS:")
    print(f"  Project Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"  Import Structure:  {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  Code Quality:      {'✅ PASS' if quality_ok else '⚠️  ISSUES'}")
    print(f"  Security Scan:     {'✅ PASS' if not issues else f'⚠️  {len(issues)} ISSUES'}")
    
    # Count critical issues
    critical_issues = [i for i in issues if i['severity'] == 'high']
    
    if structure_ok and imports_ok and not critical_issues:
        print("\n🎉 Overall Status: READY FOR PRODUCTION")
        return 0
    else:
        print("\n⚠️  Overall Status: NEEDS ATTENTION")
        return 1


if __name__ == "__main__":
    sys.exit(main())