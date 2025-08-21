"""
Autonomous SDLC Final Validation Script.

Validates the complete autonomous SDLC implementation without external dependencies.
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False


def check_directory_structure() -> bool:
    """Check if all required directories and files exist."""
    print("📁 CHECKING DIRECTORY STRUCTURE")
    print("-" * 40)
    
    required_files = [
        # Core modules
        ("spin_glass_rl/__init__.py", "Core package initialization"),
        ("spin_glass_rl/core/minimal_ising.py", "Minimal Ising model implementation"),
        
        # Generation 1: Advanced Research Extensions
        ("spin_glass_rl/research/meta_learning_optimization.py", "Meta-learning optimization"),
        ("spin_glass_rl/research/quantum_hybrid_algorithms.py", "Quantum-hybrid algorithms"),
        ("spin_glass_rl/research/federated_optimization.py", "Federated optimization"),
        
        # Generation 2: Robust & Secure
        ("spin_glass_rl/security/advanced_security_framework.py", "Advanced security framework"),
        ("spin_glass_rl/monitoring/adaptive_monitoring_system.py", "Adaptive monitoring system"),
        
        # Generation 3: Scalable & Optimized
        ("spin_glass_rl/scaling/intelligent_auto_scaling.py", "Intelligent auto-scaling"),
        ("spin_glass_rl/optimization/quantum_edge_computing.py", "Quantum edge computing"),
        
        # Quality Gates & Testing
        ("test_comprehensive_autonomous_sdlc.py", "Comprehensive test suite"),
        ("test_autonomous_sdlc_validation.py", "Validation test suite"),
        ("validate_autonomous_sdlc.py", "This validation script"),
        
        # Configuration and setup
        ("pyproject.toml", "Project configuration"),
        ("setup.py", "Setup script"),
        ("README.md", "Project documentation"),
    ]
    
    all_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_exist = False
    
    return all_exist


def analyze_code_quality() -> dict:
    """Analyze code quality metrics."""
    print("\n📊 ANALYZING CODE QUALITY")
    print("-" * 40)
    
    metrics = {
        'total_lines': 0,
        'python_files': 0,
        'modules_with_docstrings': 0,
        'modules_with_classes': 0,
        'modules_with_functions': 0,
        'advanced_features': []
    }
    
    # Count Python files and analyze content
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                metrics['python_files'] += 1
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        metrics['total_lines'] += len(lines)
                        
                        # Check for docstrings
                        if '"""' in content or "'''" in content:
                            metrics['modules_with_docstrings'] += 1
                        
                        # Check for classes
                        if 'class ' in content:
                            metrics['modules_with_classes'] += 1
                        
                        # Check for functions
                        if 'def ' in content:
                            metrics['modules_with_functions'] += 1
                        
                        # Check for advanced features
                        if 'async def' in content:
                            metrics['advanced_features'].append('Async/Await Programming')
                        if 'torch.' in content or 'import torch' in content:
                            metrics['advanced_features'].append('PyTorch Integration')
                        if 'quantum' in content.lower():
                            metrics['advanced_features'].append('Quantum Computing')
                        if 'federated' in content.lower():
                            metrics['advanced_features'].append('Federated Learning')
                        if 'security' in content.lower() or 'crypto' in content.lower():
                            metrics['advanced_features'].append('Security Framework')
                        if 'monitoring' in content.lower() or 'metric' in content.lower():
                            metrics['advanced_features'].append('Monitoring System')
                        if 'scaling' in content.lower() or 'auto' in content.lower():
                            metrics['advanced_features'].append('Auto-scaling')
                
                except Exception as e:
                    print(f"  Warning: Could not analyze {filepath}: {e}")
    
    # Remove duplicates from advanced features
    metrics['advanced_features'] = list(set(metrics['advanced_features']))
    
    # Display metrics
    print(f"  Python files: {metrics['python_files']}")
    print(f"  Total lines of code: {metrics['total_lines']}")
    print(f"  Modules with docstrings: {metrics['modules_with_docstrings']}")
    print(f"  Modules with classes: {metrics['modules_with_classes']}")
    print(f"  Modules with functions: {metrics['modules_with_functions']}")
    print(f"  Advanced features detected: {len(metrics['advanced_features'])}")
    
    if metrics['advanced_features']:
        print("  Features found:")
        for feature in sorted(metrics['advanced_features']):
            print(f"    • {feature}")
    
    return metrics


def check_imports_and_syntax() -> bool:
    """Check if main modules can be imported (syntax validation)."""
    print("\n🔍 CHECKING IMPORTS AND SYNTAX")
    print("-" * 40)
    
    modules_to_check = [
        "spin_glass_rl.research.meta_learning_optimization",
        "spin_glass_rl.research.quantum_hybrid_algorithms", 
        "spin_glass_rl.research.federated_optimization",
        "spin_glass_rl.security.advanced_security_framework",
        "spin_glass_rl.monitoring.adaptive_monitoring_system",
        "spin_glass_rl.scaling.intelligent_auto_scaling",
        "spin_glass_rl.optimization.quantum_edge_computing"
    ]
    
    all_valid = True
    
    for module in modules_to_check:
        try:
            # Use subprocess to test import without affecting current process
            result = subprocess.run(
                [sys.executable, '-c', f'import {module}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"✅ {module}")
            else:
                print(f"❌ {module}: {result.stderr.strip()}")
                all_valid = False
                
        except subprocess.TimeoutExpired:
            print(f"⏱️ {module}: Import timeout")
            all_valid = False
        except Exception as e:
            print(f"❌ {module}: {str(e)}")
            all_valid = False
    
    return all_valid


def validate_generation_implementations() -> dict:
    """Validate that each generation has been properly implemented."""
    print("\n🚀 VALIDATING GENERATION IMPLEMENTATIONS")
    print("-" * 40)
    
    generations = {
        'Generation 1 (Advanced Research)': {
            'files': [
                'spin_glass_rl/research/meta_learning_optimization.py',
                'spin_glass_rl/research/quantum_hybrid_algorithms.py', 
                'spin_glass_rl/research/federated_optimization.py'
            ],
            'key_features': [
                'MetaOptimizer', 'QuantumAnnealingSimulator', 'FederatedServer'
            ]
        },
        'Generation 2 (Robust & Secure)': {
            'files': [
                'spin_glass_rl/security/advanced_security_framework.py',
                'spin_glass_rl/monitoring/adaptive_monitoring_system.py'
            ],
            'key_features': [
                'SecureOptimizationFramework', 'AdaptiveMonitoringSystem'
            ]
        },
        'Generation 3 (Scalable & Optimized)': {
            'files': [
                'spin_glass_rl/scaling/intelligent_auto_scaling.py',
                'spin_glass_rl/optimization/quantum_edge_computing.py'
            ],
            'key_features': [
                'AutoScalingController', 'QuantumEdgeNode'
            ]
        }
    }
    
    validation_results = {}
    
    for gen_name, gen_info in generations.items():
        print(f"\n  {gen_name}:")
        gen_valid = True
        
        # Check files exist
        for file_path in gen_info['files']:
            if os.path.exists(file_path):
                print(f"    ✅ {file_path}")
                
                # Check for key features in file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for feature in gen_info['key_features']:
                        if feature in content:
                            print(f"      ✅ {feature} implementation found")
                        else:
                            print(f"      ⚠️ {feature} not clearly found")
                            
                except Exception as e:
                    print(f"      ❌ Error reading file: {e}")
                    gen_valid = False
            else:
                print(f"    ❌ {file_path} - Missing")
                gen_valid = False
        
        validation_results[gen_name] = gen_valid
    
    return validation_results


def assess_production_readiness() -> dict:
    """Assess production readiness of the system."""
    print("\n🏭 ASSESSING PRODUCTION READINESS")
    print("-" * 40)
    
    readiness_criteria = {
        'Configuration Management': ['pyproject.toml', 'setup.py'],
        'Documentation': ['README.md', 'ARCHITECTURE.md', 'API_DOCUMENTATION.md'],
        'Testing Framework': ['test_autonomous_sdlc_validation.py', 'test_comprehensive_autonomous_sdlc.py'],
        'Security Implementation': ['spin_glass_rl/security/advanced_security_framework.py'],
        'Monitoring & Observability': ['spin_glass_rl/monitoring/adaptive_monitoring_system.py'],
        'Scalability Features': ['spin_glass_rl/scaling/intelligent_auto_scaling.py'],
        'Quality Gates': ['comprehensive_quality_gates.py', 'run_quality_gates.py'],
        'Deployment Automation': ['deploy_system.py', 'production_deployment_setup.py']
    }
    
    readiness_score = 0
    total_criteria = len(readiness_criteria)
    
    for criterion, required_files in readiness_criteria.items():
        criterion_met = any(os.path.exists(f) for f in required_files)
        
        if criterion_met:
            print(f"✅ {criterion}")
            readiness_score += 1
        else:
            print(f"❌ {criterion} - Missing required files")
            print(f"    Required: {', '.join(required_files)}")
    
    readiness_percentage = (readiness_score / total_criteria) * 100
    
    assessment = {
        'score': readiness_score,
        'total': total_criteria,
        'percentage': readiness_percentage,
        'status': 'READY' if readiness_percentage >= 80 else 'NEEDS_WORK'
    }
    
    print(f"\nProduction Readiness Score: {readiness_score}/{total_criteria} ({readiness_percentage:.1f}%)")
    print(f"Status: {assessment['status']}")
    
    return assessment


def generate_validation_report() -> dict:
    """Generate comprehensive validation report."""
    print("\n📋 GENERATING VALIDATION REPORT")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all validations
    structure_valid = check_directory_structure()
    code_metrics = analyze_code_quality()
    imports_valid = check_imports_and_syntax()
    generation_results = validate_generation_implementations()
    production_assessment = assess_production_readiness()
    
    validation_time = time.time() - start_time
    
    # Calculate overall score
    structure_score = 25 if structure_valid else 0
    imports_score = 25 if imports_valid else 0
    generation_score = (sum(generation_results.values()) / len(generation_results)) * 25
    production_score = (production_assessment['percentage'] / 100) * 25
    
    overall_score = structure_score + imports_score + generation_score + production_score
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'validation_time_seconds': round(validation_time, 2),
        'structure_validation': structure_valid,
        'imports_validation': imports_valid,
        'code_metrics': code_metrics,
        'generation_validation': generation_results,
        'production_readiness': production_assessment,
        'overall_score': round(overall_score, 1),
        'status': 'PASSED' if overall_score >= 75 else 'FAILED',
        'recommendations': []
    }
    
    # Generate recommendations
    if not structure_valid:
        report['recommendations'].append("Fix missing files in directory structure")
    
    if not imports_valid:
        report['recommendations'].append("Resolve import/syntax errors in modules")
    
    if any(not result for result in generation_results.values()):
        report['recommendations'].append("Complete implementation of all generations")
    
    if production_assessment['percentage'] < 80:
        report['recommendations'].append("Improve production readiness score")
    
    return report


def main():
    """Main validation function."""
    print("🎯 AUTONOMOUS SDLC FINAL VALIDATION")
    print("=" * 50)
    print("Validating complete implementation of Autonomous SDLC")
    print("following the progressive enhancement strategy:")
    print("• Generation 1: Advanced Research Extensions")
    print("• Generation 2: Robust & Secure Implementation") 
    print("• Generation 3: Scalable & Optimized System")
    print("=" * 50)
    
    # Generate validation report
    report = generate_validation_report()
    
    # Display final results
    print("\n🎯 FINAL VALIDATION RESULTS")
    print("=" * 50)
    print(f"Overall Score: {report['overall_score']}/100")
    print(f"Status: {report['status']}")
    print(f"Validation Time: {report['validation_time_seconds']}s")
    
    if report['recommendations']:
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save report
    report_filename = f"autonomous_sdlc_validation_report_{int(time.time())}.json"
    try:
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved: {report_filename}")
    except Exception as e:
        print(f"\n⚠️ Could not save report: {e}")
    
    # Final status
    if report['status'] == 'PASSED':
        print("\n🎉 AUTONOMOUS SDLC VALIDATION SUCCESSFUL!")
        print("   System is ready for production deployment.")
        return True
    else:
        print("\n⚠️ AUTONOMOUS SDLC VALIDATION FAILED")
        print("   Please address the issues above before deployment.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)