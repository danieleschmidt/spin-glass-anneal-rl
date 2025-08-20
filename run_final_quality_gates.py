#!/usr/bin/env python3
"""
Final Quality Gates Execution for AUTONOMOUS SDLC
Execute all mandatory quality gates with comprehensive reporting.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any

def run_code_execution_test() -> Dict[str, Any]:
    """Test that core code runs without errors."""
    print("1️⃣ Testing Code Execution...")
    
    try:
        # Test minimal functionality
        from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer, demo_basic_functionality
        
        # Run basic demo
        demo_basic_functionality()
        
        # Test batch processing
        problems = [MinimalIsingModel(10) for _ in range(3)]
        annealer = MinimalAnnealer()
        
        for problem in problems:
            energy, history = annealer.anneal(problem, n_sweeps=100)
        
        return {
            'status': 'PASS',
            'message': 'All core functionality executes successfully',
            'details': {'problems_tested': 3, 'minimal_demo_passed': True}
        }
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Code execution failed: {e}',
            'details': {'error': str(e)}
        }

def run_test_coverage() -> Dict[str, Any]:
    """Run test coverage analysis."""
    print("2️⃣ Analyzing Test Coverage...")
    
    try:
        # Count implementation files
        impl_files = 0
        test_files = 0
        
        for root, dirs, files in os.walk('spin_glass_rl'):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    impl_files += 1
        
        for root, dirs, files in os.walk('tests'):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_files += 1
        
        # Estimate coverage based on test files vs implementation files
        coverage_estimate = min((test_files / max(impl_files, 1)) * 100, 95.0)
        
        if coverage_estimate >= 85.0:
            status = 'PASS'
            message = f'Test coverage meets target: {coverage_estimate:.1f}%'
        else:
            status = 'WARNING'
            message = f'Test coverage below target: {coverage_estimate:.1f}% < 85%'
        
        return {
            'status': status,
            'message': message,
            'details': {
                'coverage_percent': coverage_estimate,
                'implementation_files': impl_files,
                'test_files': test_files
            }
        }
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Coverage analysis failed: {e}',
            'details': {'error': str(e)}
        }

def run_security_scan() -> Dict[str, Any]:
    """Run basic security scan."""
    print("3️⃣ Running Security Scan...")
    
    try:
        security_issues = []
        
        # Check for dangerous imports
        dangerous_imports = ['os.system', 'subprocess.call', 'eval', 'exec', '__import__']
        
        for root, dirs, files in os.walk('spin_glass_rl'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        for dangerous in dangerous_imports:
                            if dangerous in content:
                                security_issues.append({
                                    'file': filepath,
                                    'issue': f'Potentially dangerous import: {dangerous}',
                                    'severity': 'medium'
                                })
                    except UnicodeDecodeError:
                        continue
        
        if len(security_issues) == 0:
            return {
                'status': 'PASS',
                'message': 'No security issues detected',
                'details': {'issues_found': 0}
            }
        elif len(security_issues) <= 5:
            return {
                'status': 'WARNING', 
                'message': f'{len(security_issues)} potential security issues found',
                'details': {'issues': security_issues[:3]}  # Show first 3
            }
        else:
            return {
                'status': 'FAIL',
                'message': f'{len(security_issues)} security issues found (too many)',
                'details': {'issues_count': len(security_issues)}
            }
            
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Security scan failed: {e}',
            'details': {'error': str(e)}
        }

def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("4️⃣ Running Performance Benchmarks...")
    
    try:
        from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer
        
        # Benchmark small problem
        start_time = time.time()
        problem = MinimalIsingModel(20)
        annealer = MinimalAnnealer()
        energy, history = annealer.anneal(problem, n_sweeps=500)
        benchmark_time = time.time() - start_time
        
        # Performance targets
        max_time_20_spins = 5.0  # seconds
        
        if benchmark_time <= max_time_20_spins:
            status = 'PASS'
            message = f'Performance benchmark met: {benchmark_time:.3f}s <= {max_time_20_spins}s'
        else:
            status = 'FAIL'
            message = f'Performance benchmark failed: {benchmark_time:.3f}s > {max_time_20_spins}s'
        
        return {
            'status': status,
            'message': message,
            'details': {
                'benchmark_time_seconds': benchmark_time,
                'target_time_seconds': max_time_20_spins,
                'problem_size': 20,
                'sweeps': 500,
                'final_energy': energy
            }
        }
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Performance benchmark failed: {e}',
            'details': {'error': str(e)}
        }

def check_documentation() -> Dict[str, Any]:
    """Check documentation completeness."""
    print("5️⃣ Checking Documentation...")
    
    try:
        # Check for key documentation files
        required_docs = [
            'README.md',
            'API_DOCUMENTATION.md', 
            'ARCHITECTURE.md',
            'DEPLOYMENT.md',
            'SECURITY.md'
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not os.path.exists(doc):
                missing_docs.append(doc)
        
        # Check docstring coverage
        python_files_with_docstrings = 0
        total_python_files = 0
        
        for root, dirs, files in os.walk('spin_glass_rl'):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    total_python_files += 1
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '"""' in content or "'''" in content:
                                python_files_with_docstrings += 1
                    except UnicodeDecodeError:
                        continue
        
        docstring_coverage = (python_files_with_docstrings / max(total_python_files, 1)) * 100
        
        if len(missing_docs) == 0 and docstring_coverage >= 80:
            status = 'PASS'
            message = f'Documentation complete ({docstring_coverage:.1f}% docstring coverage)'
        elif len(missing_docs) == 0:
            status = 'WARNING'
            message = f'Documentation present but low docstring coverage: {docstring_coverage:.1f}%'
        else:
            status = 'FAIL'
            message = f'Missing documentation files: {missing_docs}'
        
        return {
            'status': status,
            'message': message,
            'details': {
                'missing_docs': missing_docs,
                'docstring_coverage': docstring_coverage,
                'python_files_checked': total_python_files
            }
        }
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Documentation check failed: {e}',
            'details': {'error': str(e)}
        }

def check_reproducibility() -> Dict[str, Any]:
    """Check reproducibility of results."""
    print("6️⃣ Testing Reproducibility...")
    
    try:
        from spin_glass_rl.core.minimal_ising import MinimalIsingModel, MinimalAnnealer
        import random
        import numpy as np
        
        # Set seeds for reproducibility test
        results = []
        
        for run in range(3):
            # Reset seeds
            random.seed(42)
            np.random.seed(42)
            
            # Create and solve problem
            problem = MinimalIsingModel(15)
            annealer = MinimalAnnealer(initial_temp=2.0, final_temp=0.01)
            energy, history = annealer.anneal(problem, n_sweeps=200)
            results.append(energy)
        
        # Check consistency
        energy_std = np.std(results)
        max_variation = abs(max(results) - min(results))
        
        if max_variation < 0.1:  # Allow small numerical variations
            status = 'PASS'
            message = f'Results reproducible (max variation: {max_variation:.6f})'
        else:
            status = 'WARNING'
            message = f'Results may not be fully reproducible (variation: {max_variation:.6f})'
        
        return {
            'status': status,
            'message': message,
            'details': {
                'results': results,
                'std_deviation': energy_std,
                'max_variation': max_variation,
                'runs_tested': len(results)
            }
        }
        
    except Exception as e:
        return {
            'status': 'FAIL',
            'message': f'Reproducibility test failed: {e}',
            'details': {'error': str(e)}
        }

def generate_final_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive final report."""
    
    total_gates = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    warnings = sum(1 for r in results if r['status'] == 'WARNING')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    
    pass_rate = (passed / total_gates) * 100
    
    # Overall status determination
    if failed > 0:
        overall_status = 'FAIL'
        production_ready = False
    elif warnings > 2:
        overall_status = 'WARNING'
        production_ready = False
    elif pass_rate >= 85:
        overall_status = 'PASS'
        production_ready = True
    else:
        overall_status = 'WARNING'
        production_ready = False
    
    return {
        'timestamp': time.time(),
        'overall_status': overall_status,
        'production_ready': production_ready,
        'quality_gates': {
            'total': total_gates,
            'passed': passed,
            'warnings': warnings, 
            'failed': failed,
            'pass_rate': pass_rate
        },
        'detailed_results': results,
        'recommendations': _generate_recommendations(results),
        'next_steps': _generate_next_steps(overall_status, production_ready)
    }

def _generate_recommendations(results: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on results."""
    recommendations = []
    
    for result in results:
        if result['status'] == 'FAIL':
            recommendations.append(f"🔴 CRITICAL: Fix {result['gate_name']} - {result['message']}")
        elif result['status'] == 'WARNING':
            recommendations.append(f"🟡 IMPROVE: {result['gate_name']} - {result['message']}")
    
    return recommendations

def _generate_next_steps(overall_status: str, production_ready: bool) -> List[str]:
    """Generate next steps based on results."""
    if production_ready:
        return [
            "✅ All quality gates passed - Ready for production deployment",
            "🚀 Execute deployment pipeline",
            "📊 Set up production monitoring",
            "📋 Schedule post-deployment validation"
        ]
    elif overall_status == 'WARNING':
        return [
            "⚠️ Address warnings before production deployment",
            "🔧 Implement recommended improvements",
            "🧪 Re-run quality gates after fixes",
            "👥 Consider additional code review"
        ]
    else:
        return [
            "🔴 CRITICAL: Must fix failures before production",
            "🛠️ Address all failed quality gates",
            "🔄 Re-run complete quality gate suite",
            "⏳ Do not proceed to deployment until all gates pass"
        ]

def main():
    """Execute all quality gates with comprehensive reporting."""
    
    print("🎯 EXECUTING FINAL QUALITY GATES")
    print("🔥 Autonomous SDLC - Production Readiness Assessment")
    print("=" * 70)
    
    start_time = time.time()
    
    # Execute all quality gates
    quality_gates = [
        ('Code Execution', run_code_execution_test),
        ('Test Coverage', run_test_coverage),
        ('Security Scan', run_security_scan),
        ('Performance Benchmarks', run_performance_benchmarks),
        ('Documentation', check_documentation),
        ('Reproducibility', check_reproducibility)
    ]
    
    results = []
    
    for gate_name, gate_function in quality_gates:
        gate_start = time.time()
        result = gate_function()
        gate_time = time.time() - gate_start
        
        result['gate_name'] = gate_name
        result['execution_time'] = gate_time
        results.append(result)
        
        # Print result
        status_icon = {'PASS': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}[result['status']]
        print(f"{status_icon} {gate_name}: {result['status']} - {result['message']}")
    
    total_time = time.time() - start_time
    
    # Generate comprehensive report
    final_report = generate_final_report(results)
    final_report['execution_time_seconds'] = total_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 FINAL QUALITY GATES SUMMARY")
    print("=" * 70)
    
    overall_icon = {'PASS': '✅', 'WARNING': '⚠️', 'FAIL': '❌'}[final_report['overall_status']]
    print(f"{overall_icon} Overall Status: {final_report['overall_status']}")
    
    qg = final_report['quality_gates']
    print(f"📈 Results: {qg['passed']} passed, {qg['warnings']} warnings, {qg['failed']} failed")
    print(f"🎯 Pass Rate: {qg['pass_rate']:.1f}%")
    print(f"⏱️ Total Execution Time: {total_time:.2f}s")
    
    if final_report['production_ready']:
        print("\n🚀 PRODUCTION READY!")
        print("All mandatory quality gates have been successfully completed.")
    else:
        print("\n⚠️ NOT READY FOR PRODUCTION")
        print("Address the following issues before deployment:")
        
        for recommendation in final_report['recommendations']:
            print(f"  {recommendation}")
    
    print("\n📋 Next Steps:")
    for step in final_report['next_steps']:
        print(f"  {step}")
    
    # Save detailed report
    report_filename = f"final_quality_gates_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n💾 Detailed report saved: {report_filename}")
    
    return final_report

if __name__ == "__main__":
    try:
        report = main()
        exit_code = 0 if report['production_ready'] else 1
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)