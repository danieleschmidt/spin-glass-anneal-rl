"""Setup script for Spin-Glass Annealing RL."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version_file = Path(__file__).parent / "spin_glass_rl" / "_version.py"
exec(version_file.read_text())

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="spin-glass-anneal-rl",
    version=__version__,
    author="Terragon Labs",
    author_email="info@terragonlabs.com",
    description="GPU-accelerated optimization framework using physics-inspired spin-glass models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/retreival-free-context-compressor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.11.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "click>=8.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "pytest-benchmark>=3.4.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "myst-parser>=0.15.0",
        ],
        "cuda": [
            "cupy-cuda11x>=9.6.0",  # CUDA 11.x support
        ],
        "visualization": [
            "plotly>=5.3.0",
            "dash>=2.0.0",
            "seaborn>=0.11.0",
        ],
        "benchmark": [
            "memory-profiler>=0.60.0",
            "line-profiler>=3.3.0",
            "py-spy>=0.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "spin-glass-rl=spin_glass_rl.cli:cli",
        ],
    },
    package_data={
        "spin_glass_rl": [
            "*.yaml",
            "*.json",
            "benchmarks/data/*.json",
            "examples/*.py",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "optimization",
        "spin-glass",
        "annealing", 
        "reinforcement-learning",
        "gpu-acceleration",
        "physics-inspired",
        "combinatorial-optimization",
        "scheduling",
        "routing",
        "machine-learning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/danieleschmidt/retreival-free-context-compressor/issues",
        "Source": "https://github.com/danieleschmidt/retreival-free-context-compressor",
        "Documentation": "https://spin-glass-rl.readthedocs.io/",
    },
)