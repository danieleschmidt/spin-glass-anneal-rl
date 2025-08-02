# Contributing to Spin-Glass-Anneal-RL

Thank you for your interest in contributing to Spin-Glass-Anneal-RL! This document provides guidelines and information about how to contribute to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Contribution Types](#contribution-types)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [conduct@terragonlabs.com](mailto:conduct@terragonlabs.com).

## Getting Started

### Prerequisites

- Python 3.9+ (3.11 recommended)
- Git
- Docker (for containerized development)
- CUDA 12.2+ (for GPU acceleration, optional)

### Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/spin-glass-anneal-rl.git
   cd spin-glass-anneal-rl
   ```
3. **Set up development environment**:
   ```bash
   # Option 1: Docker (recommended)
   docker-compose up dev
   
   # Option 2: Local setup
   make setup-env
   source venv/bin/activate
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

We follow a Git Flow-inspired workflow:

- **main**: Production-ready code
- **develop**: Integration branch for features
- **feature/***: New features and enhancements
- **bugfix/***: Bug fixes
- **hotfix/***: Critical production fixes
- **release/***: Release preparation

### Branch Naming Convention

- `feature/gpu-acceleration-improvements`
- `bugfix/memory-leak-in-annealing`
- `docs/api-documentation-update`
- `test/integration-test-coverage`
- `refactor/coupling-matrix-optimization`

## Contribution Types

We welcome various types of contributions:

### 🚀 Features
- New algorithms and optimization techniques
- GPU acceleration improvements
- RL integration enhancements
- Performance optimizations

### 🐛 Bug Fixes
- Memory leaks or performance issues
- Incorrect algorithm implementations
- Platform-specific bugs
- Documentation errors

### 📚 Documentation
- API documentation improvements
- Tutorial and example updates
- Architecture documentation
- Performance guides

### 🧪 Testing
- Unit test coverage improvements
- Integration tests
- Performance benchmarks
- GPU testing

### 🏗️ Infrastructure
- Build system improvements
- CI/CD enhancements
- Docker optimizations
- Development tooling

## Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/terragonlabs/spin-glass-anneal-rl.git
cd spin-glass-anneal-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
make check
```

### Docker Development

```bash
# Start development environment
docker-compose up dev

# Access the container
docker-compose exec dev bash

# Run tests
docker-compose up test
```

### GPU Development

```bash
# Check GPU availability
make check-cuda

# Start GPU-enabled development
docker-compose up dev  # Automatically includes GPU support
```

## Testing Guidelines

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# GPU tests (requires CUDA)
make test-gpu

# Fast tests (exclude slow tests)
make test-fast
```

### Writing Tests

1. **Unit Tests**: Test individual functions and classes
   ```python
   @pytest.mark.unit
   def test_ising_energy_calculation():
       # Test implementation
       pass
   ```

2. **Integration Tests**: Test component interactions
   ```python
   @pytest.mark.integration
   def test_annealing_pipeline():
       # Test implementation
       pass
   ```

3. **GPU Tests**: Test CUDA functionality
   ```python
   @pytest.mark.gpu
   def test_cuda_annealing(skip_if_no_cuda):
       # Test implementation
       pass
   ```

### Test Coverage

- Maintain >80% code coverage
- All new features must include tests
- GPU code requires both CPU and GPU tests
- Performance-critical code needs benchmark tests

## Code Style

We use automated code formatting and linting:

### Python Code Style

- **Formatter**: Black (line length: 88)
- **Import sorting**: isort
- **Linting**: Ruff
- **Type checking**: mypy

### CUDA Code Style

- **Formatter**: clang-format
- **Style**: LLVM-based with CUDA modifications
- **Line length**: 100 characters

### Running Code Quality Checks

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
mypy spin_glass_rl/

# Run all quality checks
make check
```

### Pre-commit Hooks

Pre-commit hooks automatically run on every commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Documentation

### Building Documentation

```bash
# Build HTML documentation
make docs

# Serve documentation locally
make serve-docs

# Live documentation with auto-reload
make docs-live
```

### Documentation Standards

1. **Docstrings**: Use Google-style docstrings
   ```python
   def calculate_energy(spins: np.ndarray, coupling_matrix: np.ndarray) -> float:
       """Calculate the energy of an Ising configuration.
       
       Args:
           spins: Spin configuration array with values -1 or 1
           coupling_matrix: Symmetric coupling matrix
           
       Returns:
           Energy value for the configuration
           
       Example:
           >>> spins = np.array([1, -1, 1])
           >>> coupling = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]])
           >>> energy = calculate_energy(spins, coupling)
       """
   ```

2. **API Documentation**: Document all public APIs
3. **Examples**: Include practical examples in docstrings
4. **Architecture**: Update architecture docs for significant changes

## Pull Request Process

### Before Submitting

1. **Check your changes**:
   ```bash
   make check  # Run all quality checks
   make test   # Run all tests
   ```

2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Update CHANGELOG.md** with your changes

### Submission Process

1. **Create a pull request** with:
   - Clear title describing the change
   - Detailed description of what and why
   - Link to related issues
   - Screenshots/examples if applicable

2. **PR Template**: Fill out the pull request template completely

3. **Review Process**:
   - Automated checks must pass
   - Code review by maintainers
   - Manual testing if needed
   - Approval from code owners

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage maintained/improved
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] Follows code style guidelines

## Release Process

### Version Numbers

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Types

- **Feature releases**: New major/minor versions
- **Patch releases**: Bug fixes and small improvements
- **Hotfixes**: Critical security or bug fixes

## Performance Considerations

### Algorithm Performance

- Use vectorized operations with NumPy
- Leverage GPU acceleration when available
- Profile performance-critical code
- Include benchmark tests for new algorithms

### Memory Management

- Be mindful of GPU memory limitations
- Use memory mapping for large datasets
- Clear CUDA cache when appropriate
- Monitor memory usage in tests

### Scaling Considerations

- Test with various problem sizes
- Consider distributed computing for large problems
- Optimize for both single-GPU and multi-GPU scenarios

## Security Guidelines

### Security Reviews

- All contributions undergo security review
- Use `bandit` for security linting
- Follow secure coding practices
- Report security issues privately

### Dependency Management

- Keep dependencies up to date
- Use `safety` for vulnerability scanning
- Pin dependency versions for security
- Review dependency licenses

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions and questions
- **Discord**: Real-time community chat
- **Email**: [contact@terragonlabs.com](mailto:contact@terragonlabs.com)

### Getting Help

1. **Documentation**: Check existing documentation first
2. **Issues**: Search existing issues
3. **Discussions**: Ask questions in GitHub Discussions
4. **Discord**: Join our community chat

### Recognition

Contributors are recognized in:

- **AUTHORS.md**: All contributors
- **Release notes**: Major contributions
- **Hall of Fame**: Outstanding contributions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License). See [LICENSE](LICENSE) for details.

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Start a discussion on GitHub Discussions
- Contact us at [contribute@terragonlabs.com](mailto:contribute@terragonlabs.com)

Thank you for contributing to Spin-Glass-Anneal-RL! 🚀