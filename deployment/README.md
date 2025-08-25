# Progressive Quality Gates - Production Deployment

## Overview

This deployment package contains everything needed to deploy Progressive Quality Gates 
in a production environment with high availability, monitoring, and auto-scaling.

## Architecture

Progressive Quality Gates implements a three-generation architecture:

1. **Generation 1**: Basic progressive quality gates with stage-based thresholds
2. **Generation 2**: Enhanced error handling, monitoring, and circuit breakers
3. **Generation 3**: Performance optimization, ML prediction, and distributed execution

## Deployment Environments

- **Development**: Single replica, basic monitoring
- **Staging**: Multi-replica, full monitoring, performance testing
- **Production**: Auto-scaling, high availability, comprehensive monitoring

## Quick Start

### Prerequisites

- Docker installed and running
- Kubernetes cluster access
- kubectl configured

### Kubernetes Deployment

```bash
# Build Docker image
./deployment/scripts/build.sh

# Deploy to Kubernetes
./deployment/scripts/deploy.sh

# Run health checks
./deployment/scripts/health-check.sh
```

## Configuration

### Environment Variables

- `ENVIRONMENT`: Deployment environment (development/staging/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `CONFIG_PATH`: Path to configuration file

### Resource Requirements

| Environment | CPU | Memory | Storage |
|-------------|-----|---------|---------|
| Development | 100m | 256Mi | 1Gi |
| Staging | 500m | 1Gi | 5Gi |
| Production | 1000m | 2Gi | 10Gi |

## Features

### Progressive Quality Gates

- **Entry Level**: Basic functionality and safety checks
- **Development Level**: Standard quality and testing requirements  
- **Staging Level**: Comprehensive validation and performance testing
- **Production Level**: Enterprise-grade quality with full compliance
- **Critical Level**: Research-grade validation with statistical rigor

### Advanced Capabilities

- **Intelligent Caching**: SQLite-backed persistent cache with TTL management
- **Machine Learning**: Quality score prediction and online learning
- **Distributed Execution**: Auto-scaling worker pool with load balancing
- **Real-time Monitoring**: Circuit breakers and performance tracking
- **Enhanced Security**: Comprehensive threat detection and vulnerability scanning

## Monitoring

### Health Endpoints

- Health Check: `GET /health`
- Readiness: `GET /ready`
- Metrics: `GET /metrics`

### Key Metrics

- Quality gate success rates
- Execution times and performance
- Cache hit rates and optimization
- Worker pool utilization
- System health and resource usage

## Scaling

### Horizontal Pod Autoscaler

- Min replicas: 2
- Max replicas: 10  
- CPU target: 70%

### Auto-scaling Features

- Worker pool auto-scaling based on queue size
- Intelligent cache size management
- ML model adaptation based on workload

## Security

### Container Security

- Non-root user execution (UID 1001)
- Minimal base image (Python slim)
- Health checks and liveness probes
- Resource limits and requests

### Application Security

- Input validation and sanitization
- Security scanning and threat detection
- Circuit breaker pattern for resilience
- Comprehensive error handling

## Troubleshooting

### Common Issues

1. **Pod startup failures**
   - Check resource limits: `kubectl describe pod <pod-name> -n progressive-quality-gates-staging`
   - Review logs: `kubectl logs -l app=progressive-quality-gates -n progressive-quality-gates-staging`

2. **Health check failures**
   - Verify service endpoints: `kubectl get endpoints -n progressive-quality-gates-staging`
   - Run diagnostic script: `./deployment/scripts/health-check.sh`

3. **Performance issues**
   - Check resource utilization: `kubectl top pods -n progressive-quality-gates-staging`
   - Scale up: `kubectl scale deployment progressive-quality-gates --replicas=5 -n progressive-quality-gates-staging`

### Support Commands

```bash
# View deployment status
kubectl get all -n progressive-quality-gates-staging

# Check logs
kubectl logs -l app=progressive-quality-gates -n progressive-quality-gates-staging --tail=100

# Scale deployment
kubectl scale deployment progressive-quality-gates --replicas=5 -n progressive-quality-gates-staging

# Port forward for local testing
kubectl port-forward service/progressive-quality-gates-service 8080:8080 -n progressive-quality-gates-staging
```

## License

MIT License - see LICENSE file for details.

## Version History

- v1.0.0: Initial production release with all three generations
- Features: Progressive stages, enhanced monitoring, ML optimization
