# Production Deployment Guide

## Overview
This guide covers deploying Spin-Glass-Anneal-RL to production with high availability, security, and monitoring.

## Prerequisites
- Docker and Docker Compose installed
- Kubernetes cluster (optional)
- SSL certificates configured
- Monitoring infrastructure (Prometheus/Grafana)

## Quick Start (Docker Compose)

1. **Build and Deploy**
   ```bash
   docker-compose up -d
   ```

2. **Verify Deployment**
   ```bash
   curl http://localhost:8080/health
   ```

3. **Access Monitoring**
   - Grafana: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090

## Kubernetes Deployment

1. **Apply Configurations**
   ```bash
   kubectl apply -f k8s/k8s-deployment.yaml
   kubectl apply -f k8s/k8s-hpa.yaml
   ```

2. **Configure Ingress**
   Update `k8s-deployment.yaml` with your domain and SSL certificates.

3. **Monitor Deployment**
   ```bash
   kubectl get pods -l app=spin-glass-rl
   kubectl logs -f deployment/spin-glass-rl
   ```

## Security Configuration

1. **API Keys**: Configure authentication in production
2. **HTTPS**: Ensure all traffic is encrypted
3. **Input Validation**: Strict validation enabled by default
4. **Network Security**: Configure firewalls and VPNs as needed

## Monitoring and Alerting

1. **Prometheus Metrics**: Available at `/metrics` endpoint
2. **Grafana Dashboards**: Pre-configured performance dashboards
3. **Health Checks**: Liveness and readiness probes configured
4. **Log Aggregation**: JSON structured logs for analysis

## Performance Tuning

1. **Resource Limits**: Adjust CPU/memory based on workload
2. **Auto Scaling**: HPA configured for CPU and memory thresholds
3. **Optimization Settings**: Enable GPU acceleration for large problems
4. **Caching**: Configure Redis for result caching (optional)

## Troubleshooting

1. **Health Check Failures**: Check application logs
2. **High Memory Usage**: Reduce batch sizes or problem complexity
3. **Slow Response Times**: Check resource allocation and scaling
4. **Authentication Errors**: Verify API key configuration

## Maintenance

1. **Updates**: Use blue-green deployment strategy
2. **Backups**: Regular configuration and data backups
3. **Log Rotation**: Automated log cleanup configured
4. **Certificate Renewal**: Automated SSL certificate renewal

## Support

For issues and support, check the troubleshooting section or file an issue on GitHub.
