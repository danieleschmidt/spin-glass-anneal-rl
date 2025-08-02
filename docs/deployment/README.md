# Deployment Guide for Spin-Glass-Anneal-RL

This guide covers various deployment scenarios for Spin-Glass-Anneal-RL, from local development to production deployments.

## Overview

Spin-Glass-Anneal-RL supports multiple deployment modes:

- **Development**: Local development with hot reloading
- **Testing**: Automated testing environments
- **Production**: Scalable production deployments
- **Cloud**: Cloud-native deployments (AWS, GCP, Azure)
- **HPC**: High-performance computing environments

## Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/terragonlabs/spin-glass-anneal-rl.git
cd spin-glass-anneal-rl

# Start development environment
docker-compose up dev
```

### Production Deployment

```bash
# Build and start production services
docker-compose up -d app postgres redis

# Or use the production profile
docker-compose --profile production up -d
```

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- CUDA 12.2+ (for GPU acceleration)

### Available Services

Our Docker Compose configuration provides multiple services:

#### Core Services

- **dev**: Development environment with live code reloading
- **app**: Production application server
- **test**: Testing environment
- **benchmark**: Performance benchmarking

#### Support Services

- **postgres**: PostgreSQL database for experiment tracking
- **redis**: Redis for caching and job queues
- **tensorboard**: TensorBoard for metrics visualization
- **mlflow**: MLflow for experiment tracking
- **prometheus**: Metrics collection
- **grafana**: Metrics visualization

#### Specialized Services

- **jupyter**: Jupyter Lab for interactive development
- **docs**: Documentation server

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Database
POSTGRES_DB=spin_glass_rl
POSTGRES_USER=sgrl_user
POSTGRES_PASSWORD=your_secure_password

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# ML Services
WANDB_API_KEY=your_wandb_key
DWAVE_API_TOKEN=your_dwave_token

# Application
LOG_LEVEL=INFO
WORKERS=4
```

### Service Management

```bash
# Start specific services
docker-compose up -d dev postgres redis

# View logs
docker-compose logs -f app

# Scale services
docker-compose up -d --scale app=3

# Stop services
docker-compose down

# Remove all data
docker-compose down -v
```

### GPU Support

Ensure NVIDIA Docker runtime is installed:

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi

# Start GPU-enabled services
docker-compose up -d dev
```

## Production Deployment

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Application   │────│    Database     │
│    (Nginx)      │    │   (FastAPI)     │    │  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                │
                       ┌─────────────────┐
                       │     Cache       │
                       │    (Redis)      │
                       └─────────────────┘
```

### Production Checklist

#### Security

- [ ] Change default passwords
- [ ] Use HTTPS/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up proper authentication
- [ ] Enable audit logging
- [ ] Scan images for vulnerabilities

#### Performance

- [ ] Configure resource limits
- [ ] Set up monitoring and alerting
- [ ] Configure auto-scaling
- [ ] Optimize database connections
- [ ] Set up caching strategies

#### Reliability

- [ ] Configure health checks
- [ ] Set up backup procedures
- [ ] Test disaster recovery
- [ ] Configure log rotation
- [ ] Set up monitoring dashboards

### Production Configuration

#### docker-compose.prod.yml

```yaml
services:
  app:
    image: spin-glass-rl:prod
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - LOG_LEVEL=WARNING
      - WORKERS=4
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped
```

#### Nginx Configuration

```nginx
upstream app {
    server app:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Cloud Deployment

### AWS Deployment

#### ECS with Fargate

```yaml
# task-definition.json
{
  "family": "spin-glass-rl",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-registry/spin-glass-rl:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint/db"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/spin-glass-rl",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### EKS Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spin-glass-rl
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spin-glass-rl
  template:
    metadata:
      labels:
        app: spin-glass-rl
    spec:
      containers:
      - name: app
        image: spin-glass-rl:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

### Google Cloud Platform

#### Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: spin-glass-rl
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 3600
      containers:
      - image: gcr.io/project/spin-glass-rl:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

#### GKE with GPUs

```yaml
# gke-gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spin-glass-rl-gpu
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spin-glass-rl-gpu
  template:
    metadata:
      labels:
        app: spin-glass-rl-gpu
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      containers:
      - name: app
        image: gcr.io/project/spin-glass-rl:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
```

### Azure Deployment

#### Container Instances

```yaml
# azure-container-instance.yaml
apiVersion: 2021-07-01
location: eastus
name: spin-glass-rl
properties:
  containers:
  - name: app
    properties:
      image: yourregistry.azurecr.io/spin-glass-rl:latest
      resources:
        requests:
          cpu: 2
          memoryInGb: 8
      ports:
      - port: 8000
        protocol: TCP
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 80
```

## HPC Deployment

### Slurm Integration

```bash
#!/bin/bash
#SBATCH --job-name=spin-glass-rl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Load modules
module load cuda/12.2
module load python/3.11
module load singularity

# Run with Singularity
singularity exec --nv spin-glass-rl.sif python -m spin_glass_rl.scripts.benchmark
```

### Singularity Container

```bash
# Build Singularity container
sudo singularity build spin-glass-rl.sif docker://spin-glass-rl:latest

# Run on HPC
singularity exec --nv spin-glass-rl.sif python your_script.py
```

## Monitoring and Observability

### Prometheus Metrics

The application exposes metrics at `/metrics`:

```
# HELP spin_glass_annealing_duration_seconds Time spent annealing
# TYPE spin_glass_annealing_duration_seconds histogram
spin_glass_annealing_duration_seconds_bucket{le="0.1"} 0
spin_glass_annealing_duration_seconds_bucket{le="1.0"} 5
spin_glass_annealing_duration_seconds_bucket{le="10.0"} 15
spin_glass_annealing_duration_seconds_bucket{le="+Inf"} 20

# HELP spin_glass_energy_convergence Energy convergence metric
# TYPE spin_glass_energy_convergence gauge
spin_glass_energy_convergence 0.95
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` with default credentials:
- Username: `admin`
- Password: configured in `.env`

Key dashboards include:
- Application Performance
- GPU Utilization
- Energy Convergence
- System Resources

### Health Checks

The application provides several health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/health/detailed

# Readiness check
curl http://localhost:8000/ready
```

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
docker-compose exec postgres pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup.sql

# Restore from backup
docker-compose exec -i postgres psql -U $POSTGRES_USER $POSTGRES_DB < backup.sql
```

### Volume Backup

```bash
# Backup persistent volumes
docker run --rm -v spin-glass-rl-postgres:/data -v $(pwd):/backup ubuntu tar czf /backup/postgres-backup.tar.gz /data
```

## Scaling

### Horizontal Scaling

```bash
# Scale application instances
docker-compose up -d --scale app=5

# Use load balancer
docker-compose up -d nginx
```

### Vertical Scaling

```yaml
# Increase resource limits
services:
  app:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

## Troubleshooting

### Common Issues

#### GPU Access

```bash
# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi

# Check NVIDIA Docker runtime
docker info | grep nvidia
```

#### Memory Issues

```bash
# Monitor memory usage
docker stats

# Check GPU memory
nvidia-smi

# Clear CUDA cache
docker-compose exec app python -c "import torch; torch.cuda.empty_cache()"
```

#### Network Issues

```bash
# Check port availability
netstat -tulpn | grep :8000

# Test container networking
docker-compose exec app curl http://localhost:8000/health
```

### Debugging

```bash
# View application logs
docker-compose logs -f app

# Access container shell
docker-compose exec app /bin/bash

# Debug with Python
docker-compose exec app python -c "import spin_glass_rl; print('OK')"
```

### Performance Tuning

#### CUDA Optimization

```bash
# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NUMBA_CUDA_DEBUGINFO=0
```

#### Memory Management

```python
# Python memory optimization
import torch
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

## Security Considerations

### Container Security

```bash
# Scan for vulnerabilities
docker scan spin-glass-rl:latest

# Use non-root user
USER appuser

# Read-only filesystem
docker run --read-only spin-glass-rl:latest
```

### Network Security

```yaml
# Network policies
networks:
  default:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: br-sgrl
```

### Secrets Management

```bash
# Use Docker secrets
echo "your_secret" | docker secret create db_password -

# Mount secrets
volumes:
  - source: db_password
    target: /run/secrets/db_password
```

This deployment guide provides comprehensive coverage of various deployment scenarios. Choose the approach that best fits your infrastructure and requirements.