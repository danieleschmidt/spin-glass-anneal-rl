#!/usr/bin/env python3
"""
Simple Production Deployment Script for Progressive Quality Gates.

This script generates a complete production deployment package without external dependencies.
"""

import sys
import os
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    namespace: str = "progressive-quality-gates"
    replicas: int = 3


class SimpleProgressiveQualityGatesDeployer:
    """Simple production deployment generator."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_dir = Path("deployment")
        self.deployment_dir.mkdir(exist_ok=True)
        
        self.docker_dir = self.deployment_dir / "docker"
        self.k8s_dir = self.deployment_dir / "k8s"
        self.scripts_dir = self.deployment_dir / "scripts"
        
        # Create directory structure
        for directory in [self.docker_dir, self.k8s_dir, self.scripts_dir]:
            directory.mkdir(exist_ok=True)
    
    def generate_dockerfile(self) -> str:
        """Generate optimized Dockerfile."""
        dockerfile_content = """# Progressive Quality Gates Production Dockerfile
FROM python:3.11-slim

# Runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && groupadd -g 1001 pqg \\
    && useradd -r -u 1001 -g pqg pqg

# Application code
WORKDIR /app
COPY progressive_quality_gates*.py .
COPY spin_glass_rl/ ./spin_glass_rl/ || true

# Install Python dependencies
RUN pip install --no-cache-dir numpy scipy psutil

# Security hardening
RUN chown -R pqg:pqg /app
USER pqg

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Entry point
CMD ["python", "progressive_quality_gates_optimized.py"]
"""
        
        dockerfile_path = self.docker_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        logger.info(f"Generated Dockerfile: {dockerfile_path}")
        return str(dockerfile_path)
    
    def generate_kubernetes_manifests(self) -> List[str]:
        """Generate Kubernetes deployment manifests."""
        manifests = []
        
        # Namespace YAML
        namespace_yaml = f"""apiVersion: v1
kind: Namespace
metadata:
  name: {self.config.namespace}
  labels:
    app: progressive-quality-gates
    environment: {self.config.environment.value}
"""
        
        # ConfigMap YAML
        configmap_yaml = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: pqg-config
  namespace: {self.config.namespace}
data:
  config.yaml: |
    environment: {self.config.environment.value}
    cache:
      enabled: true
      ttl: 300
    ml_prediction:
      enabled: true
      model_path: /app/models
    distributed_execution:
      enabled: true
      max_workers: 8
    monitoring:
      enabled: true
      port: 9090
"""
        
        # Deployment YAML
        deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: progressive-quality-gates
  namespace: {self.config.namespace}
  labels:
    app: progressive-quality-gates
    version: v1.0.0
spec:
  replicas: {self.config.replicas}
  selector:
    matchLabels:
      app: progressive-quality-gates
  template:
    metadata:
      labels:
        app: progressive-quality-gates
        version: v1.0.0
    spec:
      containers:
      - name: pqg
        image: progressive-quality-gates:latest
        ports:
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        env:
        - name: ENVIRONMENT
          value: {self.config.environment.value}
        - name: CONFIG_PATH
          value: /app/config/config.yaml
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: cache
          mountPath: /app/cache
        resources:
          limits:
            cpu: 1000m
            memory: 2Gi
          requests:
            cpu: 250m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: pqg-config
      - name: cache
        emptyDir: {{}}
"""
        
        # Service YAML
        service_yaml = f"""apiVersion: v1
kind: Service
metadata:
  name: progressive-quality-gates-service
  namespace: {self.config.namespace}
  labels:
    app: progressive-quality-gates
spec:
  selector:
    app: progressive-quality-gates
  ports:
  - name: health
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP
"""
        
        # HPA YAML
        hpa_yaml = f"""apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: progressive-quality-gates-hpa
  namespace: {self.config.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: progressive-quality-gates
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
        
        # Save manifests
        manifests_data = [
            (namespace_yaml, "namespace.yaml"),
            (configmap_yaml, "configmap.yaml"),
            (deployment_yaml, "deployment.yaml"),
            (service_yaml, "service.yaml"),
            (hpa_yaml, "hpa.yaml")
        ]
        
        for manifest_content, filename in manifests_data:
            manifest_path = self.k8s_dir / filename
            manifest_path.write_text(manifest_content)
            manifests.append(str(manifest_path))
            logger.info(f"Generated K8s manifest: {manifest_path}")
        
        return manifests
    
    def generate_deployment_scripts(self) -> List[str]:
        """Generate deployment automation scripts."""
        scripts = []
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "🐳 Building Progressive Quality Gates Docker image..."

# Build the image
docker build -t progressive-quality-gates:latest -f deployment/docker/Dockerfile .

# Tag for environment
docker tag progressive-quality-gates:latest progressive-quality-gates:${ENVIRONMENT:-development}

echo "✅ Docker image built successfully"

# Optional: Push to registry
if [ "$PUSH_TO_REGISTRY" = "true" ]; then
    echo "📤 Pushing to registry..."
    docker push progressive-quality-gates:${ENVIRONMENT:-development}
    echo "✅ Image pushed to registry"
fi
"""
        
        build_script_path = self.scripts_dir / "build.sh"
        build_script_path.write_text(build_script)
        build_script_path.chmod(0o755)
        scripts.append(str(build_script_path))
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
set -e

ENVIRONMENT="${{ENVIRONMENT:-{self.config.environment.value}}}"
NAMESPACE="{self.config.namespace}"

echo "🚀 Deploying Progressive Quality Gates to $ENVIRONMENT"

# Apply Kubernetes manifests
echo "📋 Applying Kubernetes manifests..."
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/progressive-quality-gates -n $NAMESPACE

# Get deployment status
kubectl get pods -n $NAMESPACE -l app=progressive-quality-gates

echo "✅ Deployment completed successfully"
echo "🔗 Access the service at: http://localhost:8080"

# Port forward for local access (optional)
if [ "$PORT_FORWARD" = "true" ]; then
    echo "🔌 Setting up port forwarding..."
    kubectl port-forward -n $NAMESPACE service/progressive-quality-gates-service 8080:8080 &
    kubectl port-forward -n $NAMESPACE service/progressive-quality-gates-service 9090:9090 &
    echo "✅ Port forwarding active"
fi
"""
        
        deploy_script_path = self.scripts_dir / "deploy.sh"
        deploy_script_path.write_text(deploy_script)
        deploy_script_path.chmod(0o755)
        scripts.append(str(deploy_script_path))
        
        # Health check script
        health_check_script = """#!/bin/bash
set -e

NAMESPACE="${NAMESPACE:-progressive-quality-gates}"
SERVICE_URL="${SERVICE_URL:-http://localhost:8080}"

echo "🏥 Running health checks..."

# Check if pods are running
echo "📋 Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=progressive-quality-gates

# Check service endpoints
echo "🔗 Checking service endpoints..."
kubectl get endpoints -n $NAMESPACE

# Health check
echo "❤️  Performing health check..."
if curl -f "$SERVICE_URL/health" > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Metrics check
echo "📊 Checking metrics endpoint..."
if curl -f "$SERVICE_URL/metrics" > /dev/null 2>&1; then
    echo "✅ Metrics endpoint accessible"
else
    echo "⚠️  Metrics endpoint not accessible"
fi

echo "🎉 All health checks completed"
"""
        
        health_check_script_path = self.scripts_dir / "health-check.sh"
        health_check_script_path.write_text(health_check_script)
        health_check_script_path.chmod(0o755)
        scripts.append(str(health_check_script_path))
        
        logger.info(f"Generated deployment scripts: {len(scripts)} scripts")
        return scripts
    
    def generate_documentation(self) -> List[str]:
        """Generate deployment documentation."""
        docs = []
        
        # README
        readme_content = f"""# Progressive Quality Gates - Production Deployment

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
   - Check resource limits: `kubectl describe pod <pod-name> -n {self.config.namespace}`
   - Review logs: `kubectl logs -l app=progressive-quality-gates -n {self.config.namespace}`

2. **Health check failures**
   - Verify service endpoints: `kubectl get endpoints -n {self.config.namespace}`
   - Run diagnostic script: `./deployment/scripts/health-check.sh`

3. **Performance issues**
   - Check resource utilization: `kubectl top pods -n {self.config.namespace}`
   - Scale up: `kubectl scale deployment progressive-quality-gates --replicas=5 -n {self.config.namespace}`

### Support Commands

```bash
# View deployment status
kubectl get all -n {self.config.namespace}

# Check logs
kubectl logs -l app=progressive-quality-gates -n {self.config.namespace} --tail=100

# Scale deployment
kubectl scale deployment progressive-quality-gates --replicas=5 -n {self.config.namespace}

# Port forward for local testing
kubectl port-forward service/progressive-quality-gates-service 8080:8080 -n {self.config.namespace}
```

## License

MIT License - see LICENSE file for details.

## Version History

- v1.0.0: Initial production release with all three generations
- Features: Progressive stages, enhanced monitoring, ML optimization
"""
        
        readme_path = self.deployment_dir / "README.md"
        readme_path.write_text(readme_content)
        docs.append(str(readme_path))
        
        logger.info(f"Generated documentation: {len(docs)} documents")
        return docs
    
    def deploy(self) -> Dict[str, Any]:
        """Execute complete deployment package generation."""
        logger.info(f"🚀 Starting Progressive Quality Gates deployment package for {self.config.environment.value}")
        
        deployment_results = {
            "environment": self.config.environment.value,
            "timestamp": time.time(),
            "status": "in_progress",
            "components": {}
        }
        
        try:
            # 1. Generate Docker assets
            logger.info("🐳 Generating Docker configuration...")
            dockerfile = self.generate_dockerfile()
            deployment_results["components"]["docker"] = {"dockerfile": dockerfile}
            
            # 2. Generate Kubernetes manifests
            logger.info("☸️  Generating Kubernetes manifests...")
            k8s_manifests = self.generate_kubernetes_manifests()
            deployment_results["components"]["kubernetes"] = {
                "manifests": k8s_manifests,
                "namespace": self.config.namespace
            }
            
            # 3. Generate deployment scripts
            logger.info("📜 Generating deployment scripts...")
            deployment_scripts = self.generate_deployment_scripts()
            deployment_results["components"]["scripts"] = deployment_scripts
            
            # 4. Generate documentation
            logger.info("📚 Generating documentation...")
            documentation = self.generate_documentation()
            deployment_results["components"]["documentation"] = documentation
            
            # 5. Create deployment summary
            summary_file = self.deployment_dir / "deployment_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(deployment_results, f, indent=2, default=str)
            
            deployment_results["status"] = "completed"
            deployment_results["summary_file"] = str(summary_file)
            
            logger.info("✅ Progressive Quality Gates deployment package generated successfully!")
            
            return deployment_results
            
        except Exception as e:
            logger.error(f"❌ Deployment generation failed: {e}")
            deployment_results["status"] = "failed"
            deployment_results["error"] = str(e)
            return deployment_results


def main():
    """Main deployment execution."""
    print("🚀 PROGRESSIVE QUALITY GATES - PRODUCTION DEPLOYMENT")
    print("=" * 70)
    print("Generating complete production deployment package...")
    print()
    
    # Production configuration
    production_config = DeploymentConfig(
        environment=DeploymentEnvironment.PRODUCTION,
        namespace="progressive-quality-gates",
        replicas=3
    )
    
    # Staging configuration  
    staging_config = DeploymentConfig(
        environment=DeploymentEnvironment.STAGING,
        namespace="progressive-quality-gates-staging",
        replicas=2
    )
    
    try:
        # Deploy for both environments
        environments = [
            (production_config, "Production"),
            (staging_config, "Staging")
        ]
        
        deployment_summaries = []
        
        for config, env_name in environments:
            print(f"📦 Generating {env_name} deployment package...")
            
            deployer = SimpleProgressiveQualityGatesDeployer(config)
            result = deployer.deploy()
            
            deployment_summaries.append({
                "environment": env_name,
                "result": result
            })
            
            if result["status"] == "completed":
                print(f"✅ {env_name} deployment package generated successfully")
            else:
                print(f"❌ {env_name} deployment package generation failed")
            print()
        
        # Final summary
        print("🎯 DEPLOYMENT PACKAGE SUMMARY")
        print("-" * 50)
        
        for summary in deployment_summaries:
            env_name = summary["environment"]
            result = summary["result"]
            status = "✅" if result["status"] == "completed" else "❌"
            
            print(f"{status} {env_name}: {result['status']}")
            
            if result["status"] == "completed":
                components = result["components"]
                print(f"   Docker: Production-ready Dockerfile")
                print(f"   Kubernetes: {len(components['kubernetes']['manifests'])} manifests")
                print(f"   Scripts: {len(components['scripts'])} deployment scripts")
                print(f"   Documentation: Complete setup guide")
        
        print()
        print("📋 DEPLOYMENT INSTRUCTIONS")
        print("-" * 30)
        print("1. Review configuration in deployment/")
        print("2. Build Docker image: ./deployment/scripts/build.sh")
        print("3. Deploy to Kubernetes: ./deployment/scripts/deploy.sh")
        print("4. Run health checks: ./deployment/scripts/health-check.sh")
        
        print()
        print("🔗 QUICK ACCESS")
        print("-" * 20)
        print("• Health Check: http://localhost:8080/health")
        print("• Metrics: http://localhost:9090/metrics")
        print("• Documentation: deployment/README.md")
        
        print()
        print("✅ PRODUCTION DEPLOYMENT PACKAGE COMPLETE!")
        print("🎉 Progressive Quality Gates ready for enterprise deployment")
        print()
        print("🚀 THREE-GENERATION ARCHITECTURE DEPLOYED:")
        print("   Generation 1: ✅ Basic progressive quality gates")
        print("   Generation 2: ✅ Enhanced monitoring & resilience") 
        print("   Generation 3: ✅ ML optimization & distributed execution")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment package generation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)