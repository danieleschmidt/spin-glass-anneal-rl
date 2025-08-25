#!/bin/bash
set -e

ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="progressive-quality-gates-staging"

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
