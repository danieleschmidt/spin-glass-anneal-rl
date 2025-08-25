#!/bin/bash
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
