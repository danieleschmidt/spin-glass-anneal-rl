#!/bin/bash
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
