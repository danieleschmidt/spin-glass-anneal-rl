#!/bin/bash

# Spin-Glass-Anneal-RL Deployment Script
# Autonomous deployment script for various environments

set -e

# Configuration
PROJECT_NAME="spin-glass-anneal-rl"
DOCKER_IMAGE="terragonlabs/spin-glass-anneal-rl"
VERSION=$(python3 -c "import spin_glass_rl; print(spin_glass_rl.__version__)" 2>/dev/null || echo "dev")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_help() {
    cat << EOF
Spin-Glass-Anneal-RL Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    local       Deploy locally using Docker Compose
    k8s         Deploy to Kubernetes cluster
    aws         Deploy to AWS ECS/EKS
    azure       Deploy to Azure Container Instances/AKS
    gcp         Deploy to Google Cloud Run/GKE
    test        Run deployment tests
    cleanup     Clean up deployment resources
    help        Show this help message

Options:
    -e, --environment ENV    Target environment (dev, staging, prod)
    -v, --version VERSION    Version to deploy (default: current)
    -f, --force             Force deployment without confirmation
    -c, --config FILE       Custom configuration file
    --dry-run               Show what would be deployed
    --skip-tests            Skip pre-deployment tests
    --no-gpu                Deploy without GPU support
    
Examples:
    $0 local -e dev                    # Local development deployment
    $0 k8s -e prod -v 1.2.3           # Production Kubernetes deployment
    $0 aws -e staging --dry-run       # Dry run AWS deployment
    $0 test                           # Run deployment tests

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="dev"
    FORCE=false
    DRY_RUN=false
    SKIP_TESTS=false
    NO_GPU=false
    CONFIG_FILE=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --no-gpu)
                NO_GPU=true
                shift
                ;;
            -h|--help|help)
                show_help
                exit 0
                ;;
            *)
                COMMAND="$1"
                shift
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check for NVIDIA GPU support if needed
    if [[ "$NO_GPU" == false ]]; then
        if command -v nvidia-smi &> /dev/null; then
            log_info "NVIDIA GPU detected"
        else
            log_warning "No NVIDIA GPU detected, will deploy CPU-only version"
            NO_GPU=true
        fi
    fi
    
    log_success "Environment validation completed"
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        log_warning "Skipping pre-deployment tests"
        return
    fi
    
    log_info "Running pre-deployment tests..."
    
    # Security scan
    if [[ -f "security_scan.py" ]]; then
        python3 security_scan.py || {
            log_error "Security scan failed"
            exit 1
        }
    fi
    
    # Build test
    docker build -t "${PROJECT_NAME}:test" --target test . || {
        log_error "Test build failed"
        exit 1
    }
    
    log_success "Pre-deployment tests completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    local tag="${DOCKER_IMAGE}:${VERSION}"
    local latest_tag="${DOCKER_IMAGE}:latest"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would build: $tag"
        return
    fi
    
    # Build production image
    if [[ "$NO_GPU" == true ]]; then
        docker build -t "$tag" --target runtime .
    else
        docker build -t "$tag" --target production .
    fi
    
    # Tag as latest for development
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        docker tag "$tag" "$latest_tag"
    fi
    
    log_success "Docker images built successfully"
}

# Deploy to local environment using Docker Compose
deploy_local() {
    log_info "Deploying to local environment..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would run: docker-compose -f $compose_file up -d"
        return
    fi
    
    # Set environment variables
    export SPIN_GLASS_VERSION="$VERSION"
    export SPIN_GLASS_ENV="$ENVIRONMENT"
    
    # Deploy with Docker Compose
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8888/health &> /dev/null; then
        log_success "Local deployment completed successfully"
        log_info "Services available at:"
        log_info "  - API: http://localhost:8888"
        log_info "  - Metrics: http://localhost:9090"
        log_info "  - Grafana: http://localhost:3000"
    else
        log_error "Health check failed"
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes cluster..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    local namespace="spin-glass-${ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would deploy to namespace: $namespace"
        return
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy using Helm if available, otherwise use kubectl
    if command -v helm &> /dev/null && [[ -d "deployment/helm" ]]; then
        log_info "Using Helm for deployment..."
        helm upgrade --install "$PROJECT_NAME" deployment/helm/ \
            --namespace "$namespace" \
            --set image.tag="$VERSION" \
            --set environment="$ENVIRONMENT"
    else
        log_info "Using kubectl for deployment..."
        # Apply Kubernetes manifests
        for manifest in deployment/k8s/*.yaml; do
            if [[ -f "$manifest" ]]; then
                sed "s/{{VERSION}}/$VERSION/g; s/{{ENVIRONMENT}}/$ENVIRONMENT/g" "$manifest" | \
                kubectl apply -n "$namespace" -f -
            fi
        done
    fi
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available deployment/"$PROJECT_NAME" \
        --namespace="$namespace" --timeout=300s
    
    log_success "Kubernetes deployment completed"
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would deploy to AWS ECS/EKS"
        return
    fi
    
    # Deploy using AWS CDK or CloudFormation
    if [[ -f "deployment/aws/cdk.json" ]]; then
        log_info "Using AWS CDK for deployment..."
        cd deployment/aws
        npx cdk deploy --require-approval never
        cd ../..
    elif [[ -f "deployment/aws/cloudformation.yaml" ]]; then
        log_info "Using CloudFormation for deployment..."
        aws cloudformation deploy \
            --template-file deployment/aws/cloudformation.yaml \
            --stack-name "$PROJECT_NAME-$ENVIRONMENT" \
            --parameter-overrides Version="$VERSION" Environment="$ENVIRONMENT" \
            --capabilities CAPABILITY_IAM
    else
        log_error "No AWS deployment configuration found"
        exit 1
    fi
    
    log_success "AWS deployment completed"
}

# Deploy to Azure
deploy_azure() {
    log_info "Deploying to Azure..."
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would deploy to Azure"
        return
    fi
    
    # Deploy using Azure Resource Manager templates
    if [[ -f "deployment/azure/template.json" ]]; then
        az deployment group create \
            --resource-group "$PROJECT_NAME-$ENVIRONMENT" \
            --template-file deployment/azure/template.json \
            --parameters version="$VERSION" environment="$ENVIRONMENT"
    else
        log_error "No Azure deployment configuration found"
        exit 1
    fi
    
    log_success "Azure deployment completed"
}

# Deploy to Google Cloud
deploy_gcp() {
    log_info "Deploying to Google Cloud..."
    
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud CLI is not installed"
        exit 1
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY-RUN] Would deploy to Google Cloud"
        return
    fi
    
    # Deploy using Cloud Run or GKE
    if [[ -f "deployment/gcp/cloudrun.yaml" ]]; then
        log_info "Deploying to Cloud Run..."
        gcloud run services replace deployment/gcp/cloudrun.yaml \
            --region=us-central1
    else
        log_error "No GCP deployment configuration found"
        exit 1
    fi
    
    log_success "GCP deployment completed"
}

# Run deployment tests
test_deployment() {
    log_info "Running deployment tests..."
    
    # Build test image
    docker build -t "${PROJECT_NAME}:test" --target test .
    
    # Run tests
    docker run --rm "${PROJECT_NAME}:test"
    
    log_success "Deployment tests completed"
}

# Cleanup deployment resources
cleanup_deployment() {
    log_info "Cleaning up deployment resources..."
    
    if [[ "$FORCE" == false ]]; then
        read -p "Are you sure you want to cleanup all resources? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Cleanup cancelled"
            exit 0
        fi
    fi
    
    # Local cleanup
    if docker-compose ps &> /dev/null; then
        docker-compose down -v
    fi
    
    # Remove images
    docker images "${DOCKER_IMAGE}" -q | xargs -r docker rmi -f
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    parse_args "$@"
    
    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
    
    log_info "Starting deployment: $COMMAND (env: $ENVIRONMENT, version: $VERSION)"
    
    case "$COMMAND" in
        local)
            validate_environment
            run_tests
            build_images
            deploy_local
            ;;
        k8s)
            validate_environment
            run_tests
            build_images
            deploy_k8s
            ;;
        aws)
            validate_environment
            run_tests
            build_images
            deploy_aws
            ;;
        azure)
            validate_environment
            run_tests
            build_images
            deploy_azure
            ;;
        gcp)
            validate_environment
            run_tests
            build_images
            deploy_gcp
            ;;
        test)
            test_deployment
            ;;
        cleanup)
            cleanup_deployment
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"