# 🚀 Spin-Glass-Anneal-RL Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Spin-Glass-Anneal-RL in production environments with enterprise-grade reliability, security, and scalability.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- 4 CPU cores
- 8 GB RAM  
- 50 GB SSD storage
- Python 3.9+

**Recommended Production:**
- 16+ CPU cores
- 32+ GB RAM
- 500+ GB NVMe SSD
- GPU support (NVIDIA with CUDA 12.x)
- High-speed network (10 Gbps+)

### Software Dependencies

```bash
# System packages
sudo apt update
sudo apt install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    build-essential \
    curl \
    git \
    htop \
    nginx \
    redis-server \
    postgresql \
    docker.io \
    docker-compose

# NVIDIA drivers (if using GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-2
```

## 🏗️ Production Architecture

### Single-Node Deployment

```
┌─────────────────────────────────────────┐
│              Load Balancer              │
│                (nginx)                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│           Application Layer             │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  FastAPI    │  │  Worker Pool    │   │
│  │  Server     │  │  (Celery)       │   │
│  └─────────────┘  └─────────────────┘   │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│            Data Layer                   │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Redis      │  │  PostgreSQL     │   │
│  │  (Cache)    │  │  (Metadata)     │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
```

### Multi-Node Cluster

```
┌─────────────────────────────────────────┐
│            Load Balancer                │
│          (nginx + HAProxy)              │
└──────────┬────────────┬─────────────────┘
           │            │
┌──────────▼──┐  ┌──────▼──────┐
│  Node 1     │  │   Node 2    │  ... Node N
│  ┌────────┐ │  │  ┌────────┐ │
│  │FastAPI │ │  │  │FastAPI │ │
│  └────────┘ │  │  └────────┘ │
│  ┌────────┐ │  │  ┌────────┐ │
│  │Workers │ │  │  │Workers │ │
│  └────────┘ │  │  └────────┘ │
└─────────────┘  └─────────────┘
           │            │
┌──────────▼────────────▼─────────────────┐
│         Shared Services                 │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Redis      │  │  PostgreSQL     │   │
│  │  Cluster    │  │  Cluster        │   │
│  └─────────────┘  └─────────────────┘   │
└─────────────────────────────────────────┘
```

## 🔧 Installation & Setup

### 1. Environment Preparation

```bash
# Create production user
sudo useradd -m -s /bin/bash sgrl
sudo usermod -aG sudo sgrl
sudo -u sgrl -i

# Create application directory
sudo mkdir -p /opt/spin-glass-rl
sudo chown sgrl:sgrl /opt/spin-glass-rl
cd /opt/spin-glass-rl

# Clone repository
git clone https://github.com/terragonlabs/spin-glass-anneal-rl.git .
git checkout production  # Use production branch
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel

# Install production dependencies
pip install -e ".[production]"

# Install additional production tools
pip install \
    gunicorn \
    uvicorn[standard] \
    celery[redis] \
    psycopg2-binary \
    prometheus-client \
    structlog
```

### 3. Configuration

```bash
# Create configuration directory
mkdir -p /opt/spin-glass-rl/config

# Production configuration
cat > /opt/spin-glass-rl/config/production.yaml << 'EOF'
# Production Configuration
environment: production
debug: false

# Application settings
app:
  name: "Spin-Glass-Anneal-RL"
  version: "1.0.0"
  workers: 16
  max_requests: 1000
  max_requests_jitter: 100

# Database configuration
database:
  url: "postgresql://sgrl:${POSTGRES_PASSWORD}@localhost/sgrl_prod"
  pool_size: 20
  max_overflow: 30
  pool_pre_ping: true

# Redis configuration  
redis:
  url: "redis://localhost:6379/0"
  max_connections: 100

# Security settings
security:
  secret_key: "${SECRET_KEY}"
  cors_origins:
    - "https://your-domain.com"
  rate_limiting:
    enabled: true
    per_minute: 60
    per_hour: 1000

# Optimization settings
optimization:
  gpu_enabled: true
  gpu_memory_fraction: 0.8
  cpu_workers: 16
  cache_size_mb: 1024
  distributed_workers: 4

# Monitoring
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_port: 8080
  log_level: "INFO"
  
# Performance
performance:
  enable_jit: true
  memory_pool_size: "4GB"
  thread_pool_size: 32
EOF
```

### 4. Database Setup

```bash
# PostgreSQL setup
sudo -u postgres createuser -s sgrl
sudo -u postgres createdb sgrl_prod -O sgrl

# Set password
sudo -u postgres psql -c "ALTER USER sgrl PASSWORD 'your_secure_password';"

# Initialize database schema
python -c "
from spin_glass_rl.database import init_database
init_database('postgresql://sgrl:your_secure_password@localhost/sgrl_prod')
"
```

### 5. Service Configuration

#### systemd Service Files

```bash
# API Service
sudo tee /etc/systemd/system/sgrl-api.service << 'EOF'
[Unit]
Description=Spin-Glass-Anneal-RL API Server
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=exec
User=sgrl
Group=sgrl
WorkingDirectory=/opt/spin-glass-rl
Environment=PATH=/opt/spin-glass-rl/venv/bin
Environment=SGRL_CONFIG=/opt/spin-glass-rl/config/production.yaml
ExecStart=/opt/spin-glass-rl/venv/bin/gunicorn \
    --bind 127.0.0.1:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --timeout 300 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile /var/log/sgrl/access.log \
    --error-logfile /var/log/sgrl/error.log \
    spin_glass_rl.api:app
    
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

# Worker Service
sudo tee /etc/systemd/system/sgrl-workers.service << 'EOF'
[Unit]
Description=Spin-Glass-Anneal-RL Workers
After=network.target redis.service
Requires=redis.service

[Service]
Type=exec
User=sgrl
Group=sgrl
WorkingDirectory=/opt/spin-glass-rl
Environment=PATH=/opt/spin-glass-rl/venv/bin
Environment=SGRL_CONFIG=/opt/spin-glass-rl/config/production.yaml
ExecStart=/opt/spin-glass-rl/venv/bin/celery worker \
    --app=spin_glass_rl.workers \
    --loglevel=info \
    --concurrency=8 \
    --max-tasks-per-child=100 \
    --logfile=/var/log/sgrl/workers.log
    
Restart=always
RestartSec=3
KillSignal=SIGTERM

[Install]
WantedBy=multi-user.target
EOF

# Create log directory
sudo mkdir -p /var/log/sgrl
sudo chown sgrl:sgrl /var/log/sgrl
```

#### nginx Configuration

```bash
sudo tee /etc/nginx/sites-available/sgrl << 'EOF'
upstream sgrl_app {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 300s;
    
    # File upload size
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://sgrl_app;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_redirect off;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://sgrl_app/health;
    }
    
    # Metrics endpoint (restrict access)
    location /metrics {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:9090/metrics;
    }
    
    # Static files
    location /static/ {
        alias /opt/spin-glass-rl/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/sgrl /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 🔒 Security Configuration

### 1. Environment Variables

```bash
# Create secure environment file
sudo tee /opt/spin-glass-rl/.env << 'EOF'
# Database
POSTGRES_PASSWORD=your_very_secure_database_password

# Application
SECRET_KEY=your_very_long_secret_key_here_use_64_chars_minimum
JWT_SECRET=another_secure_secret_for_jwt_tokens

# Redis
REDIS_PASSWORD=secure_redis_password

# External APIs (if needed)
# EXTERNAL_API_KEY=your_api_key_here

# Monitoring
PROMETHEUS_TOKEN=monitoring_token_here
EOF

# Secure the environment file
sudo chmod 600 /opt/spin-glass-rl/.env
sudo chown sgrl:sgrl /opt/spin-glass-rl/.env
```

### 2. Firewall Configuration

```bash
# Configure UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow internal services (adjust network as needed)
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL
sudo ufw allow from 10.0.0.0/8 to any port 6379  # Redis
sudo ufw allow from 10.0.0.0/8 to any port 9090  # Metrics

# Enable firewall
sudo ufw --force enable
```

### 3. SSL/TLS Setup

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring & Observability

### 1. Prometheus Configuration

```bash
# Install Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xzf prometheus-2.45.0.linux-amd64.tar.gz
sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus
sudo useradd --no-create-home --shell /bin/false prometheus
sudo chown -R prometheus:prometheus /opt/prometheus

# Configuration
sudo tee /opt/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'sgrl-api'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'nginx-exporter'
    static_configs:
      - targets: ['localhost:9113']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093
EOF

# Alert rules
sudo tee /opt/prometheus/alert_rules.yml << 'EOF'
groups:
  - name: sgrl-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
EOF

# Systemd service
sudo tee /etc/systemd/system/prometheus.service << 'EOF'
[Unit]
Description=Prometheus
After=network.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/opt/prometheus/prometheus \
    --config.file=/opt/prometheus/prometheus.yml \
    --storage.tsdb.path=/opt/prometheus/data \
    --web.console.templates=/opt/prometheus/consoles \
    --web.console.libraries=/opt/prometheus/console_libraries \
    --web.listen-address=0.0.0.0:9091 \
    --web.enable-lifecycle

[Install]
WantedBy=multi-user.target
EOF
```

### 2. Grafana Dashboard

```bash
# Install Grafana
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
echo "deb https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt update
sudo apt install -y grafana

# Configure Grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

# Access: http://your-server:3000 (admin/admin)
```

## 🚀 Deployment Process

### 1. Automated Deployment Script

```bash
#!/bin/bash
# /opt/spin-glass-rl/deploy.sh

set -euo pipefail

# Configuration
DEPLOY_USER="sgrl"
APP_DIR="/opt/spin-glass-rl"
BACKUP_DIR="/opt/sgrl-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check disk space
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 80 ]; then
        error "Disk usage is ${DISK_USAGE}%. Free up space before deploying."
    fi
    
    # Check memory
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    if [ "$MEMORY_USAGE" -gt 90 ]; then
        warn "Memory usage is ${MEMORY_USAGE}%. Consider restarting services."
    fi
    
    # Check services
    for service in postgresql redis-server nginx; do
        if ! systemctl is-active --quiet $service; then
            error "Service $service is not running"
        fi
    done
    
    log "Pre-deployment checks passed"
}

# Backup current deployment
backup_current() {
    log "Creating backup..."
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup application
    tar -czf "$BACKUP_DIR/sgrl_app_$TIMESTAMP.tar.gz" -C "$APP_DIR" \
        --exclude=venv \
        --exclude=.git \
        --exclude=__pycache__ \
        .
    
    # Backup database
    sudo -u postgres pg_dump sgrl_prod | gzip > "$BACKUP_DIR/sgrl_db_$TIMESTAMP.sql.gz"
    
    log "Backup completed: $BACKUP_DIR/sgrl_*_$TIMESTAMP.*"
}

# Deploy new version
deploy() {
    log "Starting deployment..."
    
    cd "$APP_DIR"
    
    # Pull latest code
    git fetch origin
    git checkout production
    git pull origin production
    
    # Update dependencies
    source venv/bin/activate
    pip install --upgrade -e ".[production]"
    
    # Run database migrations (if any)
    python -m spin_glass_rl.database migrate
    
    # Collect static files
    python -m spin_glass_rl.static collect
    
    log "Code deployment completed"
}

# Restart services
restart_services() {
    log "Restarting services..."
    
    # Restart application services
    sudo systemctl restart sgrl-workers
    sleep 5
    sudo systemctl restart sgrl-api
    
    # Reload nginx
    sudo nginx -t && sudo systemctl reload nginx
    
    log "Services restarted"
}

# Post-deployment verification
verify_deployment() {
    log "Verifying deployment..."
    
    # Wait for services to start
    sleep 10
    
    # Check service status
    for service in sgrl-api sgrl-workers; do
        if ! systemctl is-active --quiet $service; then
            error "Service $service failed to start"
        fi
    done
    
    # Health check
    if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
        error "Health check failed"
    fi
    
    # Run smoke tests
    cd "$APP_DIR"
    source venv/bin/activate
    python -m pytest tests/smoke/ -v
    
    log "Deployment verification completed"
}

# Cleanup old backups
cleanup() {
    log "Cleaning up old backups..."
    
    # Keep only last 7 days of backups
    find "$BACKUP_DIR" -name "sgrl_*" -mtime +7 -delete
    
    log "Cleanup completed"
}

# Main deployment flow
main() {
    log "Starting Spin-Glass-Anneal-RL deployment"
    
    pre_deployment_checks
    backup_current
    deploy
    restart_services
    verify_deployment
    cleanup
    
    log "Deployment completed successfully!"
}

# Run main function
main "$@"
```

### 2. Rolling Deployment (Zero Downtime)

```bash
#!/bin/bash
# /opt/spin-glass-rl/rolling_deploy.sh

# Blue-Green deployment for zero downtime
BLUE_PORT=8000
GREEN_PORT=8001

# Current active deployment
CURRENT=$(curl -s http://localhost:$BLUE_PORT/health && echo "blue" || echo "green")
NEW_PORT=$([[ $CURRENT == "blue" ]] && echo $GREEN_PORT || echo $BLUE_PORT)

log "Current active: $CURRENT, deploying to: $([[ $NEW_PORT == $BLUE_PORT ]] && echo "blue" || echo "green")"

# Deploy to inactive slot
deploy_to_slot $NEW_PORT

# Switch traffic
switch_traffic $NEW_PORT

# Verify and cleanup
verify_new_deployment $NEW_PORT
```

## 📈 Scaling & Performance

### 1. Horizontal Scaling

```bash
# Add new compute nodes
ansible-playbook -i inventory/production add_compute_nodes.yml

# Scale workers
sudo systemctl edit sgrl-workers
# Add:
# [Service]
# ExecStart=
# ExecStart=/opt/spin-glass-rl/venv/bin/celery worker --concurrency=16
```

### 2. Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

SELECT pg_reload_conf();
```

### 3. Redis Optimization

```bash
# Redis configuration
sudo tee -a /etc/redis/redis.conf << 'EOF'
# Memory optimization
maxmemory 4gb
maxmemory-policy allkeys-lru

# Performance
save 900 1
save 300 10
save 60 10000

# Network
tcp-keepalive 300
timeout 0
EOF

sudo systemctl restart redis-server
```

## 🔧 Maintenance & Operations

### 1. Health Checks

```bash
#!/bin/bash
# /opt/spin-glass-rl/health_check.sh

# Comprehensive health check script
check_api() {
    curl -f -s http://localhost:8000/health || return 1
}

check_database() {
    sudo -u postgres psql -d sgrl_prod -c "SELECT 1;" > /dev/null || return 1
}

check_redis() {
    redis-cli ping | grep -q PONG || return 1
}

check_workers() {
    pgrep -f "celery worker" > /dev/null || return 1
}

check_disk_space() {
    [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -lt 90 ] || return 1
}

# Run all checks
for check in api database redis workers disk_space; do
    if check_$check; then
        echo "$check: OK"
    else
        echo "$check: FAILED"
        exit 1
    fi
done

echo "All health checks passed"
```

### 2. Log Management

```bash
# Configure logrotate
sudo tee /etc/logrotate.d/sgrl << 'EOF'
/var/log/sgrl/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 sgrl sgrl
    postrotate
        systemctl reload sgrl-api sgrl-workers
    endscript
}
EOF
```

### 3. Backup Strategy

```bash
#!/bin/bash
# /opt/spin-glass-rl/backup.sh

# Full backup script
BACKUP_DIR="/opt/sgrl-backups"
S3_BUCKET="your-backup-bucket"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Database backup
sudo -u postgres pg_dump sgrl_prod | gzip > "$BACKUP_DIR/db_$TIMESTAMP.sql.gz"

# Application backup
tar -czf "$BACKUP_DIR/app_$TIMESTAMP.tar.gz" -C /opt/spin-glass-rl .

# Upload to S3
aws s3 cp "$BACKUP_DIR/db_$TIMESTAMP.sql.gz" "s3://$S3_BUCKET/db/"
aws s3 cp "$BACKUP_DIR/app_$TIMESTAMP.tar.gz" "s3://$S3_BUCKET/app/"

# Cleanup local backups older than 7 days
find "$BACKUP_DIR" -mtime +7 -delete
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   free -h
   ps aux --sort=-%mem | head -10
   
   # Restart workers if needed
   sudo systemctl restart sgrl-workers
   ```

2. **Database Connection Issues**
   ```bash
   # Check connections
   sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Kill idle connections
   sudo -u postgres psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < now() - interval '5 minutes';"
   ```

3. **Performance Issues**
   ```bash
   # Check system load
   htop
   iostat -x 1
   
   # Check application metrics
   curl http://localhost:9090/metrics
   ```

### Emergency Procedures

1. **Service Recovery**
   ```bash
   # Emergency restart
   sudo systemctl restart sgrl-api sgrl-workers nginx postgresql redis-server
   ```

2. **Rollback Deployment**
   ```bash
   # Restore from backup
   cd /opt/spin-glass-rl
   tar -xzf /opt/sgrl-backups/sgrl_app_TIMESTAMP.tar.gz
   sudo systemctl restart sgrl-api sgrl-workers
   ```

## 📞 Support & Contacts

- **Emergency Contact**: ops@terragonlabs.com
- **Documentation**: https://docs.terragonlabs.com/sgrl
- **Monitoring Dashboard**: https://monitoring.your-domain.com
- **Status Page**: https://status.your-domain.com

---

## ✅ Deployment Checklist

- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Configuration files updated
- [ ] Database initialized
- [ ] Services configured and started
- [ ] SSL certificates installed
- [ ] Firewall configured
- [ ] Monitoring deployed
- [ ] Backups configured
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team notified

**Deployment Date**: ___________  
**Deployed By**: ___________  
**Version**: ___________  
**Notes**: ___________