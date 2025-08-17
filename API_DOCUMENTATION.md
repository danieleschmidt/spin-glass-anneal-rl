# 🔌 Spin-Glass-Anneal-RL API Documentation

## Overview

The Spin-Glass-Anneal-RL API provides a RESTful interface for solving complex optimization problems using physics-inspired annealing algorithms. This API supports multi-agent scheduling, vehicle routing, resource allocation, and custom optimization problems.

## Base URL

```
Production: https://api.spin-glass-rl.com/v1
Development: http://localhost:8000/v1
```

## Authentication

The API uses JWT tokens for authentication:

```bash
curl -X POST /auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

Use the token in subsequent requests:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" /endpoint
```

---

## 🧮 Core Optimization API

### Create Optimization Job

**Endpoint**: `POST /optimization/jobs`

Create a new optimization job with problem specification.

**Request Body**:
```json
{
  "problem_type": "scheduling|routing|allocation|custom",
  "problem_data": {
    "n_tasks": 10,
    "n_agents": 3,
    "time_horizon": 100.0,
    "constraints": [
      {
        "type": "assignment",
        "penalty_weight": 100.0
      }
    ]
  },
  "optimization_config": {
    "algorithm": "simulated_annealing",
    "n_sweeps": 1000,
    "initial_temp": 10.0,
    "final_temp": 0.01,
    "gpu_enabled": true
  },
  "priority": 1
}
```

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "queued",
  "created_at": "2025-08-17T18:30:00Z",
  "estimated_completion": "2025-08-17T18:35:00Z"
}
```

**Example**:
```bash
curl -X POST /optimization/jobs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_type": "scheduling",
    "problem_data": {
      "n_tasks": 5,
      "n_agents": 2,
      "time_horizon": 50.0
    },
    "optimization_config": {
      "n_sweeps": 500,
      "gpu_enabled": false
    }
  }'
```

### Get Job Status

**Endpoint**: `GET /optimization/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "completed|running|failed|queued",
  "progress": 85.5,
  "created_at": "2025-08-17T18:30:00Z",
  "started_at": "2025-08-17T18:30:05Z",
  "completed_at": "2025-08-17T18:33:42Z",
  "execution_time": 217.3,
  "node_id": "worker_node_1"
}
```

### Get Job Results

**Endpoint**: `GET /optimization/jobs/{job_id}/results`

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "completed",
  "solution": {
    "objective_value": 42.5,
    "is_feasible": true,
    "variables": {
      "assignments": {
        "agent_0": [0, 2, 4],
        "agent_1": [1, 3]
      }
    },
    "metadata": {
      "makespan": 42.5,
      "total_cost": 127.3,
      "resource_utilization": 0.85
    }
  },
  "optimization_stats": {
    "initial_energy": -15.2,
    "final_energy": -42.5,
    "energy_improvement": 27.3,
    "convergence_sweep": 823,
    "acceptance_rate": 0.234
  }
}
```

### Cancel Job

**Endpoint**: `DELETE /optimization/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "job_12345",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

---

## 📊 Problem-Specific APIs

### Multi-Agent Scheduling

**Endpoint**: `POST /problems/scheduling`

Solve multi-agent scheduling problems with task assignments and resource constraints.

**Request Body**:
```json
{
  "tasks": [
    {
      "id": 0,
      "duration": 5.0,
      "priority": 1.0,
      "due_date": 20.0,
      "resource_requirements": {"cpu": 1, "memory": 2}
    }
  ],
  "agents": [
    {
      "id": 0,
      "cost_per_hour": 10.0,
      "capacity": {"cpu": 4, "memory": 8},
      "availability": [[0, 100]]
    }
  ],
  "objective": "makespan|cost|weighted",
  "constraints": {
    "assignment": {"type": "exactly_one", "penalty": 100.0},
    "capacity": {"type": "resource_limit", "penalty": 50.0},
    "time_windows": {"type": "due_dates", "penalty": 75.0}
  }
}
```

### Vehicle Routing Problem

**Endpoint**: `POST /problems/routing`

Solve vehicle routing problems with capacity and time window constraints.

**Request Body**:
```json
{
  "depot": {"x": 0, "y": 0},
  "customers": [
    {"id": 1, "x": 10, "y": 15, "demand": 5, "time_window": [8, 18]}
  ],
  "vehicles": [
    {"id": 1, "capacity": 20, "cost_per_km": 1.5}
  ],
  "objective": "distance|cost|time",
  "constraints": {
    "capacity": {"penalty": 100.0},
    "time_windows": {"penalty": 75.0}
  }
}
```

### Resource Allocation

**Endpoint**: `POST /problems/allocation`

Solve resource allocation problems with budget and capacity constraints.

**Request Body**:
```json
{
  "resources": [
    {"id": 1, "capacity": 100, "cost": 50}
  ],
  "demands": [
    {"id": 1, "amount": 25, "value": 100, "priority": 1}
  ],
  "budget": 500,
  "objective": "maximize_value|minimize_cost",
  "constraints": {
    "budget": {"penalty": 100.0},
    "capacity": {"penalty": 75.0}
  }
}
```

---

## ⚙️ Configuration API

### Get Available Algorithms

**Endpoint**: `GET /config/algorithms`

**Response**:
```json
{
  "algorithms": [
    {
      "name": "simulated_annealing",
      "description": "Classical simulated annealing with GPU acceleration",
      "parameters": {
        "n_sweeps": {"type": "int", "min": 1, "max": 1000000, "default": 1000},
        "initial_temp": {"type": "float", "min": 0.001, "max": 1000, "default": 10.0},
        "final_temp": {"type": "float", "min": 0.0001, "max": 10, "default": 0.01},
        "schedule_type": {"type": "enum", "values": ["linear", "exponential", "geometric"], "default": "geometric"}
      }
    },
    {
      "name": "parallel_tempering",
      "description": "Parallel tempering with replica exchange",
      "parameters": {
        "n_replicas": {"type": "int", "min": 2, "max": 100, "default": 8},
        "temp_min": {"type": "float", "min": 0.01, "max": 10, "default": 0.1},
        "temp_max": {"type": "float", "min": 1, "max": 100, "default": 10.0}
      }
    }
  ]
}
```

### Get System Configuration

**Endpoint**: `GET /config/system`

**Response**:
```json
{
  "version": "1.0.0",
  "capabilities": {
    "gpu_enabled": true,
    "gpu_memory_gb": 8,
    "cpu_cores": 16,
    "max_problem_size": 100000,
    "distributed_computing": true
  },
  "limits": {
    "max_jobs_per_user": 10,
    "max_job_duration": 3600,
    "max_file_size_mb": 100
  }
}
```

---

## 📈 Monitoring API

### System Health

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-17T18:30:00Z",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "workers": "healthy",
    "gpu": "available"
  },
  "metrics": {
    "active_jobs": 5,
    "queued_jobs": 2,
    "completed_jobs_24h": 1247,
    "error_rate_24h": 0.001
  }
}
```

### System Metrics

**Endpoint**: `GET /metrics`

Returns Prometheus-compatible metrics:

```
# HELP sgrl_jobs_total Total number of jobs processed
# TYPE sgrl_jobs_total counter
sgrl_jobs_total{status="completed"} 1247
sgrl_jobs_total{status="failed"} 3

# HELP sgrl_job_duration_seconds Job execution time
# TYPE sgrl_job_duration_seconds histogram
sgrl_job_duration_seconds_bucket{le="1"} 45
sgrl_job_duration_seconds_bucket{le="10"} 523
sgrl_job_duration_seconds_bucket{le="60"} 1180
```

### Performance Stats

**Endpoint**: `GET /stats/performance`

**Response**:
```json
{
  "time_range": "24h",
  "job_statistics": {
    "total_jobs": 1250,
    "successful_jobs": 1247,
    "failed_jobs": 3,
    "success_rate": 0.9976,
    "avg_execution_time": 45.3,
    "median_execution_time": 12.7
  },
  "system_performance": {
    "cpu_utilization": 65.4,
    "memory_utilization": 42.1,
    "gpu_utilization": 78.9,
    "network_io_mbps": 125.3
  },
  "algorithm_performance": {
    "simulated_annealing": {
      "avg_time": 23.1,
      "success_rate": 0.998,
      "avg_quality": 0.89
    }
  }
}
```

---

## 🔐 User Management API

### Create User

**Endpoint**: `POST /users` (Admin only)

**Request Body**:
```json
{
  "username": "researcher",
  "email": "researcher@university.edu",
  "password": "secure_password",
  "role": "user|admin|readonly",
  "limits": {
    "max_jobs_per_hour": 100,
    "max_cpu_hours_per_day": 24
  }
}
```

### Get User Info

**Endpoint**: `GET /users/me`

**Response**:
```json
{
  "user_id": "user_12345",
  "username": "researcher",
  "email": "researcher@university.edu",
  "role": "user",
  "created_at": "2025-08-01T10:00:00Z",
  "usage_stats": {
    "jobs_submitted": 156,
    "cpu_hours_used": 23.4,
    "storage_used_mb": 450.2
  },
  "limits": {
    "max_jobs_per_hour": 100,
    "max_cpu_hours_per_day": 24
  }
}
```

---

## 📁 File Management API

### Upload Problem Data

**Endpoint**: `POST /files/upload`

Upload problem data files (CSV, JSON, or custom formats).

**Request**: Multipart form data
```bash
curl -X POST /files/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@problem_data.csv" \
  -F "type=problem_data" \
  -F "description=Customer locations for VRP"
```

**Response**:
```json
{
  "file_id": "file_12345",
  "filename": "problem_data.csv",
  "size_bytes": 15420,
  "type": "problem_data",
  "uploaded_at": "2025-08-17T18:30:00Z",
  "url": "/files/file_12345/download"
}
```

### Download Results

**Endpoint**: `GET /files/{file_id}/download`

Downloads result files in various formats (JSON, CSV, Excel).

---

## 🚀 Batch Processing API

### Submit Batch Job

**Endpoint**: `POST /batch/jobs`

Submit multiple optimization problems for batch processing.

**Request Body**:
```json
{
  "jobs": [
    {
      "name": "problem_1",
      "problem_type": "scheduling",
      "problem_data": {...},
      "config": {...}
    },
    {
      "name": "problem_2", 
      "problem_type": "routing",
      "problem_data": {...},
      "config": {...}
    }
  ],
  "batch_config": {
    "parallel_execution": true,
    "max_concurrent": 4,
    "timeout_per_job": 300
  }
}
```

**Response**:
```json
{
  "batch_id": "batch_12345",
  "status": "queued",
  "job_count": 2,
  "estimated_completion": "2025-08-17T19:00:00Z"
}
```

---

## 📊 Analytics API

### Algorithm Comparison

**Endpoint**: `POST /analytics/compare`

Compare performance of different algorithms on the same problem.

**Request Body**:
```json
{
  "problem_data": {...},
  "algorithms": [
    {"name": "simulated_annealing", "config": {...}},
    {"name": "parallel_tempering", "config": {...}}
  ],
  "metrics": ["solution_quality", "execution_time", "convergence_rate"]
}
```

### Performance Benchmarks

**Endpoint**: `GET /analytics/benchmarks`

Get performance benchmarks for different problem sizes and types.

**Response**:
```json
{
  "benchmarks": [
    {
      "problem_type": "scheduling",
      "problem_size": 100,
      "algorithm": "simulated_annealing",
      "avg_time": 12.3,
      "avg_quality": 0.89,
      "sample_size": 50
    }
  ]
}
```

---

## 🔧 Advanced Features

### Real-time Problem Solving

**WebSocket Endpoint**: `ws://api.spin-glass-rl.com/v1/ws/solve`

Connect to real-time optimization with progress updates:

```javascript
const ws = new WebSocket('ws://api.spin-glass-rl.com/v1/ws/solve');

ws.send(JSON.stringify({
  "problem_type": "scheduling",
  "problem_data": {...},
  "stream_progress": true
}));

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log(`Progress: ${update.progress}%`);
  console.log(`Current energy: ${update.current_energy}`);
};
```

### Custom Algorithms

**Endpoint**: `POST /algorithms/custom`

Upload and register custom optimization algorithms:

```json
{
  "name": "my_algorithm",
  "description": "Custom genetic algorithm variant",
  "code": "base64_encoded_python_code",
  "parameters": {
    "population_size": {"type": "int", "default": 100},
    "mutation_rate": {"type": "float", "default": 0.01}
  }
}
```

---

## 📝 Error Handling

### Error Response Format

All API errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid problem configuration",
    "details": {
      "field": "n_sweeps",
      "issue": "Must be between 1 and 1000000"
    },
    "request_id": "req_12345",
    "timestamp": "2025-08-17T18:30:00Z"
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Invalid request parameters | 400 |
| `AUTHENTICATION_ERROR` | Invalid or missing token | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `OPTIMIZATION_ERROR` | Problem solving failed | 422 |
| `SYSTEM_ERROR` | Internal server error | 500 |
| `SERVICE_UNAVAILABLE` | System maintenance | 503 |

---

## 🔗 SDKs and Libraries

### Python SDK

```python
from sgrl_client import SpinGlassClient

client = SpinGlassClient(
    api_url="https://api.spin-glass-rl.com/v1",
    api_key="your_api_key"
)

# Submit optimization job
job = client.optimize.scheduling(
    n_tasks=10,
    n_agents=3,
    algorithm="simulated_annealing",
    n_sweeps=1000
)

# Wait for results
result = job.wait_for_completion(timeout=300)
print(f"Optimal makespan: {result.solution.metadata['makespan']}")
```

### JavaScript SDK

```javascript
import { SpinGlassClient } from '@terragon/sgrl-client';

const client = new SpinGlassClient({
  apiUrl: 'https://api.spin-glass-rl.com/v1',
  apiKey: 'your_api_key'
});

// Submit job
const job = await client.optimize.routing({
  customers: [...],
  vehicles: [...],
  algorithm: 'simulated_annealing'
});

// Get results
const result = await job.getResults();
console.log('Total distance:', result.solution.objective_value);
```

---

## 📚 Examples and Tutorials

### Complete Example: Multi-Agent Scheduling

```bash
# 1. Authenticate
TOKEN=$(curl -X POST /auth/login \
  -d '{"username":"demo","password":"demo"}' | jq -r .access_token)

# 2. Submit problem
JOB_ID=$(curl -X POST /optimization/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "problem_type": "scheduling",
    "problem_data": {
      "n_tasks": 8,
      "n_agents": 3,
      "time_horizon": 100.0
    },
    "optimization_config": {
      "n_sweeps": 2000,
      "initial_temp": 10.0,
      "final_temp": 0.01
    }
  }' | jq -r .job_id)

# 3. Monitor progress
while true; do
  STATUS=$(curl -H "Authorization: Bearer $TOKEN" \
    /optimization/jobs/$JOB_ID | jq -r .status)
  echo "Status: $STATUS"
  [[ "$STATUS" == "completed" ]] && break
  sleep 5
done

# 4. Get results
curl -H "Authorization: Bearer $TOKEN" \
  /optimization/jobs/$JOB_ID/results | jq .solution
```

For more examples and tutorials, visit: https://docs.spin-glass-rl.com/tutorials

---

## 🆘 Support

- **API Documentation**: https://api.spin-glass-rl.com/docs
- **Status Page**: https://status.spin-glass-rl.com
- **Support Email**: api-support@terragonlabs.com
- **Community Forum**: https://community.spin-glass-rl.com

**API Version**: v1.0.0  
**Last Updated**: August 17, 2025