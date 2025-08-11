"""Production deployment configuration and management."""

import os
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import json

from spin_glass_rl.annealing.temperature_scheduler import ScheduleType
from spin_glass_rl.rl_integration.reward_shaping import RewardType


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "spin_glass_rl"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 20
    connection_timeout: int = 30
    query_timeout: int = 300


@dataclass
class RedisConfig:
    """Redis configuration for caching and queuing."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    ssl: bool = False
    connection_pool_size: int = 50
    socket_timeout: int = 30
    socket_connect_timeout: int = 30


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-change-this"
    jwt_secret_key: str = "jwt-secret-key-change-this"
    jwt_expiration_hours: int = 24
    api_rate_limit: str = "1000/hour"
    allowed_hosts: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    enable_https_redirect: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    enable_grafana: bool = True
    grafana_port: int = 3000
    
    # Health check settings
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # Alert thresholds
    cpu_alert_threshold: float = 80.0
    memory_alert_threshold: float = 85.0
    disk_alert_threshold: float = 90.0
    gpu_alert_threshold: float = 90.0
    
    # Notification settings
    slack_webhook_url: Optional[str] = None
    email_alerts: List[str] = field(default_factory=list)
    pagerduty_key: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Default optimization parameters."""
    # Annealing settings
    default_n_sweeps: int = 5000
    default_initial_temp: float = 10.0
    default_final_temp: float = 0.01
    default_schedule_type: ScheduleType = ScheduleType.GEOMETRIC
    
    # Parallel tempering
    default_n_replicas: int = 8
    default_exchange_interval: int = 10
    
    # Performance settings
    max_problem_size: int = 10000
    auto_scaling_enabled: bool = True
    memory_limit_gb: int = 32
    gpu_memory_limit_gb: int = 16
    
    # RL settings
    default_reward_type: RewardType = RewardType.ENERGY_IMPROVEMENT
    rl_training_enabled: bool = False
    model_checkpoint_interval: int = 100


@dataclass
class ServerConfig:
    """Web server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_connections: int = 1000
    max_requests: int = 10000
    max_requests_jitter: int = 100
    keepalive: int = 5
    timeout: int = 300
    graceful_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    enable_file_logging: bool = True
    log_file: str = "/var/log/spin_glass_rl/app.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Structured logging
    enable_json_logging: bool = True
    enable_correlation_ids: bool = True
    
    # External logging
    elasticsearch_url: Optional[str] = None
    sentry_dsn: Optional[str] = None
    datadog_api_key: Optional[str] = None


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    enable_api: bool = True
    enable_web_ui: bool = True
    enable_distributed_computing: bool = False
    enable_gpu_acceleration: bool = True
    enable_quantum_integration: bool = False
    
    # Performance settings
    max_concurrent_optimizations: int = 10
    optimization_queue_size: int = 1000
    result_cache_ttl_hours: int = 24
    
    # Backup settings
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "ProductionConfig":
        """Load configuration from file."""
        config_path = Path(config_path)
        
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductionConfig":
        """Create configuration from dictionary."""
        # Convert nested dictionaries to dataclass instances
        config = cls()
        
        for key, value in data.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if isinstance(attr, DatabaseConfig):
                    setattr(config, key, DatabaseConfig(**value))
                elif isinstance(attr, RedisConfig):
                    setattr(config, key, RedisConfig(**value))
                elif isinstance(attr, SecurityConfig):
                    setattr(config, key, SecurityConfig(**value))
                elif isinstance(attr, MonitoringConfig):
                    setattr(config, key, MonitoringConfig(**value))
                elif isinstance(attr, OptimizationConfig):
                    setattr(config, key, OptimizationConfig(**value))
                elif isinstance(attr, ServerConfig):
                    setattr(config, key, ServerConfig(**value))
                elif isinstance(attr, LoggingConfig):
                    setattr(config, key, LoggingConfig(**value))
                else:
                    setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_environment(cls) -> "ProductionConfig":
        """Create configuration from environment variables."""
        config = cls()
        
        # Environment detection
        env_name = os.getenv('DEPLOYMENT_ENVIRONMENT', 'production').lower()
        config.environment = DeploymentEnvironment(env_name)
        config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Database configuration
        if os.getenv('DATABASE_URL'):
            # Parse DATABASE_URL
            import urllib.parse
            url = urllib.parse.urlparse(os.getenv('DATABASE_URL'))
            config.database.host = url.hostname or 'localhost'
            config.database.port = url.port or 5432
            config.database.database = url.path.lstrip('/') or 'spin_glass_rl'
            config.database.username = url.username or 'postgres'
            config.database.password = url.password or ''
        
        # Redis configuration
        config.redis.host = os.getenv('REDIS_HOST', 'localhost')
        config.redis.port = int(os.getenv('REDIS_PORT', '6379'))
        config.redis.password = os.getenv('REDIS_PASSWORD')
        
        # Security configuration
        config.security.secret_key = os.getenv('SECRET_KEY', config.security.secret_key)
        config.security.jwt_secret_key = os.getenv('JWT_SECRET_KEY', config.security.jwt_secret_key)
        
        # Server configuration
        config.server.host = os.getenv('HOST', '0.0.0.0')
        config.server.port = int(os.getenv('PORT', '8000'))
        config.server.workers = int(os.getenv('WORKERS', '4'))
        
        # Monitoring configuration
        config.monitoring.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        config.monitoring.pagerduty_key = os.getenv('PAGERDUTY_KEY')
        
        # Feature flags
        config.enable_gpu_acceleration = os.getenv('ENABLE_GPU', 'true').lower() == 'true'
        config.enable_distributed_computing = os.getenv('ENABLE_DISTRIBUTED', 'false').lower() == 'true'
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment.value,
            'debug': self.debug,
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'security': self.security.__dict__,
            'monitoring': self.monitoring.__dict__,
            'optimization': self.optimization.__dict__,
            'server': self.server.__dict__,
            'logging': self.logging.__dict__,
            'enable_api': self.enable_api,
            'enable_web_ui': self.enable_web_ui,
            'enable_distributed_computing': self.enable_distributed_computing,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_quantum_integration': self.enable_quantum_integration,
            'max_concurrent_optimizations': self.max_concurrent_optimizations,
            'optimization_queue_size': self.optimization_queue_size,
            'result_cache_ttl_hours': self.result_cache_ttl_hours,
            'backup_enabled': self.backup_enabled,
            'backup_schedule': self.backup_schedule,
            'backup_retention_days': self.backup_retention_days
        }
    
    def save_to_file(self, config_path: Union[str, Path], format: str = 'yaml'):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        if format == 'yaml':
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
        elif format == 'json':
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Security validation
        if self.security.secret_key == "your-secret-key-change-this":
            issues.append("Security: SECRET_KEY must be changed from default value")
        
        if self.security.jwt_secret_key == "jwt-secret-key-change-this":
            issues.append("Security: JWT_SECRET_KEY must be changed from default value")
        
        # Database validation
        if not self.database.host:
            issues.append("Database: host cannot be empty")
        
        if not self.database.database:
            issues.append("Database: database name cannot be empty")
        
        # Redis validation (if caching enabled)
        if not self.redis.host:
            issues.append("Redis: host cannot be empty")
        
        # Monitoring validation
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if not self.monitoring.email_alerts and not self.monitoring.slack_webhook_url:
                issues.append("Monitoring: No alert notifications configured for production")
        
        # SSL validation for production
        if (self.environment == DeploymentEnvironment.PRODUCTION and 
            self.security.enable_https_redirect and 
            not self.security.ssl_cert_path):
            issues.append("Security: SSL certificate path required when HTTPS redirect enabled")
        
        return issues
    
    def get_environment_specific_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            return {
                'debug': True,
                'logging': {'level': LogLevel.DEBUG},
                'server': {'workers': 1},
                'monitoring': {'enable_prometheus': False, 'enable_grafana': False}
            }
        elif self.environment == DeploymentEnvironment.STAGING:
            return {
                'debug': False,
                'logging': {'level': LogLevel.INFO},
                'server': {'workers': 2},
                'monitoring': {'enable_prometheus': True, 'enable_grafana': True}
            }
        else:  # Production
            return {
                'debug': False,
                'logging': {'level': LogLevel.WARNING},
                'server': {'workers': 4},
                'monitoring': {'enable_prometheus': True, 'enable_grafana': True},
                'security': {'enable_https_redirect': True}
            }


def create_default_configs():
    """Create default configuration files for different environments."""
    configs_dir = Path(__file__).parent / "configs"
    configs_dir.mkdir(exist_ok=True)
    
    # Development configuration
    dev_config = ProductionConfig()
    dev_config.environment = DeploymentEnvironment.DEVELOPMENT
    dev_config.debug = True
    dev_config.server.workers = 1
    dev_config.logging.level = LogLevel.DEBUG
    dev_config.save_to_file(configs_dir / "development.yaml")
    
    # Staging configuration
    staging_config = ProductionConfig()
    staging_config.environment = DeploymentEnvironment.STAGING
    staging_config.debug = False
    staging_config.server.workers = 2
    staging_config.logging.level = LogLevel.INFO
    staging_config.save_to_file(configs_dir / "staging.yaml")
    
    # Production configuration
    prod_config = ProductionConfig()
    prod_config.environment = DeploymentEnvironment.PRODUCTION
    prod_config.debug = False
    prod_config.server.workers = 4
    prod_config.logging.level = LogLevel.WARNING
    prod_config.security.enable_https_redirect = True
    prod_config.save_to_file(configs_dir / "production.yaml")
    
    print(f"Default configuration files created in: {configs_dir}")


def main():
    """Main function for configuration management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production configuration management")
    parser.add_argument("--create-defaults", action="store_true", 
                       help="Create default configuration files")
    parser.add_argument("--validate", type=str, 
                       help="Validate configuration file")
    parser.add_argument("--from-env", action="store_true",
                       help="Generate configuration from environment variables")
    
    args = parser.parse_args()
    
    if args.create_defaults:
        create_default_configs()
    elif args.validate:
        config = ProductionConfig.from_file(args.validate)
        issues = config.validate()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
    elif args.from_env:
        config = ProductionConfig.from_environment()
        print("Configuration from environment:")
        print(yaml.dump(config.to_dict(), default_flow_style=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()