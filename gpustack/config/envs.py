"""Configurable environment variables for GPUStack."""

import os

# Database configuration
DB_ECHO = os.getenv("GPUSTACK_DB_ECHO", "false").lower() == "true"
DB_POOL_SIZE = int(os.getenv("GPUSTACK_DB_POOL_SIZE", 10))
DB_MAX_OVERFLOW = int(os.getenv("GPUSTACK_DB_MAX_OVERFLOW", 20))
DB_POOL_TIMEOUT = int(os.getenv("GPUSTACK_DB_POOL_TIMEOUT", 30))
DB_POOL_RECYCLE = int(os.getenv("GPUSTACK_DB_POOL_RECYCLE", 300))

# Proxy configuration
PROXY_TIMEOUT = int(os.getenv("GPUSTACK_PROXY_TIMEOUT_SECONDS", 1800))

# HTTP client TCP connector configuration
TCP_CONNECTOR_LIMIT = int(os.getenv("GPUSTACK_TCP_CONNECTOR_LIMIT", 1000))

# JWT Expiration
JWT_TOKEN_EXPIRE_MINUTES = int(os.getenv("GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES", 120))
