# Docker Deployment Guide for az-ai-bff

## Table of Contents
1. [Azure DevOps Integration](#azure-devops-integration)
2. [Production Deployment](#production-deployment)
3. [Development Setup](#development-setup)
4. [Configuration](#configuration)
5. [Service URLs](#service-urls)
6. [Management](#management)
7. [Troubleshooting](#troubleshooting)

## Azure DevOps Integration

This BFF service is designed to work with the [Azure DevOps Build AI Reviewer](https://github.com/aQaJoooN/azure-devops-build-ai-reviewer) extension.

### Prerequisites:
1. **Install the Azure DevOps AI Analyzer extension** from the marketplace
2. **Configure the extension** in your Azure DevOps project to point to this BFF service URL
3. **Configure CORS settings** to allow your Azure DevOps organization URL
4. **Enable HTTPS** if Azure DevOps uses HTTPS (recommended for production)

### Key Configuration for Azure DevOps:
- Set `AZ_AI_BFF_ALLOWED_EXTENSION_ORIGINS` to include your Azure DevOps URLs
- Enable HTTPS by configuring SSL certificates in nginx/ssl/
- Ensure your BFF URL matches what's configured in the Azure DevOps extension

## Production Deployment

### Using Pre-built Docker Hub Image
For production, use the pre-built image from Docker Hub:

1. **Create deployment directory:**
   ```bash
   mkdir az-ai-bff-deployment && cd az-ai-bff-deployment
   ```

2. **Create configuration files:**
   ```bash
   # Copy production docker-compose
   curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/docker-compose.prod.yml
   
   # Choose your nginx configuration:
   # For HTTP-only:
   curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-http -o nginx.conf
   # For HTTPS with HTTP-to-HTTPS redirect (recommended for Azure DevOps):
   curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-https -o nginx.conf
   
   # Create directories
   mkdir -p nginx/ssl nginx/logs
   
   # Create .env file
   cp .env.example .env
   ```

3. **Configure environment:**
   Edit `.env` file with your production settings:
   ```bash
   # Docker image configuration
   DOCKER_IMAGE=docker.io/aqajooon/az-ai-bff:latest

   # Application configuration
   AZ_AI_BFF_TRUSTED_PROXY_ADDRESSES=["nginx"]
   AZ_AI_BFF_ACCESS_CREDENTIAL=your_production_access_token
   AZ_AI_BFF_AZURE_TOKEN=your-azure-devops-pat          # Required PAT for on-prem TFS/Azure DevOps
   AZ_AI_BFF_BACKEND_URL=http://your-production-ai-backend
   AZ_AI_BFF_BACKEND_MODEL=your-model-name
   AZ_AI_BFF_BACKEND_CREDENTIAL=your-backend-token
   AZ_AI_BFF_BACKEND_FORMAT=openai
   AZ_AI_BFF_RESPONSE_TIMEOUT_SECONDS=30

   # Nginx configuration (optional)
   NGINX_PORT=80
   NGINX_SSL_PORT=443
   NGINX_CONFIG=./nginx.conf
   NGINX_SSL_DIR=./nginx/ssl
   NGINX_LOG_DIR=./nginx/logs
   ENV_FILE=./.env
   ```

4. **Start services:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Environment Variables Reference
Create a `.env` file with these required variables:
```
# Docker Configuration
DOCKER_IMAGE=docker.io/aqajooon/az-ai-bff:latest

# Application Configuration
AZ_AI_BFF_TRUSTED_PROXY_ADDRESSES=["nginx"]
AZ_AI_BFF_ACCESS_CREDENTIAL=your_access_token
AZ_AI_BFF_AZURE_TOKEN=your-azure-devops-pat
AZ_AI_BFF_BACKEND_URL=http://your-ai-backend-url
AZ_AI_BFF_BACKEND_MODEL=your-model-name
AZ_AI_BFF_BACKEND_CREDENTIAL=your-backend-token
AZ_AI_BFF_BACKEND_FORMAT=openai  # or "simple"
AZ_AI_BFF_RESPONSE_TIMEOUT_SECONDS=30

# Azure DevOps CORS Configuration (REQUIRED)
# Add your specific Azure DevOps organization URLs
AZ_AI_BFF_ALLOWED_EXTENSION_ORIGINS='["https://dev.azure.com", "https://*.visualstudio.com"]'

# Optional Nginx Configuration
NGINX_PORT=80
NGINX_SSL_PORT=443  # Set to 443 for HTTPS
NGINX_CONFIG=./nginx.conf
NGINX_SSL_DIR=./nginx/ssl  # Required for HTTPS
NGINX_LOG_DIR=./nginx/logs
ENV_FILE=./.env
```

## SSL/HTTPS Setup (Required for Azure DevOps HTTPS)

**Important**: If your Azure DevOps organization uses HTTPS (most do), you must enable HTTPS on this BFF service.

To enable HTTPS:

1. Generate or obtain SSL certificates
2. Place certificates in `nginx/ssl/` directory:
   - `cert.pem` - SSL certificate
   - `key.pem` - Private key
3. Use the HTTPS example configuration:
   ```bash
   curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-https -o nginx.conf
   ```
4. Update `.env` file:
   ```
   NGINX_PORT=443
   NGINX_SSL_PORT=443
   ```
5. Ensure port 443 is exposed in your docker-compose configuration

**Note**: For development/testing with self-signed certificates, you may need to add your BFF's certificate to Azure DevOps trusted certificates.

## Logs

View logs for each service:
```bash
# BFF service logs
docker-compose logs az-ai-bff

# Nginx logs
docker-compose logs nginx

# All logs
docker-compose logs -f
```

## Management Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# View running containers
docker-compose ps

# Execute command in running container
docker-compose exec az-ai-bff bash

# Remove everything (containers, networks, volumes)
docker-compose down -v
```

## Development

For development, you might want to mount the source code as a volume:

```yaml
# In docker-compose.yml under az-ai-bff service
volumes:
  - ./.env:/app/.env:ro
  - ./src:/app/src:ro  # Add this for development
```

## Troubleshooting

1. **Service won't start:** Check `.env` file exists and has all required variables — including `AZ_AI_BFF_AZURE_TOKEN`
2. **Connection refused:** Verify BFF service is running inside container
3. **Authentication errors:** Ensure `AZ_AI_BFF_ACCESS_CREDENTIAL` matches the token in requests
4. **Backend connection errors:** Check `AZ_AI_BFF_BACKEND_URL` and network connectivity
5. **Nginx 502 errors:** BFF service might not be running or health check failing
6. **Analyze 502 azure_auth_error:** `AZ_AI_BFF_AZURE_TOKEN` is invalid or lacks Build read permissions

## Production Considerations

1. Use proper SSL certificates
2. Set up monitoring and logging
3. Configure resource limits in docker-compose.yml
4. Use environment-specific .env files
5. Consider using Docker secrets for sensitive data
6. Implement proper backup strategy for logs and data
## Development Setup

For local development and testing:

1. **Clone and setup:**
   ```bash
   git clone https://github.com/aqajooon/az-ai-bff.git
   cd az-ai-bff
   cp .env.example .env
   # Edit .env with local configuration
   ```

2. **Build and run locally:**
   ```bash
   # Using development docker-compose (builds from source)
   docker-compose up -d --build
   
   # Or using production compose with local image
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Run tests:**
   ```bash
   # Inside container
   docker-compose exec az-ai-bff python -m pytest tests/ -v
   
   # Or locally with virtual environment
   python -m pytest tests/ -v
   ```

## Service URLs

- **Nginx (Reverse Proxy):** http://localhost:${NGINX_PORT:-80}
- **Direct BFF Service:** http://localhost:8000 (for debugging/monitoring)
- **Health Check:** http://localhost:${NGINX_PORT:-80}/health
- **Analyze Endpoint:** http://localhost:${NGINX_PORT:-80}/api/v1/analyze
- **Chat Endpoint:** http://localhost:${NGINX_PORT:-80}/api/v1/chat

## Management

### Docker Compose Commands
```bash
# Start services
docker-compose -f docker-compose.prod.yml up -d

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restart services
docker-compose -f docker-compose.prod.yml restart

# View logs
docker-compose -f docker-compose.prod.yml logs -f
docker-compose -f docker-compose.prod.yml logs az-ai-bff
docker-compose -f docker-compose.prod.yml logs nginx

# View running containers
docker-compose -f docker-compose.prod.yml ps

# Execute command in container
docker-compose -f docker-compose.prod.yml exec az-ai-bff bash

# Pull latest image
docker-compose -f docker-compose.prod.yml pull az-ai-bff
```

### Health Checks
Both services include health checks:
- **BFF Service:** Checks `/health` endpoint
- **Nginx:** Validates configuration with `nginx -t`

### Monitoring
```bash
# Check service health
curl http://localhost/health

# Check Docker health status
docker inspect --format='{{.State.Health.Status}}' az-ai-bff
docker inspect --format='{{.State.Health.Status}}' az-ai-nginx
```

## Troubleshooting

### Common Issues

1. **Container won't start:**
   ```bash
   # Check logs
   docker-compose -f docker-compose.prod.yml logs az-ai-bff
   
   # Verify .env file exists and has required variables
   cat .env
   ```

2. **Image pull failures:**
   ```bash
   # Check Docker Hub credentials
   echo $DOCKERHUB_USERNAME
   
   # Manually pull image
   docker pull ${DOCKER_IMAGE}
   ```

3. **Nginx 502 errors:**
   ```bash
   # Check if BFF service is healthy
   docker-compose -f docker-compose.prod.yml ps
   
   # Test BFF directly
   curl http://localhost:8000/health
   ```

4. **Authentication errors:**
   - Verify `AZ_AI_BFF_ACCESS_CREDENTIAL` matches the token in requests
   - Check `AZ_AI_BFF_BACKEND_CREDENTIAL` if backend requires authentication
   - Verify `AZ_AI_BFF_AZURE_TOKEN` is a valid PAT with **Read** permissions on Build for your Azure DevOps project

5. **Backend connection errors:**
   ```bash
   # Test backend connectivity from container
   docker-compose -f docker-compose.prod.yml exec az-ai-bff \
     curl -v ${AZ_AI_BFF_BACKEND_URL}
   ```

### Logs Location
- **Application logs:** Docker container logs
- **Nginx access logs:** `./nginx/logs/access.log`
- **Nginx error logs:** `./nginx/logs/error.log`

### SSL/HTTPS Setup
To enable HTTPS:

1. Generate or obtain SSL certificates
2. Place in `nginx/ssl/` directory:
   - `cert.pem` - SSL certificate
   - `key.pem` - Private key
3. Update `.env` file:
   ```
   NGINX_PORT=443
   NGINX_SSL_PORT=443
   ```
4. Uncomment HTTPS section in `nginx.conf`
5. Restart services

## Security Considerations

1. **Never commit `.env` file** to version control
2. Use strong, randomly generated credentials
3. Regularly update Docker images with security patches
4. Configure firewall rules appropriately
5. Monitor logs for suspicious activity
6. Consider using Docker secrets for sensitive data in production

## Updating to New Versions

1. **Pull latest image:**
   ```bash
   docker-compose -f docker-compose.prod.yml pull az-ai-bff
   ```

2. **Restart services:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Verify update:**
   ```bash
   docker-compose -f docker-compose.prod.yml images
   curl http://localhost/health
   ```

For major version updates, review changelog and test in staging first.
