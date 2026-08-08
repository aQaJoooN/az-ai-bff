# Azure DevOps AI Backend-for-Frontend (BFF)

A FastAPI-based backend-for-frontend service for Azure DevOps AI integrations.

## Integration with Azure DevOps AI Analyzer Extension

This BFF service is designed to work with the [Azure DevOps Build AI Reviewer](https://github.com/aQaJoooN/azure-devops-build-ai-reviewer) extension. To enable AI analysis in Azure DevOps:

1. **Install the extension**: Get the AI Analyzer extension from the Azure DevOps Marketplace
2. **Configure the extension**: Set the BFF URL in the extension settings
3. **Deploy this BFF**: Follow the deployment instructions below
4. **Configure CORS**: Ensure your Azure DevOps organization URL is in the allowed origins
5. **Enable HTTPS**: If Azure DevOps uses HTTPS, this BFF must also use HTTPS

## Features

- FastAPI-based REST API
- OpenAI-compatible AI backend integration
- Token management and budget control
- Multi-AI backend support
- Dockerized deployment
- Nginx reverse proxy configuration
- GitHub Actions CI/CD pipeline

## Quick Start

### Production Deployment (Using Docker Hub)

```bash
# Create deployment directory
mkdir az-ai-bff-deployment && cd az-ai-bff-deployment

# Download production configuration
curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/docker-compose.prod.yml

# Choose your nginx configuration:
# For HTTP-only: 
curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-http -o nginx.conf
# For HTTPS with HTTP-to-HTTPS redirect (recommended for Azure DevOps):
curl -O https://raw.githubusercontent.com/aqajooon/az-ai-bff/main/nginx.conf.example-https -o nginx.conf

# Create configuration
mkdir -p nginx/ssl nginx/logs
cp .env.example .env  # Edit with your settings

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

### Development

```bash
# Clone repository
git clone https://github.com/aqajooon/az-ai-bff.git
cd az-ai-bff

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Run with Docker
docker-compose up -d --build

# Or run locally
pip install -e .
python -m az_ai_bff
```

## API Usage

```bash
# Health check
curl http://localhost/health

# Chat endpoint
curl http://localhost/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{"message": "Hello"}'
```

## Documentation

- [Detailed Deployment Guide](DEPLOYMENT.md) - Complete setup and configuration
- [API Reference](#) - API documentation (generate with `make docs`)

## Configuration

### Required for Azure DevOps Integration:
- `AZ_AI_BFF_ALLOWED_EXTENSION_ORIGINS` - Azure DevOps organization URLs (e.g., `["https://dev.azure.com/your-org"]`)
- HTTPS enabled with valid SSL certificates
- Proper CORS headers configured in nginx

### Required environment variables (see `.env.example`):

- `AZ_AI_BFF_ACCESS_CREDENTIAL` - Authentication token for API
- `AZ_AI_BFF_BACKEND_URL` - AI backend URL (e.g., OpenAI-compatible)
- `AZ_AI_BFF_BACKEND_MODEL` - AI model name
- `AZ_AI_BFF_BACKEND_CREDENTIAL` - Backend authentication token
- `AZ_AI_BFF_BACKEND_FORMAT` - Backend format (`simple` or `openai`)

### Azure DevOps Specific:
- If Azure DevOps uses HTTPS, this BFF must also use HTTPS
- Add your specific Azure DevOps URLs to CORS allowed origins
- Ensure the BFF URL matches what's configured in the Azure DevOps extension

## CI/CD Pipeline

The repository includes GitHub Actions workflow that:
- Automatically builds Docker images on pushes
- Pushes to Docker Hub
- Supports multi-architecture builds (amd64/arm64)
- Runs tests on pull requests

## Project Structure

```
├── src/az_ai_bff/           # Application source code
├── tests/                   # Test suite
├── .github/workflows/       # GitHub Actions workflows
├── Dockerfile               # Docker build configuration
├── docker-compose.yml       # Development Docker Compose
├── docker-compose.prod.yml  # Production Docker Compose
├── nginx.conf.example-http  # HTTP-only example configuration
├── nginx.conf.example-https # HTTPS with HTTP-to-HTTPS redirect example
├── pyproject.toml           # Python dependencies
└── .env.example             # Environment configuration template
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

[License Name] - See LICENSE file for details.