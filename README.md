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

# Analyze a failed Azure DevOps build pipeline
curl https://your-bff-host/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "message": "azure build pipeline error",
    "data": "https://your-tfs/tfs/Collection/ProjectId/_build/results?buildId=12345",
    "role": "user",
    "type": "normal"
  }'

# Chat endpoint (generic AI chat)
curl http://localhost/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{"message": "Hello"}'
```

### Analyze endpoint logic

`POST /api/v1/analyze` applies the following decision tree:

| `role` | `type` | `message` | Result |
|--------|--------|-----------|--------|
| `user` | `super` | any | `{"answer": "unsupported"}` |
| `user` | `normal` | `azure build pipeline improvement` | `{"answer": "unsupported"}` |
| `user` | `normal` | `azure build pipeline error` | Fetches build logs → AI analysis |

When processing `azure build pipeline error`, the service:
1. Validates the `data` URL against your on-premise Azure DevOps using `AZ_AI_BFF_AZURE_TOKEN`
2. Finds all failed tasks in the build timeline
3. Fetches their logs and greps for `error` with 5 lines of context around each match
4. Sends the extracted logs to the configured AI backend and returns the answer

## Documentation

- [Detailed Deployment Guide](DEPLOYMENT.md) - Complete setup and configuration
- [API Reference](#) - API documentation (generate with `make docs`)

## Configuration

### Required for Azure DevOps Integration:
- `AZ_AI_BFF_ALLOWED_EXTENSION_ORIGINS` - Azure DevOps organization URLs (e.g., `["https://dev.azure.com/your-org"]`)
- HTTPS enabled with valid SSL certificates
- Proper CORS headers configured in nginx

### Required environment variables (see `.env.example`):

- `AZ_AI_BFF_AZURE_TOKEN` - **Required.** Personal Access Token (PAT) for your on-premise Azure DevOps / TFS server. Used by the `/api/v1/analyze` endpoint to fetch build logs.
- `AZ_AI_BFF_ACCESS_CREDENTIAL` - Authentication token for the API (optional — leave empty to disable)
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
│   ├── app.py               # Routes: /health, /api/v1/chat, /api/v1/analyze
│   ├── analyze.py           # Azure DevOps log fetcher and error grep
│   ├── backend.py           # AI backend HTTP client (simple + openai formats)
│   ├── chat.py              # Chat service (orchestration + token management)
│   ├── config.py            # Pydantic settings (AZ_AI_BFF_* env vars)
│   ├── models.py            # Request/response Pydantic models
│   ├── orchestration.py     # Multi-AI fan-out and response synthesis
│   ├── security.py          # Bearer token auth
│   ├── middleware.py        # Request ID propagation + structured logging
│   ├── token_management.py  # Token budget estimation and context trimming
│   ├── errors.py            # Centralized exception handlers
│   └── logging_config.py   # JSON structured logging
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