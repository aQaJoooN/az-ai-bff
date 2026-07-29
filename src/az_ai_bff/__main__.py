import uvicorn

from az_ai_bff.app import create_app
from az_ai_bff.config import Settings
from az_ai_bff.logging_config import configure_logging


def main() -> None:
    """Run the service with proxy headers trusted only from configured addresses."""
    configure_logging()
    settings = Settings()
    uvicorn.run(
        create_app(settings),
        host=settings.internal_host,
        port=settings.internal_port,
        proxy_headers=True,
        forwarded_allow_ips=",".join(settings.trusted_proxy_addresses),
    )


if __name__ == "__main__":
    main()