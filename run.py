#!/usr/bin/env python3
"""Run the API server."""
import uvicorn
from src.config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1,
        log_level=settings.LOG_LEVEL.lower(),
    )
