#!/usr/bin/env python3
"""Run the Liveness API with uvicorn (config from env)."""

import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.host,
        port=s.port,
        workers=s.workers,
        reload=s.debug,
        log_level=s.log_level.lower(),
    )
