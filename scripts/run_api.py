#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    root = Path(__file__).resolve().parent.parent
    uvicorn.run(
        "apps.api_app:app",
        host=host,
        port=port,
        reload=False,
        app_dir=str(root),
    )


if __name__ == "__main__":
    main()
