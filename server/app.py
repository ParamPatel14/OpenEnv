from __future__ import annotations

import uvicorn

from supportdesk_env.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
