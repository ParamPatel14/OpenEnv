from __future__ import annotations

import os
import secrets
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Security, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ..env_logic import SupportDeskEnvironment
from ..models import StepResult, SupportDeskAction, SupportDeskState

app = FastAPI(title="SupportDesk Env (OpenEnv)")

# -- Security Configuration --
SUPPORTDESK_API_KEY = os.getenv("SUPPORTDESK_API_KEY")
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if SUPPORTDESK_API_KEY:
        if not api_key or not secrets.compare_digest(api_key, SUPPORTDESK_API_KEY):
            raise HTTPException(status_code=403, detail="Could not validate API key")
    return api_key

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        return response

app.add_middleware(SecurityHeadersMiddleware)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -----------------------------


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    result: StepResult


class StepRequest(BaseModel):
    session_id: str
    action: SupportDeskAction


class StateRequest(BaseModel):
    session_id: str


_sessions: Dict[str, SupportDeskEnvironment] = {}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset", dependencies=[Depends(get_api_key)])
def http_reset(req: ResetRequest) -> ResetResponse:
    env = SupportDeskEnvironment()
    result = env.reset(task_id=req.task_id)
    session_id = str(env.state.episode_id or result.observation.ticket.ticket_id)
    _sessions[session_id] = env
    return ResetResponse(session_id=session_id, result=result)


@app.post("/step", dependencies=[Depends(get_api_key)])
def http_step(req: StepRequest) -> StepResult:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call /reset first.")
    return env.step(req.action)


@app.post("/state", dependencies=[Depends(get_api_key)])
def http_state(req: StateRequest) -> SupportDeskState:
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call /reset first.")
    return env.state


@app.get("/state", dependencies=[Depends(get_api_key)])
def http_state_get(session_id: str) -> SupportDeskState:
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Unknown session_id. Call /reset first.")
    return env.state


@app.get("/web")
def web() -> HTMLResponse:
    html = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SupportDesk Env</title>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
      code { background: #f4f4f5; padding: 2px 6px; border-radius: 6px; }
      .box { background: #fafafa; border: 1px solid #e5e7eb; padding: 16px; border-radius: 12px; }
      a { color: #2563eb; }
    </style>
  </head>
  <body>
    <h1>SupportDesk Env (OpenEnv)</h1>
    <div class="box">
      <p>This is an OpenEnv-compatible environment server.</p>
      <ul>
        <li>WebSocket session endpoint: <code>/ws</code></li>
        <li>HTTP endpoints: <code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/health</code></li>
        <li>OpenAPI docs: <a href="/docs">/docs</a></li>
      </ul>
      <p>Tip: use the provided Python client (<code>supportdesk_env.SupportDeskEnv</code>) for typed access.</p>
    </div>
  </body>
</html>
"""
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    if SUPPORTDESK_API_KEY:
        provided_key = websocket.headers.get(API_KEY_NAME.lower())
        if not provided_key or not secrets.compare_digest(provided_key, SUPPORTDESK_API_KEY):
            await websocket.close(code=1008, reason="Could not validate API key")
            return
            
    await websocket.accept()
    env = SupportDeskEnvironment()
    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            if msg_type == "reset":
                task_id = msg.get("task_id")
                result = env.reset(task_id=task_id)
                await websocket.send_json(result.model_dump(mode="json"))
            elif msg_type == "step":
                action_payload = msg.get("action")
                if action_payload is None:
                    await websocket.send_json({"error": "Missing action"})
                    continue
                action = SupportDeskAction.model_validate(action_payload)
                result = env.step(action)
                await websocket.send_json(result.model_dump(mode="json"))
            elif msg_type == "state":
                await websocket.send_json(env.state.model_dump(mode="json"))
            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})
    except WebSocketDisconnect:
        return


def run() -> None:
    import uvicorn

    uvicorn.run("supportdesk_env.server.app:app", host="0.0.0.0", port=8000, reload=False)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
