from __future__ import annotations

import asyncio
import json
import os
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Optional

import httpx
import websockets

from .models import StepResult, SupportDeskAction, SupportDeskState


class SupportDeskEnv:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self._session_id: Optional[str] = None
        
        self._api_key = api_key or os.getenv("SUPPORTDESK_API_KEY")
        headers = {}
        if self._api_key:
            headers["X-API-Key"] = self._api_key
            
        self._http = httpx.Client(timeout=30.0, headers=headers)

    def close(self) -> None:
        self._http.close()

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        resp = self._http.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        payload = resp.json()
        self._session_id = payload["session_id"]
        return StepResult.model_validate(payload["result"])

    def step(self, action: SupportDeskAction) -> StepResult:
        if not self._session_id:
            raise RuntimeError("Call reset() first.")
        resp = self._http.post(
            f"{self.base_url}/step",
            json={"session_id": self._session_id, "action": action.model_dump(mode="json")},
        )
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> SupportDeskState:
        if not self._session_id:
            raise RuntimeError("Call reset() first.")
        resp = self._http.post(f"{self.base_url}/state", json={"session_id": self._session_id})
        resp.raise_for_status()
        return SupportDeskState.model_validate(resp.json())

    def sync(self) -> "SupportDeskEnvSync":
        return SupportDeskEnvSync(self)

    async def ws(self) -> "SupportDeskEnvWS":
        return await SupportDeskEnvWS.connect(self.base_url, api_key=self._api_key)


class SupportDeskEnvSync(AbstractContextManager["SupportDeskEnv"]):
    def __init__(self, env: SupportDeskEnv):
        self._env = env

    def __enter__(self) -> SupportDeskEnv:
        return self._env

    def __exit__(self, exc_type, exc, tb) -> None:
        self._env.close()


class SupportDeskEnvWS(AbstractAsyncContextManager["SupportDeskEnvWS"]):
    def __init__(self, base_url: str, ws):
        self.base_url = base_url.rstrip("/")
        self._ws = ws

    @classmethod
    async def connect(cls, base_url: str, api_key: Optional[str] = None) -> "SupportDeskEnvWS":
        ws_url = cls._to_ws_url(base_url.rstrip("/") + "/ws")
        
        headers = {}
        _key = api_key or os.getenv("SUPPORTDESK_API_KEY")
        if _key:
            headers["X-API-Key"] = _key
            
        ws = await websockets.connect(ws_url, open_timeout=30, extra_headers=headers)
        return cls(base_url=base_url, ws=ws)

    @staticmethod
    def _to_ws_url(url: str) -> str:
        if url.startswith("https://"):
            return "wss://" + url[len("https://") :]
        if url.startswith("http://"):
            return "ws://" + url[len("http://") :]
        return url

    async def __aenter__(self) -> "SupportDeskEnvWS":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._ws.close()

    async def reset(self, task_id: Optional[str] = None) -> StepResult:
        await self._ws.send(json.dumps({"type": "reset", "task_id": task_id}))
        payload = json.loads(await self._ws.recv())
        return StepResult.model_validate(payload)

    async def step(self, action: SupportDeskAction) -> StepResult:
        await self._ws.send(json.dumps({"type": "step", "action": action.model_dump(mode="json")}))
        payload = json.loads(await self._ws.recv())
        return StepResult.model_validate(payload)

    async def state(self) -> SupportDeskState:
        await self._ws.send(json.dumps({"type": "state"}))
        payload = json.loads(await self._ws.recv())
        return SupportDeskState.model_validate(payload)


def run_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        raise RuntimeError("run_sync cannot be called from a running event loop.")
    return asyncio.run(coro)
