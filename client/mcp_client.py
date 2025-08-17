# client/mcp_client.py
from __future__ import annotations

import os
import json
import shlex
import asyncio
import shutil
import contextlib
from typing import Any, Dict, List, Optional

# NOTE: This requires mcp>=1.1.0,<2.0.0
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


def _split_cmd(cmd: str | List[str]) -> tuple[str, List[str]]:
    """Split a command into (executable, args)."""
    if isinstance(cmd, list):
        if not cmd:
            return "python", []
        return cmd[0], cmd[1:]
    parts = shlex.split(cmd or "")
    return (parts[0], parts[1:]) if parts else ("python", [])


class MCPClient:
    """
    Minimal stdio MCP client that spawns the MCP server (pbm_server.py) per call.

    Respects:
      - MCP_SERVER_CMD  e.g. `python -u /app/server/pbm_server.py` OR `node /app/server_node/pbm_server.mjs`
      - MCP_SERVER_PY   path to the Python server .py (used if MCP_SERVER_CMD not set)
    """

    def __init__(
        self,
        server_cmd: Optional[str | List[str]] = None,
        server_py: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        # Build env for the child process
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        # If you need module imports to resolve inside /app, uncomment:
        # child_env["PYTHONPATH"] = child_env.get("PYTHONPATH", "/app")

        # Resolve command preference: explicit -> env -> python fallback
        cmd_spec = server_cmd or os.getenv("MCP_SERVER_CMD")
        if not cmd_spec:
            # Use MCP_SERVER_PY or default to our Python server
            server_py = server_py or os.getenv("MCP_SERVER_PY") or "/app/server/pbm_server.py"
            server_py = os.path.abspath(server_py)
            py = shutil.which("python") or shutil.which("python3") or "python"
            cmd_spec = f"{py} -u {server_py}"

        command, args = _split_cmd(cmd_spec)

        # IMPORTANT: the current `mcp` Python client expects StdioServerParameters
        self.server_params = StdioServerParameters(command=command, args=args, env=child_env)

    # ---- Session plumbing ----
    def _run(self, coro):
        return asyncio.run(coro)

    async def _with_session(self, fn):
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    return await fn(session)
                finally:
                    with contextlib.suppress(Exception):
                        await session.shutdown()

    # ---- Helpers ----
    @staticmethod
    def _to_python(value: Any) -> Any:
        """Best-effort extraction of JSON/text from MCP tool results."""
        if value is None:
            return None

        # Pydantic-like content blocks
        t = getattr(value, "type", None)
        if t in ("json", "object"):
            return getattr(value, "json", None)
        if t in ("text", "string"):
            txt = getattr(value, "text", None) or getattr(value, "data", None)
            if txt is None:
                return None
            try:
                return json.loads(txt)
            except Exception:
                return txt

        # Dict-like
        if isinstance(value, dict):
            if "json" in value:
                return value["json"]
            if "text" in value:
                try:
                    return json.loads(value["text"])
                except Exception:
                    return value["text"]

        # Plain str / bytes
        if isinstance(value, (str, bytes)):
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except Exception:
                    return value
            try:
                return json.loads(value)
            except Exception:
                return value

        return value

    # ---- Public API ----
    def list_tools(self) -> List[Dict[str, Any]]:
        async def runner(session: ClientSession):
            res = await session.list_tools()
            tools = getattr(res, "tools", res)
            out: List[Dict[str, Any]] = []
            for t in tools:
                if hasattr(t, "model_dump"):
                    d = t.model_dump()
                elif isinstance(t, dict):
                    d = t
                else:
                    d = {"name": getattr(t, "name", str(t)), "description": getattr(t, "description", "")}
                out.append({"name": d.get("name", ""), "description": d.get("description", "")})
            return out

        return self._run(self._with_session(runner))

    def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        async def runner(session: ClientSession):
            res = await session.call_tool(name, arguments=arguments)
            content = getattr(res, "content", None) or getattr(res, "outputs", None) or res
            if isinstance(content, list):
                for c in content:
                    v = MCPClient._to_python(c)
                    if v is not None:
                        return v
            return MCPClient._to_python(content)

        return self._run(self._with_session(runner))

    # Convenience wrappers
    def list_plans(self) -> List[str]:
        v = self._call_tool("list_plans", {})
        return list(v) if isinstance(v, (list, tuple)) else []

    def list_drugs(self, plan: str) -> List[str]:
        v = self._call_tool("list_drugs", {"plan": plan})
        return list(v) if isinstance(v, (list, tuple)) else []

    def evaluate_pa(self, plan: str, drug: str, diagnosis_text: str, patient: Dict[str, Any]) -> Dict[str, Any]:
        v = self._call_tool(
            "evaluate_pa",
            {"plan": plan, "drug": drug, "diagnosis_text": diagnosis_text, "patient": patient},
        )
        return dict(v) if isinstance(v, dict) else {"raw": v}
