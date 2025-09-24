# client/mcp_client.py
"""Minimal stdio MCP client used by the Gradio frontend to reach the PA backend.

The helper intentionally hides all asyncio plumbing so the UI code can make
plain method calls that:

* Spawn the MCP server as a subprocess (using stdio transport).
* Establish a session via the official MCP Python SDK.
* Marshal tool responses back into Python-native dict/list primitives.

Environment variables respected:
- ``MCP_SERVER_CMD`` – explicit command to run the server (string or list form).
- ``MCP_SERVER_PY`` – fallback path to a Python server file if
  ``MCP_SERVER_CMD`` is not provided.
"""

from __future__ import annotations

import os
import json
import shlex
import asyncio
import shutil
import contextlib
from typing import Any, Dict, List, Optional

# NOTE: Requires mcp>=1.1.0,<2.0.0
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession


def _split_cmd(cmd: str | List[str]) -> tuple[str, List[str]]:
    """
    Split a command string or list into (executable, args).

    Args:
        cmd: Either a shell command string (e.g. "python server.py")
             or a list form (["python", "server.py"]).

    Returns:
        (executable, args) tuple.
    """
    if isinstance(cmd, list):
        if not cmd:
            return "python", []
        return cmd[0], cmd[1:]
    parts = shlex.split(cmd or "")
    return (parts[0], parts[1:]) if parts else ("python", [])


class MCPClient:
    """
    Minimal stdio MCP client that spawns the MCP server (pbm_server.py) per call.

    Respects environment variables:
      - MCP_SERVER_CMD: explicit command (Python or Node.js).
      - MCP_SERVER_PY:  path to Python server file (used if CMD not set).
    """

    def __init__(
        self,
        server_cmd: Optional[str | List[str]] = None,
        server_py: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the MCP client configuration.

        Args:
            server_cmd: Explicit command to launch server (string or list).
            server_py: Path to Python server file (fallback).
            env: Extra environment variables to pass to subprocess.
        """
        # Build child process environment
        child_env = os.environ.copy()
        if env:
            child_env.update(env)

        # If extra Python imports need resolving inside /app:
        # child_env["PYTHONPATH"] = child_env.get("PYTHONPATH", "/app")

        # Resolve which command to run
        cmd_spec = server_cmd or os.getenv("MCP_SERVER_CMD")
        if not cmd_spec:
            # Fall back to Python server path
            server_py = server_py or os.getenv("MCP_SERVER_PY") or "/app/server/pbm_server.py"
            server_py = os.path.abspath(server_py)
            py = shutil.which("python") or shutil.which("python3") or "python"
            cmd_spec = f"{py} -u {server_py}"

        command, args = _split_cmd(cmd_spec)

        # Construct MCP server parameters object
        self.server_params = StdioServerParameters(command=command, args=args, env=child_env)

    # ---- Session plumbing ----
    def _run(self, coro):
        """Synchronously execute an async coroutine using ``asyncio.run``.

        The Gradio layer is entirely synchronous; this helper keeps the public
        API ergonomics simple while still leveraging the async MCP SDK under
        the hood.
        """
        return asyncio.run(coro)

    async def _with_session(self, fn):
        """Open a stdio session, run ``fn(session)``, and guarantee shutdown.

        Parameters
        ----------
        fn:
            Awaitable that accepts a :class:`~mcp.client.session.ClientSession`.

        Returns
        -------
        Any
            Whatever the provided callback returns.
        """
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                try:
                    return await fn(session)
                finally:
                    # Ensure clean shutdown even if errors occur
                    with contextlib.suppress(Exception):
                        await session.shutdown()

    # ---- Helpers ----
    @staticmethod
    def _to_python(value: Any) -> Any:
        """
        Convert MCP tool output into Python-native objects if possible.

        Handles:
        - Pydantic-style content blocks with type/json/text fields.
        - Dicts containing "json" or "text".
        - Strings or bytes that may contain JSON.
        - Falls back to returning the raw value.
        """
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
        """
        List all tools exposed by the MCP server.

        Returns:
            A list of dicts with {"name": str, "description": str}.
        """
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
        """Invoke ``name`` with ``arguments`` and coerce the response.

        The implementation normalizes the various content containers that MCP
        servers return (content blocks, raw JSON strings, etc.) so callers can
        handle plain Python structures without worrying about transport format.
        """
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
        """Return all plan names via MCP tool."""
        v = self._call_tool("list_plans", {})
        return list(v) if isinstance(v, (list, tuple)) else []

    def list_drugs(self, plan: str) -> List[str]:
        """Return drugs requiring PA for a given plan."""
        v = self._call_tool("list_drugs", {"plan": plan})
        return list(v) if isinstance(v, (list, tuple)) else []

    def evaluate_pa(self, plan: str, drug: str, diagnosis_text: str, patient: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the `evaluate_pa` tool on the MCP server.

        Args:
            plan: Plan name.
            drug: Drug name.
            diagnosis_text: Clinical diagnosis text.
            patient: Patient dict with fields like age, labs, tried_failed, etc.

        Returns:
            Dict with decision_code, outcome, criteria_evaluation, etc.
        """
        v = self._call_tool(
            "evaluate_pa",
            {"plan": plan, "drug": drug, "diagnosis_text": diagnosis_text, "patient": patient},
        )
        return dict(v) if isinstance(v, dict) else {"raw": v}
