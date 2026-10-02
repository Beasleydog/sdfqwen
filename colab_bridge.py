"""Project-local control queue using Google's official Colab MCP implementation.

Run with the project Python; connection.json is ephemeral and gitignored.
The browser connects through the official Colab MCP URL. Requests are JSON
files in .colab/requests; results are .colab/responses with the same filename.
"""
import asyncio
import json
import logging
from pathlib import Path
import colab_mcp_local  # Apply the shared Windows loopback compatibility fix.

from fastmcp import Client, FastMCP
from colab_mcp.session import ColabSessionProxy

ROOT = Path(__file__).resolve().parent / ".colab"


async def main():
    for folder in (ROOT, ROOT / "requests", ROOT / "responses"):
        folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=ROOT / "bridge.log", level=logging.INFO)
    proxy = ColabSessionProxy()
    await proxy.start_proxy_server()
    server = FastMCP("SDF Colab MCP")
    server.mount(proxy.proxy_server)
    for middleware in proxy.middleware:
        server.add_middleware(middleware)
    (ROOT / "connection.json").write_text(json.dumps({
        "fragment": f"mcpProxyToken={proxy.wss.token}&mcpProxyPort={proxy.wss.port}"
    }), encoding="utf-8")
    print("Colab MCP bridge ready", flush=True)
    try:
        async with Client(server) as client:
            while True:
                for path in sorted((ROOT / "requests").glob("*.json")):
                    target = ROOT / "responses" / path.name
                    if target.exists():
                        continue
                    try:
                        request = json.loads(path.read_text(encoding="utf-8-sig"))
                        if request["op"] == "list":
                            result = [tool.model_dump(mode="json") for tool in await client.list_tools()]
                        else:
                            response = await asyncio.wait_for(
                                client.call_tool(request["name"], request.get("args", {})), timeout=150)
                            result = {"content": [c.model_dump(mode="json") for c in response.content],
                                      "data": response.data, "is_error": response.is_error}
                        payload = {"ok": True, "result": result}
                    except Exception as exc:
                        payload = {"ok": False, "error": str(exc)}
                    temporary = target.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
                    temporary.replace(target)
                    print("Completed", path.name, flush=True)
                await asyncio.sleep(0.25)
    finally:
        await proxy.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
