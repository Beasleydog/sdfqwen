"""Official Google Colab MCP with a Windows loopback binding compatibility fix."""
import colab_mcp
import colab_mcp.session as session
from websockets.asyncio.server import serve
from websockets.typing import Subprotocol
from colab_mcp.websocket_server import ColabWebSocketServer


class IPv4ColabWebSocketServer(ColabWebSocketServer):
    def __init__(self):
        # localhost+port=0 otherwise creates IPv4/IPv6 sockets on different ports.
        super().__init__(host="127.0.0.1")

    async def __aenter__(self):
        # Artifact chunks exceed websockets' default 1 MiB response limit.
        # Keep a finite 16 MiB cap and the vendor's origin/token checks.
        self._server = await serve(self._connection_handler, host=self.host, port=0,
                                   subprotocols=[Subprotocol('mcp')], origins=self.allowed_origins,
                                   process_request=self._validate_authorization, max_size=16 * 1024 * 1024)
        self.port = self._server.sockets[0].getsockname()[1]
        return self


session.ColabWebSocketServer = IPv4ColabWebSocketServer

if __name__ == "__main__":
    colab_mcp.main()
