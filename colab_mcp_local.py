"""Official Google Colab MCP with a Windows loopback binding compatibility fix."""
import colab_mcp
import colab_mcp.session as session
from colab_mcp.websocket_server import ColabWebSocketServer


class IPv4ColabWebSocketServer(ColabWebSocketServer):
    def __init__(self):
        # localhost+port=0 otherwise creates IPv4/IPv6 sockets on different ports.
        super().__init__(host="127.0.0.1")


session.ColabWebSocketServer = IPv4ColabWebSocketServer

if __name__ == "__main__":
    colab_mcp.main()
