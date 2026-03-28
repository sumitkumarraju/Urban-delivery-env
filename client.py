"""
Urban Delivery Environment Client.

Provides the client for connecting to an Urban Delivery Environment server.
"""

from openenv.core.mcp_client import MCPToolClient


class UrbanDeliveryEnv(MCPToolClient):
    """Client for the Urban Delivery Environment.

    This client provides an interface for interacting with the environment
    via MCP tools. Inherits all functionality from MCPToolClient:
    - list_tools(): Discover available tools
    - call_tool(name, **kwargs): Call a tool by name
    - reset(**kwargs): Reset the environment
    - step(action): Execute an action

    Example:
        >>> with UrbanDeliveryEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("move", direction="up")
        ...     print(result)

    Example with HuggingFace Space:
        >>> env = UrbanDeliveryEnv.from_env("username/urban-delivery-env")
        >>> try:
        ...     env.reset()
        ...     result = env.call_tool("set_task", difficulty="hard")
        ...     result = env.call_tool("move", direction="right")
        ... finally:
        ...     env.close()
    """

    pass
