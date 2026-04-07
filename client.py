"""MCP client for connecting to a running Urban Delivery Environment server."""

from openenv.core.mcp_client import MCPToolClient


class UrbanDeliveryEnv(MCPToolClient):
    """Thin wrapper that gives the delivery env its own importable name.

    All real logic (list_tools, call_tool, reset, step) lives in the
    parent MCPToolClient — this subclass exists so users can write
    ``from urban_delivery_env import UrbanDeliveryEnv`` and get a
    client pre-typed to this specific environment.
    """

    pass
