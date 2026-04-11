"""FastAPI entry point — wires the delivery environment to the HTTP server."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from server.urban_delivery_environment import UrbanDeliveryEnvironment
from server.observation import DeliveryObservationResponse

app = create_app(
    UrbanDeliveryEnvironment,
    CallToolAction,
    DeliveryObservationResponse,
    env_name="urban_delivery_env",
    max_concurrent_envs=4,
)


def main():
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
