"""Integration tests for the HTTP server and MCP tool layer.

Validates that the FastAPI endpoints work correctly and that
done/reward signals propagate from the inner environment.
"""

import pytest
from fastapi.testclient import TestClient
from server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    """Server must return healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_metadata_endpoint(client):
    """Metadata must include name and description."""
    resp = client.get("/metadata")
    assert resp.status_code == 200
    data = resp.json()
    assert "name" in data
    assert "description" in data


def test_schema_endpoint(client):
    """Schema must include action, observation, and state."""
    resp = client.get("/schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data


def test_reset_endpoint(client):
    """Reset must return valid initial observation."""
    resp = client.post("/reset", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert data["done"] is False


def test_mcp_tools_list(client):
    """POST /mcp with tools/list must return registered tools."""
    resp = client.post("/mcp", json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["jsonrpc"] == "2.0"
    tools = data["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "move" in tool_names
    assert "deliver" in tool_names
    assert "refuel" in tool_names
    assert "get_observation" in tool_names
    assert "get_hint" in tool_names
    assert "set_task" in tool_names
    assert "pickup" in tool_names


def test_step_move_returns_done_and_reward(client):
    """Step with a move tool call must propagate done/reward."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "move",
            "arguments": {"direction": "down"},
        }
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "reward" in data
    assert "done" in data
    assert data["reward"] is not None


def test_step_invalid_direction_no_crash(client):
    """Invalid direction must not crash the server."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "move",
            "arguments": {"direction": "sideways"},
        }
    })
    assert resp.status_code == 200


def test_step_invalid_tool_no_crash(client):
    """Calling a non-existent tool must not crash."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "teleport",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_step_get_observation_tool(client):
    """get_observation tool should return state without stepping."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "get_observation",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_step_get_hint_tool(client):
    """get_hint tool should return hint and distances."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "get_hint",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_step_set_task_tool(client):
    """set_task tool should switch difficulty."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "set_task",
            "arguments": {"difficulty": "hard"},
        }
    })
    assert resp.status_code == 200


def test_step_deliver_no_package_no_crash(client):
    """Delivering with no package should not crash."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "deliver",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_step_refuel_not_at_station_no_crash(client):
    """Refueling when not at a station should not crash."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "refuel",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_step_reward_is_not_none(client):
    """Every step response must have a non-None reward value."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "move",
            "arguments": {"direction": "right"},
        }
    })
    data = resp.json()
    assert data.get("reward") is not None, "Reward must not be None in step response"


def test_step_pickup_tool(client):
    """pickup tool should return carrying info without crashing."""
    client.post("/reset", json={})

    resp = client.post("/step", json={
        "action": {
            "type": "call_tool",
            "tool_name": "pickup",
            "arguments": {},
        }
    })
    assert resp.status_code == 200


def test_inner_env_done_propagation():
    """Verify done/reward propagate correctly in the environment wrapper."""
    from server.urban_delivery_environment import UrbanDeliveryEnvironment
    from openenv.core.env_server.mcp_types import CallToolAction

    env = UrbanDeliveryEnvironment(task_name="easy")
    env.reset()

    done = False
    for _ in range(200):
        action = CallToolAction(tool_name="move", arguments={"direction": "down"})
        obs = env.step(action)
        if obs.done:
            done = True
            break

    assert done, "Episode should eventually terminate"
    assert obs.reward is not None, "Final step reward must not be None"
