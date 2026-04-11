---
title: Urban Delivery Optimization Environment
emoji: "\U0001F69A"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
license: bsd-3-clause
---

# Urban Delivery Optimization Environment

A reinforcement learning environment that simulates urban package delivery on a grid.
Agents must navigate traffic, manage fuel, respect deadlines, and prioritise urgent
packages — built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

---

## What makes it interesting

| Constraint | Detail |
|------------|--------|
| Multi-resource budgets | Fuel depletes per move; rain/fog add extra cost |
| Dynamic traffic | Grid congestion pattern changes every 5 steps (hard mode) |
| Package priorities | Urgent and fragile packages carry bonus/penalty deadlines |
| Vehicle capacity | Can only carry N packages at once — forces routing decisions |
| LLM-native hints | Natural-language observation summaries for language-model agents |
| Transparent grading | Every score comes with a full factor-level breakdown |

---

## Action Space (6 discrete)

| Index | Action |
|-------|--------|
| 0 | Move up (row - 1) |
| 1 | Move down (row + 1) |
| 2 | Move left (col - 1) |
| 3 | Move right (col + 1) |
| 4 | Deliver carried package |
| 5 | Refuel at fuel station |

---

## Observation Space

```python
{
    "vehicle": {"position": [row, col], "fuel": float, "carrying": [int, ...]},
    "packages": [
        {
            "package_id": int,
            "pickup_position": [row, col],
            "delivery_position": [row, col],
            "picked_up": bool,
            "delivered": bool,
            "priority": int,          # 0=normal, 1=urgent, 2=fragile
            "deadline": int | None,
            "delivery_step": int | None,
        }
    ],
    "traffic_grid": [[int, ...]],
    "fuel_stations": [[row, col]],
    "grid_size": int,
    "time_elapsed": int,
    "max_steps": int,
    "weather": int,               # 0=clear, 1=rain, 2=fog
    "total_reward": float,
    "packages_delivered": int,
    "packages_total": int,
    "done": bool,
    "hint": str,                  # NL advice for LLM agents
}
```

---

## Reward Shaping

| Event | Reward |
|-------|--------|
| Deliver package | +20 |
| Pick up package | +2 |
| All delivered bonus | +50 |
| Movement cost | -1 |
| Wall collision | -2 |
| Enter traffic cell | -5 |
| Move in rain | -2 |
| Move in fog | -1 |
| Refuel at station | +5 |
| Refuel (full tank) | -0.5 |
| Wrong refuel spot | -1 |
| Fuel empty (terminal) | -10 |
| Urgent bonus | +15 |
| Fragile bonus | +10 |
| Deadline met | +10 |
| Deadline missed | -5 |

---

## Tasks

### Easy
5x5 grid, 2 packages, 100 fuel, no traffic, no deadlines, 100 steps.

### Medium
8x8 grid, 3 packages, 60 fuel, static traffic, some deadlines, 200 steps.

### Hard
10x10 grid, 5 packages, 40 fuel, dynamic traffic, all deadlines, priorities, weather, 300 steps.

---

## Grading

All graders are deterministic and return scores strictly in (0, 1).

**Easy** — `completion_ratio`

**Medium** — `0.5 * completion + 0.3 * fuel_efficiency + 0.2 * time_efficiency`

**Hard** — `0.4 * completion + 0.2 * fuel_efficiency + 0.2 * deadline_compliance + 0.1 * priority_accuracy + 0.1 * reward_normalized`

Every grader also returns a breakdown dict showing each factor's raw value, weight, and contribution.

---

## Project Layout

```
urban_delivery_env/
  env.py                      Core simulation engine
  inference.py                LLM agent runner (OpenAI-compatible)
  client.py                   MCP client wrapper
  models/                     Pydantic action / observation / reward
  tasks/                      easy / medium / hard configs
  graders/                    Deterministic scoring + explanations
  server/                     FastAPI + MCP tool definitions
  scripts/                    Benchmark & interactive debugger
  tests/                      pytest suite (22 tests)
  Dockerfile                  Multi-stage production build
  openenv.yaml                Environment manifest
```

---

## Quick Start

```bash
git clone https://github.com/sumitkumarraju/Urban-delivery-env.git
cd urban_delivery_env

pip install -e .
make test      # 22 tests
make bench     # steps-per-second profiler
make start     # local server on :8000
```

### Docker

```bash
docker build -t urban-delivery-env .
docker run -p 8000:8000 urban-delivery-env
```

### Run Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...
python inference.py
```

---

## MCP Tools

| Tool | Args | Description |
|------|------|-------------|
| `move` | `direction: str` | up / down / left / right |
| `pickup` | — | Check carrying status and nearby packages |
| `deliver` | — | Deliver carried package at delivery location |
| `refuel` | — | Refuel at fuel station |
| `get_observation` | — | Full state + NL hint |
| `get_hint` | — | Strategic NL advice with distances |
| `set_task` | `difficulty: str` | easy / medium / hard |

---

## License

BSD 3-Clause
