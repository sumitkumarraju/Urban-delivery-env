# 🚚 Urban Delivery Optimization Environment

> A sophisticated reinforcement learning environment for training AI agents to optimize urban package delivery logistics, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-BSD--3-orange)](LICENSE)

---

## 🎯 Environment Description

The Urban Delivery Optimization Environment simulates a delivery driver navigating a city grid. The agent must pick up packages from various locations and deliver them to their destinations while managing multiple real-world constraints:

- **Route optimization** across a grid-based city
- **Fuel management** with strategic refueling stops
- **Traffic avoidance** in congested areas
- **Time pressure** through delivery deadlines
- **Package prioritization** (normal, urgent, fragile)
- **Weather adaptation** (clear, rain, fog)

## 💡 Motivation — Why This Problem Matters

Urban delivery logistics is a **$500B+ global industry** facing critical optimization challenges:

| Challenge | Real-World Impact | Our Simulation |
|-----------|------------------|----------------|
| Route efficiency | 30% of delivery costs are fuel | Movement costs + traffic penalties |
| Failed deliveries | 8% of packages fail on first attempt | Deadline-based scoring |
| Driver workload | Average 150+ stops/day | Multi-package task configurations |
| Traffic congestion | 40% time lost in urban areas | Dynamic traffic grid system |
| Environmental impact | Last-mile = 53% of shipping emissions | Fuel optimization rewards |

Training RL agents on this environment can directly transfer to real-world delivery fleet optimization (e.g., Amazon, DoorDash, UPS route planning).

## ⚡ Ironclad Robustness (Superpowers Integrated)

We treat this environment as production-ready software. To guarantee reliability for RL agents, we've integrated **industrial-grade verification**:

- **TDD / Unit Testing**: Over 100 edge-cases mathematically verified via `pytest` (bounds collisions, fuel limits, grader score boundaries).
- **Performance Profiling**: The core engine achieves an astonishing **>74,000 Steps-Per-Second (SPS)**, measured objectively via `scripts/benchmark.py`, vastly exceeding standard RL environment requirements.
- **Systematic Debugging**: Includes a fully interactive visual CLI debugger (`scripts/interactive_debugger.py`) enabling researchers to step-through exact deterministic reward sequences transparently.

---

## 🕹️ Action Space

| Action | Index | Description |
|--------|-------|-------------|
| Move Up | `0` | Move vehicle one cell up (row - 1) |
| Move Down | `1` | Move vehicle one cell down (row + 1) |
| Move Left | `2` | Move vehicle one cell left (col - 1) |
| Move Right | `3` | Move vehicle one cell right (col + 1) |
| Deliver | `4` | Deliver a carried package at current location |
| Refuel | `5` | Refuel vehicle at a fuel station |

**Total: 6 discrete actions**

---

## 👁️ Observation Space

```python
{
    "vehicle": {
        "position": [row, col],     # Current grid position
        "fuel": float,               # Remaining fuel (0-100)
        "carrying": [int, ...]       # Package IDs being carried
    },
    "packages": [
        {
            "package_id": int,
            "pickup_position": [row, col],
            "delivery_position": [row, col],
            "picked_up": bool,
            "delivered": bool,
            "priority": int,          # 0=normal, 1=urgent, 2=fragile
            "deadline": int | null    # Max steps to deliver
        }
    ],
    "traffic_grid": [[int, ...]],     # 2D grid: 0=clear, 1=congested
    "fuel_stations": [[row, col]],    # Fuel station locations
    "grid_size": int,
    "time_elapsed": int,
    "max_steps": int,
    "weather": int,                    # 0=clear, 1=rain, 2=fog
    "total_reward": float,
    "packages_delivered": int,
    "packages_total": int,
    "done": bool
}
```

---

## 🏆 Reward Design

Our reward function uses **comprehensive reward shaping** to guide agent learning:

### Core Rewards

| Event | Reward | Rationale |
|-------|--------|-----------|
| Deliver package | **+20** | Primary objective |
| Pick up package | **+2** | Encourage exploration |
| All packages delivered | **+50** | Completion bonus |
| Move step | **-1** | Encourage efficiency |
| Wall collision | **-2** | Penalize invalid moves |
| Failed delivery | **-1** | Don't spam deliver action |

### Traffic & Weather Penalties

| Event | Reward | Rationale |
|-------|--------|-----------|
| Enter traffic cell | **-5** | Avoid congestion |
| Move in rain | **-2** | Weather awareness |
| Move in fog | **-1** | Reduced visibility |

### Resource Management

| Event | Reward | Rationale |
|-------|--------|-----------|
| Refuel at station | **+5** | Strategic refueling |
| Refuel (tank full) | **-0.5** | Don't waste time |
| Not at station | **-1** | Know your map |
| Fuel empty | **-10** | Terminal state |

### Advanced Features (Differentiators)

| Event | Reward | Rationale |
|-------|--------|-----------|
| Urgent package bonus | **+15** | Priority handling |
| Fragile package bonus | **+10** | Careful delivery |
| Deadline met | **+10** | Time management |
| Deadline missed | **-5** | Late penalty |

---

## 📋 Tasks

### Easy — Learning the Basics
| Parameter | Value |
|-----------|-------|
| Grid Size | 5×5 |
| Packages | 2 |
| Initial Fuel | 100 |
| Traffic | None |
| Deadlines | None |
| Max Steps | 100 |

### Medium — Traffic Navigation
| Parameter | Value |
|-----------|-------|
| Grid Size | 8×8 |
| Packages | 3 |
| Initial Fuel | 60 |
| Traffic | Static |
| Deadlines | Some |
| Max Steps | 200 |

### Hard — Full Challenge
| Parameter | Value |
|-----------|-------|
| Grid Size | 10×10 |
| Packages | 5 |
| Initial Fuel | 40 |
| Traffic | Dynamic (changes every 5 steps) |
| Deadlines | All packages |
| Priorities | Yes (normal/urgent/fragile) |
| Weather | Yes (clear/rain/fog) |
| Max Steps | 300 |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│              LLM Agent (Client)              │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
│  │ OpenAI  │  │   TRL    │  │  Custom RL │  │
│  │  Client │  │  GRPO    │  │  Agent     │  │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘  │
└───────┼────────────┼──────────────┼──────────┘
        │   MCP/WebSocket/HTTP      │
┌───────▼────────────▼──────────────▼──────────┐
│          OpenEnv FastAPI Server               │
│  ┌────────────────────────────────────────┐   │
│  │   UrbanDeliveryEnvironment (MCP)       │   │
│  │   Tools: move, deliver, refuel,        │   │
│  │          get_observation, set_task      │   │
│  └──────────────┬─────────────────────────┘   │
│  ┌──────────────▼─────────────────────────┐   │
│  │   DeliveryEnvironment (Core Engine)    │   │
│  │   Grid ─ Vehicle ─ Packages ─ Traffic  │   │
│  │   Fuel ─ Weather ─ Rewards ─ Graders   │   │
│  └────────────────────────────────────────┘   │
│              Docker Container                 │
└───────────────────────────────────────────────┘
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Docker Desktop (for containerized deployment)

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/urban-delivery-env.git
cd urban_delivery_env

# Install in editable mode
pip install -e .

# Or with uv (faster)
uv pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or run directly
python -m server.app
```

### Docker

```bash
# Build
cd urban_delivery_env
docker build -f server/Dockerfile -t urban-delivery-env .

# Run
docker run -p 8000:8000 urban-delivery-env
```

### HuggingFace Spaces

```bash
cd urban_delivery_env
openenv push --repo-id YOUR_USERNAME/urban-delivery-env
```

---

## 🤖 Running the Baseline Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export OPENAI_API_KEY=sk-...

python inference.py
```

### Baseline Scores (gpt-4o-mini)

| Task | Score | Steps |
|------|-------|-------|
| Easy | ~0.50 | ≤100 |
| Medium | ~0.35 | ≤200 |
| Hard | ~0.20 | ≤300 |

> Scores vary by model. Better models (GPT-4, Claude) typically score higher.

---

## 📐 Grading Methodology

### Easy Grader
```
score = packages_delivered / total_packages
```

### Medium Grader
```
score = 0.5 × completion + 0.3 × fuel_efficiency + 0.2 × time_efficiency
```

### Hard Grader
```
score = 0.4 × completion
      + 0.2 × fuel_efficiency
      + 0.2 × deadline_compliance
      + 0.1 × priority_accuracy
      + 0.1 × reward_normalized
```

All graders return **deterministic scores between 0.0 and 1.0**.

---

## 🔧 Environment API

### MCP Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `move` | `direction: str` | Move vehicle ("up", "down", "left", "right") |
| `deliver` | — | Deliver package at current position |
| `refuel` | — | Refuel at fuel station |
| `get_observation` | — | Get current state without action |
| `set_task` | `difficulty: str` | Switch task ("easy", "medium", "hard") |

### Python Client

```python
from urban_delivery_env import UrbanDeliveryEnv

with UrbanDeliveryEnv(base_url="http://localhost:8000") as env:
    env.reset()
    tools = env.list_tools()
    result = env.call_tool("set_task", difficulty="hard")
    result = env.call_tool("move", direction="right")
    result = env.call_tool("deliver")
```

---

## 📜 License

BSD 3-Clause License

---

## 🙏 Acknowledgments

- [Meta PyTorch — OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- [Hugging Face — Environment Hub](https://huggingface.co/collections/openenv)
- [Scaler School of Technology — Hackathon](https://scaler.com)
