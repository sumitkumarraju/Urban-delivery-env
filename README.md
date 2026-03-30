# 🚚 Urban Delivery Optimization Environment

> A production-grade reinforcement learning environment for training AI agents to optimize urban package delivery logistics, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-13%2F13_Passing-brightgreen)]()
[![SPS](https://img.shields.io/badge/Performance-76%2C690_SPS-orange)]()
[![License](https://img.shields.io/badge/License-BSD--3-orange)](LICENSE)

---

## 🎯 Why This Environment Is Different

Most hackathon RL environments are toy grid worlds with move-and-score mechanics. **This environment simulates a real logistics problem** — the kind Amazon, UPS, and DoorDash solve daily for billions of dollars.

What sets us apart:

| Differentiator | What It Means |
|----------------|---------------|
| 🎯 **Multi-constraint optimization** | Agent balances fuel, time, traffic, weather simultaneously — not just "reach the goal" |
| 📦 **Package priority system** | Urgent/fragile packages with deadlines force the agent to make strategic trade-offs |
| 🚦 **Dynamic traffic** | Traffic patterns change every 5 steps on hard mode — agent must adapt in real-time |
| 🌧️ **Weather effects** | Rain increases fuel cost, fog reduces visibility — environmental uncertainty |
| 🏋️ **Vehicle capacity limits** | Cannot carry unlimited packages — forces route optimization |
| 🗣️ **LLM-native hints** | Natural language observation hints built-in — designed for LLM agents, not just RL |
| 📊 **Transparent grading** | Every grader explains its score with a full factor breakdown |
| ⚡ **76,000+ SPS** | Industrial-grade performance — zero bottleneck for agent training |

---

## 💡 Motivation — Why This Problem Matters

Urban delivery logistics is a **$500B+ global industry** facing critical optimization challenges:

| Challenge | Real-World Impact | Our Simulation |
|-----------|------------------|----------------|
| Route efficiency | 30% of delivery costs are fuel | Movement costs + traffic penalties |
| Failed deliveries | 8% of packages fail on first attempt | Deadline-based scoring |
| Driver workload | Average 150+ stops/day | Multi-package task configurations |
| Traffic congestion | 40% time lost in urban areas | Dynamic traffic grid system |
| Environmental impact | Last-mile = 53% of shipping emissions | Fuel optimization rewards |
| Fleet capacity | Vehicles have weight/volume limits | Max carrying capacity constraint |

Training RL agents on this environment can directly transfer to real-world delivery fleet optimization (e.g., Amazon, DoorDash, UPS route planning).

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
            "deadline": int | null,   # Max steps to deliver
            "delivery_step": int | null  # Exact step when delivered (for grading)
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
    "done": bool,

    # LLM-Friendly Fields
    "carrying_count": int,             # Current packages being carried
    "max_carrying": int,               # Vehicle capacity limit
    "nearest_package_distance": int,   # Manhattan distance to closest target
    "nearest_fuel_station_distance": int,
    "hint": str                        # Natural language advice (see below)
}
```

### 🗣️ Natural Language Hints (LLM-Native Design)

Unlike traditional RL environments, this environment generates **natural language hints** for LLM agents:

```
"CRITICAL: Fuel at 3! Refuel immediately or you will die."
"Carrying 2 package(s): pkg 0 → [4,7], pkg 2 → [1,3]. Navigate to delivery location and use DELIVER."
"URGENT package 1 at [2,5] — pick it up first!"
"DEADLINE ALERT: Package 3 must be delivered in 4 steps!"
```

This bridges the gap between structured observations and LLM comprehension.

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
| Vehicle full (can't pick up) | **msg** | Capacity awareness |

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
| Max Carrying | 3 |
| Traffic | None |
| Deadlines | None |
| Max Steps | 100 |

### Medium — Traffic Navigation
| Parameter | Value |
|-----------|-------|
| Grid Size | 8×8 |
| Packages | 3 |
| Initial Fuel | 60 |
| Max Carrying | 3 |
| Traffic | Static |
| Deadlines | Some |
| Max Steps | 200 |

### Hard — Full Challenge
| Parameter | Value |
|-----------|-------|
| Grid Size | 10×10 |
| Packages | 5 |
| Initial Fuel | 40 |
| Max Carrying | 3 |
| Traffic | Dynamic (changes every 5 steps) |
| Deadlines | All packages |
| Priorities | Yes (normal/urgent/fragile) |
| Weather | Yes (clear/rain/fog) |
| Max Steps | 300 |

---

## 📐 Grading Methodology

All graders return **deterministic scores between 0.0 and 1.0** and provide a full **explanation breakdown**.

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
      + 0.2 × deadline_compliance    ← uses per-package delivery_step
      + 0.1 × priority_accuracy
      + 0.1 × reward_normalized
```

### Example Grader Explanation Output
```json
{
  "completion": {"raw": 0.8, "weight": 0.4, "weighted": 0.32},
  "fuel_efficiency": {"raw": 0.45, "weight": 0.2, "weighted": 0.09},
  "deadline_compliance": {"raw": 1.0, "weight": 0.2, "weighted": 0.20, "met": 3, "total": 3},
  "priority_accuracy": {"raw": 0.5, "weight": 0.1, "weighted": 0.05, "delivered": 1, "total": 2},
  "reward_normalized": {"raw": 0.3, "weight": 0.1, "weighted": 0.03},
  "final_score": 0.69
}
```

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
│  │          get_observation, get_hint,     │   │
│  │          set_task                       │   │
│  └──────────────┬─────────────────────────┘   │
│  ┌──────────────▼─────────────────────────┐   │
│  │   DeliveryEnvironment (Core Engine)    │   │
│  │   Grid ─ Vehicle ─ Packages ─ Traffic  │   │
│  │   Fuel ─ Weather ─ Rewards ─ Capacity  │   │
│  │   NL Hints ─ Graders ─ Explanations    │   │
│  └────────────────────────────────────────┘   │
│              Docker Container                 │
└───────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
urban_delivery_env/
├── openenv.yaml                ← OpenEnv manifest (spec_version: 1)
├── pyproject.toml              ← Package config + dependencies
├── Makefile                    ← Dev commands: test, bench, start, debug
├── README.md                   ← This file
├── .gitignore
├── .dockerignore
│
├── __init__.py                 ← Package exports (defensive imports)
├── env.py                      ← Core simulation engine (530+ lines)
│                                  └─ Grid, Vehicle, Packages, Traffic, Weather
│                                  └─ Reward shaping, NL hints, capacity
├── client.py                   ← MCPToolClient subclass for remote usage
├── inference.py                ← Baseline LLM inference script (OpenAI API)
│
├── models/                     ← Pydantic typed models
│   ├── __init__.py
│   ├── action.py               ← 6 actions (move×4, deliver, refuel)
│   ├── observation.py          ← Vehicle, Package, Weather, delivery_step
│   └── reward.py               ← Reward breakdown per step
│
├── tasks/                      ← Task difficulty configurations
│   ├── __init__.py             ← ALL_TASKS registry
│   ├── easy.py                 ← 5×5, 2 pkgs, 100 fuel, no traffic
│   ├── medium.py               ← 8×8, 3 pkgs, 60 fuel, static traffic
│   └── hard.py                 ← 10×10, 5 pkgs, 40 fuel, dynamic everything
│
├── graders/                    ← Deterministic scoring (0.0–1.0)
│   ├── __init__.py
│   ├── base_grader.py          ← Abstract base + grade_with_explanation()
│   ├── easy_grader.py          ← score = completion ratio
│   ├── medium_grader.py        ← score = 0.5×completion + 0.3×fuel + 0.2×time
│   └── hard_grader.py          ← score = 5-factor weighted (completion, fuel,
│                                        deadline, priority, reward)
│
├── server/                     ← OpenEnv server + Docker
│   ├── urban_delivery_environment.py  ← MCPEnvironment wrapper (6 MCP tools)
│   ├── app.py                  ← FastAPI entry point
│   ├── Dockerfile              ← Multi-stage production build
│   └── requirements.txt        ← Docker dependencies
│
├── scripts/                    ← Dev utilities
│   ├── benchmark.py            ← Raw SPS performance profiler
│   └── interactive_debugger.py ← Visual CLI state stepping tool
│
└── tests/                      ← pytest test suite (13 tests)
    ├── __init__.py
    ├── conftest.py             ← Shared fixtures (easy_config, hard_config)
    ├── test_env.py             ← Env logic: walls, fuel, capacity, hints
    └── test_graders.py         ← Grader: bounds, explanations, determinism
```

---

## ⚡ Performance & Quality Metrics

| Metric | Result |
|--------|--------|
| **Steps/Second (SPS)** | 76,690 (7.6× above 10K threshold) |
| **Unit Tests** | 13/13 passing |
| **Test Coverage** | Environment logic, graders, hints, capacity |
| **Grading** | Deterministic, explainable, 0.0–1.0 |
| **Test Speed** | Full suite: 0.02 seconds |

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

# Run tests
make test

# Run performance benchmark
make bench

# Start the server
make start

# Interactive debugger
make debug
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

### Baseline Scores

| Task | Random Agent | GPT-4o (via Bytez) | Improvement |
|------|-------------|-------------------|-------------|
| Easy | 0.0000 | **1.0000** ✅ | ♾️ |
| Medium | 0.1390 | **0.5583** | 4× |
| Hard | 0.2000 | **0.3900** | 2× |
| **Average** | 0.1130 | **0.6494** | **5.7×** |

> GPT-4o scored a **perfect 1.0 on Easy** (13 steps, 34s), demonstrating the environment's
> LLM-native design. The NL hints and structured observations enable strong reasoning.
> Total inference time: 214s — well within the 20-minute limit.

---

## 🔧 Environment API

### MCP Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `move` | `direction: str` | Move vehicle ("up", "down", "left", "right") |
| `deliver` | — | Deliver package at current position |
| `refuel` | — | Refuel at fuel station |
| `get_observation` | — | Get current state + NL hint |
| `get_hint` | — | Get natural language advice only |
| `set_task` | `difficulty: str` | Switch task ("easy", "medium", "hard") |

### Python Client

```python
from urban_delivery_env import UrbanDeliveryEnv

with UrbanDeliveryEnv(base_url="http://localhost:8000") as env:
    env.reset()
    tools = env.list_tools()
    result = env.call_tool("set_task", difficulty="hard")
    result = env.call_tool("move", direction="right")
    result = env.call_tool("get_hint")  # Get NL advice
    result = env.call_tool("deliver")
```

---

## 🧪 Development

```bash
make test          # Run full test suite (13 tests)
make bench         # Performance benchmark (target: >10K SPS)
make debug         # Interactive CLI debugger
make start         # Run the server with hot-reload
make validate      # OpenEnv spec validation
make inference     # Run baseline LLM inference
```

---

## 📜 License

BSD 3-Clause License

---

## 🙏 Acknowledgments

- [Meta PyTorch — OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- [Hugging Face — Environment Hub](https://huggingface.co/collections/openenv)
- [Scaler School of Technology — Hackathon](https://scaler.com)
