#!/usr/bin/env python3
"""
Baseline Inference Script for Urban Delivery Optimization Environment.

Uses OpenAI-compatible API to run an LLM agent on all 3 tasks.

Required environment variables:
    API_BASE_URL  — Base URL for the API (e.g., https://api.openai.com/v1)
    MODEL_NAME    — Model to use (e.g., gpt-4o-mini)
    OPENAI_API_KEY — API key

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export OPENAI_API_KEY=sk-...
    python inference.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

from env import DeliveryEnvironment
from models.action import DeliveryAction, ACTION_NAMES
from tasks import ALL_TASKS
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader


SYSTEM_PROMPT = """You are an expert delivery driver AI navigating a city grid.

Your goal: Pick up and deliver all packages efficiently while managing fuel.

ACTIONS (respond with ONLY the action number):
0 = Move Up (row - 1)
1 = Move Down (row + 1)
2 = Move Left (col - 1)
3 = Move Right (col + 1)
4 = Deliver Package (at delivery location)
5 = Refuel (at fuel station)

STRATEGY:
- Navigate to package pickup locations first
- Packages are auto-picked up when you step on them
- Navigate to delivery target locations, then use action 4
- Watch your fuel! Refuel at fuel stations (action 5) before running out
- Avoid traffic cells (they cost extra fuel and give penalties)
- Urgent packages have deadlines — deliver them first
- Moving into a wall costs a penalty — stay within grid bounds

RESPONSE FORMAT: Reply with ONLY a single digit (0-5). Nothing else."""


def get_llm_action(client: OpenAI, model: str, state_summary: dict) -> int:
    """Get an action from the LLM based on current state."""
    state_str = json.dumps(state_summary, indent=2)
    user_msg = f"Current state:\n{state_str}\n\nWhat action do you take? (0-5)"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        action = int(answer[0]) if answer and answer[0].isdigit() else 0
        return max(0, min(5, action))
    except Exception as e:
        print(f"  LLM error: {e}, using random action")
        import random
        return random.randint(0, 3)


def run_task(task_name: str, client: OpenAI, model: str, max_llm_steps: int = 50) -> list[int]:
    """Run a single task using the LLM agent.

    Args:
        task_name: Task difficulty level.
        client: OpenAI client.
        model: Model name.
        max_llm_steps: Max steps to query the LLM.

    Returns:
        List of action integers taken.
    """
    config = ALL_TASKS[task_name]
    env = DeliveryEnvironment(config)
    obs = env.reset()
    actions: list[int] = []

    print(f"\n{'='*50}")
    print(f"  Task: {task_name.upper()}")
    print(f"  Grid: {config.grid_size}x{config.grid_size}")
    print(f"  Packages: {config.num_packages}")
    print(f"  Fuel: {config.initial_fuel}")
    print(f"  Features: traffic={config.has_traffic}, deadlines={config.has_deadlines}, "
          f"priorities={config.has_priorities}, weather={config.has_weather}")
    print(f"{'='*50}")

    step = 0
    while not obs.done and step < max_llm_steps:
        state = env.get_state_summary()
        action_int = get_llm_action(client, model, state)
        actions.append(action_int)

        action = DeliveryAction(action=action_int)
        obs, reward = env.step(action)

        if step % 10 == 0 or obs.done:
            print(f"  Step {step:3d} | Action: {ACTION_NAMES.get(action_int, '?'):15s} | "
                  f"Delivered: {obs.packages_delivered}/{obs.packages_total} | "
                  f"Fuel: {obs.vehicle.fuel:5.1f} | Reward: {obs.total_reward:+7.1f}")

        step += 1

    return actions


def main():
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        print("WARNING: OPENAI_API_KEY not set. Running with random actions as fallback.")
        use_llm = False
    else:
        use_llm = True

    if use_llm:
        client = OpenAI(base_url=api_base, api_key=api_key)
    else:
        client = None

    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }

    print("\n" + "=" * 60)
    print("  URBAN DELIVERY OPTIMIZATION — BASELINE INFERENCE")
    print("=" * 60)
    print(f"  Model: {model}")
    print(f"  API Base: {api_base}")
    print(f"  LLM Enabled: {use_llm}")

    total_start = time.time()
    results = {}

    for task_name in ["easy", "medium", "hard"]:
        task_start = time.time()

        if use_llm:
            actions = run_task(task_name, client, model)
        else:
            import random
            random.seed(ALL_TASKS[task_name].seed + 999)
            config = ALL_TASKS[task_name]
            env = DeliveryEnvironment(config)
            obs = env.reset()
            actions = []
            for _ in range(config.max_steps):
                if obs.done:
                    break
                a = random.randint(0, 3)
                actions.append(a)
                action = DeliveryAction(action=a)
                obs, _ = env.step(action)

        score = graders[task_name].grade(actions)
        elapsed = time.time() - task_start

        results[task_name] = {
            "score": score,
            "steps": len(actions),
            "time_seconds": round(elapsed, 2),
        }

        print(f"\n  ✅ {task_name.upper()} — Score: {score:.4f} | "
              f"Steps: {len(actions)} | Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    for task, res in results.items():
        print(f"  {task:8s} → Score: {res['score']:.4f}")
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    print(f"  {'AVERAGE':8s} → Score: {avg_score:.4f}")
    print(f"\n  Total Time: {total_elapsed:.1f}s")
    print("=" * 60)

    if total_elapsed > 1200:
        print("  ⚠️  WARNING: Exceeded 20-minute limit!")
    else:
        print(f"  ✅ Within time limit ({1200 - total_elapsed:.0f}s remaining)")


if __name__ == "__main__":
    main()
