#!/usr/bin/env python3
"""
Baseline Inference Script for Urban Delivery Optimization Environment.

Uses OpenAI-compatible API to run an LLM agent on all 3 tasks.
Emits structured [START]/[STEP]/[END] output blocks to stdout for the
OpenEnv validator pipeline.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Print to stdout with flush for validator visibility."""
    print(msg, flush=True)


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
        print(f"  LLM error: {e}, using random action", file=sys.stderr)
        import random
        return random.randint(0, 5)


# ---------------------------------------------------------------------------
# Task runner — emits [STEP] lines
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    client: OpenAI | None,
    model: str,
    use_llm: bool,
) -> list[int]:
    """Run a single task and emit [STEP] structured output for every step.

    Returns:
        List of action integers taken.
    """
    config = ALL_TASKS[task_name]
    env = DeliveryEnvironment(config)
    obs = env.reset()
    actions: list[int] = []

    step = 0

    if use_llm and client is not None:
        # ---- LLM-driven loop ----
        while not obs.done and step < config.max_steps:
            state = env.get_state_summary()
            action_int = get_llm_action(client, model, state)
            actions.append(action_int)

            action = DeliveryAction(action=action_int)
            obs, reward_info = env.step(action)
            step += 1

            # Emit structured [STEP] line exactly as requested
            log(f"[STEP] task={task_name} step={step} action={action_int} reward={reward_info.step_reward:.4f} cumulative_reward={reward_info.cumulative_reward:.4f} done={obs.done}")
    else:
        # ---- Random fallback loop ----
        import random
        random.seed(config.seed + 999)
        while not obs.done and step < config.max_steps:
            a = random.randint(0, 5)
            actions.append(a)

            action = DeliveryAction(action=a)
            obs, reward_info = env.step(action)
            step += 1

            # Emit structured [STEP] line exactly as requested
            log(f"[STEP] task={task_name} step={step} action={a} reward={reward_info.step_reward:.4f} cumulative_reward={reward_info.cumulative_reward:.4f} done={obs.done}")

    return actions


# ---------------------------------------------------------------------------
# Main — emits [START] and [END] blocks
# ---------------------------------------------------------------------------

def main():
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        log("WARNING: OPENAI_API_KEY not set. Running with random actions as fallback.")
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

    total_start = time.time()
    results = {}

    for task_name in ["easy", "medium", "hard"]:
        task_start = time.time()

        # ---- [START] block ----
        config = ALL_TASKS[task_name]
        log(f"[START] task={task_name}")

        # Run the episode (emits [STEP] lines inside)
        actions = run_task(task_name, client, model, use_llm)

        # Grade
        score, explanation = graders[task_name].grade_with_explanation(actions)
        elapsed = time.time() - task_start

        results[task_name] = {
            "score": score,
            "steps": len(actions),
            "time_seconds": round(elapsed, 2),
            "explanation": explanation,
        }

        # ---- [END] block ----
        log(f"[END] task={task_name} score={score:.4f} steps={len(actions)}")

    # ---- Summary ----
    total_elapsed = time.time() - total_start
    avg_score = sum(r["score"] for r in results.values()) / len(results)

    log("")
    log("=" * 60)
    log("  FINAL RESULTS")
    log("=" * 60)
    for task, res in results.items():
        log(f"  {task:8s} -> Score: {res['score']:.4f} | Steps: {res['steps']} | Time: {res['time_seconds']}s")
    log(f"  {'AVERAGE':8s} -> Score: {avg_score:.4f}")
    log(f"  Total Time: {total_elapsed:.1f}s")
    log("=" * 60)

    if total_elapsed > 1200:
        log("  WARNING: Exceeded 20-minute limit!")
    else:
        log(f"  Within time limit ({1200 - total_elapsed:.0f}s remaining)")


if __name__ == "__main__":
    main()
