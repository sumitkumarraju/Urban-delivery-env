#!/usr/bin/env python3
"""Run an LLM agent through all three delivery tasks and report scores.

Env vars:
    API_BASE_URL   — chat-completions endpoint (default: OpenAI)
    MODEL_NAME     — model id
    HF_TOKEN / OPENAI_API_KEY — bearer token
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


def log(msg: str) -> None:
    print(msg, flush=True)


SYSTEM_PROMPT = """You are an expert delivery driver AI navigating a grid-based city.

GOAL: Pick up ALL packages and deliver them to their destinations as fast as possible while managing fuel.

ACTIONS (respond with ONLY the action number):
0 = Move Up (row - 1)
1 = Move Down (row + 1)
2 = Move Left (col - 1)
3 = Move Right (col + 1)
4 = Deliver Package (when at delivery location)
5 = Refuel (when at fuel station)

KEY RULES:
- Packages auto-pickup when you step on their pickup_position
- To DELIVER: navigate to the package's delivery_position, then use action 4
- Refuel ONLY at fuel_station locations using action 5
- Moving out of bounds wastes a turn with a -2 penalty — stay in grid [0, grid_size-1]
- Traffic cells cost -5 reward and extra fuel — avoid them if possible
- URGENT packages (priority=1) give +15 bonus — deliver them first
- If fuel drops below 10, refuel immediately or you'll die (-10 penalty)

STRATEGY:
1. Check fuel first — if low, go to nearest fuel station
2. If carrying packages, navigate to the closest delivery point
3. If not carrying, navigate to the closest undelivered package pickup
4. Prefer URGENT packages over normal ones
5. Avoid traffic cells when possible
6. Use Manhattan distance to find closest targets

RESPONSE: Reply with ONLY a single digit 0-5. No explanation."""


def compute_optimal_direction(current_pos, target_pos, grid_size, traffic_grid=None):
    """Compute best single-step move toward target, avoiding traffic if possible."""
    cr, cc = current_pos
    tr, tc = target_pos

    candidates = []
    if tr < cr:
        candidates.append((0, cr - 1, cc))
    if tr > cr:
        candidates.append((1, cr + 1, cc))
    if tc < cc:
        candidates.append((2, cr, cc - 1))
    if tc > cc:
        candidates.append((3, cr, cc + 1))

    if not candidates:
        return 0

    best_action = candidates[0][0]
    best_dist = float("inf")
    best_traffic = True

    for action, nr, nc in candidates:
        if not (0 <= nr < grid_size and 0 <= nc < grid_size):
            continue
        dist = abs(nr - tr) + abs(nc - tc)
        has_traffic = traffic_grid and traffic_grid[nr][nc] == 1 if traffic_grid else False

        if (not has_traffic and best_traffic) or (has_traffic == best_traffic and dist < best_dist):
            best_action = action
            best_dist = dist
            best_traffic = has_traffic

    return best_action


def heuristic_action(state: dict) -> int:
    """Deterministic heuristic agent as fallback when no LLM is available."""
    pos = state["position"]
    fuel = state["fuel"]
    carrying = state["carrying"]
    packages = state["packages"]
    fuel_stations = state["fuel_stations"]
    grid_size = state["grid_size"]
    traffic_grid_data = state.get("traffic_cells", [])

    traffic_grid = [[0] * grid_size for _ in range(grid_size)]
    for r, c in traffic_grid_data:
        traffic_grid[r][c] = 1

    nearest_station_dist = float("inf")
    nearest_station = None
    for s in fuel_stations:
        d = abs(s[0] - pos[0]) + abs(s[1] - pos[1])
        if d < nearest_station_dist:
            nearest_station_dist = d
            nearest_station = s

    if fuel <= nearest_station_dist + 3 and nearest_station:
        if nearest_station_dist == 0:
            return 5
        return compute_optimal_direction(pos, nearest_station, grid_size, traffic_grid)

    if carrying:
        best_dist = float("inf")
        best_target = None
        for pkg in packages:
            if pkg["id"] in carrying and not pkg["delivered"]:
                d = abs(pkg["delivery"][0] - pos[0]) + abs(pkg["delivery"][1] - pos[1])
                if d < best_dist:
                    best_dist = d
                    best_target = pkg["delivery"]

        if best_target:
            if best_dist == 0:
                return 4
            return compute_optimal_direction(pos, best_target, grid_size, traffic_grid)

    undelivered = [p for p in packages if not p["picked_up"] and not p["delivered"]]
    if undelivered:
        urgent = [p for p in undelivered if p.get("priority") == "urgent"]
        targets = urgent if urgent else undelivered

        best_dist = float("inf")
        best_target = None
        for pkg in targets:
            d = abs(pkg["pickup"][0] - pos[0]) + abs(pkg["pickup"][1] - pos[1])
            if d < best_dist:
                best_dist = d
                best_target = pkg["pickup"]

        if best_target:
            return compute_optimal_direction(pos, best_target, grid_size, traffic_grid)

    carried_undelivered = [p for p in packages if p["picked_up"] and not p["delivered"]]
    if carried_undelivered:
        best_dist = float("inf")
        best_target = None
        for pkg in carried_undelivered:
            d = abs(pkg["delivery"][0] - pos[0]) + abs(pkg["delivery"][1] - pos[1])
            if d < best_dist:
                best_dist = d
                best_target = pkg["delivery"]
        if best_target:
            if best_dist == 0:
                return 4
            return compute_optimal_direction(pos, best_target, grid_size, traffic_grid)

    return 0


def get_llm_action(client: OpenAI, model: str, state_summary: dict, history: list) -> int:
    """Get an action from the LLM based on current state."""
    compact_state = {
        "position": state_summary["position"],
        "fuel": state_summary["fuel"],
        "carrying": state_summary["carrying"],
        "grid_size": state_summary["grid_size"],
        "weather": state_summary["weather"],
        "step": state_summary["step"],
        "max_steps": state_summary["max_steps"],
        "delivered": state_summary["delivered"],
        "total": state_summary["total"],
        "reward": state_summary["reward"],
        "hint": state_summary["hint"],
        "packages": [
            {
                "id": p["id"],
                "pickup": p["pickup"],
                "delivery": p["delivery"],
                "picked_up": p["picked_up"],
                "delivered": p["delivered"],
                "priority": p["priority"],
            }
            for p in state_summary["packages"]
            if not p["delivered"]
        ],
        "fuel_stations": state_summary["fuel_stations"],
    }

    state_str = json.dumps(compact_state, separators=(",", ":"))
    user_msg = f"State: {state_str}\n\nAction?"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-4:]:
        messages.append({"role": "user", "content": h["state"]})
        messages.append({"role": "assistant", "content": str(h["action"])})
    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=5,
            temperature=0.0,
        )
        answer = response.choices[0].message.content.strip()
        action = int(answer[0]) if answer and answer[0].isdigit() else 0
        return max(0, min(5, action))
    except Exception as e:
        print(f"  LLM error: {e}, using heuristic fallback", file=sys.stderr)
        return heuristic_action(state_summary)


def run_task(
    task_name: str,
    client: OpenAI | None,
    model: str,
    use_llm: bool,
) -> list[int]:
    """Execute one task. Returns the action list."""
    config = ALL_TASKS[task_name]
    env = DeliveryEnvironment(config)
    obs = env.reset()
    actions: list[int] = []
    history: list[dict] = []

    step = 0

    while not obs.done and step < config.max_steps:
        state = env.get_state_summary()

        if use_llm and client is not None:
            action_int = get_llm_action(client, model, state, history)
            history.append({"state": json.dumps(state, separators=(",", ":")), "action": action_int})
        else:
            action_int = heuristic_action(state)

        actions.append(action_int)
        action = DeliveryAction(action=action_int)
        obs, reward_info = env.step(action)
        step += 1
        log(f"[STEP] task={task_name} step={step} action={action_int} reward={reward_info.step_reward:.4f} cumulative_reward={reward_info.cumulative_reward:.4f} done={obs.done}")

    return actions


def main():
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")

    api_key = HF_TOKEN or os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        log("WARNING: No API key set. Running with heuristic agent.")
        use_llm = False
    else:
        use_llm = True

    if use_llm:
        client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
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

        config = ALL_TASKS[task_name]
        log(f"[START] task={task_name}")

        actions = run_task(task_name, client, MODEL_NAME, use_llm)
        score, explanation = graders[task_name].grade_with_explanation(actions)
        elapsed = time.time() - task_start

        results[task_name] = {
            "score": score,
            "steps": len(actions),
            "time_seconds": round(elapsed, 2),
            "explanation": explanation,
        }

        log(f"[END] task={task_name} score={score:.6f} steps={len(actions)}")

    total_elapsed = time.time() - total_start
    avg_score = sum(r["score"] for r in results.values()) / len(results)

    log("")
    log("=" * 60)
    log("  FINAL RESULTS")
    log("=" * 60)
    for task, res in results.items():
        log(f"  {task:8s} -> Score: {res['score']:.6f} | Steps: {res['steps']} | Time: {res['time_seconds']}s")
    log(f"  {'AVERAGE':8s} -> Score: {avg_score:.6f}")
    log(f"  Total Time: {total_elapsed:.1f}s")
    log("=" * 60)

    if total_elapsed > 1200:
        log("  WARNING: Exceeded 20-minute limit!")
    else:
        log(f"  Within time limit ({1200 - total_elapsed:.0f}s remaining)")


if __name__ == "__main__":
    main()
