#!/usr/bin/env python3
"""
Systematic Debugging: Interactive CLI visualizer.
Provides a human-in-the-loop debugging mechanism to trace the step-by-step
logic of state transitions, reward shapes, and edge-cases (wall collisions, etc.).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DeliveryEnvironment
from tasks import EASY_TASK, HARD_TASK, MEDIUM_TASK
from models.action import DeliveryAction, ActionType


def print_state(obs, reward, last_action):
    """Render grid out to debugging terminal."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("="*60)
    print(" 🛠️  Systematic Debugger  —  Urban Delivery Environment ")
    print("="*60)
    print(f"Step: {obs.time_elapsed:3d}/{obs.max_steps} | Fuel: {obs.vehicle.fuel:5.1f} | Reward: {obs.total_reward:+.1f}")
    print(f"Packages Delivered: {obs.packages_delivered}/{obs.packages_total}")
    
    weather_str = ["☀️ Clear", "🌧️ Rain", "🌫️ Fog"][obs.weather]
    print(f"Weather: {weather_str}")
    
    if last_action:
        print(f"Last Action: {ActionType(last_action.action).name}")
    print(f"Env Msg: {obs.message}")
    
    if reward and reward.breakdown:
        print("\nReward Breakdown Triggered:")
        for k, v in reward.breakdown.items():
            print(f"   -> {k}: {v:+.1f}")
    
    print("\n--- Grid Map ---")
    gs = obs.grid_size
    vr, vc = obs.vehicle.position
    
    station_locs = set(tuple(s) for s in obs.fuel_stations)
    
    # Pre-compute packages
    unpicked = {} 
    delivery = {}
    for p in obs.packages:
        if not p.picked_up:
            unpicked[tuple(p.pickup_position)] = p.package_id
        if p.picked_up and not p.delivered:
            delivery[tuple(p.delivery_position)] = p.package_id

    for r in range(gs):
        row_str = ""
        for c in range(gs):
            pos = (r, c)
            if pos == (vr, vc):
                row_str += " 🚚 "
            elif pos in unpicked:
                row_str += f" P{unpicked[pos]} "
            elif pos in delivery:
                row_str += f" D{delivery[pos]} "
            elif pos in station_locs:
                row_str += " ⛽ "
            elif obs.traffic_grid[r][c] == 1:
                row_str += " 🛑 "
            else:
                row_str += " .  "
        print(row_str)
    
    print("\nControls:")
    print(" w/a/s/d = Move | q = Deliver | e = Refuel | x = Exit")
    print("="*60)


def main():
    print("Select Difficulty: 1=Easy, 2=Medium, 3=Hard")
    diff = input("> ").strip()
    config = EASY_TASK
    if diff == '2': config = MEDIUM_TASK
    elif diff == '3': config = HARD_TASK
    
    env = DeliveryEnvironment(config)
    obs = env.reset(seed=int(os.getenv("DEBUG_SEED", 42)))
    
    reward = None
    last_act = None
    
    while True:
        print_state(obs, reward, last_act)
        
        if obs.done:
            print("\n*** EPISODE LOGIC DONE ***")
            break
            
        cmd = input("Action: ").strip().lower()
        
        # Map input
        act_val = None
        if cmd == 'w': act_val = 0
        elif cmd == 's': act_val = 1
        elif cmd == 'a': act_val = 2
        elif cmd == 'd': act_val = 3
        elif cmd == 'q': act_val = 4
        elif cmd == 'e': act_val = 5
        elif cmd == 'x': break
        else:
            continue
            
        last_act = DeliveryAction(action=act_val)
        obs, reward = env.step(last_act)


if __name__ == "__main__":
    main()
