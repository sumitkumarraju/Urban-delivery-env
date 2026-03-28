#!/usr/bin/env python3
"""
Performance Profiling: Benchmark steps per second (SPS).
For OpenEnv hackathon, ensuring high throughput is crucial for RL training.
"""
import time
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DeliveryEnvironment
from tasks import HARD_TASK
from models.action import DeliveryAction

NUM_STEPS = 100_000

def run_benchmark():
    print(f"Benchmarking Urban Delivery Environment ({NUM_STEPS:,} steps)...")
    env = DeliveryEnvironment(HARD_TASK)
    
    # We want to measure RAW throughput of `step()`, ignoring episodes ending. 
    # To do this cleanly, we'll reset whenever done = True, 
    # but still track total steps across episodes.
    
    obs = env.reset()
    start_time = time.time()
    resets = 0

    for _ in range(NUM_STEPS):
        # Pick random action 0-5
        act = DeliveryAction(action=random.randint(0, 5))
        obs, _ = env.step(act)
        
        if obs.done:
            obs = env.reset()
            resets += 1

    elapsed = time.time() - start_time
    sps = NUM_STEPS / elapsed
    
    print("-" * 50)
    print(f"Total Steps: {NUM_STEPS:,}")
    print(f"Time Elapsed: {elapsed:.3f} s")
    print(f"Episodes Run: {resets:,}")
    print(f"Steps/Sec (SPS): {sps:,.0f}")
    print("-" * 50)
    
    if sps > 10000:
        print("✅ PERFORMANCE PASSED: > 10,000 SPS achieved. Industrial grade.")
    else:
        print("❌ PERFORMANCE FAILED: Bottleneck found. Requires immediate profiling.")
        sys.exit(1)

if __name__ == "__main__":
    run_benchmark()
