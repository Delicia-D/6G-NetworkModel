from __future__ import annotations
from Datageneration import User,CallGenerator,CallSimulator
from nonpredictor import NonPredictiveCallAdmissionController
from predictor import PredictiveCallAdmissionController, AdmissionMetrics,RAT_CAPACITIES,RATPool,RATState,RB_PER_SERVICE

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import heapq
import pandas as pd
import numpy as np
from math import exp, log, sin, cos, pi
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from nn import CallDurationPredictor

print("=" * 70)

# Create users with gamma-distributed relationships
N = 100
users = {i: User(i) for i in range(1, N+1)}
_rng = np.random.default_rng(42)

# Create gamma-distributed relationships
print(" Creating gamma-distributed user relationships...")
gamma_shape, gamma_scale = 1.2,4  # Parameters for right-skewed distribution

for i in range(1, N+1):
    for j in range(i+1, N+1):
        if _rng.random() < 0.35:  # 35% connection rate
            # Generate gamma-distributed relationship score
            score = _rng.gamma(gamma_shape, gamma_scale)
            
            # Cap at maximum and ensure minimum
            score = min(score, 20)  
            score = max(score, 1)  
            score=round(score)
            
            users[i].addRelationship(j, score)
            users[j].addRelationship(i, score)

# Analyze relationship distribution
all_scores = []
all_calc_scores = []  # For the calculation-scale scores
for user in users.values():
    all_scores.extend(list(user.relationships.values()))
    # Get the calculation-scale scores too
    for other_id in user.relationships:
        all_calc_scores.append(user.getRelationshipScore(other_id))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(all_scores, bins=20, alpha=0.7, edgecolor='black', color='blue', range=(1, 20))
plt.xlabel('Relationship Score (1-20 scale)')
plt.ylabel('Number of Relationships')
plt.title('All Assigned Relationships (1-20 scale)')
plt.axvline(np.mean(all_scores), color='red', linestyle='--', 
           label=f'Mean: {np.mean(all_scores):.1f}')
plt.legend()
plt.grid(alpha=0.3)
print(f"Relationship score distribution (1-20 scale):")
print(f"   Mean: {np.mean(all_scores):.1f}")
print(f"   Median: {np.median(all_scores):.1f}")
print(f"   Std: {np.std(all_scores):.1f}")
print(f"   Min: {np.min(all_scores):.1f}")
print(f"   Max: {np.max(all_scores):.1f}")
print(f"   1-7 (acquaintances): {sum(1 for s in all_scores if 1 <= s <= 7) / len(all_scores):.1%}")
print(f"   8-12 (regular friends): {sum(1 for s in all_scores if 8 <= s <= 12) / len(all_scores):.1%}")
print(f"   13-20 (close friends): {sum(1 for s in all_scores if 13 <= s <= 20) / len(all_scores):.1%}")


# Run ultra-predictable simulation with gamma relationships
sim = CallSimulator(
    users=users,
    start_dt=datetime(2025, 1, 1),
    end_dt=datetime(2025, 10, 30),
    rng=np.random.default_rng(123),
    calls_per_day_mean=100,
    noise_level=0.02,
    gamma_shape=gamma_shape,
    gamma_scale=gamma_scale
)

rows = sim.run()

# Create and save DataFrame
df = pd.DataFrame(rows)
df.to_csv("gamma_relationship_call_data.csv", index=False)

print(f"\n GAMMA-RELATIONSHIP DATA GENERATED!")
print(f"Total calls: {len(df):,}")

