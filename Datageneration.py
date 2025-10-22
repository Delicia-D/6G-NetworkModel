import pandas as pd
import numpy as np
from math import exp, log, sin, cos, pi
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# =============================
# User in network

# =============================
# User in network
# =============================
class User:
    def __init__(self, userId: int):
        self.userId: int = userId
        self.relationships: Dict[int, float] = {}
        self.callHistory: Dict[int, List[Tuple[
            Optional[float], datetime, str, str, str, Optional[float]
        ]]] = {}

    def addRelationship(self, otherUserId: int, score: float) -> None:
        self.relationships[otherUserId] = float(score)

    def getRelationshipScore(self, otherUserId: int) -> float:
        # Convert from 1-20 scale to 0.1-2.0 scale for calculations
        if otherUserId not in self.relationships:
            return 0.1  # Minimum if no relationship
        return self.relationships[otherUserId] / 10.0  # Convert 1-20 → 0.1-2.0

    def isContact(self, otherUserId: int) -> bool:
        return otherUserId in self.relationships

    def getContacts(self) -> List[int]:
        return list(self.relationships.keys())

    def logCall(self, otherUserId: int, durationSec: Optional[float], timestamp: datetime,
               serviceType: str, callerLocation: str, calleeLocation: str, 
               timeSinceLastCallSec: Optional[float]) -> None:
        if otherUserId not in self.callHistory:
            self.callHistory[otherUserId] = []
        self.callHistory[otherUserId].append((
            durationSec, timestamp, serviceType, callerLocation, calleeLocation, timeSinceLastCallSec
        ))

# =============================
# CallGenerator
# =============================
# =============================

class CallGenerator:
    serviceTypes = ["voice", "video"]
    locations = ["home", "campus", "work", "inTransit", "other_outdoor"]


    def __init__(self, callerId: int, calleeId: int, timestamp: datetime, rng: np.random.Generator, service_type: str = None):
        self.callerId = callerId
        self.calleeId = calleeId
        self.timestamp = timestamp
        self.rng = rng
        self.service_type = service_type  # ← ADD THIS LINE
        
        # Validate service_type if provided
        if self.service_type is not None and self.service_type not in self.serviceTypes:
            raise ValueError(f"service_type must be one of {self.serviceTypes}")

    def getHourOfDay(self) -> int:
        return int(self.timestamp.hour)

    def getDayOfWeek(self) -> Tuple[int, str]:
        return self.timestamp.weekday(), self.timestamp.strftime("%A")

    def isWeekend(self) -> bool:
        return self.timestamp.weekday() >= 5

    def getHourBucket(self) -> str:
        h = self.getHourOfDay()
        if 0 <= h < 6:
            return "night"
        if 6 <= h < 9:
            return "morning"
        if 9 <= h <= 17:
            return "day"
        return "evening"

    def serviceType(self) -> str:
        # If service_type was specified in constructor, use it consistently
        if self.service_type is not None:
            return self.service_type
        
        # Otherwise, use realistic distribution: 70% voice, 30% video
        if self.rng.random() < 0.70:
            return "voice"
        else:
            return "video"

    def _locationProbs(self) -> np.ndarray:
        hour = self.getHourOfDay()
        dayIdx, _ = self.getDayOfWeek()

        if  self.isWeekend():
            probs = np.array([0.4, 0.1, 0.05, 0.25, 0.2], dtype=float)
        else:
            if 9 <= hour <= 17 and dayIdx < 5:
                probs = np.array([0.15, 0.35, 0.35, 0.1, 0.05], dtype=float)
            else:
                probs = np.array([0.5, 0.15, 0.1, 0.15, 0.1], dtype=float)
        return probs / probs.sum()

    def chooseBothLocations(self) -> Tuple[str, str]:
        probs = self._locationProbs()
        callerLoc = str(self.rng.choice(self.locations, p=probs))
        calleeLoc = str(self.rng.choice(self.locations, p=probs))
        return callerLoc, calleeLoc

    def contextDict(self) -> dict:
        """
        Generate context dictionary with consistent service type.
        """
        dayIdx, dayName = self.getDayOfWeek()
        svc = self.serviceType()  # This will use the consistent service type if specified
        callerLoc, calleeLoc = self.chooseBothLocations()
        return {
            "caller_id": self.callerId,
            "callee_id": self.calleeId,
            "timestamp": self.timestamp,
            "hour": self.getHourOfDay(),
            "day_index": dayIdx,
            "day_name": dayName,
            "hour_bucket": self.getHourBucket(),
            "is_weekend": self.isWeekend(),
            "service_type": svc,
            "caller_location": callerLoc,
            "callee_location": calleeLoc,
        }
# =============================
 #CallSimulator with Gamma Distribution Relationships
# =============================
import math
import hashlib
class CallSimulator:
    """
    - Gamma distribution for relationships: few close friends, many acquaintances
    - Deterministic base duration formula
    - Simple, strong multiplicative effects
    - Minimal self.rngness, maximum signal
    """

    def __init__(
        self,
        users: Dict[int, "User"],
        start_dt: datetime,
        end_dt: datetime,
        rng: Optional[np.random.Generator] = None,
        calls_per_day_mean: int = 120,
        noise_level: float = 0.05,  # Very low noise for high predictability
        strict_relationships: bool = True,
        gamma_shape: float = 0.7,  # Shape parameter for gamma distribution (creates right-skewed distribution)
        gamma_scale: float = 1.2,  # Scale parameter for gamma distribution
        max_relationship_score: float = 2.0,  
        pair_variation_range: float = 0.03   # ±30% variation for same relationship score
    ):
        self.users = users
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.calls_per_day_mean = calls_per_day_mean
        self.noise_level = noise_level
        self.strict_relationships = strict_relationships
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.max_relationship_score = max_relationship_score
        
        # Per-pair data
        self._pair_rel: Dict[Tuple[int,int], float] = {}
        self._pair_call_count: Dict[Tuple[int,int], int] = {}
        self._pair_last_ts: Dict[Tuple[int,int], Optional[datetime]] = {}
        self._pair_total_duration: Dict[Tuple[int,int], float] = {}
        self.pair_variation_range = pair_variation_range
        self._pair_seed: Dict[Tuple[int,int], float] = {}  # NEW: Pair-specific seed
        
    @staticmethod
    def _unordered(u: int, v: int) -> Tuple[int,int]:
        return (u, v) if u < v else (v, u)
    @staticmethod
    def _get_pair_seed(u: int, v: int) -> float:
        """Generate a deterministic but unique seed for each pair"""
        # Create a hash from the sorted user IDs
        pair_str = f"{min(u, v)}_{max(u, v)}"
        hash_val = hashlib.md5(pair_str.encode()).hexdigest()
        # Convert first 8 characters to a float between 0 and 1
        return int(hash_val[:8], 16) / 0xFFFFFFFF
    def _ensure_pair_params(self, u: int, v: int) -> bool:
        key = self._unordered(u, v)
        if key in self._pair_rel:
            return True

        # Check both directions
        known = (
            u in self.users
            and v in self.users
            and v in self.users[u].relationships
            and u in self.users[v].relationships
        )
    
        if not known:
            if self.strict_relationships:
                raise ValueError(
                    f"No relationship defined between users {u} and {v}."
                )
            else:
                return False
    
        score = self.users[u].getRelationshipScore(v)
        self._pair_rel[key] = float(score)
        self._pair_call_count[key] = 0
        self._pair_total_duration[key] = 0.0
        self._pair_last_ts[key] = None
        self._pair_seed[key] = self._get_pair_seed(u, v)  # NEW: Store pair seed
        return True
    
    def compute_deterministic_duration(
        self,
        relationship_score: float,
        service_type: str,
        caller_location: str,
        callee_location: str,
        hour: int,
        is_weekend: bool,
        call_count: int,
        hours_since_last: Optional[float],
         pair_seed: float  ,# NEW: Pair-specific seed
        pair_total_duration: float = 0.0  # NEW: match the call site
       
    )  -> Tuple[float, float]:
        """
         duration in seconds with gamma-distributed relationships
        """
        
        # 1. BASE DURATION - now exponential to match gamma distribution pattern
        # relationship_score follows gamma: most values low, few high
        # Apply pair-specific variation using the seed
        base= relationship_score*240 # 300 seconds
        
        # 3. SERVICE TYPE
        service_factor = 1.05 if service_type == "video" else 1.0
        
        # 4. LOCATION FACTOR
        location_scores = {
            "home": 1.3,
            "campus": 1, 
            "work": 0.9,
            "other_outdoor": 0.8,
            "inTransit": 0.5
        }
        
        caller_loc_score = location_scores.get(caller_location, 1.0)
        callee_loc_score = location_scores.get(callee_location, 1.0)
        location_factor = (caller_loc_score * callee_loc_score) ** 0.5
            
        if call_count==0:
            past_avg_duration=0.
            base_duration = relationship_score*240 # 300 seconds
        else:
            # Calculate the actual observed mean for this pair
            total = float(pair_total_duration) if pair_total_duration is not None else 0.0
            past_avg_duration = max(0.0, total / call_count)
            base_duration = (base*0.7) +(past_avg_duration *0.3)
            
        # 5. TIME FACTOR 
    
        if is_weekend:
            # On weekends all hours are good for longer calls
            time_factor = 2 # Uniformly boost duration all day
        
        else:
            # It's a weekday
            if 9 <= hour < 17:   # Standard Work Hours (9 AM - 5 PM)
                time_factor = 0.9
            elif 17 <= hour < 23: # Prime After-Work Hours (6 PM - 11 PM)
                time_factor = 1.3
            elif 23 <= hour or hour < 9: # Late Night / Early Morning (11 PM - 7 AM)
                time_factor = 0.85
            else: # Catch-all for other hours (5-6 PM, etc.)
                time_factor = 1.0    
        # 6. FREQUENCY FACTOR
        frequency_factor = 0.7 + 0.2 * exp(-call_count / 10.0)
        
        # 7. RECENCY FACTOR
     
        duration = (base_duration * relationship_score* service_factor * 
                   location_factor * time_factor * frequency_factor )
        
        
        return max(30, min(duration, 9000.0)), past_avg_duration
    
    def add_controlled_noise(self, duration: float) -> float:
        """Add minimal noise while preserving predictability"""
        if self.noise_level <= 0:
            return duration
            
        noise_factor = self.rng.lognormal(0, self.noise_level)
        noisy_duration = (duration+10) * noise_factor
        
        return max(30, min(noisy_duration, 9000.0))
    
    def run(self) -> List[dict]:
        """Generate """
        rows: List[dict] = []
        day = self.start_dt

        print(" Generating data")
        print(f"Gamma distribution: shape={self.gamma_shape}, scale={self.gamma_scale}")
        print(f"Noise level: {self.noise_level*100:.1f}%")

        while day.date() <= self.end_dt.date():
            # Deterministic daily call count
            base_calls = self.calls_per_day_mean
            if day.weekday() >= 5:
                n_calls = int(base_calls * 1.2)
            else:
                n_calls = base_calls
            
            n_calls = max(1, int(self.rng.poisson(n_calls * 0.95) + n_calls * 0.05))

            # Generate events
            events = []
            for _ in range(n_calls):
                candidates = [u for u in self.users.values() if u.relationships]
                if not candidates:
                    break
                    
                caller = self.rng.choice(candidates)
                contacts = caller.getContacts()
                
                # Weight by gamma-distributed relationship scores (cubic for strong preference)
                weights = np.array([caller.getRelationshipScore(c)**2 for c in contacts])
                weights = weights / weights.sum()
                
                callee_id = int(self.rng.choice(contacts, p=weights))
                caller_id = int(caller.userId)
                
                # Time distribution
                hour_weights = np.array([
                    0.5, 0.3, 0.2, 0.2, 0.3, 0.8,
                    2.0, 3.5, 4.0, 5.0, 4.5, 3.5,
                    3.0, 3.5, 4.0, 4.5, 4.0, 3.5,
                    3.0, 5.0, 6.0, 4.5, 2.5, 1.0
                ])
                hour_weights = hour_weights / hour_weights.sum()
                hour = int(self.rng.choice(24, p=hour_weights))
                
                minute = int(self.rng.integers(0, 60))
                second = int(self.rng.integers(0, 60))
                ts = day.replace(hour=hour, minute=minute, second=second)
                
                cg = CallGenerator(caller_id, callee_id, ts,self.rng)
                ctx = cg.contextDict()
                events.append({
                    "caller_id": caller_id,
                    "callee_id": callee_id, 
                    "timestamp": ts,
                    "ctx": ctx
                })

            # Sort chronologically
            events.sort(key=lambda e: e["timestamp"])

            # Process events
            for e in events:
                caller_id, callee_id, ts, ctx = e["caller_id"], e["callee_id"], e["timestamp"], e["ctx"]
                if not self._ensure_pair_params(caller_id, callee_id):
                    continue
                    
                key = self._unordered(caller_id, callee_id)

                # Calculate time since last call
                last_ts = self._pair_last_ts.get(key)
                hours_since_last = None if last_ts is None else (ts - last_ts).total_seconds() / 3600.0

                # Deterministic duration calculation
                deterministic_duration, past_avg_duration = self.compute_deterministic_duration(
                    relationship_score=self._pair_rel[key],
                    service_type=ctx["service_type"],
                    caller_location=ctx["caller_location"],
                    callee_location=ctx["callee_location"],
                    hour=ctx["hour"],
                    is_weekend=ctx["is_weekend"],
                    call_count=self._pair_call_count[key],
                    hours_since_last=hours_since_last,
                    pair_total_duration=self._pair_total_duration.get(key, 0.0),
                    pair_seed=self._pair_seed[key]
                )
                
                # Add controlled noise
                final_duration = self.add_controlled_noise(deterministic_duration)

                # Update counters
                self._pair_call_count[key] += 1
                self._pair_total_duration[key] = self._pair_total_duration.get(key, 0.0) + final_duration
                self._pair_last_ts[key] = ts
          
                # Log for users
                tsl_sec = None if hours_since_last is None else hours_since_last * 3600.0
                for uid in [caller_id, callee_id]:
                    other_id = callee_id if uid == caller_id else caller_id
                    other_loc = ctx["callee_location"] if uid == caller_id else ctx["caller_location"]
                    user_loc = ctx["caller_location"] if uid == caller_id else ctx["callee_location"]
                    self.users[uid].logCall(other_id, final_duration, ts, ctx["service_type"], user_loc, other_loc, tsl_sec)

                # Store result
                rows.append({
                    "caller_id": caller_id,
                    "callee_id": callee_id,
                    "timestamp": ts,  
                    "service_type": ctx["service_type"],
                    "relationship_score": self._pair_rel[key],
                     "caller_location": ctx["caller_location"],
                    "callee_location": ctx["callee_location"],
                     "past_avg_duration": past_avg_duration, 
                    "pair_call_count": self._pair_call_count[key],
                    "time_since_last_call_hours": hours_since_last,
                    "duration_sec": final_duration , 
                })

            day += timedelta(days=1)
            
            if day.day == 1:
                print(f"Generated {day.strftime('%B %Y')}")

        return rows

# =============================
# MAIN EXECUTION with Gamma Relationships
# =============================
