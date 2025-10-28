# =============================
# Standalone Dual Call Simulator
# =============================
from __future__ import annotations
from datetime import datetime
import hashlib
from math import exp
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Literal
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import heapq
from Datageneration import CallGenerator, User
from classifier import CoverageClassifier
from nn import CallDurationPredictor
from nonpredictor import NonPredictiveCallAdmissionController
from predictor import HANDOFF_TIME_SEC, RAT_CAPACITIES, RB_PER_SERVICE, PredictiveCallAdmissionController
from visibility import LEOWindowManager, SatelliteParams



class DualCallSimulator:
    """
    Runs both predictive and non-predictive controllers simultaneously
    on the same call stream for fair comparison.
    """

    def __init__(
        self,
        users: Dict[int, "User"],
        start_dt: datetime,
        end_dt: datetime,
        coverage_classifier: CoverageClassifier,
        rng: Optional[np.random.Generator] = None,
        arrival_rate_per_second: float = 0.05,
        noise_level: float = 0.05,
        strict_relationships: bool = True,
        gamma_shape: float = 0.7,
        gamma_scale: float = 1.2,
        max_relationship_score: float = 2.0,
        
        pair_variation_range: float = 0.03,
        satelliteHeight:int =1300,
        predictor: Optional[CallDurationPredictor] = None,
        service_type:Optional[str]=None
       
    ):
        self.users = users
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.arrival_rate_per_second = arrival_rate_per_second
        self.noise_level = noise_level
        self.strict_relationships = strict_relationships
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.max_relationship_score = max_relationship_score
        self.pair_variation_range = pair_variation_range
        self.predictor = predictor
        self.classifier = coverage_classifier 
        self.service_type=service_type
        self.satelliteHeight=satelliteHeight 
        self.alldurations = []
        # Per-pair data 
        self._pair_rel: Dict[Tuple[int,int], float] = {}
        self._pair_call_count: Dict[Tuple[int,int], int] = {}
        self._pair_last_ts: Dict[Tuple[int,int], Optional[datetime]] = {}
        self._pair_total_duration: Dict[Tuple[int,int], float] = {}
        self._pair_seed: Dict[Tuple[int,int], float] = {}

    @staticmethod
    def _unordered(u: int, v: int) -> Tuple[int,int]:
        return (u, v) if u < v else (v, u)
        
    @staticmethod
    def _get_pair_seed(u: int, v: int) -> float:
        """Generate a deterministic but unique seed for each pair"""
        pair_str = f"{min(u, v)}_{max(u, v)}"
        hash_val = hashlib.md5(pair_str.encode()).hexdigest()
        return int(hash_val[:8], 16) / 0xFFFFFFFF

    def _ensure_pair_params(self, u: int, v: int) -> bool:
        key = self._unordered(u, v)
        if key in self._pair_rel:
            return True

        if not (u in self.users and v in self.users and 
                v in self.users[u].relationships and u in self.users[v].relationships):
            if self.strict_relationships:
                raise ValueError(f"No relationship defined between users {u} and {v}.")
            else:
                return False

        score = self.users[u].getRelationshipScore(v)
        self._pair_rel[key] = float(score)
        self._pair_call_count[key] = 0
        self._pair_total_duration[key] = 0.0
        self._pair_last_ts[key] = None
        self._pair_seed[key] = self._get_pair_seed(u, v)
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
        pair_seed: float,
        pair_total_duration: float = 0.0
    ) -> Tuple[float, float]:
        """Calculate actual call duration using deterministic formula"""
        
        base= relationship_score*240 
        service_factor = 1.05 if service_type == "video" else 1.0
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
            base_duration = relationship_score*240 
        else:
            # Calculate the actual observed mean for this pair
            total = float(pair_total_duration) if pair_total_duration is not None else 0.0
            past_avg_duration = max(0.0, total / call_count)
            base_duration = (base*0.7) +(past_avg_duration *0.3)
             
    
        if is_weekend :
            # On weekends, all hours are good for longer calls
            time_factor = 2 # Uniformly boost duration all day
        
        else:
            if 9 <= hour < 17:   # Standard Work Hours (9 AM - 5 PM)
                time_factor = 0.9
            elif 17 <= hour < 23: # Prime After-Work Hours (6 PM - 11 PM)
                time_factor = 1.3
            elif 23 <= hour or hour < 9: # Late Night / Early Morning 
                time_factor = 0.85
            else: # Catch-all for other hours 
                time_factor = 1.0    
        
        frequency_factor = 0.7 + 0.2 * exp(-call_count / 10.0)
     
        duration = (base_duration * relationship_score* service_factor * 
                   location_factor * time_factor * frequency_factor )
        
        return max(30, min(duration, 9000.0)), past_avg_duration

    def add_controlled_noise(self, duration: float) -> float:
        """Add minimal noise while preserving predictability"""
        if self.noise_level <= 0:
            return duration
        noise_factor = self.rng.lognormal(0, self.noise_level)
        noisy_duration = (duration + 10) * noise_factor
        return max(30, min(noisy_duration, 9000.0))

    def run(self) -> Tuple[List[dict], List[dict]]:
        """Run both controllers simultaneously, return results for both"""
        predictive_rows = []
        nonpredictive_rows = []
        current_time = self.start_dt
        
             # Create separate LEO window managers for each controller
        predictive_leo_window = LEOWindowManager(
            start_ts=current_time.timestamp(),
            T_max_sec=72*60*60,
            gap_sec=0
        )
        
        nonpredictive_leo_window = LEOWindowManager(
            start_ts=current_time.timestamp(),
            T_max_sec=72*60*60, 
            gap_sec=0
        )
        
        predictive_controller = PredictiveCallAdmissionController(
            leo_window=predictive_leo_window,  
            ratCapacities=RAT_CAPACITIES.copy(),
            rbPerService=RB_PER_SERVICE,
            handoffTimeSec=HANDOFF_TIME_SEC
        )
        
        nonpredictive_controller = NonPredictiveCallAdmissionController(
            rng=self.rng,
            leo_window=nonpredictive_leo_window, 
            ratCapacities=RAT_CAPACITIES.copy(),
            rbPerService=RB_PER_SERVICE,
            handoffTimeSec=HANDOFF_TIME_SEC
        )
        self.predictive_controller = predictive_controller
        self.nonpredictive_controller = nonpredictive_controller
        
        sat_params = SatelliteParams(H_km=self.satelliteHeight, eps_min_deg=10.0,inc_deg=0)
        self.alldurations = []
        
        print("Starting DUAL simulation (Predictive + Non-Predictive)")
        print(f"Start: {self.start_dt}, End: {self.end_dt}")
        
        print(f"\n{'='*60}")
        print(f"ARRIVAL RATE: {self.arrival_rate_per_second} calls/second")
        print(f"{'='*60}")
        calls_per_second = self.arrival_rate_per_second 
        next_call_time = current_time
        total_calls = 0
        day_stats = {}
        max_call_duration = timedelta(minutes=30)
        call_generation_cutoff = self.end_dt - max_call_duration
        print(f"Call generation: {self.start_dt.strftime('%H:%M:%S')} to {call_generation_cutoff.strftime('%H:%M:%S')}")
        print(f"Call completion: until {self.end_dt.strftime('%H:%M:%S')}")
        
                
        while current_time <= self.end_dt:
    # Release due sessions in both controllers
            
            if current_time <= call_generation_cutoff:
                while next_call_time <= current_time and next_call_time <= call_generation_cutoff:  
                    call_generated = self._generate_and_process_dual_call(
                        next_call_time, calls_per_second, 
                        predictive_controller, nonpredictive_controller,
                        sat_params, 
                        predictive_rows, nonpredictive_rows, day_stats,self.alldurations
                    )
                    
                    if call_generated:
                        total_calls += 1
                    
                    # Schedule next call
                    inter_arrival_time = self.rng.exponential(1.0 / calls_per_second)
                    next_call_time += timedelta(seconds=inter_arrival_time)

            predictive_controller.releaseDueSessions(current_time)
            nonpredictive_controller.releaseDueSessions(current_time)
            current_time += timedelta(seconds=1)  
            # Progress reporting
            if current_time.second == 0 and current_time.minute == 0:
                if current_time.hour == 0:
                    prev_day = (current_time - timedelta(days=1)).strftime("%Y-%m-%d")
                    calls_yesterday = day_stats.get(prev_day, 0)
                    predictive_active = len(predictive_controller.activeSessions)
                    nonpredictive_active = len(nonpredictive_controller.activeSessions)
                    print(f"[{current_time.strftime('%Y-%m-%d %H:%M')}] "
                          f"Day: {calls_yesterday} calls, "
                          f"Active: P={predictive_active} NP={nonpredictive_active}")
                    
                elif current_time.hour % 1 == 0:
                    predictive_active = len(predictive_controller.activeSessions)
                    nonpredictive_active = len(nonpredictive_controller.activeSessions)
                    next_call_str = next_call_time.strftime('%H:%M:%S') if next_call_time <= self.end_dt else "END"
                    print(f"[{current_time.strftime('%H:%M')}] "
                          f"Active: P={predictive_active} NP={nonpredictive_active}, "
                          f"Next: {next_call_str}")
        
        self._print_comparative_results(predictive_controller, nonpredictive_controller, total_calls)
        
        # Initialize counters
        count_0_to_1_min = 0
        count_1_to_3_min = 0
        count_3_to_6_min = 0
        count_6_to_12_min = 0

        # Count values in each range
        for time in self.alldurations:
            if 0 <= time <= 60:  # 0 to 1 minute 
                count_0_to_1_min += 1
            elif 61 <= time <= 180:  # 1 to 3 minutes 
                count_1_to_3_min += 1
            elif 181 <= time <= 360:  # 3 to 6 minutes 
                count_3_to_6_min += 1
            elif 361 <= time <= 720:  # 6 to 12 minutes 
                count_6_to_12_min += 1

        # Print results
        print(f"0 to 1 min: {count_0_to_1_min}")
        print(f"1 to 3 min: {count_1_to_3_min}")
        print(f"3 to 6 min: {count_3_to_6_min}")
        print(f"6 to 12 min: {count_6_to_12_min}")
        return predictive_rows, nonpredictive_rows
        
    # =============================
    # Call processing methods 
    # =============================

    def _generate_and_process_dual_call(self, call_time, calls_per_second,
                                      predictive_controller, nonpredictive_controller,
                                      sat_params,
                                      predictive_rows, nonpredictive_rows, day_stats,alldurations):
        """Generate and process the same call through both controllers"""
        
        
        # Get all users with relationships
        candidates = [u for u in self.users.values() if u.relationships]
        if not candidates:
            return False
        
        # Randomly select caller
        caller = self.rng.choice(candidates)
        caller_id = caller.userId
        
        # Get all contacts 
        contacts = caller.getContacts()
        if not contacts:
            return False
        
        # Randomly select callee 
        callee_id = self.rng.choice(contacts)
    
       
        
        # Generate call context (same for both controllers)
        cg = CallGenerator(caller_id, callee_id, call_time,self.rng, service_type=self.service_type)
        ctx = cg.contextDict()
        
        caller_lat, caller_lon = self.classifier.generate_coordinates()
        callee_lat, callee_lon = self.classifier.generate_coordinates()
    
        # Determine coverage groups using the approach set during initialization
        caller_coverage_data = self.classifier.get_user_coverage(caller_lat, caller_lon)
        callee_coverage_data = self.classifier.get_user_coverage(callee_lat, callee_lon)
    
        caller_coverage_group = caller_coverage_data['group']
        callee_coverage_group = callee_coverage_data['group']
    
        ctx.update({
            "caller_lat": caller_lat, "caller_lon": caller_lon,
            "callee_lat": callee_lat, "callee_lon": callee_lon,
            "caller_coverage_group": caller_coverage_group,
            "callee_coverage_group": callee_coverage_group
        })
        
        # Process through both controllers
        self._process_single_dual_call(
            caller_id, callee_id, call_time, ctx,
            predictive_controller, nonpredictive_controller,
            sat_params, predictive_rows, nonpredictive_rows, day_stats,self.alldurations
        )
        return True

    def _process_single_dual_call(self, caller_id, callee_id, call_time, ctx,
                                predictive_controller, nonpredictive_controller,
                                sat_params, predictive_rows, nonpredictive_rows, day_stats,alldurations):
        """Process the same call through both controllers"""
        
        if not self._ensure_pair_params(caller_id, callee_id):
            return
            
        key = self._unordered(caller_id, callee_id)
        
        # Calculate time since last call
        last_ts = self._pair_last_ts.get(key)
        hours_since_last = None if last_ts is None else (call_time - last_ts).total_seconds() / 3600.0
        
        # Calculate deterministic duration 
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
        
        # Get neural network prediction for predictive controller
        if self.predictor and self.predictor.model is not None:
            predicted = {
                'relationship_score': self._pair_rel[key],
                'pair_call_count': self._pair_call_count[key],
                'time_since_last_call_hours': hours_since_last,
                'service_type': ctx["service_type"],
                'caller_location': ctx["caller_location"],
                'callee_location': ctx["callee_location"],
                'timestamp':call_time,
                'past_avg_duration': past_avg_duration
            }
            predicted_duration = float(self.predictor.predict(predicted)[0])
        else:
            print("failed")
        
        # Satellite visibility 
        visibility_sec = sat_params.visibility_time_seconds(
            user_lat_deg=ctx["caller_lat"],
            user_lon_deg=ctx["caller_lon"],
        )
        
        actual_duration = self.add_controlled_noise(deterministic_duration)
        self.alldurations.append(actual_duration)
        # Common call context
        common_callCtx = {
            "caller_id": caller_id,
            "callee_id": callee_id,
            "service_type": ctx["service_type"],
            "predicted_duration_sec": float(predicted_duration),
            "actual_duration_sec": float(actual_duration),
            "visibility_sec": float(visibility_sec),
            "user_group": ctx["caller_coverage_group"],
            "timestamp": call_time,
        }
        #print(f"[{call_time.strftime('%H:%M:%S')}] DUAL_CALL users=({caller_id},{callee_id}) group={ctx['caller_coverage_group']} predicted={predicted_duration:.1f}, actual={actual_duration:.1f}, visibility={visibility_sec:.1f}, service={ctx['service_type']}, caller_lat={ctx['caller_lat']:.4f}, caller_lon={ctx['caller_lon']:.4f}")        # Process through PREDICTIVE controller
        predictive_decision = predictive_controller.handleNewCallRequest(common_callCtx.copy())
        
       

        # Process through NON-PREDICTIVE controller  
        nonpredictive_decision = nonpredictive_controller.handleNewCallRequest(common_callCtx.copy())

        
        # Update counters if admitted by either controller
        predictive_admitted = predictive_decision.startswith("Admitted")
        nonpredictive_admitted = nonpredictive_decision.startswith("Admitted")
        
        if predictive_admitted or nonpredictive_admitted:
            self._pair_call_count[key] += 1
            self._pair_total_duration[key] = self._pair_total_duration.get(key, 0.0) + actual_duration
            self._pair_last_ts[key] = call_time
            
            # Log for users
            tsl_sec = None if hours_since_last is None else hours_since_last * 3600.0
            for uid in [caller_id, callee_id]:
                other_id = callee_id if uid == caller_id else caller_id
                other_loc = ctx["callee_location"] if uid == caller_id else ctx["caller_location"]
                user_loc = ctx["caller_location"] if uid == caller_id else ctx["callee_location"]
                self.users[uid].logCall(other_id, actual_duration, call_time, 
                                      ctx["service_type"], user_loc, other_loc, tsl_sec)
        
        # Store results for both controllers
        row_data = {
            "caller_id": caller_id,
            "callee_id": callee_id,
            "timestamp": call_time,  
            "service_type": ctx["service_type"],
            "relationship_score": self._pair_rel[key],
            "caller_location": ctx["caller_location"],
            "callee_location": ctx["callee_location"],
            "past_avg_duration": past_avg_duration, 
            "pair_call_count": self._pair_call_count[key],
            "time_since_last_call_hours": hours_since_last,
            "predicted_duration_sec": predicted_duration,
            "actual_duration_sec": actual_duration,
            "day_index": call_time.weekday(),
            "day_name": call_time.strftime("%A"),
            "is_weekend": ctx["is_weekend"],
            "hour": call_time.hour,
            "visibility_sec": visibility_sec,
            "caller_coverage_group": ctx["caller_coverage_group"],
        }
        
        predictive_rows.append({**row_data, "admission_decision": predictive_decision, "controller": "predictive"})
        nonpredictive_rows.append({**row_data, "admission_decision": nonpredictive_decision, "controller": "nonpredictive"})
        
        # Track daily statistics
        day_key = call_time.strftime("%Y-%m-%d")
        day_stats[day_key] = day_stats.get(day_key, 0) + 1

    def _print_comparative_results(self, predictive_controller, nonpredictive_controller, total_calls):
        """Print side-by-side comparison with new handoff metrics"""
    
        
        predictive_metrics = predictive_controller.getMetrics()
        nonpredictive_metrics = nonpredictive_controller.getMetrics()
        
        total_days = (self.end_dt - self.start_dt).days + 1
        
        print(f"\n=== DUAL SIMULATION COMPLETE ===")
        print(f"Total time: {total_days} days")
        print(f"Total calls generated: {total_calls}")
        print(f"Average calls per day: {total_calls / total_days:.1f}")
        
        print(f"\n=== COMPARATIVE RESULTS ===")
        print(f"{'METRIC':<25} {'PREDICTIVE':<15} {'NON-PREDICTIVE':<15}")
        print(f"{'-'*55}")
        print(f"{'Admission attempts':<25} {predictive_metrics.attempts:<15} {nonpredictive_metrics.attempts:<15}")
        print(f"{'Admitted':<25} {predictive_metrics.admitted:<15} {nonpredictive_metrics.admitted:<15}")
        print(f"{'Blocked':<25} {predictive_metrics.blocked:<15} {nonpredictive_metrics.blocked:<15}")  
        print(f"{'handoffs':<25} {predictive_metrics.handoffs:<15} {nonpredictive_metrics.handoffs:<15}")
        print(f"{'Handoff probability':<25} {predictive_metrics.handoffs/predictive_metrics.calls_admitted_to_satellite:<15.3f} {nonpredictive_metrics.handoffs/nonpredictive_metrics.calls_admitted_to_satellite:<15.3f}")

        print(f"{'Admitted to satellites':<25} {predictive_metrics.calls_admitted_to_satellite:<15} {nonpredictive_metrics.calls_admitted_to_satellite:<15}")

        predictive_block_rate = predictive_metrics.blocked / max(1, predictive_metrics.attempts) * 100
        nonpredictive_block_rate = nonpredictive_metrics.blocked / max(1, nonpredictive_metrics.attempts) * 100
        print(f"{'Blocking rate':<25} {predictive_block_rate:.1f}%{'':<9} {nonpredictive_block_rate:.1f}%{'':<9}")

        print(f"P Voice calls admitted: {predictive_metrics.voice_calls_admitted}")
        print(f"P Video calls admitted: {predictive_metrics.video_calls_admitted}")
        print(f"NP Voice calls admitted: {nonpredictive_metrics.voice_calls_admitted}")
        print(f"NP Video calls admitted: {nonpredictive_metrics.video_calls_admitted}")
        print("\n=== GROUP DISTRIBUTION ===")
        print("PREDICTIVE GROUP COUNTS:")
        for group, count in predictive_controller.group_counts.items():
            percentage = (count / predictive_controller.metrics.attempts) * 100
            print(f"  {group}: {count} ({percentage:.1f}%)")
        
        print("NON-PREDICTIVE GROUP COUNTS:")  
        for group, count in nonpredictive_controller.group_counts.items():
            percentage = (count / nonpredictive_controller.metrics.attempts) * 100
            print(f"  {group}: {count} ({percentage:.1f}%)")
   
                
