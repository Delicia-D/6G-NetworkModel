# Run ultra-predictable simulation with gamma relationships
from datetime import datetime, timedelta
from Datageneration import CallSimulator, User
from classifier import CoverageClassifier
from dualsimulator import DualCallSimulator
from nn import CallDurationPredictor
import numpy as np
import pandas as pd
import pickle
import json
from joblib import Parallel, delayed
import multiprocessing
import os
import time

def run_single_simulation(arrival_rate, i, total_rates, users, predictor, rng_seed, scenario_name, start_dt, end_dt):
    """Run a single simulation for a given arrival rate and scenario"""
    try:
        print(f"\n{'='*60}")
        print(f" {scenario_name} - Simulation {i+1}/{total_rates}: {arrival_rate} calls/second (PID: {os.getpid()})")
        print(f" Time period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")
        
        # Create new RNG for this process to avoid shared state issues
        _rng = np.random.default_rng(rng_seed)
        
        # Create simulator with current arrival rate
        classifier = CoverageClassifier(mode="real_simulation", rng=_rng)

        # Create and run dual simulator
        sim = DualCallSimulator(
            users=users,
            start_dt=start_dt,
            end_dt=end_dt,
            coverage_classifier=classifier,
            arrival_rate_per_second=arrival_rate,
            predictor=predictor,
            rng=_rng 
        )
        
        # Run simulation
        predictive_rows, nonpredictive_rows = sim.run()
        
        predictive_metrics = sim.predictive_controller.getMetrics()
        nonpredictive_metrics = sim.nonpredictive_controller.getMetrics()
        
        # Calculate rates
        predictive_blocking_prob = predictive_metrics.blocked / max(1, predictive_metrics.attempts)
        nonpredictive_blocking_prob = nonpredictive_metrics.blocked / max(1, nonpredictive_metrics.attempts)
        
        predictive_handoff_prob= predictive_metrics.handoffs / max(1, predictive_metrics.calls_admitted_to_satellite)
        nonpredictive_handoff_prob = nonpredictive_metrics.handoffs / max(1, nonpredictive_metrics.calls_admitted_to_satellite)
        
        # Calculate simulation duration in hours
        duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
        
        # Store results
        result = {
            'scenario': scenario_name,
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'start_dt': start_dt.isoformat(),
            'end_dt': end_dt.isoformat(),
            'duration_hours': duration_hours,
            'predictive_blocking_prob': predictive_blocking_prob,
            'nonpredictive_blocking_prob': nonpredictive_blocking_prob,
            'total_calls': predictive_metrics.attempts,
            'predictive_admitted': predictive_metrics.admitted,
            'nonpredictive_admitted': nonpredictive_metrics.admitted,
            'predictive_handoffs': predictive_metrics.handoffs,
            'nonpredictive_handoffs': nonpredictive_metrics.handoffs,
            'predictive_calls_admitted_to_satellite': predictive_metrics.calls_admitted_to_satellite,
            'nonpredictive_calls_admitted_to_satellite': nonpredictive_metrics.calls_admitted_to_satellite,
            'predictive_handoff_prob': predictive_handoff_prob,
            'nonpredictive_handoff_prob_': nonpredictive_handoff_prob,
            'predictive_blocking_reasons': dict(predictive_metrics.blockedReason),
            'nonpredictive_blocking_reasons': dict(nonpredictive_metrics.blockedReason)
        }
        
        print(f" COMPLETED: {scenario_name} - {arrival_rate} calls/sec | "
              f"Duration: {duration_hours:.1f}h | "
              f"Blocking P: {predictive_blocking_prob:.3f}, NP: {nonpredictive_blocking_prob:.3f} | "
              f"Handoff P: {predictive_handoff_prob:.3f}, NP: {nonpredictive_handoff_prob:.3f}")
        
        return result
        
    except Exception as e:
        print(f" ERROR in {scenario_name} - {arrival_rate} calls/sec: {e}")
        import traceback
        traceback.print_exc()
        return {
            'scenario': scenario_name,
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'start_dt': start_dt.isoformat(),
            'end_dt': end_dt.isoformat(),
            'duration_hours': (end_dt - start_dt).total_seconds() / 3600.0,
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0
        }

def run_parallel_scenario_experiment(users, predictor, base_rng, scenario_name, start_dt, end_dt):
    """Run simulations for a specific scenario in parallel"""
    
    # Define arrival rates to test (calls per second)
    arrival_rates = [0.5,1,2,3,4,5]
    
    # Get number of CPU cores (use all available)
    n_jobs = multiprocessing.cpu_count()
    print(f"\n STARTING {scenario_name.upper()} - MULTI-CORE PARALLEL EXECUTION")
    print(f"   Running {len(arrival_rates)} simulations using {n_jobs} CPU cores")
    print(f"   Arrival rates: {arrival_rates}")
    print(f"   Time period: {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}")
    print(f"   Duration: {(end_dt - start_dt).total_seconds()/3600:.1f} hours")
    print(f"   Using CONSISTENT random seed across all rates for fair comparison")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    base_seed = 42
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            arrival_rate,
            i,
            len(arrival_rates),
            users,
            predictor,
            base_seed,  # Same seed for all
            scenario_name,
            start_dt,
            end_dt
        )
        for i, arrival_rate in enumerate(arrival_rates)
    )
    
    total_time = time.time() - start_time
    print(f"\n {scenario_name.upper()} - ALL {len(all_results)} SIMULATIONS COMPLETED!")
    print(f"   Total parallel time: {total_time:.1f}s")
    print(f"   End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    print("=" * 70)
    
    # Create users with normal-distributed relationships
    N = 100
    users = {i: User(i) for i in range(1, N+1)}
    _rng = np.random.default_rng(42)

    # Create normal-distributed relationships
    print(" Creating normal-distributed user relationships...")
    mean_relationship = 10   # Mean relationship score
    std_relationship = 3     # Standard deviation

    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if _rng.random() < 0.35:  # 35% connection rate
                # Generate normal-distributed relationship score
                score = _rng.normal(mean_relationship, std_relationship)
                
                # Clip to reasonable range (1-20) and round
                score = np.clip(score, 1, 20)
                score = round(score)
                
                users[i].addRelationship(j, score)
                users[j].addRelationship(i, score)

    # Generate training data
    sim = CallSimulator(
        users=users,
        start_dt=datetime(2025, 1, 1),
        end_dt=datetime(2025, 7, 30),
        rng=_rng,
        calls_per_day_mean=400,
        noise_level=0.04,
    )

    rows = sim.run()

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    df.to_csv("data.csv", index=False)

    print(f"\n  DATA GENERATED!")
    print(f"Total calls: {len(df):,}")
    
    # Load data and train predictor
    predictor = CallDurationPredictor()
    df = pd.read_csv('data.csv')
    
    # STEP 1: TRAIN THE MODEL (separate from evaluation)
    predictor.train(df, verbose=True)

    # STEP 2: EVALUATE THE MODEL (separate call)
    train_metrics, test_metrics = predictor.evaluate(verbose=True)

    # Save the trained model
    with open('trained_predictor.pkl', 'wb') as f:
        pickle.dump(predictor, f)

    # Save the metrics
    metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Define scenarios
    scenarios = [
        {
            'name': 'weekday_working_hours_short',
            'start_dt': datetime(2025, 8, 11, 10, 0, 0),  # Monday 10:00 AM
            'end_dt': datetime(2025, 8, 11, 12, 45, 0),   # Monday 1:45 PM
        },
        {
            'name': 'weekend_evening_long', 
            'start_dt': datetime(2025, 8, 9, 20, 0, 0),   # Saturday 8:00 PM
            'end_dt': datetime(2025, 8, 9, 22, 45, 0),    # Saturday 10:45 PM
        }
    ]

    # Run experiments for both scenarios
    all_results = {}
    
    for scenario in scenarios:
        print("\n" + "="*70)
        print(f"STARTING {scenario['name'].upper()} EXPERIMENT")
        print("="*70)
        
        results = run_parallel_scenario_experiment(
            users, 
            predictor, 
            _rng,
            scenario['name'],
            scenario['start_dt'], 
            scenario['end_dt']
        )
        
        all_results[scenario['name']] = results
    
    # Save combined results
    with open('varyingduration.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    with open('varyingduration.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = {}
        for scenario_name, results in all_results.items():
            json_results[scenario_name] = results
        json.dump(json_results, f, indent=2, default=str)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL EXPERIMENT SUMMARY - SCENARIO COMPARISON")
    print(f"{'='*70}")
    
    for scenario_name, results in all_results.items():
        print(f"\n{scenario_name.upper()}:")
        for result in results:
            if 'error' not in result:
                print(f"  Rate {result['arrival_rate_per_second']}/s: "
                      f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                      f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                      f"Handoff P={result['predictive_handoff_prob']:.3f}, "
                      f"NP={result['nonpredictive_handoff_prob_']:.3f} | "
                      f"Calls={result['total_calls']}")
            else:
                print(f"  Rate {result['arrival_rate_per_second']}/s: ERROR - {result['error']}")
    
    print(f"\nResults saved to 'varyingduration.pkl' for analysis")
    print(f"Scenarios compared:")
    for scenario in scenarios:
        duration = (scenario['end_dt'] - scenario['start_dt']).total_seconds() / 3600
        print(f"  - {scenario['name']}: {scenario['start_dt'].strftime('%Y-%m-%d %H:%M')} to {scenario['end_dt'].strftime('%Y-%m-%d %H:%M')} ({duration:.1f}h)")