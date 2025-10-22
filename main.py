# Run ultra-predictable simulation with gamma relationships
from datetime import datetime
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

def run_single_simulation(arrival_rate, i, total_rates, users, predictor, rng_seed):
    """Run a single simulation for a given arrival rate"""
    try:
        print(f"\n{'='*60}")
        print(f" Simulation {i+1}/{total_rates}: {arrival_rate} calls/second (PID: {os.getpid()})")
        print(f"{'='*60}")
        
        # Create new RNG for this process to avoid shared state issues
        _rng = np.random.default_rng(rng_seed)
        
        # Create simulator with current arrival rate
        classifier = CoverageClassifier(mode="real_simulation", rng=_rng)

        # Create and run dual simulator
        sim = DualCallSimulator(
            users=users,
            start_dt=datetime(2025, 8, 10,20, 0, 0),
            end_dt=datetime(2025, 8, 10, 22, 45, 0),
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
        
        # Store results
        result = {
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
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
        
        print(f" COMPLETED: {arrival_rate} calls/sec | "
              f"Blocking P: {predictive_blocking_prob:.3f}, NP: {nonpredictive_blocking_prob:.3f} | "
              f"Handoff P: {predictive_metrics.handoffs:.3f}, NP: {nonpredictive_metrics.handoffs:.3f}")
        
        return result
        
    except Exception as e:
        print(f" ERROR in {arrival_rate} calls/sec: {e}")
        import traceback
        traceback.print_exc()
        return {
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0
        }

def run_parallel_arrival_rate_experiment(users, predictor, base_rng):
    """Run simulations with different arrival rates in parallel using CONSISTENT randomness"""
    
    # Define arrival rates to test (calls per second)
    arrival_rates = [0.1,1,2,3,4,5]
    #arrival_rates = [ 6]
    # Get number of CPU cores (use all available)
    n_jobs = multiprocessing.cpu_count()
    print(f"\n STARTING MULTI-CORE PARALLEL EXECUTION")
    print(f"   Running {len(arrival_rates)} simulations using {n_jobs} CPU cores")
    print(f"   Arrival rates: {arrival_rates}")
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
            base_seed  # ← SAME for all
        )
        for i, arrival_rate in enumerate(arrival_rates)
    )


    
    total_time = time.time() - start_time
    print(f"\n ALL {len(all_results)} SIMULATIONS COMPLETED!")
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
    with open('trained_predictor1.pkl', 'wb') as f:
        pickle.dump(predictor, f)

    # Save the metrics
    metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    with open('training_metrics1.json', 'w') as f:
        json.dump(metrics, f)

    # NOW run the parallel experiment
    print("\n" + "="*70)
    print("STARTING PARALLEL ARRIVAL RATE EXPERIMENT")
    print("="*70)
    
    results = run_parallel_arrival_rate_experiment(users, predictor, _rng)
    
    # Save results for Jupyter plotting
    with open('arrival_rate_resultsmain.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    '''with open('arrival_rate_resultsmain.json', 'w') as f:
        json.dump(results, f, indent=2)'''
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    for result in results:
        if 'error' not in result:
            print(f"Rate {result['arrival_rate_per_second']}/s: "
                  f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                  f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                  f"Handoff P={result['predictive_handoff_prob']:.3f}, "
                  f"NP={result['nonpredictive_handoff_prob_']:.3f} | "
                  f"Calls={result['total_calls']}")
        else:
            print(f"Rate {result['arrival_rate_per_second']}/s: ERROR - {result['error']}")
    
    print(f"\nResults saved to 'arrival_rate_resultsmain.pkl' for Jupyter plotting")