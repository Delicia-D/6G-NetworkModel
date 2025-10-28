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

def run_single_height_simulation(arrival_rate, height, sim_id, total_sims, users, predictor, rng_seed):
    """Run a single simulation for a given arrival rate and satellite height"""
    try:
        print(f"\n{'='*60}")
        print(f" Simulation {sim_id+1}/{total_sims}: {arrival_rate} calls/sec, {height}km height (PID: {os.getpid()})")
        print(f"{'='*60}")
        
        # Create new RNG for this process
        _rng = np.random.default_rng(rng_seed)
        
        # Create classifier
        classifier = CoverageClassifier(mode="real_simulation", rng=_rng)

        # Create and run dual simulator with specific height
        sim = DualCallSimulator(
            users=users,
            start_dt=datetime(2025, 8, 9, 9, 0, 0),
            end_dt=datetime(2025, 8, 9, 22, 45, 0),
            coverage_classifier=classifier,
            arrival_rate_per_second=arrival_rate,
            predictor=predictor,
            rng=_rng,
            satelliteHeight=height
        )
        
        # Run simulation
        predictive_rows, nonpredictive_rows = sim.run()
        
        predictive_metrics = sim.predictive_controller.getMetrics()
        nonpredictive_metrics = sim.nonpredictive_controller.getMetrics()
        
        # Calculate probabilities
        predictive_blocking_prob = predictive_metrics.blocked / max(1, predictive_metrics.attempts)
        nonpredictive_blocking_prob = nonpredictive_metrics.blocked / max(1, nonpredictive_metrics.attempts)
        
        predictive_handoff_prob = predictive_metrics.handoffs / max(1, predictive_metrics.calls_admitted_to_satellite)
        nonpredictive_handoff_prob = nonpredictive_metrics.handoffs / max(1, nonpredictive_metrics.calls_admitted_to_satellite)
        
        # Store results with height information
        result = {
            'satellite_height_km': height,
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
            'nonpredictive_handoff_prob': nonpredictive_handoff_prob,
            'predictive_blocking_reasons': dict(predictive_metrics.blockedReason),
            'nonpredictive_blocking_reasons': dict(nonpredictive_metrics.blockedReason)
        }
        
        print(f" COMPLETED: {height}km, {arrival_rate}/s | "
              f"Blocking P: {predictive_blocking_prob:.3f}, NP: {nonpredictive_blocking_prob:.3f} | "
              f"Handoff P: {predictive_handoff_prob:.3f}, NP: {nonpredictive_handoff_prob:.3f}")
        
        return result
        
    except Exception as e:
        print(f" ERROR in {height}km, {arrival_rate}/s: {e}")
        import traceback
        traceback.print_exc()
        return {
            'satellite_height_km': height,
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0
        }

def run_height_experiment(users, predictor, base_rng):
    """Run simulations with different satellite heights and arrival rates"""
    
    # Define test parameters
    satellite_heights = [1300,1400,1600]  # km
    arrival_rates = [0.3,1,1.5,2,2.5,3,3.5,4,4.5,5]  # calls per second
    
    # Generate all combinations
    test_scenarios = []
    for height in satellite_heights:
        for rate in arrival_rates:
            test_scenarios.append((rate, height))
    
    n_jobs = multiprocessing.cpu_count()
    print(f"\n STARTING SATELLITE HEIGHT EXPERIMENT")
    print(f"   Testing {len(test_scenarios)} scenarios:")
    print(f"   Heights: {satellite_heights} km")
    print(f"   Arrival rates: {arrival_rates} calls/sec")
    print(f"   Using {n_jobs} CPU cores")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    base_seed = 42
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_height_simulation)(
            scenario[0],  # arrival_rate
            scenario[1],  # height  
            i,            # sim_id
            len(test_scenarios),
            users,
            predictor,
            base_seed + i
        )
        for i, scenario in enumerate(test_scenarios)
    )
    
    total_time = time.time() - start_time
    print(f"\n ALL {len(all_results)} SIMULATIONS COMPLETED!")
    print(f"   Total parallel time: {total_time:.1f}s")
    print(f"   End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return all_results

def save_raw_results(results, outputs_folder):
    """Simply save the raw results without any calculations"""
    
    # Save detailed results to outputs folder
    results_file_path = os.path.join(outputs_folder, 'satellite_height_results2.pkl')
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {results_file_path}")
    return results_file_path

def print_detailed_results(results):
    """Print all results in a detailed table"""
    
    successful_results = [r for r in results if 'error' not in r]
    
    print(f"\n{'='*120}")
    print("DETAILED RESULTS - ALL SCENARIOS")
    print(f"{'='*120}")
    print(f"{'Height':>8} {'Arrival Rate':>12} {'Blocking P':>12} {'Blocking NP':>12} {'Handoff P':>12} {'Handoff NP':>12} {'Handoffs P':>12} {'Handoffs NP':>12}")
    print(f"{'(km)':>8} {'(calls/sec)':>12} {'':>12} {'':>12} {'Prob':>12} {'Prob':>12} {'Count':>12} {'Count':>12}")
    print(f"{'-'*120}")
    
    # Sort by height and arrival rate
    successful_results.sort(key=lambda x: (x['satellite_height_km'], x['arrival_rate_per_second']))
    
    for result in successful_results:
        print(f"{result['satellite_height_km']:>8} "
              f"{result['arrival_rate_per_second']:>12.1f} "
              f"{result['predictive_blocking_prob']:>12.4f} "
              f"{result['nonpredictive_blocking_prob']:>12.4f} "
              f"{result['predictive_handoff_prob']:>12.4f} "
              f"{result['nonpredictive_handoff_prob']:>12.4f} "
              f"{result['predictive_handoffs']:>12.0f} "
              f"{result['nonpredictive_handoffs']:>12.0f}")

if __name__ == "__main__":
    print("=" * 70)
    print("SATELLITE HEIGHT COMPARISON EXPERIMENT")
    print("=" * 70)
    
    # Create outputs folder if it doesn't exist
    outputs_folder = 'outputs'
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
        print(f"Created outputs folder: {outputs_folder}")
    
    # Load or create users
    N = 100
    users = {i: User(i) for i in range(1, N+1)}
    _rng = np.random.default_rng(42)

    # Create normal-distributed relationships
    print(" Creating normal-distributed user relationships...")
    mean_relationship = 10
    std_relationship = 3

    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if _rng.random() < 0.35:
                score = _rng.normal(mean_relationship, std_relationship)
                score = np.clip(score, 1, 20)
                score = round(score)
                users[i].addRelationship(j, score)
                users[j].addRelationship(i, score)

    # Generate training data and save to outputs folder
    sim = CallSimulator(
        users=users,
        start_dt=datetime(2025, 1, 1),
        end_dt=datetime(2025, 7, 30),
        rng=_rng,
        calls_per_day_mean=400,
        noise_level=0.04,
    )
    rows = sim.run()
    df = pd.DataFrame(rows)
    data_file_path = os.path.join(outputs_folder, "data.csv")
    df.to_csv(data_file_path, index=False)
    print(f"Training data generated: {len(df):,} calls")
    print(f"Data saved to: {data_file_path}")

    # Load and train predictor
    predictor = CallDurationPredictor()
    df = pd.read_csv(data_file_path)
    predictor.train(df, verbose=True)
    train_metrics, test_metrics = predictor.evaluate(verbose=True)

    # Save model to outputs folder
    model_file_path = os.path.join(outputs_folder, 'trained_predictor.pkl')
    with open(model_file_path, 'wb') as f:
        pickle.dump(predictor, f)
    print(f"Model saved to: {model_file_path}")

    # Run height experiment
    print("\n" + "="*70)
    print("STARTING SATELLITE HEIGHT COMPARISON EXPERIMENT")
    print("="*70)
    
    results = run_height_experiment(users, predictor, _rng)
    
    # Save and display results 
    results_file_path = save_raw_results(results, outputs_folder)
    print_detailed_results(results)
    
    # Simple final message
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Total scenarios simulated: {len([r for r in results if 'error' not in r])}")
    print(f"Results saved to:")
    print(f"  - {data_file_path}")
    print(f"  - {model_file_path}")
    print(f"  - {results_file_path}")
    print(f"All files saved in '{outputs_folder}' folder")
    print(f"{'='*70}")