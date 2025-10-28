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

def run_single_simulation(arrival_rate, i, total_rates, users, predictor, rng_seed, group_config, config_name):
    """Run a single simulation for a given arrival rate and group configuration"""
    try:
        print(f"\n{'='*60}")
        print(f" Simulation {i+1}/{total_rates} [{config_name}]: {arrival_rate} calls/second (PID: {os.getpid()})")
        print(f"{'='*60}")
        
        _rng = np.random.default_rng(rng_seed)
        
        # Create classifier with current group configuration
        classifier = CoverageClassifier(mode="group_assignment", group_config=group_config, rng=_rng)

        # Create and run dual simulator
        sim = DualCallSimulator(
            users=users,
            start_dt=datetime(2025, 8, 11, 0, 0, 0),
            end_dt=datetime(2025, 8, 11, 23, 30, 0),
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
            'config_name': config_name,  
            'group_config': group_config,
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
        print(f" ERROR in {arrival_rate} calls/sec [{config_name}]: {e}")
        import traceback
        traceback.print_exc()
        return {
            'config_name': config_name,  
            'group_config': group_config,
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0,
            'total_calls': 0,
            'predictive_admitted': 0,
            'nonpredictive_admitted': 0,
            'predictive_handoffs': 0,
            'nonpredictive_handoffs': 0,
            'predictive_calls_admitted_to_satellite': 0,
            'nonpredictive_calls_admitted_to_satellite': 0,
            'predictive_handoff_prob': 0,
            'nonpredictive_handoff_prob_': 0,
            'predictive_blocking_reasons': {},
            'nonpredictive_blocking_reasons': {}
        }

def run_parallel_arrival_rate_experiment(users, predictor, base_rng, group_configs):
    """Run simulations with different arrival rates and group configurations in parallel"""
    
    # Define arrival rates to test 
    arrival_rates = [0.3,1,2,3,4,5]
    
    # Prepare all combinations of configurations and arrival rates
    all_simulations = []
    for config_name, group_config in group_configs.items():
        for arrival_rate in arrival_rates:
            all_simulations.append({
                'arrival_rate': arrival_rate,
                'config_name': config_name,
                'group_config': group_config
            })
    
    # Get number of CPU cores 
    n_jobs = multiprocessing.cpu_count()
    print(f"\n STARTING MULTI-CORE PARALLEL EXECUTION")
    print(f"   Running {len(all_simulations)} simulations using {n_jobs} CPU cores")
    print(f"   Group configurations: {list(group_configs.keys())}")
    print(f"   Arrival rates: {arrival_rates}")
    print(f"   Using CONSISTENT random seed across all runs for fair comparison")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    base_seed = 42
    
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            sim_config['arrival_rate'],
            i,
            len(all_simulations),
            users,
            predictor,
            base_seed,
            sim_config['group_config'],
            sim_config['config_name']
        )
        for i, sim_config in enumerate(all_simulations)
    )
    
    total_time = time.time() - start_time
    print(f"\n ALL {len(all_results)} SIMULATIONS COMPLETED!")
    print(f"   Total parallel time: {total_time:.1f}s")
    print(f"   End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return all_results

if __name__ == "__main__":
    print("=" * 70)
    
    # Create outputs folder if it doesn't exist
    outputs_folder = 'outputs'
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
        print(f"Created outputs folder: {outputs_folder}")
    
    # Create users with gamma-distributed relationships
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

    # Create and save DataFrame to outputs folder
    df = pd.DataFrame(rows)
    data_file_path = os.path.join(outputs_folder, "data.csv")
    df.to_csv(data_file_path, index=False)

    print(f"\n GAMMA-RELATIONSHIP DATA GENERATED!")
    print(f"Total calls: {len(df):,}")
    print(f"Data saved to: {data_file_path}")
    
    # Load data and train predictor
    predictor = CallDurationPredictor()
    df = pd.read_csv(data_file_path)
    
    # Train the model 
    predictor.train(df, verbose=True)

    # Evaluate the model 
    train_metrics, test_metrics = predictor.evaluate(verbose=True)

    # Save the trained model to outputs folder
    model_file_path = os.path.join(outputs_folder, 'trained_predictor.pkl')
    with open(model_file_path, 'wb') as f:
        pickle.dump(predictor, f)

    # Save the metrics to outputs folder
    metrics_file_path = os.path.join(outputs_folder, 'training_metrics.json')
    metrics = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics
    }
    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f)

    # Define group configurations to test
    group_configs = {
        'Group_A_100%': {'Group A': 1.0},      
        'Group_B_100%': {'Group B': 1.0} ,   
        'Group_C_100%': {'Group C': 1.0} , 
        'Group_D_100%': {'Group D': 1.0} 
    }       

    # Run the parallel experiment with all configurations
    print("\n" + "="*70)
    print("STARTING PARALLEL ARRIVAL RATE EXPERIMENT WITH MULTIPLE GROUP CONFIGURATIONS")
    print("="*70)
    
    results = run_parallel_arrival_rate_experiment(users, predictor, _rng, group_configs)
    
    # Save results to outputs folder for Jupyter plotting
    results_file_path = os.path.join(outputs_folder, 'arrival_rate_results_multiple_groupsvary1.pkl')
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    # Group results by configuration for better readability
    results_by_config = {}
    for result in results:
        config_name = result['config_name']
        if config_name not in results_by_config:
            results_by_config[config_name] = []
        results_by_config[config_name].append(result)
    
    for config_name, config_results in results_by_config.items():
        print(f"\n{config_name}:")
        for result in config_results:
            if 'error' not in result:
                print(f"  Rate {result['arrival_rate_per_second']}/s: "
                      f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                      f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                      f"Handoff P={result['nonpredictive_handoffs']:.3f}, "
                      f"NP={result['nonpredictive_handoffs']:.3f} | "
                      f"Calls={result['total_calls']}")
            else:
                print(f"  Rate {result['arrival_rate_per_second']}/s: ERROR - {result['error']}")
    
    print(f"\nResults saved to '{results_file_path}' for Jupyter plotting")
    print(f"All files saved in '{outputs_folder}' folder:")
    print(f"  - {data_file_path}")
    print(f"  - {model_file_path}")
    print(f"  - {metrics_file_path}")
    print(f"  - {results_file_path}")