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

def run_single_simulation(arrival_rate, i, total_rates, users, predictor, rng_seed, accuracy_label, r2_score):
    """Run a single simulation for a given arrival rate"""
    try:
        print(f"\n{'='*60}")
        print(f" Simulation {i+1}/{total_rates}: {arrival_rate} calls/second | Accuracy: {accuracy_label} (PID: {os.getpid()})")
        print(f"{'='*60}")
        
        _rng = np.random.default_rng(rng_seed)
        
        # Create simulator with current arrival rate
        classifier = CoverageClassifier(mode="real_simulation", rng=_rng)

        # Create and run dual simulator
        sim = DualCallSimulator(
            users=users,
            start_dt=datetime(2025, 8, 11,20, 0, 0),
            end_dt=datetime(2025, 8, 11, 22, 45, 0),
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
        
        # Store results with accuracy information
        result = {
            'accuracy_label': accuracy_label,
            'r2_score': r2_score,
            'accuracy_percentage': r2_score * 100,  # R² as percentage
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
        
        print(f" COMPLETED: {arrival_rate} calls/sec | Accuracy: {accuracy_label} ({r2_score:.1%}) | "
              f"Blocking P: {predictive_blocking_prob:.3f}, NP: {nonpredictive_blocking_prob:.3f} | "
              f"Handoff P: {predictive_metrics.handoffs:.3f}, NP: {nonpredictive_metrics.handoffs:.3f}")
        
        return result
        
    except Exception as e:
        print(f" ERROR in {arrival_rate} calls/sec (Accuracy: {accuracy_label}): {e}")
        import traceback
        traceback.print_exc()
        return {
            'accuracy_label': accuracy_label,
            'r2_score': r2_score,
            'accuracy_percentage': r2_score * 100,
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0
        }

def run_parallel_arrival_rate_experiment(users, predictors_dict, base_rng):
    """Run simulations with different arrival rates and accuracy levels in parallel"""
    
    # Define arrival rates to test 
    arrival_rates = [0.6, 2, 4, 6, 8, 10]
    
    # Get number of CPU cores
    n_jobs = multiprocessing.cpu_count()
    
    print(f"\n STARTING MULTI-ACCURACY MULTI-CORE PARALLEL EXECUTION")
    print(f"   Testing {len(predictors_dict)} accuracy levels")
    print(f"   Testing {len(arrival_rates)} arrival rates: {arrival_rates}")
    print(f"   Total simulations: {len(predictors_dict) * len(arrival_rates)}")
    print(f"   Using {n_jobs} CPU cores")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    start_time = time.time()
    
    all_results = []
    
    # Run experiments for each accuracy level
    for accuracy_label, predictor_info in predictors_dict.items():
        print(f"\n{'='*50}")
        print(f"STARTING ACCURACY LEVEL: {accuracy_label}")
        print(f"{'='*50}")
        
        # Get the predictor and its R² score
        predictor = predictor_info['predictor']
        r2_score = predictor_info['r2_score']
        
        base_seed = 42  
        
        # Run parallel simulations for this accuracy level
        accuracy_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_simulation)(
                arrival_rate,
                i,
                len(arrival_rates),
                users,
                predictor,
                base_seed,
                accuracy_label,
                r2_score
            )
            for i, arrival_rate in enumerate(arrival_rates)
        )
        
        all_results.extend(accuracy_results)
    
    total_time = time.time() - start_time
    print(f"\n ALL {len(all_results)} SIMULATIONS COMPLETED!")
    print(f"   Total parallel time: {total_time:.1f}s")
    print(f"   End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return all_results

def generate_data_for_noise_level(noise_level, users, base_rng, outputs_folder):
    """Generate data and train predictor for a specific noise level"""
    print(f"\nGenerating data with {noise_level*100}% noise...")
    
    sim = CallSimulator(
        users=users,
        start_dt=datetime(2025, 1, 1),
        end_dt=datetime(2025, 7, 30),
        rng=base_rng,
        calls_per_day_mean=400,
        noise_level=noise_level,  
    )

    rows = sim.run()
    df = pd.DataFrame(rows)
    
    print(f"  Data generated with {noise_level*100}% noise: {len(df):,} calls")
    
    predictor = CallDurationPredictor()
    predictor.train(df, verbose=False)
    
    # Evaluate the model
    train_metrics, test_metrics = predictor.evaluate(verbose=False)
    
    r2_score = test_metrics['r2']
    
    # Create accuracy label based on R² score
    if r2_score >= 0.9:
        accuracy_label = "High Accuracy"
    elif r2_score >= 0.3:
        accuracy_label = "Medium Accuracy" 
    else:
        accuracy_label = "Poor Accuracy"
    
    print(f"  Model trained with {noise_level*100}% noise data:")
    print(f"    Testing R²: {r2_score:.4f} ({r2_score*100:.1f}%) - {accuracy_label}")
    print(f"    Testing RMSE: {test_metrics['rmse']:.2f}s")
    
    return {
        'predictor': predictor,
        'data': df,
        'test_metrics': test_metrics,
        'r2_score': r2_score,
        'accuracy_label': accuracy_label
    }

if __name__ == "__main__":
    print("=" * 70)
    
    # Create outputs folder if it doesn't exist
    outputs_folder = 'outputs'
    if not os.path.exists(outputs_folder):
        os.makedirs(outputs_folder)
        print(f"Created outputs folder: {outputs_folder}")
    
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

    # Define noise levels to test
    noise_levels = {
        'Low_Noise': 0.02,    
        'Medium_Noise': 0.60, 
        'High_Noise': 1    
    }
    
    predictors_dict = {}
    all_metrics = {}
    
    # Generate data and train predictors for each noise level
    for noise_label, noise_value in noise_levels.items():
        result = generate_data_for_noise_level(noise_value, users, _rng, outputs_folder)
        
        # Use the accuracy label from the training results
        accuracy_label = result['accuracy_label']
        predictors_dict[accuracy_label] = {
            'predictor': result['predictor'],
            'r2_score': result['r2_score'],
            'noise_level': noise_value,
            'original_noise_label': noise_label
        }
        all_metrics[accuracy_label] = result['test_metrics']
        
        # Save individual datasets and models to outputs folder
        data_file_path = os.path.join(outputs_folder, f"call_data_{accuracy_label.replace(' ', '_')}.csv")
        result['data'].to_csv(data_file_path, index=False)
        
        model_file_path = os.path.join(outputs_folder, f'trained_predictor_{accuracy_label.replace(' ', '_')}.pkl')
        with open(model_file_path, 'wb') as f:
            pickle.dump(result['predictor'], f)
    
    # Save all metrics to outputs folder
    metrics_file_path = os.path.join(outputs_folder, 'all_accuracy_levels_metrics.json')
    with open(metrics_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary of all models
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE SUMMARY ACROSS ACCURACY LEVELS")
    print(f"{'='*70}")
    for accuracy_label, predictor_info in predictors_dict.items():
        metrics = all_metrics[accuracy_label]
        print(f"{accuracy_label:16}: R² = {metrics['r2']:.4f} ({metrics['r2']*100:.1f}%) | RMSE = {metrics['rmse']:.2f}s | Noise = {predictor_info['noise_level']*100:.0f}%")
    
    # Run the parallel experiment with all accuracy levels
    print("\n" + "="*70)
    print("STARTING PARALLEL ARRIVAL RATE EXPERIMENT ACROSS ACCURACY LEVELS")
    print("="*70)
    
    results = run_parallel_arrival_rate_experiment(users, predictors_dict, _rng)
    
    # Save results to outputs folder for Jupyter plotting
    results_filename = 'arrival_rate_results_multiaccuracy_20251021_234346.pkl'
    results_file_path = os.path.join(outputs_folder, results_filename)
    
    with open(results_file_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL EXPERIMENT SUMMARY ACROSS ACCURACY LEVELS")
    print(f"{'='*70}")
    
    # Group results by accuracy level 
    results_by_accuracy = {}
    for result in results:
        accuracy_label = result['accuracy_label']
        if accuracy_label not in results_by_accuracy:
            results_by_accuracy[accuracy_label] = []
        results_by_accuracy[accuracy_label].append(result)
    
    for accuracy_label, accuracy_results in results_by_accuracy.items():
        print(f"\n{accuracy_label.upper()} (R² = {accuracy_results[0]['r2_score']:.3f}):")
        for result in accuracy_results:
            if 'error' not in result:
                print(f"  Rate {result['arrival_rate_per_second']}/s: "
                      f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                      f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                      f"Handoff P={result['predictive_handoff_prob']:.3f}, "
                      f"NP={result['nonpredictive_handoff_prob_']:.3f}")
            else:
                print(f"  Rate {result['arrival_rate_per_second']}/s: ERROR - {result['error']}")
    
    print(f"\nResults saved to '{results_file_path}' for Jupyter plotting")
    print(f"All files saved in '{outputs_folder}' folder:")
    print(f"  - Individual datasets: call_data_<accuracy_level>.csv")
    print(f"  - Individual models: trained_predictor_<accuracy_level>.pkl")
    print(f"  - All metrics: {metrics_file_path}")
    print(f"  - Main results: {results_file_path}")