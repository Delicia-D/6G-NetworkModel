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

def run_single_simulation(arrival_rate, service_type, i, total_runs, users, predictor, rng_seed):
    """Run a single simulation for a given arrival rate and service type"""
    try:
        print(f"\n{'='*60}")
        print(f" Simulation {i+1}/{total_runs}: {arrival_rate} calls/second | Service: {service_type or 'Mixed'} (PID: {os.getpid()})")
        print(f"{'='*60}")
        
        # Create new RNG for this process to avoid shared state issues
        _rng = np.random.default_rng(rng_seed)
        
        # Create simulator with current arrival rate
        classifier = CoverageClassifier(mode="real_simulation", rng=_rng)

        # Create and run dual simulator WITH SERVICE TYPE
        sim = DualCallSimulator(
            users=users,
            start_dt=datetime(2025, 8,9, 0, 0, 0),
            end_dt=datetime(2025, 8, 9, 22, 45, 0),
            coverage_classifier=classifier,
            arrival_rate_per_second=arrival_rate,
            predictor=predictor,
            rng=_rng,
            service_type=service_type  # ← ADDED: Pass service type
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
            'service_type': service_type or 'mixed',  # ← ADDED: Track service type
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
        
        print(f" COMPLETED: {arrival_rate} calls/sec | Service: {service_type or 'Mixed'} | "
              f"Blocking P: {predictive_blocking_prob:.3f}, NP: {nonpredictive_blocking_prob:.3f} | "
              f"Handoff P: {predictive_handoff_prob:.3f}, NP: {nonpredictive_handoff_prob:.3f}")
        
        return result
        
    except Exception as e:
        print(f" ERROR in {arrival_rate} calls/sec | Service: {service_type or 'Mixed'}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'arrival_rate_per_second': arrival_rate,
            'arrival_rate_per_hour': arrival_rate * 3600,
            'service_type': service_type or 'mixed',
            'error': str(e),
            'predictive_blocking_prob': 1.0,
            'nonpredictive_blocking_prob': 1.0
        }

def run_service_type_different_rates(users, predictor, base_rng):
    """Run video calls at lower rates (0.1-0.9) and voice calls at higher rates (1-5)"""
    
    # Define different arrival rates for video and voice
    video_arrival_rates = [0.1,0.5,1,2,3,4,5]
    voice_arrival_rates = [0.5,1,2,3,4,5]
                                                                                                                                                                                                                                                                   
    # Get number of CPU cores (use all available)
    n_jobs = multiprocessing.cpu_count()
    
    all_results = []
    
    # Run VIDEO-ONLY first (lower arrival rates)
    print(f"\n{'='*70}")
    print("STARTING VIDEO-ONLY EXPERIMENTS (Lower Arrival Rates: 0.1-0.9 calls/sec)")
    print(f"{'='*70}")
    print(f"   Running {len(video_arrival_rates)} video arrival rates in parallel")
    print(f"   Using {n_jobs} CPU cores")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    video_start_time = time.time()
    
    # Run all video experiments in parallel
    video_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            arrival_rate,
            'video',  # Video-only
            i,
            len(video_arrival_rates),
            users,
            predictor,
            100 + i  # Consistent seeds for video
        )
        for i, arrival_rate in enumerate(video_arrival_rates)
    )
    
    video_time = time.time() - video_start_time
    all_results.extend(video_results)
    
    print(f"\n VIDEO-ONLY EXPERIMENTS COMPLETED!")
    print(f"   Video time: {video_time:.1f}s")
    print(f"   Tested rates: {video_arrival_rates}")
    
    # Run VOICE-ONLY next (higher arrival rates)
    print(f"\n{'='*70}")
    print("STARTING VOICE-ONLY EXPERIMENTS (Higher Arrival Rates: 1-5 calls/sec)")
    print(f"{'='*70}")
    print(f"   Running {len(voice_arrival_rates)} voice arrival rates in parallel")
    print(f"   Using {n_jobs} CPU cores")
    print(f"   Start time: {datetime.now().strftime('%H:%M:%S')}")
    
    voice_start_time = time.time()
    
    # Run all voice experiments in parallel
    voice_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            arrival_rate,
            'voice',  # Voice-only
            i,
            len(voice_arrival_rates),
            users,
            predictor,
            200 + i  # Consistent seeds for voice (different from video)
        )
        for i, arrival_rate in enumerate(voice_arrival_rates)
    )
    
    voice_time = time.time() - voice_start_time
    all_results.extend(voice_results)
    
    total_time = video_time + voice_time
    
    print(f"\n ALL SERVICE TYPE EXPERIMENTS COMPLETED!")
    print(f"   Video-only time: {video_time:.1f}s (rates: {video_arrival_rates})")
    print(f"   Voice-only time: {voice_time:.1f}s (rates: {voice_arrival_rates})")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   End time: {datetime.now().strftime('%H:%M:%S')}")
    
    return all_results

def run_mixed_service_experiment(users, predictor, base_rng):
    """Run mixed service type experiments (optional)"""
    arrival_rates = [0.9, 2, 4, 6, 8]
    
    print(f"\n{'='*70}")
    print("STARTING MIXED SERVICE EXPERIMENTS (70% voice, 30% video)")
    print(f"{'='*70}")
    
    n_jobs = multiprocessing.cpu_count()
    
    mixed_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_simulation)(
            arrival_rate,
            None,  # Mixed service
            i,
            len(arrival_rates),
            users,
            predictor,
            300 + i  # Consistent seeds for mixed
        )
        for i, arrival_rate in enumerate(arrival_rates)
    )
    
    return mixed_results

if __name__ == "__main__":
    print("=" * 70)
    
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

    # Create and save DataFrame
    df = pd.DataFrame(rows)
    df.to_csv("data.csv", index=False)

    print(f"\n GAMMA-RELATIONSHIP DATA GENERATED!")
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

    # NOW run the service type experiments with different arrival rates
    print("\n" + "="*70)
    print("STARTING SERVICE TYPE EXPERIMENTS WITH DIFFERENT ARRIVAL RATES")
    print("VIDEO-ONLY (0.1-0.9 calls/sec) → VOICE-ONLY (1-5 calls/sec)")
    print("="*70)
    
    # Run video-only first (lower rates), then voice-only (higher rates)
    results = run_service_type_different_rates(users, predictor, _rng)
    
    # Optional: Add mixed service type if needed
    # print("\n" + "="*70)
    # print("ADDING MIXED SERVICE EXPERIMENTS")
    # print("="*70)
    # mixed_results = run_mixed_service_experiment(users, predictor, _rng)
    # results.extend(mixed_results)
    
    # Save results for Jupyter plotting
    with open('service_type_different_rates_results1.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    '''with open('service_type_different_rates_results1.json', 'w') as f:
        json.dump(results, f, indent=2)'''
    
    # Print final summary by service type
    print(f"\n{'='*70}")
    print("FINAL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    
    # Group results by service type
    results_by_service = {}
    for result in results:
        if 'error' not in result:
            service = result['service_type']
            if service not in results_by_service:
                results_by_service[service] = []
            results_by_service[service].append(result)
    
    # Print video results (lower rates)
    print(f"\nVIDEO-ONLY (Lower Arrival Rates: 0.1-0.9 calls/sec):")
    print("-" * 60)
    
    if 'video' in results_by_service:
        for result in sorted(results_by_service['video'], key=lambda x: x['arrival_rate_per_second']):
            print(f"  Rate {result['arrival_rate_per_second']:.1f}/s: "
                  f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                  f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                  f"Handoff P={result['predictive_handoff_prob']:.3f}, "
                  f"NP={result['nonpredictive_handoff_prob_']:.3f} | "
                  f"Calls={result['total_calls']}")
    else:
        print(f"  No results for video")
    
    # Print voice results (higher rates)
    print(f"\nVOICE-ONLY (Higher Arrival Rates: 1-5 calls/sec):")
    print("-" * 60)
    
    if 'voice' in results_by_service:
        for result in sorted(results_by_service['voice'], key=lambda x: x['arrival_rate_per_second']):
            print(f"  Rate {result['arrival_rate_per_second']}/s: "
                  f"Blocking P={result['predictive_blocking_prob']:.3f}, "
                  f"NP={result['nonpredictive_blocking_prob']:.3f} | "
                  f"Handoff P={result['predictive_handoff_prob']:.3f}, "
                  f"NP={result['nonpredictive_handoff_prob_']:.3f} | "
                  f"Calls={result['total_calls']}")
    else:
        print(f"  No results for voice")
    
    print(f"\nResults saved to 'service_type_different_rates_results.pkl' for Jupyter plotting")