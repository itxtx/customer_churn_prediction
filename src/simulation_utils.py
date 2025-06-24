import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from multiprocessing import Pool
import functools
import copy
import os
import contextlib
import sys
from io import StringIO

def generate_simulated_customer(distributions):
    """
    Generates a single customer by sampling from the defined distributions.

    Args:
        distributions (dict): A dictionary where keys are feature names and values
                              are tuples defining the sampling method and parameters.

    Returns:
        dict: A dictionary representing a single, simulated customer.
    """
    customer = {}
    for feature, (dist_type, params) in distributions.items():
        if dist_type == 'uniform':
            customer[feature] = np.random.uniform(*params)
        elif dist_type == 'normal':
            customer[feature] = np.random.normal(*params)
        elif dist_type == 'choice':
            customer[feature] = np.random.choice(params['options'], p=params['probabilities'])
        elif dist_type == 'lognormal':
            customer[feature] = np.random.lognormal(*params)
        elif dist_type == 'exponential':
            customer[feature] = np.random.exponential(*params)
        elif dist_type == 'poisson':
            customer[feature] = np.random.poisson(*params)
        elif dist_type == 'binomial':
            customer[feature] = np.random.binomial(*params)
        elif dist_type == 'beta':
            customer[feature] = np.random.beta(*params)
        elif dist_type == 'gamma':
            customer[feature] = np.random.gamma(*params)
        elif dist_type == 'chi2':
            customer[feature] = np.random.chisquare(*params)
        elif dist_type == 'f':
            customer[feature] = np.random.f(*params)
        # Can be extended with more distribution types if needed
    return customer

def _run_single_price_scenario(predictor, baseline_distributions, price_factor, num_simulations):
    """
    Run simulation for a single price factor (for parallel processing).
    
    Args:
        predictor: ChurnPredictor instance
        baseline_distributions: Base feature distributions
        price_factor: Price multiplier factor
        num_simulations: Number of customers to simulate
        
    Returns:
        dict: Results for this price scenario
    """
    # Create scenario-specific distribution
    scenario_dist = copy.deepcopy(baseline_distributions)
    original_mean, original_std = scenario_dist['MonthlyCharges'][1]
    scenario_dist['MonthlyCharges'] = ('normal', [original_mean * price_factor, original_std * price_factor])
    
    # Run simulation
    results_df = run_simulation(predictor, scenario_dist, num_simulations, n_jobs=1)
    
    # Calculate key metrics
    avg_churn = results_df['churn_probability'].mean()
    retained_revenue = ((1 - results_df['churn_probability']) * results_df['MonthlyCharges']).sum()
    
    return {
        'price_factor': price_factor,
        'avg_churn_rate': avg_churn,
        'total_retained_revenue': retained_revenue
    }

def run_price_sensitivity_analysis(predictor, baseline_distributions, price_factors, num_simulations, n_jobs=-1):
    """
    Run parallel price sensitivity analysis across multiple price factors.
    
    Args:
        predictor (ChurnPredictor): An instantiated churn predictor object.
        baseline_distributions (dict): Base feature distributions.
        price_factors (list): List of price multiplier factors to test.
        num_simulations (int): Number of customers to simulate per scenario.
        n_jobs (int): Number of jobs to run in parallel. -1 means use all processors.
        
    Returns:
        pd.DataFrame: DataFrame with results for all price scenarios.
    """
    print(f"Running price sensitivity analysis for {len(price_factors)} scenarios using {n_jobs} cores...")
    
    # Run all price scenarios in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_run_single_price_scenario)(predictor, baseline_distributions, factor, num_simulations)
        for factor in price_factors
    )
    
    print("Price sensitivity analysis complete.")
    return pd.DataFrame(results)

def _predict_chunk(predictor, data_chunk: pd.DataFrame) -> list:
    """
    Worker function for parallel processing. It predicts a whole chunk of data.
    
    Because it uses `predict_batch`, it's efficient and logging is not noisy,
    as summary logs are produced per-chunk, not per-customer.
    """
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ['src.data_processing', 'src.predict']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    # Prepare the data chunk for prediction
    df_prepared, _ = predictor.prepare_batch_customers(data_chunk)
    
    # Return only the raw numpy array
    return predictor.model.predict_proba(df_prepared)
def run_simulation(predictor, distributions: dict, num_simulations: int, n_jobs: int = -1) -> pd.DataFrame:
    """
    The definitive simulation runner, using Python's built-in multiprocessing.Pool
    to bypass the suspected bug in joblib/loky.
    """
    print(f"Generating {num_simulations} simulated customers...")
    
    customer_list = [generate_simulated_customer(distributions) for _ in range(num_simulations)]
    simulated_df = pd.DataFrame(customer_list)

    if n_jobs == 1:
        print("Running simulation in single-threaded mode...")
        df_prepared, _ = predictor.prepare_batch_customers(simulated_df)
        proba_matrix = predictor.model.predict_proba(df_prepared)
    else:
        # --- Using multiprocessing.Pool instead of joblib.Parallel ---
        cpu_cores = os.cpu_count() or 1
        if n_jobs == -1: n_jobs = cpu_cores
        else: n_jobs = min(n_jobs, cpu_cores)
            
        chunks = [chunk for chunk in np.array_split(simulated_df, n_jobs) if not chunk.empty]
        
        print(f"Running simulation with multiprocessing.Pool using {n_jobs} workers...")
        
        # Use functools.partial to create a function with the 'predictor' argument already filled in
        worker_func = functools.partial(_predict_chunk, predictor)

        # Create a pool of workers and map the function across the data chunks
        with Pool(processes=n_jobs) as pool:
            proba_chunks = pool.map(worker_func, chunks)
            
        # Safely concatenate the raw numpy arrays into one matrix
        proba_matrix = np.vstack(proba_chunks)

    # --- All DataFrame construction happens safely in the main process ---
    print("Assembling final results...")
    
    results_df = simulated_df.copy()
    results_df['churn_probability'] = proba_matrix[:, 1]
    results_df['confidence'] = np.max(proba_matrix, axis=1)
    predictions = np.argmax(proba_matrix, axis=1)
    results_df['churn_prediction'] = pd.Series(predictions, index=results_df.index).map({0: 'No', 1: 'Yes'})
    conditions = [
        (results_df['churn_probability'] >= 0.8), (results_df['churn_probability'] >= 0.6),
        (results_df['churn_probability'] >= 0.4), (results_df['churn_probability'] >= 0.2)
    ]
    choices = ['Very High', 'High', 'Medium', 'Low']
    results_df['risk_level'] = np.select(conditions, choices, default='Very Low')
    results_df['customer_id'] = [f"sim_customer_{i}" for i in range(len(results_df))]

    print("Simulation complete.")
    
    final_columns = ['customer_id', 'churn_prediction', 'churn_probability', 'risk_level', 'confidence']
    return results_df[final_columns]