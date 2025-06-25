import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from multiprocessing import Pool
import functools
import copy
import os
from typing import Dict, List, Any, Tuple, Union

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
    # Suppress INFO logging during simulation
    original_levels = {}
    logger_names = ['src.data_processing', 'src.predict', 'root']
    
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.WARNING)
    
    try:
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
        
        # Return all columns to allow for revenue calculations and other analyses
        final_columns = ['customer_id', 'churn_prediction', 'churn_probability', 'risk_level', 'confidence']
        
        # Add original simulated features if they're useful for analysis
        for col in simulated_df.columns:
            if col not in final_columns:
                final_columns.append(col)
        
        return results_df[final_columns]
    
    finally:
        # Restore original logging levels
        for logger_name, original_level in original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)

def create_distribution_from_data(df: pd.DataFrame, feature: str, dist_type: str = 'auto') -> Tuple[str, List[Any]]:
    """
    Create a distribution definition from real data.
    
    Args:
        df: DataFrame containing the data
        feature: Feature name to analyze
        dist_type: Type of distribution ('auto', 'normal', 'uniform', 'choice')
        
    Returns:
        Tuple of (distribution_type, parameters)
    """
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame")
    
    series = df[feature].dropna()
    
    if dist_type == 'auto':
        # Auto-detect distribution type
        if series.dtype == 'object' or len(series.unique()) < 10:
            dist_type = 'choice'
        else:
            dist_type = 'normal'
    
    if dist_type == 'choice':
        values, counts = np.unique(series, return_counts=True)
        probabilities = counts / counts.sum()
        return ('choice', {'options': values.tolist(), 'probabilities': probabilities.tolist()})
    
    elif dist_type == 'normal':
        mean = series.mean()
        std = series.std()
        return ('normal', [mean, std])
    
    elif dist_type == 'uniform':
        min_val = series.min()
        max_val = series.max()
        return ('uniform', [min_val, max_val])
    
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

def validate_distributions(distributions: Dict[str, Tuple]) -> bool:
    """
    Validate that distribution definitions are properly formatted.
    
    Args:
        distributions: Dictionary of feature distributions
        
    Returns:
        bool: True if all distributions are valid
        
    Raises:
        ValueError: If any distribution is invalid
    """
    supported_types = {
        'uniform', 'normal', 'choice', 'lognormal', 'exponential', 
        'poisson', 'binomial', 'beta', 'gamma', 'chi2', 'f'
    }
    
    for feature, (dist_type, params) in distributions.items():
        if dist_type not in supported_types:
            raise ValueError(f"Unsupported distribution type '{dist_type}' for feature '{feature}'")
        
        if dist_type == 'choice':
            if not isinstance(params, dict) or 'options' not in params or 'probabilities' not in params:
                raise ValueError(f"Choice distribution for '{feature}' must have 'options' and 'probabilities'")
            
            if len(params['options']) != len(params['probabilities']):
                raise ValueError(f"Mismatch between options and probabilities for feature '{feature}'")
            
            if not np.isclose(sum(params['probabilities']), 1.0):
                raise ValueError(f"Probabilities for feature '{feature}' must sum to 1.0")
        
        elif dist_type in ['uniform', 'normal']:
            if not isinstance(params, (list, tuple)) or len(params) != 2:
                raise ValueError(f"Distribution '{dist_type}' for feature '{feature}' must have exactly 2 parameters")
    
    return True

def run_scenario_comparison(predictor, baseline_distributions: Dict, scenarios: Dict[str, Dict], 
                          num_simulations: int = 10000, n_jobs: int = -1) -> pd.DataFrame:
    """
    Run multiple scenarios and compare their results.
    
    Args:
        predictor: ChurnPredictor instance
        baseline_distributions: Base feature distributions
        scenarios: Dict of scenario_name -> distribution_modifications
        num_simulations: Number of customers to simulate per scenario
        n_jobs: Number of parallel jobs
        
    Returns:
        pd.DataFrame: Comparison results across scenarios
    """
    results = []
    
    # Run baseline
    print("Running baseline scenario...")
    baseline_results = run_simulation(predictor, baseline_distributions, num_simulations, n_jobs)
    baseline_churn = baseline_results['churn_probability'].mean()
    
    results.append({
        'scenario': 'Baseline',
        'avg_churn_rate': baseline_churn,
        'churn_increase': 0.0,
        'relative_increase': 0.0
    })
    
    # Run scenarios
    for scenario_name, modifications in scenarios.items():
        print(f"Running scenario: {scenario_name}...")
        
        # Apply modifications to baseline
        scenario_dist = copy.deepcopy(baseline_distributions)
        for feature, new_dist in modifications.items():
            scenario_dist[feature] = new_dist
        
        # Run simulation
        scenario_results = run_simulation(predictor, scenario_dist, num_simulations, n_jobs)
        scenario_churn = scenario_results['churn_probability'].mean()
        
        # Calculate metrics
        churn_increase = scenario_churn - baseline_churn
        relative_increase = (churn_increase / baseline_churn) * 100 if baseline_churn > 0 else 0
        
        results.append({
            'scenario': scenario_name,
            'avg_churn_rate': scenario_churn,
            'churn_increase': churn_increase,
            'relative_increase': relative_increase
        })
    
    return pd.DataFrame(results)

def calculate_revenue_impact(simulation_results: pd.DataFrame, monthly_charges_column: str = 'MonthlyCharges') -> Dict[str, float]:
    """
    Calculate revenue impact from simulation results.
    
    Args:
        simulation_results: DataFrame with simulation results
        monthly_charges_column: Name of the monthly charges column
        
    Returns:
        Dict with revenue impact metrics
    """
    if monthly_charges_column not in simulation_results.columns:
        raise ValueError(f"Column '{monthly_charges_column}' not found in simulation results")
    
    total_customers = len(simulation_results)
    expected_churners = simulation_results['churn_probability'].sum()
    expected_retained = total_customers - expected_churners
    
    # Calculate revenue metrics
    total_potential_revenue = simulation_results[monthly_charges_column].sum()
    expected_lost_revenue = (simulation_results['churn_probability'] * 
                           simulation_results[monthly_charges_column]).sum()
    expected_retained_revenue = total_potential_revenue - expected_lost_revenue
    
    return {
        'total_customers': total_customers,
        'expected_churners': expected_churners,
        'expected_retained': expected_retained,
        'churn_rate': expected_churners / total_customers,
        'retention_rate': expected_retained / total_customers,
        'total_potential_revenue': total_potential_revenue,
        'expected_lost_revenue': expected_lost_revenue,
        'expected_retained_revenue': expected_retained_revenue,
        'revenue_retention_rate': expected_retained_revenue / total_potential_revenue
    }
