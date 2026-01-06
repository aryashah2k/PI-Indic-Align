import os
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import sys
import math
import krippendorff
import statsmodels.stats.inter_rater as ir

# Create a log file for output
log_file = open('kappa_log.txt', 'w')

def log_print(message):
    """Print to console and log file"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

def calculate_cohen_kappa(file_path):
    """
    Calculate Cohen's Kappa score and other inter-annotator agreement metrics
    for a CSV file containing annotations from two experts.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        tuple: (file_name, metrics_dict)
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if the required columns exist
        if 'expert_1' not in df.columns or 'expert_2' not in df.columns:
            log_print(f"Warning: {os.path.basename(file_path)} does not contain expert_1 and expert_2 columns")
            return os.path.basename(file_path), None
        
        # Extract annotations
        expert1 = df['expert_1'].values
        expert2 = df['expert_2'].values
        
        # Calculate percentage of agreement
        agreement = sum(expert1 == expert2) / len(expert1) * 100
        
        # Create a mapping of annotation categories to numbers
        # This is needed because cohen_kappa_score works with numeric labels
        unique_annotations = sorted(list(set(np.concatenate((expert1, expert2)))))
        annotation_to_id = {annotation: i for i, annotation in enumerate(unique_annotations)}
        
        log_print(f"  - Found annotation categories: {unique_annotations}")
        log_print(f"  - Category mapping: {annotation_to_id}")
        
        # Map annotations to numeric IDs
        expert1_ids = np.array([annotation_to_id[annotation] for annotation in expert1])
        expert2_ids = np.array([annotation_to_id[annotation] for annotation in expert2])
        
        # Results dictionary
        metrics = {}
        
        # 1. Calculate Cohen's Kappa score
        kappa = cohen_kappa_score(expert1_ids, expert2_ids)
        metrics['kappa_score'] = kappa
        metrics['agreement_percentage'] = agreement
        metrics['interpretation'] = interpret_kappa(kappa)
        
        # 2. Calculate Krippendorff's Alpha using the library
        # Prepare data for Krippendorff's alpha (needs a reliability data matrix)
        reliability_data = np.array([expert1_ids, expert2_ids])
        
        # Calculate Krippendorff's alpha using the krippendorff library
        alpha = krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
        metrics['krippendorff_alpha'] = alpha
        
        # 3. Calculate Gwet's AC1
        # For nominal data, we use AC1
        table = np.zeros((len(unique_annotations), len(unique_annotations)))
        for i, j in zip(expert1_ids, expert2_ids):
            table[i, j] += 1
            
        # Calculate observed agreement (same as simple percentage agreement)
        po = np.trace(table) / np.sum(table)
        
        # Calculate expected agreement under Gwet's model
        pi = np.sum(table, axis=1) / np.sum(table)  # proportion of items in each category
        pe = np.sum(pi * (1 - pi)) / (len(unique_annotations) - 1)
        
        # Calculate Gwet's AC1
        ac1 = (po - pe) / (1 - pe) if pe < 1 else 1.0
        metrics['gwet_ac1'] = ac1
        
        # 4. Calculate confidence intervals for Cohen's Kappa using statsmodels
        try:
            # Use statsmodels implementation
            kappa_result = ir.cohens_kappa(table)
            
            # The return structure might vary depending on the statsmodels version
            try:
                if hasattr(kappa_result, 'kappa_ci'):
                    # Newer versions return an object with attributes
                    kappa_ci_lower, kappa_ci_upper = kappa_result.kappa_ci
                elif isinstance(kappa_result, tuple) and len(kappa_result) > 1 and isinstance(kappa_result[1], tuple):
                    # Older versions return a tuple with CI as second element
                    kappa_ci_lower, kappa_ci_upper = kappa_result[1]
                else:
                    # Fall back to manual calculation
                    raise ValueError("Unexpected return format from statsmodels.cohens_kappa")
            except Exception as e:
                log_print(f"  - Warning: Could not extract confidence intervals from statsmodels: {e}")
                # Fall back to manual calculation
                raise ValueError("Failed to extract confidence intervals")
        except Exception as e:
            # If statsmodels fails completely, use the manual calculation
            log_print(f"  - Warning: statsmodels confidence interval calculation failed: {e}")
            n = len(expert1_ids)
            po = agreement / 100.0  # Convert percentage to proportion
            
            # Calculate pe (probability of chance agreement) for Cohen's Kappa
            cat_freq1 = np.zeros(len(unique_annotations))
            cat_freq2 = np.zeros(len(unique_annotations))
            
            for i in expert1_ids:
                cat_freq1[i] += 1
            for i in expert2_ids:
                cat_freq2[i] += 1
                
            cat_freq1 = cat_freq1 / n
            cat_freq2 = cat_freq2 / n
            
            pe = sum(cat_freq1[i] * cat_freq2[i] for i in range(len(unique_annotations)))
            
            # Calculate standard error based on Fleiss, Cohen, and Everitt (1969)
            var_kappa = (po * (1 - po)) / (n * (1 - pe)**2)
            se_kappa = math.sqrt(var_kappa)
            
            # 95% confidence interval (using normal approximation)
            z = 1.96  # Z-score for 95% CI
            kappa_ci_lower = max(-1, kappa - z * se_kappa)
            kappa_ci_upper = min(1, kappa + z * se_kappa)
        
        metrics['kappa_ci_lower'] = kappa_ci_lower
        metrics['kappa_ci_upper'] = kappa_ci_upper
        
        # Log all metrics
        log_print(f"  - Cohen's Kappa: {kappa:.4f} (95% CI: {kappa_ci_lower:.4f}-{kappa_ci_upper:.4f})")
        log_print(f"  - Krippendorff's Alpha: {alpha:.4f}")
        log_print(f"  - Gwet's AC1: {ac1:.4f}")
        log_print(f"  - Agreement percentage: {agreement:.2f}%")
        
        return os.path.basename(file_path), metrics
    
    except Exception as e:
        log_print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        import traceback
        log_print(traceback.format_exc())
        return os.path.basename(file_path), None

def interpret_kappa(kappa):
    """
    Interpret Cohen's Kappa score according to common guidelines.
    
    Args:
        kappa (float): Cohen's Kappa score
        
    Returns:
        str: Interpretation of the kappa score
    """
    if kappa < 0:
        return "Poor agreement (less than chance)"
    elif kappa < 0.2:
        return "Slight agreement"
    elif kappa < 0.4:
        return "Fair agreement"
    elif kappa < 0.6:
        return "Moderate agreement"
    elif kappa < 0.8:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"

def main():
    """
    Main function to process all CSV files in the annotator_validations folder.
    """
    # Path to the directory containing validation CSV files
    validation_dir = 'annotator_validation'
    
    log_print(f"Looking for CSV files in directory: {os.path.abspath(validation_dir)}")
    
    # Check if the directory exists
    if not os.path.exists(validation_dir):
        log_print(f"Error: Directory '{validation_dir}' does not exist")
        return
    
    # Get all CSV files in the directory
    csv_files = [os.path.join(validation_dir, f) for f in os.listdir(validation_dir) if f.endswith('.csv')]
    
    log_print(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
    
    if not csv_files:
        log_print(f"No CSV files found in '{validation_dir}'")
        return
    
    # Process each CSV file
    results = []
    for file_path in csv_files:
        log_print(f"\nProcessing file: {file_path}")
        file_name, metrics = calculate_cohen_kappa(file_path)
        
        if metrics is not None:
            result_dict = {'file_name': file_name}
            result_dict.update(metrics)
            results.append(result_dict)
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)
    
    # Save results to a CSV file
    results_df.to_csv('agreement_metrics_results.csv', index=False)
    
    # Print results
    log_print("\nInter-Annotator Agreement Results:")
    log_print("=" * 100)
    log_print(results_df.to_string(index=False))
    log_print("=" * 100)
    log_print("\nResults saved to 'agreement_metrics_results.csv'")
    
    # Close the log file
    log_file.close()

if __name__ == "__main__":
    main()
