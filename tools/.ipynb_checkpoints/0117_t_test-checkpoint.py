import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

def read_excel(file_path):
    # Read the Excel file
    return pd.read_excel(file_path)

def single_tail_t_test(data1, data2):
    # Perform t-test
    _, two_tail_p_val = ttest_ind(data1, data2)
    one_tail_p_val = two_tail_p_val / 2  # Convert to single-tail
    return one_tail_p_val

def calculate_means_and_confidence(data):
    # Calculate mean and 95% confidence interval
    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error
    return mean, ci_lower, ci_upper

def process_data(df):
    results = []
    columns = df.columns
    for col in columns:
        if col.startswith('inactive_'):
            process_name = col.replace('inactive_', '')
            active_col = 'active_' + process_name 
            if active_col in columns:
                inactive_mean, inactive_ci_lower, inactive_ci_upper = calculate_means_and_confidence(df[col])
                active_mean, active_ci_lower, active_ci_upper = calculate_means_and_confidence(df[active_col])
                p_value = single_tail_t_test(df[col], df[active_col])

                results.append({
                    "Process": process_name,
                    "Group": "quiescent",
                    "Mean": f"{inactive_mean:.3f} [{inactive_ci_lower:.3f},{inactive_ci_upper:.3f}]",
                    "P Value": p_value
                })
                results.append({
                    "Process": process_name,
                    "Group": "exudate",
                    "Mean": f"{active_mean:.3f} [{active_ci_lower:.3f},{active_ci_upper:.3f}]",
                    "P Value": ""
                })
    
    return pd.DataFrame(results)

# Example usage
file_path = "/raid/jupyter-charlielibear.md09-24f36/entropy/excel/0117_entropy_treatment.xlsx"  # Replace with your file path
df = read_excel(file_path)

results_df = process_data(df)
results_df.to_excel("/raid/jupyter-charlielibear.md09-24f36/entropy/excel/0117_entropy_treatment_ttest.xlsx", index=False)  # Replace with your desired output file path
