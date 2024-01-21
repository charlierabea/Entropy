import pandas as pd
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import statsmodels.api as sm

def read_excel(file_path):
    return pd.read_excel(file_path)

def paired_t_test_one_tail(data1, data2):
    t_statistic, two_tail_p_val = ttest_rel(data1, data2)
    
    if t_statistic > 0:  # assuming you're testing if data1 > data2
        one_tail_p_val = two_tail_p_val / 2
    else:
        one_tail_p_val = 1 - (two_tail_p_val / 2)

    return one_tail_p_val

def calculate_means_and_confidence(data):
    mean = np.mean(data)
    std_error = np.std(data) / np.sqrt(len(data))
    ci_lower = mean - 1.96 * std_error
    ci_upper = mean + 1.96 * std_error
    return mean, ci_lower, ci_upper

def calculate_r_squared(data1, data2):
    X = sm.add_constant(data1)
    model = sm.OLS(data2, X)
    results = model.fit()
    return results.rsquared

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
                p_value = paired_t_test_one_tail(df[col], df[active_col])
                # r_squared = calculate_r_squared(df[col], df[active_col])

                results.append({
                    "Process": process_name,
                    "Group": "quiescent",
                    "Mean": f"{inactive_mean:.2f} [{inactive_ci_lower:.2f},{inactive_ci_upper:.2f}]",
                    "P Value": f"{p_value:.3f}",
                    # "R Squared": f"{r_squared:.3f}"
                })
                results.append({
                    "Process": process_name,
                    "Group": "exudate",
                    "Mean": f"{active_mean:.2f} [{active_ci_lower:.2f},{active_ci_upper:.2f}]",
                    "P Value": "",
                    # "R Squared": ""  # R Squared is only relevant for the pair, not each group individually
                })
    
    return pd.DataFrame(results)

# Example usage
for n in ["entropy", "bvd", "calibre", "fd", "tortuosity", "fazarea", "fazcircularity"]:
    for k in ["_reactivate", "_treatment", ""]:
        # Load your data
        fullpath = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/biomarker_' + n + '/0117_' + n + k + '.xlsx'
        print(fullpath)
        df = read_excel(fullpath)
        
        results_df = process_data(df)
        results_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/eval_ttest/'+ n + k + '_ttest.xlsx'
        print(results_path)
        results_df.to_excel(results_path, index=False)  # Replace with your desired output file path
