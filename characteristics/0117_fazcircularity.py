import os
import pandas as pd
from skimage import io
import util2
import numpy as np
import cv2

parent_dir = "/raid/jupyter-charlielibear.md09-24f36/entropy/data/"
excel_path = "/raid/jupyter-charlielibear.md09-24f36/entropy/excel/patient_list.xlsx"  # Replace with the path to your Excel file

# Read the Excel file with patient numbers
patient_df = pd.read_excel(excel_path)
print(patient_df.columns)

# Prepare the DataFrame for results
results_df = pd.DataFrame(index=patient_df.index)

# Copy the patient numbers to the results DataFrame
results_df['inactive'] = patient_df['inactive']
results_df['active'] = patient_df['active']

# Loop through each preprocessing method directory
for dirname in os.listdir(parent_dir):
    if dirname.startswith("."):
        continue

    if dirname == 'faz' or dirname == 'gabor_faz':
        pre_dir = os.path.join(parent_dir, dirname)
        print(f"Processing directory: {pre_dir}")
    
        # Prepare columns for the current preprocessing method
        results_df[f'inactive_{dirname}'] = np.nan
        results_df[f'active_{dirname}'] = np.nan
    
        # Loop through each patient number and status
        for index, row in patient_df.iterrows():
            for status in ['inactive', 'active']:
                patient_number = row[status]
                image_pattern = f"{patient_number}_*.jpg"  # Replace with the actual pattern if needed
                image_path = os.path.join(pre_dir, status, image_pattern)
                
                # Find the image file that matches the pattern
                for file in os.listdir(os.path.join(pre_dir, status)):
                    if file.startswith(str(patient_number) + "_"):
                        full_image_path = os.path.join(pre_dir, status, file)
                        print(f"Processing image: {full_image_path}")
                        
                        # Read the image and calculate the entropy
                        img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
                        circularity = util2.calculate_faz_ci(img)[1]
                        
                        # Store the entropy value in the DataFrame
                        results_df.at[index, f'{status}_{dirname}'] = circularity
                        break

# Calculate entropy difference or any other required calculation
# Example: results_df['entropy_difference'] = results_df['active_preprocessmethod'] - results_df['inactive_preprocessmethod']

# Save the results to an Excel file
results_df.to_excel('/raid/jupyter-charlielibear.md09-24f36/entropy/excel/fazcircularity/0117_fazcircularity.xlsx')
