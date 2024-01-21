import pandas as pd

# Paths to your Excel files
main_excel_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/fazcircularity/0117_fazcircularity.xlsx'  # Replace with your main file path
pairs_excel_path1 = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/treatment.xlsx'  # Replace with your pairs file path
pairs_excel_path2 = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/reactivate.xlsx'  # Replace with your pairs file path

# Load the Excel files
main_df = pd.read_excel(main_excel_path)
pairs_df1 = pd.read_excel(pairs_excel_path1)
pairs_df2 = pd.read_excel(pairs_excel_path2)

# Print column names to verify
print("Column names in the file:", pairs_df1.columns.tolist())
print("Column names in the file:", pairs_df2.columns.tolist())

# Extract pairs from the pairs DataFrame
# Assuming the columns in pairs Excel are named 'inactive' and 'active'
pairs1 = list(pairs_df1[['inactive', 'active']].itertuples(index=False, name=None))
pairs2 = list(pairs_df2[['inactive', 'active']].itertuples(index=False, name=None))

# Filter rows in the main DataFrame
filtered_rows1 = main_df[main_df.apply(lambda row: (row['inactive'], row['active']) in pairs1, axis=1)]
filtered_rows2 = main_df[main_df.apply(lambda row: (row['inactive'], row['active']) in pairs2, axis=1)]

# Output the filtered DataFrame
filtered_rows1.to_excel("/raid/jupyter-charlielibear.md09-24f36/entropy/excel/fazcircularity/0117_fazcircularity_treatment.xlsx", index=False)
filtered_rows2.to_excel("/raid/jupyter-charlielibear.md09-24f36/entropy/excel/fazcircularity/0117_fazircularity_reactivate.xlsx", index=False)