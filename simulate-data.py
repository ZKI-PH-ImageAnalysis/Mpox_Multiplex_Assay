import numpy as np
import pandas as pd

# Define the columns
columns = [
    "sampleID_metadata", "panel", "panel_detail", "IgG_A27L", "IgG_A29", "IgG_A33R", "IgG_A35R", "IgG_A5L", "IgG_ATI-C", "IgG_ATI-N",
    "IgG_B5R", "IgG_B6", "IgG_D8L", "IgG_Delta", "IgG_E8", "IgG_H3L", "IgM_A27L", "IgM_A29", "IgM_A33R",
    "IgM_A35R", "IgM_A5L", "IgM_ATI-C", "IgM_ATI-N", "IgM_B5R", "IgM_B6", "IgM_D8L", "IgM_Delta", "IgM_E8", "IgM_H3L"
]

# Define the number of rows
num_rows = 1000

# Set a seed for reproducibility
np.random.seed(385)

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns)

# Set the 'ID' column with unique string IDs
df['sampleID_metadata'] = [f"Sample_{i+1}" for i in range(num_rows)]

# Set the 'panel' and 'panel_detail' columns
panel_values = ["CPXV", "MPXV", "MVA", "Pre", "Pre_New", "SPox", "SPox_Rep"]

# Randomly assign values to 'panel'
df['panel'] = np.random.choice(panel_values, size=num_rows)

# Define possible values for 'panel_detail'
panel_detail_values = ["CPXV", "MPXV", "MVA", "Pre", "Pre_New", "SPox", "SPox_Rep"]

# Randomly assign values to 'panel_detail'
df['panel_detail'] = np.random.choice(panel_detail_values, size=num_rows)

# Generate random data for the rest of the columns
for col in columns[3:]:
    # Generating random floats with a range similar to the provided data
    df[col] = np.random.uniform(low=0.5, high=5.0, size=num_rows)

# Save the DataFrame to a CSV file
output_file = 'simulated_data.csv'
df.to_csv(output_file, index=False)

print(f"Simulated data has been written to {output_file}")