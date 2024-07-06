import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('suprize_faces_blendshapes.csv')

# Set the 'Image' column as the index to facilitate plotting
df.set_index('Image', inplace=True)

# Transpose the DataFrame to have blendshapes as columns and images as rows
df_transposed = df.T

# Plotting
plt.figure(figsize=(14, 8))
for blendshape in df_transposed.columns:
    plt.plot(df_transposed.index, df_transposed[blendshape], marker='o', linestyle='-', linewidth=0)  # Removed label argument

plt.title('Blendshape Scores per Image - Suprize')
plt.xlabel('Blendshape Name')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')

# Assuming scores are between 0 and 1, adjust this range as necessary
plt.ylim(0, 1)  # Set the limits of the y-axis
plt.yticks(np.arange(0, 1.1, 0.2))  # Set y-axis ticks to be at intervals of 0.2

# plt.legend() call has been removed to not show labels
plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust layout to make room for the legend

plt.savefig("Analysis_Suprize.png")
