import numpy as np
from scipy import stats
import pandas as pd

# Load csv-files
model2_results = pd.read_csv("t_test/validation_results_model_SpectrHybridNet2_sparkling-planet-95.csv")
# model1_results = pd.read_csv("t_test/validation_results_model_SpectrHybridNet_sweet-grass-93.csv")
# model1_results = pd.read_csv("t_test/validation_results_model_SpectrHybridNet3_crisp-universe-105.csv")
# model1_results = pd.read_csv("t_test/validation_results_model_SpectrHybridNet4_confused-shadow-106.csv")
# model1_results = pd.read_csv("t_test/validation_results_model_SpectrHybridNet5_quiet-bee-107.csv")
# model1_results = pd.read_csv("t_test/validation_results_model_SpectrVelCNNRegr_trim-firebrand-3.csv")
model1_results = pd.read_csv("t_test/validation_results_model_SpectrRNN_revived-vortex-92.csv")

# Extract rmse loss
model1_losses = model1_results["rmse"]
model2_losses = model2_results["rmse"]

# Calculate differences (loss differences)
differences = model1_losses - model2_losses

stat, p = stats.shapiro(differences)
print(f"Shapiro-Wilk Test: stat={stat}, p={p}")
if p < 0.05:
    print("Differences may not be normally distributed.")

import matplotlib.pyplot as plt
plt.hist(model1_losses, bins=20, edgecolor='k')
plt.title('Distribution of Differences')
plt.xlabel('Difference (Model 1 - Model 2)')
plt.ylabel('Frequency')
plt.show()

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(model1_losses, model2_losses)

print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Check the mean difference
mean_diff = np.mean(differences)
print(f"Mean Difference: {mean_diff}")

# Determine the better model
if p_value < 0.05:
    if mean_diff > 0:
        print("Model 2 performs significantly better (lower loss) than Model 1.")
    elif mean_diff < 0:
        print("Model 1 performs significantly better (lower loss) than Model 2.")
else:
    print("No significant difference between Model 1 and Model 2.")