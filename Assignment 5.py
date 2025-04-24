# Artificial Intelligence with Python
# Assignment 5

# Problem 1: Diabetes
# Investigate the model for predicting Diabetes disease progression by adding more explanatory variables to it in addition to bmi and s5.
# a) Which variable would you add next? Why?
# Ans:- If I had to pick the next variable to add, I'd go with 'bp' (blood pressure).
# It just makes sense that blood pressure would be linked to how diabetes progresses,
# and it tells us something different than just body size ('bmi') or that one blood test ('s5').

# b) How does adding it affect the model's performance?
# Compute metrics and compare to having just bmi and s5.
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data and initial model
diabetes = load_diabetes(as_frame=True)
df_initial = diabetes.frame[['bmi', 's5', 'target']]
X_initial = df_initial[['bmi', 's5']]
y = df_initial['target']
X_train_initial, X_test_initial, y_train, y_test = train_test_split(X_initial, y, test_size=0.2, random_state=42)
model_initial = LinearRegression()
model_initial.fit(X_train_initial, y_train)
y_pred_initial = model_initial.predict(X_test_initial)
mse_initial = mean_squared_error(y_test, y_pred_initial)
rmse_initial = np.sqrt(mse_initial)  # Calculate RMSE by taking the square root
r2_initial = r2_score(y_test, y_pred_initial)
print(f"Problem 1:")
print(f"Initial Model (bmi and s5) - RMSE: {rmse_initial:.2f}, R2: {r2_initial:.2f}")

# Extended model with 'bp'
df_extended = diabetes.frame[['bmi', 's5', 'bp', 'target']]
X_extended = df_extended[['bmi', 's5', 'bp']]
X_train_extended, X_test_extended, y_train, y_test = train_test_split(X_extended, y, test_size=0.2, random_state=42)
model_extended = LinearRegression()
model_extended.fit(X_train_extended, y_train)
y_pred_extended = model_extended.predict(X_test_extended)
mse_extended = mean_squared_error(y_test, y_pred_extended)
rmse_extended = np.sqrt(mse_extended)  # Calculate RMSE by taking the square root
r2_extended = r2_score(y_test, y_pred_extended)
print(f"Extended Model (bmi, s5, bp) - RMSE: {rmse_extended:.2f}, R2: {r2_extended:.2f}")

print(f"\nRMSE changed by: {rmse_initial - rmse_extended:.2f}")
print(f"R2 changed by: {r2_extended - r2_initial:.2f}")

# d) Does it help if you add even more variables?
# Further extended model with 'age' and 's3'
df_further = diabetes.frame[['bmi', 's5', 'bp', 'age', 's3', 'target']]
X_further = df_further[['bmi', 's5', 'bp', 'age', 's3']]
X_train_further, X_test_further, y_train, y_test = train_test_split(X_further, y, test_size=0.2, random_state=42)
model_further = LinearRegression()
model_further.fit(X_train_further, y_train)
y_pred_further = model_further.predict(X_test_further)
mse_further = mean_squared_error(y_test, y_pred_further)
rmse_further = np.sqrt(mse_further)  # Calculate RMSE by taking the square root
r2_further = r2_score(y_test, y_pred_further)
print(f"Further Extended Model (bmi, s5, bp, age, s3) - RMSE: {rmse_further:.2f}, R2: {r2_further:.2f}")

print(f"\nRMSE changed (vs. initial): {rmse_initial - rmse_further:.2f}")
print(f"R2 changed (vs. initial): {r2_further - r2_initial:.2f}")
print(f"RMSE changed (vs. extended): {rmse_extended - rmse_further:.2f}")
print(f"R2 changed (vs. extended): {r2_further - r2_extended:.2f}")

# Problem 2: Profit prediction
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

print("\n--- Problem 2: Profit prediction ---")

# 0) Read the dataset
try:
    startups = pd.read_csv('50_Startups.csv')
    print("\nAlright, got the startup data loaded!")
    print(startups.head())
except FileNotFoundError:
    print("\nOops, couldn't find the '50_Startups.csv' file. Double-check it's in the right place!")
    # Just a fallback in case the file isn't there
    startups = pd.DataFrame({
        'R&D Spend': [1000, 2000, 1500, 2500, 1200],
        'Administration': [500, 600, 550, 700, 520],
        'State': ['NY', 'CA', 'NY', 'FL', 'CA'],
        'Profit': [10000, 15000, 12000, 18000, 11000]
    })

# 1) Let's see what we're working with here - the variables (columns)
print("\nWhat's in this dataset:")
print(startups.columns)

# 2) Time to see how these things relate to each other
# I'm curious which spending areas might really drive profit
correlation_matrix = startups[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']].corr()

# A little visual to make it easier to grasp
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('How the Spending and Profit Numbers Move Together')
plt.show()

print("\nJust the numbers on how they correlate:")
print(correlation_matrix)

# 3) Based on that, which spending bits should we use to guess the profit?
# 'R&D Spend' looks like a big one, and 'Marketing Spend' seems important too.
# 'Administration'? Not so much from the looks of it.
explanatory_vars = ['R&D Spend', 'Marketing Spend']
target_var = 'Profit'
print(f"\nI'm gonna try using: {explanatory_vars} to predict {target_var}")

# 4) Let's make some plots to see if it looks like a straight-ish line relationship
# If it's all over the place, a linear model might not be the best bet
plt.figure(figsize=(8, 6))
sns.scatterplot(x='R&D Spend', y='Profit', data=startups)
plt.title('R&D Spend vs Profit - Does it look like a line?')
plt.xlabel('R&D Spend (in dollars)')
plt.ylabel('Profit (in dollars)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Marketing Spend', y='Profit', data=startups)
plt.title('Marketing Spend vs Profit - What about this one?')
plt.xlabel('Marketing Spend (in dollars)')
plt.ylabel('Profit (in dollars)')
plt.grid(True)
plt.show()

# 5) Gotta split the data so we can train and then test on fresh data
train_data = startups.sample(frac=0.8, random_state=42)
test_data = startups.drop(train_data.index)

X_train = train_data[explanatory_vars]
y_train = train_data[target_var]
X_test = test_data[explanatory_vars]
y_test = test_data[target_var]

print(f"\nTraining data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")

# 6) Time to teach our model using the training data
model = LinearRegression()
model.fit(X_train, y_train)

print("\nAlright, model trained! Hopefully it learned something.")

# 7) Let's see how well it did on both the data it learned from and the new data
y_train_predicted = model.predict(X_train)
y_test_predicted = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_predicted)  # Calculate MSE
rmse_train = np.sqrt(mse_train)  # Then take the square root for RMSE
r2_train = r2_score(y_train, y_train_predicted)

mse_test = mean_squared_error(y_test, y_test_predicted)    # Calculate MSE
rmse_test = np.sqrt(mse_test)      # Then take the square root for RMSE
r2_test = r2_score(y_test, y_test_predicted)

print(f"\nHow'd it do on the training stuff?")
print(f"RMSE (lower is better): {rmse_train:.2f}")
print(f"R-squared (closer to 1 is better): {r2_train:.2f}")

print(f"\nAnd on the new, unseen testing data?")
print(f"RMSE: {rmse_test:.2f}")
print(f"R-squared: {r2_test:.2f}")

# Problem 3: Car mpg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

print("\n--- Problem 3: Car mpg ---")

# 1) Read the data into pandas dataframe
try:
    auto = pd.read_csv('Auto.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Auto.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Clean up potential '?' entries and missing values
auto.replace('?', np.nan, inplace=True)
auto.dropna(inplace=True)

# Convert relevant columns to numeric
numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']
auto[numeric_cols] = auto[numeric_cols].apply(pd.to_numeric)

# 2) Setup multiple regression X and y
# Using all variables except 'mpg', 'name', and 'origin' as features
X = auto[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']]
y = auto['mpg']

# 3) Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Implement both ridge regression and LASSO regression using several values for alpha
alphas = np.logspace(-3, 3, 100)  # Range of alpha values to test
ridge_r2_scores_test = []
lasso_r2_scores_test = []

# Standardize the features (important for Ridge and Lasso)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for alpha in alphas:
    # Ridge Regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_r2_scores_test.append(r2_score(y_test, ridge.predict(X_test_scaled)))

    # LASSO Regression
    lasso = Lasso(alpha=alpha, max_iter=10000)  # Increase max_iter for convergence
    lasso.fit(X_train_scaled, y_train)
    lasso_r2_scores_test.append(r2_score(y_test, lasso.predict(X_test_scaled)))

# 5) Search optimal value for alpha (in terms of R2 score)
optimal_ridge_alpha = alphas[np.argmax(ridge_r2_scores_test)]
best_ridge_r2_test = np.max(ridge_r2_scores_test)

optimal_lasso_alpha = alphas[np.argmax(lasso_r2_scores_test)]
best_lasso_r2_test = np.max(lasso_r2_scores_test)

print(f"Optimal alpha for Ridge Regression: {optimal_ridge_alpha:.4f} (R2 = {best_ridge_r2_test:.4f})")
print(f"Optimal alpha for LASSO Regression: {optimal_lasso_alpha:.4f} (R2 = {best_lasso_r2_test:.4f})")

# 6) Plot the R2 scores for both regressors as functions of alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_r2_scores_test, label='Ridge R2 (Test)')
plt.plot(alphas, lasso_r2_scores_test, label='LASSO R2 (Test)')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score (Test)')
plt.title('R2 Score vs. Alpha for Ridge and LASSO Regression')
plt.legend()
plt.grid(True)
plt.show()

# 7) Identify, as accurately as you can, the value for alpha which gives the best score
'''
Findings and Explanations:

Data Loading and Preprocessing:
The code successfully reads the 'Auto.csv' file into a pandas DataFrame. It handles potential missing values represented by '?' by replacing them with NaN and then dropping rows with any NaN values. The necessary columns ('mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year') are converted to numeric types to be used in the regression models.

Feature and Target Setup:
The features (X) for the multiple regression are selected as 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', and 'year'. The target variable (y) is 'mpg' (miles per gallon), which we aim to predict. The 'name' and 'origin' columns are excluded as they are categorical and might require different preprocessing techniques or might not be directly linearly related to 'mpg' in a simple way.

Data Splitting:
The dataset is split into training (80%) and testing (20%) sets using a random state for reproducibility. The training data is used to fit the Ridge and LASSO models, and the testing data is used to evaluate their performance on unseen data.

Ridge and LASSO Implementation:
Both Ridge and LASSO regression are implemented using a range of alpha values generated logarithmically using `np.logspace`. This allows us to explore a wide range of regularization strengths. Feature scaling using `StandardScaler` is applied to both the training and testing features. This is crucial for Ridge and LASSO as they are sensitive to the scale of the input features. Without scaling, features with larger values might disproportionately influence the model.

Optimal Alpha Search:
The code iterates through the alpha values, fits both Ridge and LASSO models on the scaled training data, and calculates the R2 score on the scaled testing data. The optimal alpha for each regressor is determined as the value that yields the highest R2 score on the testing set. The corresponding best R2 scores are also printed.

Plotting R2 Scores:
A plot is generated to visualize how the R2 scores for both Ridge and LASSO regressors change as a function of alpha. The x-axis (alpha) is on a logarithmic scale to better visualize the effect of alpha across a wide range of values. The plot helps in visually identifying the range of alpha values that lead to good performance.

Identifying Optimal Alpha:
Based on the printed optimal alpha values and the plot, we can identify the alpha values that give the best R2 scores on the testing data for both Ridge and LASSO regression. The printed values provide a more accurate numerical estimate of these optimal alphas. The plot visually confirms these values and shows the trend of model performance with varying regularization strength.

In this specific run, the optimal alpha for Ridge Regression is approximately 0.0010, achieving an R2 score of around 0.7942. The optimal alpha for LASSO Regression is around 0.7055, achieving a slightly better R2 score of around 0.8054. These values indicate that for this dataset and feature set, a small amount of regularization (very small alpha for Ridge, a moderate alpha for LASSO) leads.'''