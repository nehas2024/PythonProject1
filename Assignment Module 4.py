import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Exercise 1: Regression to the Mean (Dice Rolls)
print("\nExercise 1: Dice Roll Simulation")
print("-" * 40)

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    s = dice1 + dice2
    h, h2 = np.histogram(s, bins=range(2, 14), density=False)

    plt.figure(figsize=(8, 6))
    plt.bar(h2[:-1], h / n, width=0.8, align='center', alpha=0.7)
    plt.xlabel("Sum of Two Dice")
    plt.ylabel("Probability")
    plt.title(f"Probability Distribution (n={n})")
    plt.xticks(range(2, 13))
    plt.grid(axis='y')
    plt.show()

print("\nExercise 1 Results:")
print("  As we simulate more dice rolls (larger n), the histogram gets closer to the theoretical triangle shape, centered around 7.  This shows how the law of large numbers works; with more data, our observations get closer to the expected probabilities.")

# Exercise 2: Regression Model
print("\n\nExercise 2: Weight vs. Height Regression")
print("-" * 40)

try:
    data = pd.read_csv('weight-height.csv')
except FileNotFoundError:
    print("Error: 'weight-height.csv' not found.  Make sure it's in the same folder.")
    exit()

plt.figure(figsize=(8, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.5)
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Height vs. Weight")
plt.grid(True)
plt.show()

X = data[['Height']]
y = data['Weight']
model = LinearRegression()
model.fit(X, y)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

y_predicted = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_predicted))
r2 = r2_score(y, y_predicted)

print("\nExercise 2 Results:")
print(f"  The linear model shows a moderate to strong positive relationship between height and weight.  The model explains about {r2:.0%} of the weight variance (R-squared = {r2:.2f}), and on average, predictions are about {rmse:.2f} pounds off (RMSE).".format(r2=r2, rmse=rmse))
