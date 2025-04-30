# 1. Import Necessary Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 2. Load Dataset
df = pd.read_csv('bank.csv', delimiter=';')

# 3. Select Relevant Columns
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']].copy()

# 4. Convert Target Variable to Numeric
df2['y'] = df2['y'].map({'yes': 1, 'no': 0})

# 5. One-Hot Encode Categorical Features
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

# 6. Plot Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 7. Split Dataset into Train and Test Sets
X = df3.drop('y', axis=1)
y = df3['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 8. Train Logistic Regression Model
log_model = LogisticRegression(solver='liblinear', random_state=42)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

# 9. Evaluate Logistic Regression Model
log_cm = confusion_matrix(y_test, log_pred)
log_acc = accuracy_score(y_test, log_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("9. Logistic Regression Accuracy:", round(log_acc, 4))

# 10. Train and Evaluate K-Nearest Neighbors (KNN) Classifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_cm = confusion_matrix(y_test, knn_pred)
knn_acc = accuracy_score(y_test, knn_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Greens', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.title("KNN Confusion Matrix (k=3)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("10. KNN Accuracy (k=3):", round(knn_acc, 4))
