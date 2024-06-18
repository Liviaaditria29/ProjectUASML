import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
df = pd.read_csv('euro2024_players.csv')

# Encode label target
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])
y = df['Country']

# Drop unnecessary columns
X = df.drop('Country', axis=1)

# Ensure there are no missing values in the features
if X.isnull().values.any():
    X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensure no NaN values in X_train and y_train
if X_train.isnull().values.any() or y_train.isnull().values.any():
    raise ValueError("NaN values found in X_train or y_train. Please handle missing values.")

# Fit model
try:
    rf.fit(X_train, y_train)
except ValueError as e:
    print(f"Error fitting the model: {e}")
    raise

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
