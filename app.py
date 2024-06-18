import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Pastikan path file CSV yang benar
csv_file = 'euro2024_players.csv'

# Periksa apakah file ada
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"File '{csv_file}' not found. Please check the file path.")

# Load data
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error loading '{csv_file}': {e}")
    raise

# Inspect data types
print("Data Types:\n", df.dtypes)

# Encode label target
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])
y = df['Country']

# Drop unnecessary columns
X = df.drop('Country', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
