import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('your_dataset.csv')

# Inspect data types
print("Data Types:\n", df.dtypes)

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Encode label target
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])
y = df['Country']

# Drop unnecessary columns
X = df.drop('Country', axis=1)

# Ensure there are no missing values in the features
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

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
