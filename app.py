import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('your_dataset.csv')

# Inspect data types
print(df.dtypes)

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

# Evaluate model
score = rf.score(X_test, y_test)
print("Accuracy:", score)
