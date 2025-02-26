import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('weatherAUS.csv')

# Replace 'NA' with NaN
df.replace("NA", np.nan, inplace=True)

# Drop rows where target variables are missing
df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Convert date to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract date components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop original Date column
df.drop(columns=['Date'], inplace=True)

num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Fill numerical columns using group mean or fallback to global mean
df[num_cols] = df.groupby('Location')[num_cols].transform(lambda x: x.fillna(x.mean()))
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())  # Fill any remaining NaN values

# Fill categorical columns using group mode or fallback to global mode
for col in cat_cols:
    df[col] = df.groupby('Location')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else df[col].mode()[0]))


# Separate features (X) and target (Y)
X = df.drop(columns=['RainToday', 'RainTomorrow'])
Y = df[['RainToday', 'RainTomorrow']].copy()

# Convert target variables to binary (0/1)
Y['RainToday'] = Y['RainToday'].map({'Yes': 1, 'No': 0})
Y['RainTomorrow'] = Y['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Split data first to prevent data leakage
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Handle missing values separately for training and testing
num_imputer = SimpleImputer(strategy='mean')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# One-hot encode categorical variables
ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

X_train_encoded = ct.fit_transform(X_train)
X_test_encoded = ct.transform(X_test)

# Convert to DataFrame
encoded_feature_names = ct.get_feature_names_out()
X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_feature_names)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_feature_names)

# Identify numeric columns after encoding
num_cols_after_encoding = [col for col in X_train_encoded.columns if col.startswith('remainder__')]

# Standardize numerical features
sc = StandardScaler()
X_train_encoded[num_cols_after_encoding] = sc.fit_transform(X_train_encoded[num_cols_after_encoding])
X_test_encoded[num_cols_after_encoding] = sc.transform(X_test_encoded[num_cols_after_encoding])

# Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train_encoded, Y_train)

# Predict
Y_pred = clf.predict(X_test_encoded)

# Evaluate Accuracy
print('RainToday')
accuracy_rain_today = accuracy_score(Y_test.iloc[:, 0], Y_pred[:, 0])
print(f'Random Forest Accuracy: {accuracy_rain_today:.4f}')

print('RainTomorrow')
accuracy_rain_tomorrow = accuracy_score(Y_test.iloc[:, 1], Y_pred[:, 1])
print(f'Random Forest Accuracy: {accuracy_rain_tomorrow:.4f}')
