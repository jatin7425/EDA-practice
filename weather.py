import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('weatherAUS.csv')

df.replace("NA", np.nan, inplace=True)
df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

# Define X and Y
X = df.drop(columns=['RainToday', 'RainTomorrow'])
Y = df[['RainToday', 'RainTomorrow']].copy()

df['Date'] = pd.to_datetime(df['Date'])

X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day

X = X.drop(columns=['Date'])

num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='mean')
X[num_cols] = num_imputer.fit_transform(X[num_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

X_encoded = ct.fit_transform(X)

encoded_feature_names = ct.get_feature_names_out()
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)
# print(X_encoded)

le = LabelEncoder()
Y['RainToday'] = le.fit_transform(Y['RainToday'])
Y['RainTomorrow'] = le.fit_transform(Y['RainTomorrow'])

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=1)

# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

num_cols_after_encoding = X_encoded.columns[-len(num_cols):]

sc = StandardScaler()
X_train[num_cols_after_encoding] = sc.fit_transform(X_train[num_cols_after_encoding])
X_test[num_cols_after_encoding] = sc.transform(X_test[num_cols_after_encoding])

clf = RandomForestClassifier(n_estimators=100, random_state=1)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print('RainToday')
accuracy_rain_today = accuracy_score(Y_test.iloc[:, 0], Y_pred[:, 0])
print(f'Random Forest Accuracy: {accuracy_rain_today:.4f}')

print('RainTomorrow')
accuracy_rain_tomorrow = accuracy_score(Y_test.iloc[:, 1], Y_pred[:, 1])
print(f'Random Forest Accuracy: {accuracy_rain_tomorrow:.4f}')

