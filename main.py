import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('parkinsons.csv')

features = ['MDVP:Fo(Hz)', 'PPE']
X = df[features]
y = df['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {acc}")

if acc >= 0.8:
    joblib.dump(model, 'parkinsons_model.joblib')
    print("Model saved as parkinsons_model.joblib")
