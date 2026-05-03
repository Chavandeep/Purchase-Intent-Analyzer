# train_and_save_model.py
# Run this ONCE on your machine to train the model and save all required files.
# After running this, you'll have: model.h5, scaler.pkl, encoder.pkl, label_encoders.pkl
#
# Usage: python train_and_save_model.py
# Requires: data/dataset.csv (the UCI Online Shoppers dataset)

import numpy as np
import pandas as pd
import pickle
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

print("Loading dataset...")
dataset = pd.read_csv('data/dataset.csv')

X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.reset_index(inplace=True)
X_test.reset_index(inplace=True)

# Drop redundant columns
drop_cols = ['Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 'Region', 'TrafficType']
X_train.drop(drop_cols, axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

# Label encode Month, VisitorType, Weekend
# We use separate encoders per column so we can reuse them at inference time
le_month = LabelEncoder()
le_visitor = LabelEncoder()
le_weekend = LabelEncoder()

X_train['Month'] = le_month.fit_transform(X_train['Month'])
X_test['Month'] = le_month.transform(X_test['Month'])

X_train['VisitorType'] = le_visitor.fit_transform(X_train['VisitorType'])
X_test['VisitorType'] = le_visitor.transform(X_test['VisitorType'])

X_train['Weekend'] = le_weekend.fit_transform(X_train['Weekend'])
X_test['Weekend'] = le_weekend.transform(X_test['Weekend'])

# OneHotEncode Month, Browser, VisitorType
ohe = OneHotEncoder(categories='auto', drop='first', sparse_output=False)
train_cat = ohe.fit_transform(X_train[['Month', 'Browser', 'VisitorType']])
test_cat = ohe.transform(X_test[['Month', 'Browser', 'VisitorType']])

train_cat_df = pd.DataFrame(train_cat, index=X_train.index)
test_cat_df = pd.DataFrame(test_cat, index=X_test.index)

X_train.drop(['Month', 'Browser', 'VisitorType'], axis=1, inplace=True)
X_train = X_train.join(train_cat_df)

X_test.drop(['Month', 'Browser', 'VisitorType'], axis=1, inplace=True)
X_test = X_test.join(test_cat_df)

# Fix column names (one-hot encoded columns come in as integers)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Feature count: {X_train_scaled.shape[1]}")
print("Training model...")

# Build model
classifier = Sequential([
    Dense(units=128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dropout(rate=0.6),
    Dense(units=128, activation='relu'),
    Dropout(rate=0.6),
    Dense(units=256, activation='relu'),
    Dropout(rate=0.6),
    Dense(units=128, activation='relu'),
    Dropout(rate=0.4),
    Dense(units=1, activation='sigmoid')
])
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train_scaled, y_train, epochs=50, shuffle=False, validation_split=0.1, verbose=1)

# Evaluate
y_pred = (classifier.predict(X_test_scaled) > 0.5)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save everything
os.makedirs('model', exist_ok=True)
classifier.save('model/model.h5')
print("Saved model/model.h5")

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved model/scaler.pkl")

with open('model/encoder.pkl', 'wb') as f:
    pickle.dump(ohe, f)
print("Saved model/encoder.pkl")

label_encoders = {
    'Month': le_month,
    'VisitorType': le_visitor,
    'Weekend': le_weekend
}
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("Saved model/label_encoders.pkl")

print("\nAll files saved. You can now run: streamlit run app.py")
