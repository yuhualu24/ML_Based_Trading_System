from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from .base import BaseModel
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ClassificationModel(BaseModel):
    def __init__(self, training_data, full_data, model_type='random_forest'):
        super().__init__(training_data, full_data)
        self.prediction_vector = None
        self.test_accuracy = None
        self.train_accuracy = None
        self.model_type = model_type.lower()
        self.scaler = None

    def train(self):
        X = self.training_data.drop(columns=['action_label'])
        y = self.training_data['action_label']

        # Drop rows with NaNs before training
        non_nan_idx = X.dropna().index
        X = X.loc[non_nan_idx]
        y = y.loc[non_nan_idx]

        label_counts = y.value_counts()
        print("Label distribution:", label_counts)

        stratify = y if label_counts.min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify)

        # Add scaling
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss', random_state=42
            )
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.test_accuracy = accuracy_score(y_test, y_pred)
        y_train_pred = self.model.predict(X_train_scaled)
        self.train_accuracy = accuracy_score(y_train, y_train_pred)

    def predict(self):
        X_full = self.full_data.drop(columns=['action_label'], errors='ignore')
        X_full = X_full.dropna()

        # Scale full data using fitted scaler
        if self.scaler:
            X_full_scaled = self.scaler.transform(X_full)
        else:
            raise ValueError("Model must be trained before calling predict (scaler not fitted).")

        self.prediction_vector = self.model.predict(X_full_scaled)
        self.full_data.loc[X_full.index, 'ml_prediction'] = self.prediction_vector

    def train_deep_learning_model(self, label_col='action_label', hidden_units=[64, 32], dropout_rate=0.3, epochs=100,
                                  batch_size=32):
        # Drop NaNs
        X = self.training_data.drop(columns=[label_col])
        y = self.training_data[label_col]
        X = X.dropna()
        y = y.loc[X.index]

        # Store the columns used for training
        training_columns = X.columns.tolist()

        # Scale features
        self.scaler = StandardScaler()  # Use the class instance variable instead of local
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
        )

        # Build model
        model = Sequential()
        model.add(Dense(hidden_units[0], input_dim=X_train.shape[1], activation='relu'))
        model.add(Dropout(dropout_rate))
        for units in hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout_rate))
        model.add(Dense(3, activation='softmax'))  # 3 classes: buy, sell, hold

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                  callbacks=[early_stopping], verbose=1)

        # Evaluation
        y_pred = np.argmax(model.predict(X_val), axis=1)
        print("Classification Report:\n", classification_report(y_val, y_pred))

        # Predict on full dataset - ensure same columns as training
        X_full = self.full_data.drop(columns=[label_col], errors='ignore')

        # Keep only the columns that were used for training and in the same order
        X_full = X_full[training_columns].copy()

        X_full = X_full.dropna()
        X_full_scaled = self.scaler.transform(X_full)

        full_pred = np.argmax(model.predict(X_full_scaled), axis=1)
        self.full_data.loc[X_full.index, 'ml_prediction'] = full_pred

        train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

        self.model = model
        self.train_accuracy = train_accuracy
        self.test_accuracy = val_accuracy
