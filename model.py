import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def train_and_predict(df_features, feature_cols):
    X = df_features[feature_cols]
    y = df_features['target']

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    price_test = df_features['mid_price'].iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_xgb = y_train + 1 
    
    weights = compute_sample_weight(class_weight='balanced', y=y_train_xgb)

    xgb_model = XGBClassifier(
        objective='multi:softprob',
        n_estimators=250,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Training XGBoost...")
    xgb_model.fit(X_train_scaled, y_train_xgb, sample_weight=weights)

    y_pred_xgb = xgb_model.predict(X_test_scaled)
    y_pred_original_scale = y_pred_xgb - 1 

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_original_scale))
    
    return y_pred_original_scale, price_test
