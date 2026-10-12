import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import StackingClassifier

def train_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    }
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f'{model_name}.pkl')
        print(f"{model_name} trained and saved.")
    
    stacking_model = StackingClassifier(
        estimators=[(name, models[name]) for name in models],
        final_estimator=LogisticRegression(max_iter=1000)
    )
    
    print("Training stacking model...")
    stacking_model.fit(X_train, y_train)
    joblib.dump(stacking_model, 'stacking_model.pkl')
    print("Stacking model trained and saved.")

if __name__ == "__main__":
    print("Running model_training.py...")
    X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')
    train_models(X_train, y_train)
    print("Model training completed.")
