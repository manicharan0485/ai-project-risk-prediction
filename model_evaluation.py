import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    print("Running model_evaluation.py...")
    X_train, X_test, y_train, y_test = joblib.load('preprocessed_data.pkl')
    model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'stacking_model']
    
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        model = joblib.load(f'{model_name}.pkl')
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        print(f"{model_name.capitalize()} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("------------------------------")
