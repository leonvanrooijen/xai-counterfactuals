from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# Import json and make a small helper function to keep the code clean
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    
def dataset_log(search: GridSearchCV, X_test, y_test, pred_model, labels):
        
    best_model = search.best_estimator_
    best_params = search.best_params_
    
    y_pred = best_model.predict(X_test)
    
    # Use a single formatted string instead of incremental additions
    
    log_model = f"""
    MODEL {pred_model}:
    Best parameters: {best_params}
    Training score: {search.best_score_}
    Test score: {search.score(X_test, y_test)}
    
    Classification report:
    {classification_report(y_test, y_pred, labels=labels)}
    Accuracy: {accuracy_score(y_test, y_pred)}
    
    Confusion matrix:
    {confusion_matrix(y_test, y_pred, labels=labels)}
    
    Labels: {labels}
    """
    return log_model