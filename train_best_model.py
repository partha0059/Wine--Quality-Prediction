import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import ssl

# Fix SSL Certificate Issue
ssl._create_default_https_context = ssl._create_unverified_context

def train_best_model():
    print("Loading Wine Quality dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    best_model_name = None
    best_rmse = float('inf')
    best_model = None
    
    print("\nComparing models using 10-Fold CV:")
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-cv_scores.mean())
        print(f"{name}: RMSE = {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model: {best_model_name} with RMSE: {best_rmse:.4f}")
    
    # Hyperparameter tuning for the best model (if it's an ensemble)
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        print(f"Tuning {best_model_name}...")
        if best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
            
        grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        tuned_rmse = np.sqrt(-grid_search.best_score_)
        print(f"Tuned RMSE: {tuned_rmse:.4f}")
    else:
        best_model.fit(X, y)
        
    # Final Model Metrics
    y_pred = best_model.predict(X)
    final_r2 = r2_score(y, y_pred)
    print(f"Final Model R2 Score (on training set): {final_r2:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'best_wine_model.pkl')
    print("Model saved as best_wine_model.pkl")
    
    return best_model_name, best_rmse

if __name__ == "__main__":
    train_best_model()
