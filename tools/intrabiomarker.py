import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

for n in ["calibre", "tortuosity"]:
    for k in [ "_reactivate", "_treatment", ""]:
        # Load your data
        fullpath = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/csv_classification/0117_' + n + k + '_classification.csv'
        print(fullpath)
        data = pd.read_csv(fullpath)  # Replace with your data file path
    
        # Preprocessing
        X_raw = data.filter(regex='^((?!gabor_).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_gabor = data.filter(regex='^gabor_')
        X_all = data.drop(['patient_number', 'label'], axis=1)
        y = data['label']
        
        # Normalize the features
        scaler = StandardScaler()
        X_raw_scaled = scaler.fit_transform(X_raw)
        X_gabor_scaled = scaler.fit_transform(X_gabor)
        X_all_scaled = scaler.fit_transform(X_all)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier()
        }
        
        # Function to calculate specificity
        def specificity(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)
        
        # Function to calculate metrics
        def calculate_metrics(model, X_train, y_train, X_val, y_val):
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
        
            metrics = {
                'Training Accuracy': accuracy_score(y_train, y_pred_train),
                'Training AUC': roc_auc_score(y_train, y_pred_train),
                'Training Sensitivity': recall_score(y_train, y_pred_train),
                'Training Specificity': specificity(y_train, y_pred_train),
                'Validation Accuracy': accuracy_score(y_val, y_pred_val),
                'Validation AUC': roc_auc_score(y_val, y_pred_val),
                'Validation Sensitivity': recall_score(y_val, y_pred_val),
                'Validation Specificity': specificity(y_val, y_pred_val)
            }
        
            return metrics
        
        # Evaluate models and record results
        results = []
        feature_importances = []
        feature_sets = {'Raw': (X_raw_scaled, X_raw.columns), 'Gabor': (X_gabor_scaled, X_gabor.columns), 'All': (X_all_scaled, X_all.columns)}

        for i in [40,41,42,43,44]:
            for feature_set, (X, columns) in feature_sets.items():
                # Split the data
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=i)
            
                for name, model in models.items():
                    # Train the model
                    model.fit(X_train, y_train)
            
                    # Calculate metrics
                    metrics = calculate_metrics(model, X_train, y_train, X_val, y_val)
            
                    # Get feature importance
                    if name == 'Logistic Regression':
                        importance = np.abs(model.coef_[0])
                    else:
                        importance = model.feature_importances_
            
                    importance_sorted = sorted(zip(columns, importance), key=lambda x: x[1], reverse=True)
                    feature_importances.append({'random_seed': i, 'Model': name, 'Feature Set': feature_set, 'Importances': importance_sorted})
            
                    results.append({
                        'random_seed': i, 
                        'Model': name,
                        'Feature Set': feature_set,
                        **metrics
                    })
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            feature_importances_df = pd.DataFrame(feature_importances)
        
            results_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/eval_intrabiomarker/'+ str(i) + '_' + n + k + '_classification.xlsx'
            # Save results to Excel
            results_df.to_excel(results_path, index=False)
            features_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/eval_intrabiomarker/'+ str(i) + '_' + n + k + '_feature.xlsx'
            # Save results to Excel
            results_df.to_excel(results_path, index=False)
            feature_importances_df.to_excel(features_path, index=False)