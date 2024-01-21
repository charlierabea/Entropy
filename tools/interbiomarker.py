import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

for n in ["pvalue"]:
    for k in ["_reactivate", "_treatment"]:
        # Load your data
        fullpath = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/csv_classification/0121_' + n + k + '_classification.csv'
        data = pd.read_csv(fullpath)  # Replace with your data file path
        print(fullpath)
        
        # Preprocessing
        X_all = data.drop(['patient_number', 'label'], axis=1)
        y = data['label']
        X_gabor = data.filter(regex='^gabor_')
        X_nogabor = data.filter(regex='^((?!gabor_).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_noentropy = data.filter(regex='^((?!entropy).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_nobvd = data.filter(regex='^((?!bvd).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_nocalibre = data.filter(regex='^((?!calibre).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_nofd = data.filter(regex='^((?!fd).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_notortuosity = data.filter(regex='^((?!tortuosity).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_nofazarea = data.filter(regex='^((?!fazarea).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        X_nofazcircularity = data.filter(regex='^((?!fazcircularity).)*$').drop(['patient_number', 'label'], axis=1, errors='ignore')
        
        # Normalize the features
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        X_gabor_scaled = scaler.fit_transform(X_gabor)
        X_nogabor_scaled = scaler.fit_transform(X_nogabor)
        X_noentropy_scaled = scaler.fit_transform(X_noentropy)
        X_nobvd_scaled = scaler.fit_transform(X_nobvd)
        X_nocalibre_scaled = scaler.fit_transform(X_nocalibre)
        X_nofd_scaled = scaler.fit_transform(X_nofd)
        X_notortuosity_scaled = scaler.fit_transform(X_notortuosity)
        X_nofazarea_scaled = scaler.fit_transform(X_nofazarea)
        X_nofazcircularity_scaled = scaler.fit_transform(X_nofazcircularity)

        
        # Define models
        models = {
            # 'Logistic Regression': LogisticRegression(max_iter=1000),
            # 'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            # 'Lasso': Lasso(alpha=1.0)  # Adjust alpha as needed
        }
        
        # Function to calculate specificity
        def specificity(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)
        
        # Function to calculate metrics
        def calculate_metrics(model, X_train, y_train, X_val, y_val):
            if isinstance(model, Lasso):
                y_pred_train = model.predict(X_train) > 0.5  # Convert to binary
                y_pred_val = model.predict(X_val) > 0.5      # Convert to binary
            else:
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
        feature_sets = {'All': (X_all_scaled, X_all.columns), 'Gabor': (X_gabor_scaled, X_gabor.columns), 'Nogabor': (X_nogabor_scaled, X_nogabor.columns), 'Noentropy': (X_noentropy_scaled, X_noentropy.columns), 'Nobvd': (X_nobvd_scaled, X_nobvd.columns), 'Nocalibre': (X_nocalibre_scaled, X_nocalibre.columns), 'Nofd': (X_nofd_scaled, X_nofd.columns), 'Notortuosity': (X_notortuosity_scaled, X_notortuosity.columns), 'Nofazarea': (X_nofazarea_scaled, X_nofazarea.columns), 'Nofazcircularity': (X_nofazcircularity_scaled, X_nofazcircularity.columns)}

        for i in [40, 41, 42, 43, 44]:
            for feature_set, (X, columns) in feature_sets.items():
                # Split the data
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=i)
            
                for name, model in models.items():
                    print("doing ", model)
                    # Train the model
                    model.fit(X_train, y_train)
            
                    # Calculate metrics
                    metrics = calculate_metrics(model, X_train, y_train, X_val, y_val)
            
                    # Get feature importance
                    if name in['Logistic Regression', 'Lasso']:
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
        
            results_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/eval_interbiomarker/'+ str(i) + '_' + n + k + '_xgboost.xlsx'
            # Save results to Excel
            results_df.to_excel(results_path, index=False)
            features_path = '/raid/jupyter-charlielibear.md09-24f36/entropy/excel/eval_interbiomarker/'+ str(i) + '_' + n + k + '_xgboost_feature.xlsx'
            # Save results to Excel
            results_df.to_excel(results_path, index=False)
            feature_importances_df.to_excel(features_path, index=False)