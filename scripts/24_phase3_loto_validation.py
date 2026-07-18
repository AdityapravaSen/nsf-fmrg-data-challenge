import sys
import os

# Add the project root (one level up from the scripts folder) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from phase3_data_loader import FeaturePreprocessor 
from src.ml.targets import Phase3TargetAligner

def flatten_windows(X_seq):
    """Flattens (samples, window_size, features) to (samples, window_size * features)"""
    samples, window_size, features = X_seq.shape
    return X_seq.reshape(samples, window_size * features)

def run_loto_validation():
    # Adjust this path if your data is stored differently
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'final_multimodal_dataset.csv')
    
    development_tracks = [8, 10, 14]
    feature_groups = ['sem_only', 'thermal_plus_sem']
    
    # Frozen RF Configuration for Phase III
    rf_config = {
        'n_estimators': 300,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    loto_results = []

    for val_track in development_tracks:
        train_tracks = [t for t in development_tracks if t != val_track]
        print(f"\n{'='*50}")
        print(f"LOTO Fold: Val Track {val_track} | Train Tracks {train_tracks}")
        print(f"{'='*50}")
        
        for feature_group in feature_groups:
            print(f"\nEvaluating Feature Group: {feature_group}")
            
            # 1. Initialize Nabarun's Preprocessor
            preprocessor = FeaturePreprocessor()
            
            # 2. Modify the target features dynamically to avoid breaking the frozen class
            if feature_group == 'sem_only':
                preprocessor.all_features = preprocessor.sem_features
            elif feature_group == 'thermal_only':
                preprocessor.all_features = preprocessor.thermal_features
            # If thermal_plus_sem, leave it as default (thermal_features + sem_features)
            
            # 3. Load, Filter, and Scale (Using Nabarun's API)
            train_df, eval_df = preprocessor.load_and_scale(
                csv_path=csv_path, 
                train_tracks=train_tracks, 
                eval_tracks=[val_track]
            )
            
            # 4. Create Sequence Windows & Metadata
            X_train_seq, train_meta = preprocessor.create_sequence_windows(train_df, window_size=5)
            X_val_seq, val_meta = preprocessor.create_sequence_windows(eval_df, window_size=5)
            
            # 5. Flatten for Random Forest
            X_train = flatten_windows(X_train_seq)
            X_val = flatten_windows(X_val_seq)
            
            # 6. Align Targets (Using Nabarun's EXACT class structure)
            aligner = Phase3TargetAligner(dataset_path=csv_path) 
            Y_train = aligner.align(meta_df=train_meta, target_group='pca_shape', return_metadata=False)
            Y_val = aligner.align(meta_df=val_meta, target_group='pca_shape', return_metadata=False)
            
            # 7. Train Frozen Baseline Model
            print(f"Training Random Forest on {X_train.shape[0]} samples...")
            model = RandomForestRegressor(**rf_config)
            model.fit(X_train, Y_train)
            
            # 8. Predict and Evaluate
            Y_pred = model.predict(X_val)
            
            mae = mean_absolute_error(Y_val, Y_pred)
            rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
            median_ae = median_absolute_error(Y_val, Y_pred)
            r2 = r2_score(Y_val, Y_pred)
            
            print(f"Results -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
            
            # Store results
            loto_results.append({
                'Val Track': val_track,
                'Feature Group': feature_group,
                'MAE': mae,
                'RMSE': rmse,
                'Median AE': median_ae,
                'R2': r2
            })

    # Aggregate and display summary
    results_df = pd.DataFrame(loto_results)
    print("\n\n" + "="*50)
    print("=== LOTO Cross-Validation Summary ===")
    print("="*50)
    print(results_df.to_string(index=False))
    
    summary = results_df.groupby('Feature Group')[['MAE', 'RMSE', 'Median AE', 'R2']].mean().reset_index()
    print("\n=== Average Performance Across Folds ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_loto_validation()