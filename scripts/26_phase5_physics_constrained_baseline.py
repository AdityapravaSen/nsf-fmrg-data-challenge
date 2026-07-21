import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.phase3_data_loader import FeaturePreprocessor
from src.ml.targets import Phase3TargetAligner

def flatten_windows(X_seq):
    if len(X_seq) == 0:
        return X_seq
    samples, window_size, features = X_seq.shape
    return X_seq.reshape(samples, window_size * features)

def run_physics_loto():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'processed_data', 'final_multimodal_dataset.csv')
    development_tracks = [8, 10, 14]
    
    # We are testing the newly built target from Nabarun
    TARGET_GROUP = 'smoothed_macro_width' 
    
    print("\n" + "="*60)
    print("PHASE 5: PHYSICS-CONSTRAINED BAYESIAN RIDGE LOTO")
    print("="*60)
    
    loto_results = []
    
    for val_track in development_tracks:
        train_tracks = [t for t in development_tracks if t != val_track]
        
        print(f"\n--- Holding out Track {val_track} ---")
        
        # 1. Initialize our new Physics Preprocessor
        preprocessor = FeaturePreprocessor()
        
        # 2. Load and Scale
        train_df, eval_df = preprocessor.load_and_scale(
            csv_path=csv_path,
            train_tracks=train_tracks,
            eval_tracks=[val_track]
        )
        
        # 3. Generate Sequences
        X_train_seq, train_meta = preprocessor.create_sequence_windows(train_df, window_size=5)
        X_val_seq, val_meta = preprocessor.create_sequence_windows(eval_df, window_size=5)
        
        # 4. Flatten for Linear Model
        X_train = flatten_windows(X_train_seq)
        X_val = flatten_windows(X_val_seq)
        
        # 5. Align Targets (Waiting on Nabarun's target_groups update)
        try:
            aligner = Phase3TargetAligner(dataset_path=csv_path)
            Y_train = aligner.align(meta_df=train_meta, target_group=TARGET_GROUP, return_metadata=False)
            Y_val = aligner.align(meta_df=val_meta, target_group=TARGET_GROUP, return_metadata=False)
        except ValueError as e:
            print(f"\n[!] Target Alignment Failed: {e}")
            print(f"[!] Tell Nabarun to ensure '{TARGET_GROUP}' is added to targets.py and the CSV!")
            return
            
        # Ensure Y is 1D for BayesianRidge
        Y_train = Y_train.ravel()
        Y_val = Y_val.ravel()
        
        # 6. Train Physics Baseline

        print(f"Training Bayesian Ridge on {X_train.shape[0]} samples...")
        model = BayesianRidge()
        model.fit(X_train, Y_train)
        
        '''Random Forest Works very bad, dont replace it'''
        # print(f"Training Random Forest on {X_train.shape[0]} samples...")
        # Swap BayesianRidge for RandomForestRegressor
        # model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        # model.fit(X_train, Y_train)
        
        # 7. Evaluate
        Y_pred = model.predict(X_val)
        
        mae = mean_absolute_error(Y_val, Y_pred)
        rmse = np.sqrt(mean_squared_error(Y_val, Y_pred))
        median_ae = median_absolute_error(Y_val, Y_pred)
        r2 = r2_score(Y_val, Y_pred)
        
        print(f"Track {val_track} Results -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")
        
        loto_results.append({
            'Val Track': val_track,
            'MAE': mae,
            'RMSE': rmse,
            'Median AE': median_ae,
            'R2': r2
        })
        
    # Summary
    results_df = pd.DataFrame(loto_results)
    print("\n\n" + "="*50)
    print("=== LOTO Cross-Validation Summary ===")
    print("="*50)
    print(results_df.to_string(index=False))
    
    summary = results_df[['MAE', 'RMSE', 'Median AE', 'R2']].mean().to_frame().T
    print("\n=== Average Performance Across Folds ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    run_physics_loto()