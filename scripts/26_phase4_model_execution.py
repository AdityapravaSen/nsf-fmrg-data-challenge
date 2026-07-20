import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Add the project root to the Python path dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the frozen pipeline modules
from phase3_data_loader import FeaturePreprocessor 
from src.ml.targets import Phase3TargetAligner

def flatten_windows(X_seq):
    """Flattens (samples, window_size, features) to (samples, window_size * features)."""
    samples, window_size, features = X_seq.shape
    return X_seq.reshape(samples, window_size * features)

def execute_frozen_baseline(csv_path, output_dir):
    """Executes Stages 2, 3, and 4 of the frozen Phase IV evaluation (Blind Inference)."""
    
    train_tracks = [8, 10, 14]
    eval_tracks = [21]
    
    print("\n" + "="*50)
    print("STAGE 2: Preprocessing & Sequence Generation")
    print("="*50)
    
    preprocessor = FeaturePreprocessor()
    preprocessor.all_features = preprocessor.sem_features  
    
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Train split strictly requires pca_ready == True
    train_df = df[(df['track_id'].isin(train_tracks)) & (df['pca_ready'] == True)].copy()
    
    # Eval split (Track 21) keeps all rows for inference (Blind Test Set)
    eval_df = df[df['track_id'].isin(eval_tracks)].copy()
    
    print(f"Train split: {len(train_df)} valid rows (Tracks: {train_tracks})")
    print(f"Eval split:  {len(eval_df)} inference rows (Tracks: {eval_tracks})")
    
    preprocessor.scaler.fit(train_df[preprocessor.all_features])
    preprocessor.is_fitted = True
    
    train_df.loc[:, preprocessor.all_features] = preprocessor.scaler.transform(train_df[preprocessor.all_features])
    eval_df.loc[:, preprocessor.all_features] = preprocessor.scaler.transform(eval_df[preprocessor.all_features])
    
    print("\nCreating 5-Frame Sequence Windows...")
    X_train_seq, train_meta = preprocessor.create_sequence_windows(train_df, window_size=5)
    X_val_seq, val_meta = preprocessor.create_sequence_windows(eval_df, window_size=5)
    
    X_train = flatten_windows(X_train_seq)
    X_val = flatten_windows(X_val_seq)

    print("\n" + "="*50)
    print("STAGE 3: Target Alignment & Model Fitting")
    print("="*50)
    
    aligner = Phase3TargetAligner(dataset_path=csv_path) 
    
    # Align training targets (these exist)
    Y_train = aligner.align(meta_df=train_meta, target_group='pca_shape', return_metadata=False)
    
    # NOTE: We DO NOT align validation targets, because Track 21 is a blind test set (NaNs).
    
    print(f"Fitting Ridge Regression (alpha=1.0) on {X_train.shape[0]} development samples...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, Y_train)
    
    print(f"Predicting Track {eval_tracks[0]} (Blind Inference)...")
    Y_pred = model.predict(X_val)

    print("\n" + "="*50)
    print("STAGE 4: Formatting & Final Export")
    print("="*50)
    
    print("Notice: Track 21 is a blind test set. Ground truth is hidden.")
    print("Skipping local metric (MAE, RMSE) calculations. Generating submission file...")
    
    # Reconstruct the output DataFrame using the preprocessor's val_meta
    output_df = val_meta[['track_id', 'frame_index', 'x_position_mm']].copy()
    target_names = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5']
    
    for i, col in enumerate(target_names):
        output_df[f'predicted_{col}'] = Y_pred[:, i]
        
    os.makedirs(output_dir, exist_ok=True)
    csv_out_path = os.path.join(output_dir, 'track21_predictions.csv')
    output_df.to_csv(csv_out_path, index=False)
    
    print(f"\n✅ SUCCESS! Sealed evaluation complete.")
    print(f"Submission predictions saved to: {csv_out_path}")

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\adity\Documents\Coursework\Projects\nsf-fmrg-data-challenge\processed_data\final_multimodal_dataset.csv"
    OUTPUT_DIRECTORY = r"C:\Users\adity\Documents\Coursework\Projects\nsf-fmrg-data-challenge\processed_data\phase4\baseline_evaluation\25_phase4_baseline_evaluation_20260719_101016"
    
    execute_frozen_baseline(csv_path=DATASET_PATH, output_dir=OUTPUT_DIRECTORY)