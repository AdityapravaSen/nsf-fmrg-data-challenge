import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np

class FeaturePreprocessor:
    def __init__(self):
        # ==========================================
        # TASK 1: FREEZE THE FEATURE SCHEMA
        # ==========================================
        # Metadata needed for merging and tracking, NOT for training
        self.metadata_cols = ['track_id', 'frame_index', 'x_position_mm']
        
        # The exact thermal inputs the model is allowed to see
        self.thermal_features = [
            'peak_temp', 
            'mean_temp', 
            'mp_area_px', 
            'mp_centroid_x', 
            'mp_centroid_y', 
            'mp_length', 
            'mp_width'
        ]
        
        # The exact SEM inputs the model is allowed to see
        self.sem_features = [
            'substrate_roughness_variance', 
            'substrate_mean_intensity'
        ]
        
        self.all_features = self.thermal_features + self.sem_features
        
        # ==========================================
        # TASK 2: INITIALIZE THE SCALER
        # ==========================================
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_and_scale(self, csv_path, train_tracks, eval_tracks):
        """
        Loads the multimodal dataset, filters for valid rows, and applies 
        strict scaling to prevent data leakage.
        """
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Step A: Apply Person B's validity mask (Only keep rows where pca_ready is True)
        if 'pca_ready' in df.columns:
            valid_df = df[df['pca_ready'] == True].copy()
            print(f"Filtered dataset to {len(valid_df)} valid rows (pca_ready == True).")
        else:
            warnings.warn("Column 'pca_ready' not found. Using all rows.")
            valid_df = df.copy()

        # Step B: Split the data by physical tracks (Leave-One-Track-Out)
        train_df = valid_df[valid_df['track_id'].isin(train_tracks)].copy()
        eval_df = valid_df[valid_df['track_id'].isin(eval_tracks)].copy()
        
        if len(train_df) == 0:
            raise ValueError(f"No data found for training tracks: {train_tracks}")
            
        print(f"Train split: {len(train_df)} rows (Tracks: {train_tracks})")
        print(f"Eval split:  {len(eval_df)} rows (Tracks: {eval_tracks})")

        # Step C: Fit the scaler ONLY on the training data to prevent data leakage!
        self.scaler.fit(train_df[self.all_features])
        self.is_fitted = True
        
        # Step D: Transform both training and evaluation datasets
        train_df.loc[:, self.all_features] = self.scaler.transform(train_df[self.all_features])
        eval_df.loc[:, self.all_features] = self.scaler.transform(eval_df[self.all_features])
        
        print("✅ Features successfully scaled.")
        
        return train_df, eval_df
    
    def create_sequence_windows(self, df, window_size=5):
        """
        Converts flat tabular track data into rolling sequential windows.
        Groups by track_id to ensure we don't mix frames from different tracks.
        """
        print(f"\nCreating sequential windows (Size: {window_size} frames)...")
        
        X_sequences = []
        metadata_sequences = []
        
        # We must process each track separately so windows don't cross over 
        # from the end of Track 8 into the beginning of Track 10
        for track_id, group in df.groupby('track_id'):
            # Sort by physical position just in case
            group = group.sort_values('x_position_mm').reset_index(drop=True)
            
            # Extract features and metadata as numpy arrays
            feature_array = group[self.all_features].values
            meta_array = group[self.metadata_cols].values
            
            # Create rolling windows
            for i in range(len(group) - window_size + 1):
                # Grab a slice of 'window_size' rows
                window_features = feature_array[i : i + window_size]
                
                # The target prediction will align with the LAST frame in the window
                target_meta = meta_array[i + window_size - 1] 
                
                X_sequences.append(window_features)
                metadata_sequences.append(target_meta)
                
        X_np = np.array(X_sequences)
        meta_df = pd.DataFrame(metadata_sequences, columns=self.metadata_cols)
        
        print(f"Generated {len(X_np)} sequences of shape {X_np.shape} (Samples, Window Size, Features)")
        
        return X_np, meta_df