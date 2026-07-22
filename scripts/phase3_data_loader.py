import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

class FeaturePreprocessor:
    def __init__(self):
        # ==========================================
        # TASK 1: CONSTRAIN THE FEATURE SCHEMA
        # ==========================================
        self.metadata_cols = ['track_id', 'frame_index', 'x_position_mm']
        
        # We explicitly drop SEM and isolate the 3 physics drivers.
        # Note: We will dynamically create 'sqrt_mp_area' in load_and_scale
        self.thermal_features = [
            'peak_temp',
            'sqrt_mp_area', 
            'mp_length'
        ]
        
        self.sem_features = [] # SEM is completely disabled (Ablation proven negative)
        
        self.all_features = self.thermal_features

        # ==========================================
        # TASK 2: INITIALIZE THE SCALER
        # ==========================================
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_and_scale(self, csv_path, train_tracks, eval_tracks):
        print(f"Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Apply Physics Constraint: Linearize the area using square root
        if 'mp_area_px' in df.columns:
            df['sqrt_mp_area'] = np.sqrt(df['mp_area_px'].clip(lower=0))
        else:
            raise KeyError("Column 'mp_area_px' missing from dataset.")

        # Bypass pca_ready filtering so we don't accidentally delete Track 21!
        # We assume validity is handled by the TargetAligner now.
        valid_df = df.copy()

        train_df = valid_df[valid_df['track_id'].isin(train_tracks)].copy()
        eval_df = valid_df[valid_df['track_id'].isin(eval_tracks)].copy()
        
        if len(train_df) == 0:
            raise ValueError(f"No data found for training tracks: {train_tracks}")

        train_df = train_df.astype({feature: float for feature in self.all_features})
        if len(eval_df) > 0:
            eval_df = eval_df.astype({feature: float for feature in self.all_features})
            
        print(f"Train split: {len(train_df)} rows (Tracks: {train_tracks})")
        print(f"Eval split:  {len(eval_df)} rows (Tracks: {eval_tracks})")

        # Fit strictly on train to prevent leakage
        self.scaler.fit(train_df[self.all_features])
        self.is_fitted = True
        
        train_df.loc[:, self.all_features] = self.scaler.transform(train_df[self.all_features])
        
        # Only transform eval_df if it has data (Track 21 blind inference safety)
        if len(eval_df) > 0:
            eval_df.loc[:, self.all_features] = self.scaler.transform(eval_df[self.all_features])
            
        print("✅ Physics features successfully engineered and scaled.")
        return train_df, eval_df

    def create_sequence_windows(self, df, window_size=5):
        print(f"\nCreating sequential windows (Size: {window_size} frames)...")
        
        if len(df) == 0:
            return np.array([]), pd.DataFrame(columns=self.metadata_cols)

        X_sequences = []
        metadata_sequences = []
        
        for track_id, group in df.groupby('track_id'):
            group = group.sort_values('x_position_mm').reset_index(drop=True)
            
            feature_array = group[self.all_features].values
            meta_array = group[self.metadata_cols].values
            
            for i in range(len(group) - window_size + 1):
                window_features = feature_array[i : i + window_size]
                target_meta = meta_array[i + window_size - 1]
                
                X_sequences.append(window_features)
                metadata_sequences.append(target_meta)
                
        X_np = np.array(X_sequences)
        meta_df = pd.DataFrame(metadata_sequences, columns=self.metadata_cols)
        
        print(f"Generated {len(X_np)} sequences of shape {X_np.shape} (Samples, Window Size, Features)")
        return X_np, meta_df