import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the existing dataset
csv_path = Path("processed_data/final_multimodal_dataset.csv") 
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

if 'amplitude_um' in df.columns:
    print("Fixing NaNs and calculating smoothed_macro_width_mm...")
    
    # 2. Group by track, interpolate missing gaps, then apply the smoothing window
    df['smoothed_macro_width_mm'] = df.groupby('track_id')['amplitude_um'].transform(
        lambda x: x.interpolate(limit_direction='both').rolling(window=11, center=True, min_periods=1).mean()
    )
    
    # 3. Final safety net: forward/backward fill any remaining edge NaNs
    df['smoothed_macro_width_mm'] = df['smoothed_macro_width_mm'].ffill().bfill()
    
    # 4. Save it back
    df.to_csv(csv_path, index=False)
    print("✅ Successfully injected NaN-free 'smoothed_macro_width_mm' into the dataset!")
else:
    print("Error: Could not find 'amplitude_um'.")