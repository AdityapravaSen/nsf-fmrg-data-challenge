import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_blind_trajectory_plots():
    print("="*60)
    print("📊 GENERATING BLIND TRACK 21 TRAJECTORIES 📊")
    print("="*60)
    
    # Path to your generated CSV
    # Update this to match your specific output folder
    target_dir = r"C:\Users\adity\Documents\Coursework\Projects\nsf-fmrg-data-challenge\processed_data\phase4\baseline_evaluation\25_phase4_baseline_evaluation_20260719_101016"
    csv_path = os.path.join(target_dir, 'track21_predictions.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df = df.sort_values('x_position_mm').reset_index(drop=True)
    
    x = df['x_position_mm']
    targets = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5']
    
    # Create subplots
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    fig.suptitle('Track 21 (Sealed): Predicted PCA Geometry Trajectories\nModel: Ridge Regression (SEM-only)', fontsize=16, fontweight='bold')
    
    for i, target in enumerate(targets):
        ax = axes[i]
        pred_col = f'predicted_{target}'
        
        # Plot Predictions
        ax.plot(x, df[pred_col], label=f'Predicted {target.upper()}', color='darkred', linewidth=2)
        
        ax.set_ylabel(target.upper() + ' Score')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    axes[-1].set_xlabel('Longitudinal Position (x_position_mm)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plot_path = os.path.join(target_dir, 'track21_blind_trajectories.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    print(f"✅ Success! Trajectory plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    generate_blind_trajectory_plots()