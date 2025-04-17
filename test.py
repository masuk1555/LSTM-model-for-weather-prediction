import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib.dates as mdates
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.lstm_model import TemperatureLSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration
MODEL_PATH = "temperature_lstm.pth"
TEST_DATA_PATH = "test_dataset.pth"
RESULTS_DIR = "results"
BATCH_SIZE = 64
PLOT_MAX_POINTS = 1000  # Maximum points to plot for better visualization


def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


def load_model_and_data():
    """Load trained model and test dataset"""
    print("Loading model and test data...")

    # Load model and data with weights_only=False for compatibility
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    test_data = torch.load(TEST_DATA_PATH, weights_only=False)

    # Initialize model
    config = checkpoint['config']
    model = TemperatureLSTM(
        input_size=1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test loader
    test_loader = DataLoader(
        test_data['test_dataset'],
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return model, test_loader, checkpoint['mean'], checkpoint['std']


def evaluate_model(model, test_loader, mean, std):
    """Evaluate model performance on test data"""
    print("\nEvaluating model...")
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in tqdm(test_loader, desc='Testing'):
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(batch_y.squeeze().tolist())

    # Convert back to original scale
    predictions = np.array(predictions) * std + mean
    actuals = np.array(actuals) * std + mean

    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print(f"\nTest Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return predictions, actuals, mse, mae, r2


def plot_results(predictions, actuals, metrics):
    """Create time-based visualization with proper date formatting"""
    plt.figure(figsize=(18, 8))

    # Create a simple index-based x-axis since date range is too large
    x_values = np.arange(len(actuals))

    # Plot a subset of points for better visualization
    step = max(1, len(actuals) // PLOT_MAX_POINTS)
    plt.plot(x_values[::step], actuals[::step], label='Actual Temperature', alpha=0.8, linewidth=0.8)
    plt.plot(x_values[::step], predictions[::step], label='Predicted Temperature', alpha=0.6, linewidth=0.8)

    # Formatting
    plt.title('Temperature Prediction vs Actual (10-day intervals)', fontsize=14, pad=20)
    plt.xlabel('Time Periods (10-day intervals)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)

    # Configure x-axis
    ax = plt.gca()

    # Add metrics box
    textstr = '\n'.join((
        f'MSE: {metrics[0]:.4f}',
        f'MAE: {metrics[1]:.4f}',
        f'R²: {metrics[2]:.4f}'))
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'predictions_vs_actuals.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Additional plot: Error distribution
    plt.figure(figsize=(10, 5))
    errors = actuals - predictions
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
    plt.title('Prediction Error Distribution', fontsize=14)
    plt.xlabel('Prediction Error (°C)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, 'error_distribution.png'), dpi=300)
    plt.close()


def save_metrics(metrics):
    """Save metrics to text file"""
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"MSE: {metrics[0]}\n")
        f.write(f"MAE: {metrics[1]}\n")
        f.write(f"R²: {metrics[2]}\n")


def main():
    """Main execution function"""
    ensure_results_dir()
    model, test_loader, mean, std = load_model_and_data()
    predictions, actuals, mse, mae, r2 = evaluate_model(model, test_loader, mean, std)
    plot_results(predictions, actuals, (mse, mae, r2))
    save_metrics((mse, mae, r2))
    print(f"\nResults saved to {RESULTS_DIR} directory")


if __name__ == "__main__":
    main()