import os
import time
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from model.lstm_model import TemperatureLSTM

# Configuration
DATA_DIR = "data"
MODEL_SAVE_PATH = "temperature_lstm.pth"
TRAIN_TEST_SPLIT = 0.8
BATCH_SIZE = 64
EPOCHS = 50
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
SEQ_LENGTH = 20
NUM_WORKERS = min(4, os.cpu_count())


def load_and_preprocess_data():
    print("Loading and preprocessing data...")

    files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.nc')])
    if not files:
        raise ValueError(f"No .nc files found in {DATA_DIR}")

    print(f"Processing {len(files)} files...")

    # Load data and inspect structure
    sample_data = xr.open_dataset(files[0])
    print("\nSample file structure:")
    print(sample_data)
    print("\nVariables:", list(sample_data.variables))

    # Load all data (assuming time is the first dimension)
    data_arrays = []
    for file in files:
        ds = xr.open_dataset(file)
        # Select a specific location (lat=0, lon=0) if needed
        if 'latitude' in ds.dims and 'longitude' in ds.dims:
            data = ds['skt'].isel(latitude=0, longitude=0).values
        else:
            data = ds['skt'].values
        data_arrays.append(data)

    # Combine all time steps
    data = np.concatenate(data_arrays)
    print(f"\nCombined data shape: {data.shape}")

    # Check for NaN values
    if np.any(np.isnan(data)):
        print(f"Found {np.isnan(data).sum()} NaN values - replacing with mean")
        data = np.nan_to_num(data, nan=np.nanmean(data))

    # Convert Kelvin to Celsius if needed
    if np.nanmean(data) > 200:
        print("Converting Kelvin to Celsius...")
        data = data - 273.15

    # Normalize
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    print(f"\nNormalization - Mean: {mean:.2f}, Std: {std:.2f}")

    # Create sequences
    X, y = [], []
    for i in range(len(data) - SEQ_LENGTH):
        X.append(data[i:i + SEQ_LENGTH])
        y.append(data[i + SEQ_LENGTH])

    if not X:
        raise ValueError(f"No sequences created. Data length: {len(data)}, SEQ_LENGTH: {SEQ_LENGTH}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Convert to tensors
    X = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dimension
    y = torch.FloatTensor(y).unsqueeze(-1)

    # Split dataset
    dataset = TensorDataset(X, y)
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    torch.save({
        'test_dataset': test_dataset,
        'mean': mean,
        'std': std
    }, "test_dataset.pth")

    print(f"\nDataset prepared: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_dataset, mean, std


def train_model():
    torch.set_num_threads(os.cpu_count())
    device = torch.device('cpu')

    train_dataset, mean, std = load_and_preprocess_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = TemperatureLSTM(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}')

    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        'config': {
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'seq_length': SEQ_LENGTH
        }
    }, MODEL_SAVE_PATH)

    print(f"\nTraining completed in {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    train_model()