from pathlib import Path
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from numpy import log10

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrHybridNet1, SpectrHybridNet2, SpectrHybridNet3, SpectrHybridNet4, SpectrHybridNet5

# CONSTANTS
MODEL = SpectrHybridNet5
DEVICE = "cuda"
assert torch.cuda.is_available(), "CUDA is not available. Please run on a GPU machine."

# Model loading setup
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_NAME = "model_SpectrHybridNet5_quiet-bee-107"
SAVED_MODEL_PATH = MODEL_DIR / MODEL_NAME

# Data paths
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def evaluate_single_samples():
    print(f"Using {DEVICE} device")
    print(f"Loading model from: {SAVED_MODEL_PATH}")

    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    # Set up validation transform
    VAL_TRANSFORM = transforms.Compose([
        LoadSpectrogram(root_dir=data_dir / "validation"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()
    ])

    # Load validation dataset
    dataset_val = MODEL.dataset(
        data_dir=data_dir / "validation",
        stmf_data_path=DATA_ROOT / STMF_FILENAME,
        transform=VAL_TRANSFORM
    )

    # Create dataloader with batch_size=1
    val_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # Load the model
    model = MODEL().to(DEVICE)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))
    model.eval()

    # Lists to store results
    losses = []
    predictions = []
    targets = []
    sample_ids = []
    rmses = []
    log_rmses = []

    # Evaluate each sample
    print("Starting evaluation...")
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            spectrogram, target = data["spectrogram"].to(DEVICE), data["target"].to(DEVICE)
            
            # Get model prediction
            output = model(spectrogram)
            
            # Calculate loss
            loss = MODEL.loss_fn(output.squeeze(), target)
            
            # Calculate RMSE and log RMSE for this sample
            rmse = float(loss.item() ** 0.5)
            log_rmse = float(log10(rmse))
            
            # Store results
            losses.append(float(loss.item()))
            predictions.append(float(output.squeeze().cpu().numpy()))
            targets.append(float(target.cpu().numpy()))
            sample_ids.append(i)
            rmses.append(rmse)
            log_rmses.append(log_rmse)
            
            if i % 100 == 0:
                print(f"Processed {i} samples...")

    # Create results DataFrame
    results_df = pd.DataFrame({
        'sample_id': sample_ids,
        'prediction': predictions,
        'target': targets,
        'loss': losses,
        'rmse': rmses,
        'log_rmse': log_rmses,
        'abs_error': np.abs(np.array(predictions) - np.array(targets)),
        'squared_error': np.square(np.array(predictions) - np.array(targets))
    })

    # Save results
    results_filename = f'validation_results_{MODEL_NAME}.csv'
    results_df.to_csv(results_filename, index=False)
    print(f"\nResults saved to: {results_filename}")

    # Calculate and print overall metrics (matching training script)
    avg_loss = np.mean(losses)
    avg_rmse = np.mean(rmses)
    avg_log_rmse = np.mean(log_rmses)

    print("\nValidation Results Summary:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average Log RMSE: {avg_log_rmse:.4f}")
    print(f"Min Loss: {np.min(losses):.4f}")
    print(f"Max Loss: {np.max(losses):.4f}")
    print(f"Loss Std Dev: {np.std(losses):.4f}")

    # Additional statistics
    print("\nPercentile Statistics:")
    print(f"25th percentile Loss: {np.percentile(losses, 25):.4f}")
    print(f"Median Loss: {np.median(losses):.4f}")
    print(f"75th percentile Loss: {np.percentile(losses, 75):.4f}")

    return results_df

if __name__ == "__main__":
    results = evaluate_single_samples()