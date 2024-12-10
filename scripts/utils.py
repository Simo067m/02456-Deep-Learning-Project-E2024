from fvcore.nn import FlopCountAnalysis
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
import torch.nn as nn
from data_management import make_dataset_name
from pathlib import Path
import time

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

def get_flops(model : nn.Module, batch_size : int):
    input_tensor = get_input_tensor(model, batch_size)

    flops = FlopCountAnalysis(model, input_tensor)
    print(f"FLOPS: {flops.total()}")
    print(f"FLOPS formatted: {format_flops(flops.total())}")
    print(flops.by_module())

def format_flops(flops_value):
    if flops_value < 1e3:
        return f"{flops_value:.2f} FLOPS"
    elif flops_value < 1e6:
        return f"{flops_value / 1e3:.2f} K FLOPS"
    elif flops_value < 1e9:
        return f"{flops_value / 1e6:.2f} M FLOPS"
    else:
        return f"{flops_value / 1e9:.2f} G FLOPS"

def get_inference_rate(model : nn.Module, batch_size : int, num_runs : int = 100):
    input_tensor = get_input_tensor(model, batch_size)
    for _ in range(10):
        _ = model(input_tensor)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_time = time.time()

    # Calculate inference rate
    total_time = end_time - start_time
    average_time_per_inference = total_time / num_runs
    inference_rate = 1 / average_time_per_inference  # Batches per second (BPS)

    print(f"Inference Rate: {inference_rate:.2f} Batches per second")

def get_input_tensor(model : nn.Module, batch_size : int):
    NUM_WORKERS = 10

    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TRAIN_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )

    dataset_train = model.dataset(data_dir= data_dir / "train",
                            stmf_data_path = DATA_ROOT / STMF_FILENAME,
                            transform=TRAIN_TRANSFORM)
    train_data_loader = DataLoader(dataset_train, 
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"], data["target"]
        break

    return spectrogram