from pathlib import Path

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)
def check_model_forward(model):
    BATCH_SIZE = 10
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
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)

    for i, data in enumerate(train_data_loader):
        spectrogram, target = data["spectrogram"], data["target"]
        break

    output = model(spectrogram)
    print(f"output : {output}")
    print(f"output shape: {output.shape}")
    return output