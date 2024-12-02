from pathlib import Path

from numpy import log10
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import wandb

from custom_transforms import LoadSpectrogram, NormalizeSpectrogram, ToTensor, InterpolateSpectrogram
from data_management import make_dataset_name
from models import SpectrVelCNNRegr, weights_init_uniform_rule
from modular_train_test import train_one_epoch

# GROUP NUMBER
GROUP_NUMBER = 74

# CONSTANTS TO MODIFY AS YOU WISH
MODEL = SpectrVelCNNRegr
EPOCHS = 250 # the model converges in test perfermance after ~250-300 epochs
NUM_WORKERS = 10
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# DEVICE = "cpu"

# CONSTANTS TO LEAVE
DATA_ROOT = Path(f"/dtu-compute/02456-p4-e24/data") 
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
STMF_FILENAME = "stmf_data_3.csv"
NFFT = 512
TS_CROPTWIDTH = (-150, 200)
VR_CROPTWIDTH = (-60, 15)

# Functions for sweep
def build_optimizer(model, optimizer, learning_rate, weight_decay):
        if optimizer == "sgd":
            optimizer_ret = torch.optim.sgd(model.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == "adam":
            optimizer_ret = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
        return optimizer_ret

def build_dataset(batch_size):
    # DATA SET SETUP
    dataset_name = make_dataset_name(nfft=NFFT, ts_crop_width=TS_CROPTWIDTH, vr_crop_width=VR_CROPTWIDTH)
    data_dir = DATA_ROOT / dataset_name

    TRAIN_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "train"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    TEST_TRANSFORM = transforms.Compose(
        [LoadSpectrogram(root_dir=data_dir / "test"),
        NormalizeSpectrogram(),
        ToTensor(),
        InterpolateSpectrogram()]
    )
    dataset_train = MODEL.dataset(data_dir= data_dir / "train",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TRAIN_TRANSFORM)

    dataset_test = MODEL.dataset(data_dir= data_dir / "test",
                           stmf_data_path = DATA_ROOT / STMF_FILENAME,
                           transform=TEST_TRANSFORM)
    
    train_data_loader = DataLoader(dataset_train, 
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=NUM_WORKERS)
    test_data_loader = DataLoader(dataset_test,
                                  batch_size=500,
                                  shuffle=False,
                                  num_workers=1)
    return train_data_loader, test_data_loader

def grid_search(config=None):
    with wandb.init(config = config):
        config = wandb.config

        model = MODEL().to(DEVICE)
        model.apply(weights_init_uniform_rule)

        train_data_loader, test_data_loader = build_dataset(config.batch_size)
        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

        ## TRAINING LOOP
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            
                print('EPOCH {}:'.format(epoch_number + 1))

                # Make sure gradient tracking is on
                model.train(True)

                # Do a pass over the training data and get the average training MSE loss
                avg_loss = train_one_epoch(MODEL.loss_fn, model, train_data_loader, optimizer)
                
                # Calculate the root mean squared error: This gives
                # us the opportunity to evaluate the loss as an error
                # in natural units of the ball velocity (m/s)
                rmse = avg_loss**(1/2)

                # Take the log as well for easier tracking of the
                # development of the loss.
                log_rmse = log10(rmse)

                # Reset test loss
                running_test_loss = 0.

                # Set the model to evaluation mode
                model.eval()

                # Disable gradient computation and evaluate the test data
                with torch.no_grad():
                    for i, vdata in enumerate(test_data_loader):
                        # Get data and targets
                        spectrogram, target = vdata["spectrogram"].to(DEVICE), vdata["target"].to(DEVICE)
                        
                        # Get model outputs
                        test_outputs = model(spectrogram)

                        # Calculate the loss
                        test_loss = MODEL.loss_fn(test_outputs.squeeze(), target)

                        # Add loss to runnings loss
                        running_test_loss += test_loss

                # Calculate average test loss
                avg_test_loss = running_test_loss / (i + 1)

                # Calculate the RSE for the training predictions
                test_rmse = avg_test_loss**(1/2)

                # Take the log as well for visualisation
                log_test_rmse = torch.log10(test_rmse)

                print('LOSS train {} ; LOSS test {}'.format(avg_loss, avg_test_loss))
                
                # log metrics to wandb
                wandb.log({
                    "loss": avg_loss,
                    "rmse": rmse,
                    "log_rmse": log_rmse,
                    "test_loss": avg_test_loss,
                    "test_rmse": test_rmse,
                    "log_test_rmse": log_test_rmse,
                })

                epoch_number += 1


if __name__ == "__main__":
    print(f"Using {DEVICE} device")

    # Configure sweep
    sweep_config = {
        "method" : "grid"
    }
    metric = {
        "name" : "loss",
        "goal" : "minimize"
    }

    sweep_config["metric"] = metric

    parameters_dict = {
        "optimizer" : {
            "values" : ["adam"]
        },
        "epochs" : {
            "values" : [EPOCHS]
        },
        "learning_rate" : {
            "values" : [10**-5, 10**-4, 10**-3]
        },
        "weight_decay" : {
            "values" : [10**-3, 10**-2, 10**-1]
        },
        "batch_size" : {
            "values" : [10, 24, 32]
        }
    }

    sweep_config["parameters"] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=f"02456_group_{GROUP_NUMBER}")

    wandb.agent(sweep_id, grid_search, count=27)

    wandb.finish()