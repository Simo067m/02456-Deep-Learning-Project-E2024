#%%
from torch.nn import Module
from models import SpectrVelCNNRegr
from models import SpectrRNN as YourModel
from check_model_forward import check_model_forward

def print_model_complexity(model: Module, return_params = False) -> None:
    """Check and print the number of parameters in the network

    Args:
        model (module): Pytorch model class
    """
    
    total_params = sum(p.numel() for p in model().parameters())
    
    print(f"Number of parameters in model {model.__name__}: {total_params} = {'{:.2e}'.format(total_params)}")

    if return_params:
        return total_params

# %%
if __name__ == "__main__":
    baseline_model = SpectrVelCNNRegr
    print_model_complexity(baseline_model)
    model = YourModel
    print_model_complexity(model)
    model = YourModel()
    check_model_forward(model)

# %%
