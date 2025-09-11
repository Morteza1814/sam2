# training/utils/freeze_utils.py
import torch

def freeze_first_memory_tokens(model, n_tokens: int = 7,
                               param_name: str = "maskmem_tpos_enc"):
    """
    Zeroes the gradient for the first `n_tokens` rows of the learnable
    temporal-positional-encoding tensor that SAM2 uses for its memory bank.

    Args
    ----
    model : nn.Module
        Your instantiated SAM2 model (SAM2Base or a wrapper).
    n_tokens : int
        How many initial slots to keep frozen. Default = 7.
    param_name : str
        Attribute name that holds the parameter. Default matches
        SAM2Base.maskmem_tpos_enc.
    """
    param: torch.nn.Parameter = getattr(model, param_name)
    if not isinstance(param, torch.nn.Parameter):
        raise AttributeError(f"{param_name} not found or not a Parameter.")

    # keep the whole tensor in the optimizer, but mask the slice’s gradient
    def _hook(grad):
        grad[:n_tokens].zero_()          # kill grad for rows 0..n_tokens-1
        return grad

    param.register_hook(_hook)
    print(f"[freeze_utils] Locked first {n_tokens} memory tokens "
          f"of {param_name}.")
