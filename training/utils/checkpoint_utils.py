# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    """
    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern

    Args:
        model: the model that might have been updated
        patterns: for the parameters we want to observe
        allowed:
    """
    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, state_dict.keys()
        )
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    logging.info(f"Loaded checkpoint from {path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys):
    if isinstance(checkpoint, RecursiveScriptModule):
        # This is a torchscript JIT model
        return checkpoint.state_dict()
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        if (isinstance(pre_train_dict, Mapping) and key not in pre_train_dict) or (
            isinstance(pre_train_dict, Sequence) and key >= len(pre_train_dict)
        ):
            key_str = (
                '["' + '"]["'.join(list(map(ckpt_state_dict_keys[:i], str))) + '"]'
            )
            raise KeyError(
                f"'{key}' not found in checkpoint{key_str} "
                f"with keys: {pre_train_dict.keys()}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str] = ("state_dict",),
    map_location: str = "cpu",
) -> nn.Module:
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
        ckpt_state_dict_keys (str): Keys containing the model state dict.
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    assert g_pathmgr.exists(checkpoint_path), "Checkpoint '{}' not found".format(
        checkpoint_path
    )

    # Load the checkpoint on CPU to avoid GPU mem spike.
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    # Not logging into info etc since it's a huge log
    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )
    # Apply kernels
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s"
        % str(", ".join(list(pre_train_dict.keys())))
    )

    return pre_train_dict


def check_load_state_dict_errors(
    missing_keys,
    unexpected_keys,
    strict: bool,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
):
    if ignore_missing_keys is not None and len(ignore_missing_keys) > 0:
        ignored_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, missing_keys
        )
        missing_keys = [key for key in missing_keys if key not in ignored_keys]

    if ignore_unexpected_keys is not None and len(ignore_unexpected_keys) > 0:
        ignored_unexpected_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, unexpected_keys
        )
        unexpected_keys = [
            key for key in unexpected_keys if key not in ignored_unexpected_keys
        ]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."

    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


# def load_state_dict_into_model(
#     state_dict: Dict,
#     model: nn.Module,
#     strict: bool = True,
#     ignore_missing_keys: List[str] = None,
#     ignore_unexpected_keys: List[str] = None,
#     checkpoint_kernels: List[Callable] = None,
# ):
#     """
#     Loads a state dict into the given model.

#     Args:
#         state_dict: A dictionary containing the model's
#             state dict, or a subset if strict is False
#         model: Model to load the checkpoint weights into
#         strict: raise if the state_dict has missing state keys
#         ignore_missing_keys: unix pattern of keys to ignore
#     """
#     # Apply kernels
#     if checkpoint_kernels is not None:
#         for f in checkpoint_kernels:
#             state_dict = f(state_dict=state_dict)
#     # state_dict.pop("maskmem_tpos_enc", None)
#     missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

#     # check_load_state_dict_errors(
#     #     missing_keys,
#     #     unexpected_keys,
#     #     strict=strict,
#     #     ignore_missing_keys=ignore_missing_keys,
#     #     ignore_unexpected_keys=ignore_unexpected_keys,
#     # )
#     return model

# training/utils/checkpoint_utils.py

import torch
import torch.nn.functional as F
from typing import Dict, List, Callable

def _find_key_ending_with(d: Dict[str, torch.Tensor], suffix: str):
    for k in d.keys():
        if k.endswith(suffix):
            return k
    return None

def _expand_linear_1d_table(old_2d: torch.Tensor, new_len: int) -> torch.Tensor:
    """
    old_2d: [old_len, dim]
    returns: [new_len, dim]
    Keeps [:old_len] EXACT; fills tail via linear interp in index space.
    """
    old_len, dim = old_2d.shape
    if new_len == old_len:
        return old_2d
    src = old_2d.T.unsqueeze(0)                        # [1, dim, old_len]
    up  = F.interpolate(src, size=new_len, mode="linear", align_corners=True)
    out = up.squeeze(0).T                              # [new_len, dim]
    out[:old_len] = old_2d                             # avoid tiny interp drift
    return out

def _copy_first_k_rows_maskmem_in_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    k: int = 7,
) -> Dict[str, torch.Tensor]:
    ck_key = _find_key_ending_with(state_dict, "maskmem_tpos_enc")
    if ck_key is None:
        return state_dict

    model_sd = model.state_dict()
    md_key = _find_key_ending_with(model_sd, "maskmem_tpos_enc")
    if md_key is None:
        return state_dict

    ck_t = state_dict[ck_key]   # e.g., [7, 1, 1, D] (checkpoint)
    md_t = model_sd[md_key]     # e.g., [15, 1, 1, D] (model)

    # Expect [T,1,1,D]
    assert ck_t.dim() == 4 and ck_t.shape[1] == 1 and ck_t.shape[2] == 1, \
        f"Unexpected maskmem_tpos_enc shape in ckpt: {ck_t.shape}"
    assert md_t.dim() == 4 and md_t.shape[1] == 1 and md_t.shape[2] == 1, \
        f"Unexpected maskmem_tpos_enc shape in model: {md_t.shape}"
    assert ck_t.shape[-1] == md_t.shape[-1], \
        f"Dim mismatch: ckpt {ck_t.shape} vs model {md_t.shape}"

    old_len, dim = ck_t.shape[0], ck_t.shape[-1]
    new_len = md_t.shape[0]

    # Work in 2D for convenience
    ck_2d = ck_t.view(old_len, dim).to(dtype=md_t.dtype, device=ck_t.device)
    md_2d = md_t.view(new_len, dim).to(dtype=md_t.dtype, device=ck_t.device)

    # Copy only the first n rows
    n = min(k, old_len, new_len)
    out_2d = md_2d.clone()      # start from model's params
    out_2d[:n] = ck_2d[:n]      # overwrite first n rows with checkpoint

    state_dict[ck_key] = out_2d.view(new_len, 1, 1, dim)
    return state_dict

def _resize_maskmem_in_state_dict(state_dict: Dict[str, torch.Tensor],
                                  model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    ck_key = _find_key_ending_with(state_dict, "maskmem_tpos_enc")
    if ck_key is None:
        return state_dict

    model_sd = model.state_dict()
    md_key   = _find_key_ending_with(model_sd, "maskmem_tpos_enc")
    if md_key is None:
        return state_dict

    ck_t = state_dict[ck_key]            # e.g., [7, 1, 1, 64]
    md_t = model_sd[md_key]              # e.g., [15, 1, 1, 64]

    # print(f"Resizing maskmem_tpos_enc: ckpt {ck_t.shape} -> model {md_t.shape}")
    # print("ckpt:", ck_t)
    # print("model:", md_t)
    if ck_t.shape == md_t.shape:
        return state_dict

    # Expect [T, 1, 1, D]; collapse to [T, D], resize, then restore shape.
    assert ck_t.dim() == 4 and ck_t.shape[1] == 1 and ck_t.shape[2] == 1, \
        f"Unexpected maskmem_tpos_enc shape: {ck_t.shape}"
    old_len, _, _, dim = ck_t.shape
    new_len, _, _, dim2 = md_t.shape
    assert dim == dim2, f"Dim mismatch: ckpt {ck_t.shape} vs model {md_t.shape}"

    old_2d = ck_t.view(old_len, dim)
    new_2d = _expand_linear_1d_table(old_2d, new_len).to(dtype=ck_t.dtype, device=ck_t.device)
    # print("Original maskmem_tpos_enc:", old_2d)
    # print("Resized maskmem_tpos_enc:", new_2d)
    new_4d = new_2d.view(new_len, 1, 1, dim)
    state_dict[ck_key] = new_4d
    return state_dict

def _collapse_tpos(t: torch.Tensor) -> torch.Tensor:
    """
    Converts [T,1,1,D] -> [T,D] (or passes through [T,D]).
    """
    if t.dim() == 4 and t.size(1) == 1 and t.size(2) == 1:
        return t.contiguous().view(t.size(0), t.size(3))
    if t.dim() == 2:
        return t
    # Best-effort fallback: squeeze singletons, then take last dim as feature dim.
    t2 = t.squeeze()
    if t2.dim() == 2:
        return t2
    raise AssertionError(f"Unexpected maskmem_tpos_enc shape {tuple(t.shape)}")

def _log_maskmem_first_col(tag: str, tens: torch.Tensor):
    arr = _collapse_tpos(tens).detach().float().cpu()   # [T, D]
    rows = arr.size(0)
    vals = arr[:, 0].tolist()                           # first feature per row
    print(
        f"{tag} maskmem_tpos_enc first elem per row 0..{rows-1}: "
        + ", ".join(f"{v:.6f}" for v in vals)
    )
    
def load_state_dict_into_model(
    state_dict: Dict,
    model: torch.nn.Module,
    strict: bool = True,
    ignore_missing_keys: List[str] = None,
    ignore_unexpected_keys: List[str] = None,
    checkpoint_kernels: List[Callable] = None,
):
        # Identify keys early so we can log meaningfully.
    md_key_pre = _find_key_ending_with(model.state_dict(), "maskmem_tpos_enc")
    ck_key_pre = _find_key_ending_with(state_dict,         "maskmem_tpos_enc")

    # --- BEFORE LOAD (model’s current values)
    if md_key_pre is not None:
        with torch.no_grad():
            _log_maskmem_first_col("[BEFORE LOAD]", model.state_dict()[md_key_pre])
            
    # Apply user kernels first (if any)
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)
    
    ckpt_raw = None
    if ck_key_pre is not None:
        ckpt_raw = state_dict[ck_key_pre]   

    # *** Shape-aware fix for maskmem_tpos_enc ***
    state_dict = _resize_maskmem_in_state_dict(state_dict, model)
    # state_dict = _copy_first_k_rows_maskmem_in_state_dict(state_dict, model, k=7)


    ck_key_post = _find_key_ending_with(state_dict, "maskmem_tpos_enc")

    # --- CKPT views (raw & resized) for visibility
    if ck_key_post is not None:
        with torch.no_grad():
            if ckpt_raw is not None:
                _log_maskmem_first_col("[CKPT RAW     ]", ckpt_raw)
            _log_maskmem_first_col("[CKPT RESIZED ]", state_dict[ck_key_post])

    # Now load; shapes match, so strict can stay True if you want
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    md_key_post = _find_key_ending_with(model.state_dict(), "maskmem_tpos_enc")
    if md_key_post is not None:
        with torch.no_grad():
            _log_maskmem_first_col("[AFTER LOAD   ]", model.state_dict()[md_key_post])
    
    # (Optionally keep your checks enabled)
    # check_load_state_dict_errors(
    #     missing_keys,
    #     unexpected_keys,
    #     strict=strict,
    #     ignore_missing_keys=ignore_missing_keys,
    #     ignore_unexpected_keys=ignore_unexpected_keys,
    # )

    return model
