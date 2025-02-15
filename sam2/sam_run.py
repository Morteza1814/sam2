import torch
import torchvision
import sys
import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from torch.profiler import profile, record_function, ProfilerActivity
import gc

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# Function for tracking memory
def log_memory_usage(tag=""):
    print(f"\n[{tag}] GPU Memory Usage")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB\n")


checkpoint = "/bigtemp/rgq5aw/samData/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
# print(predictor)

# # #  profiling all layers of SAM2
# def shape_memory_hook(module, input, output):
#     torch.cuda.synchronize()  # Ensure accurate memory readings
#     input_shapes = [inp.shape for inp in input]
#     output_shapes = [out.shape for out in output] if isinstance(output, (tuple, list)) else [output.shape]

#     # Find the module's hierarchical name in the model
#     for name, mod in predictor.named_modules():
#         if mod is module:
#             module_name = name  # Get full hierarchical module name
    
#     print(f"\n Layer: {module.__class__.__name__}")
#     print(f" Full Module Path: {module_name}")  # Now prints full hierarchy!
#     print(f" Input Shapes: {input_shapes}")
#     print(f" Output Shapes: {output_shapes}")
#     print(f" Allocated Memory: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
#     print(f" Max Allocated: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB\n")


# # Register hooks on relevant layers
# for layer in predictor.modules():
#     if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, torch.nn.LayerNorm, torch.nn.ReLU, torch.nn.Sigmoid)):
#         layer.register_forward_hook(shape_memory_hook)

def cross_attn_sub_layer_memory_hook(module, name, is_start, memory_records, index):
    try:
        if module is None:
            print(f"Skipping hook for {name} because the module is None!")
            return

        torch.cuda.synchronize()  # Ensure accurate memory readings
        allocated_memory = torch.cuda.memory_allocated()
        
        if is_start:
            memory_records[index]['before'] = allocated_memory
        else:
            layer_memory = allocated_memory - memory_records[index]['before']
            memory_records[index]['max'] = max(memory_records[index]['max'], layer_memory)
    except Exception as e:
        print(f"Hook execution failed for {name}: {e}")

def hook_registerar(predictor, layers):
    memory_records = [{'before': 0, 'max': 0, 'layer': layer} for layer in layers]
    
    for name, mod in predictor.named_modules():
        for idx, layer in enumerate(layers):
            layer_name, start, end = layer
            if layer_name in name and mod is not None:
                try:
                    if start in name:
                        mod.register_forward_hook(lambda module, _, __, idx=idx: cross_attn_sub_layer_memory_hook(module, name, True, memory_records, idx))
                        print(f"Hook registered on: {name} (Start)")
                    if end in name:
                        mod.register_forward_hook(lambda module, _, __, idx=idx: cross_attn_sub_layer_memory_hook(module, name, False, memory_records, idx))
                        print(f"Hook registered on: {name} (End)")
                except Exception as e:
                    print(f"Failed to register hook on {name}: {e}")
    
    return memory_records

# Example Usage:
layers = [
    # memory attention
    ("memory_attention.layers.", "norm1", "self_attn.out_proj"), # self attention
    ("memory_attention.layers.", "norm2", "cross_attn_image.out_proj"), # cross attention
    ("memory_attention.layers.", "norm3", "linear2"), # linear layers
    # mask decoder
    ("sam_mask_decoder.transformer.layers.", "self_attn.q_proj", "self_attn.out_proj"), # self attention
    ("sam_mask_decoder.transformer.layers.", "norm1", "cross_attn_token_to_image.out_proj"), # cross_attn_token_to_image
    ("sam_mask_decoder.transformer.layers.", "norm2", "mlp.layers.1"), # mlp layers
    ("sam_mask_decoder.transformer.layers.", "norm3", "cross_attn_image_to_token.out_proj"), # cross_attn_image_to_token
    # image encoder
    ("image_encoder.trunk.blocks.", "norm1", "proj"), # qkv and proj
    ("image_encoder.trunk.blocks.", "norm2", "mlp.layers.1"), # MLP layers
    ("image_encoder.neck.convs.", "0.conv", "3.conv"), # final CONV layers
    # memory encoder
    ("memory_encoder.mask_downsampler.", "encoder.0", "encoder.12"), # mask downsampler
    ("memory_encoder.fuser.layers.", "0.dwconv", "1.pwconv2"), # conv
]

memory_records = hook_registerar(predictor, layers)

video_dir = "/bigtemp/rgq5aw/samData/videos/bedroom"
# video_dir = "/bigtemp/rgq5aw/samData/videos/sav_dataset/sav_test/JPEGImages_24fps/sav_010681"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]

frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# Select the first video frame
# frame_idx = 0
# frame_path = os.path.join(video_dir, frame_names[frame_idx])
# frame = Image.open(frame_path)

# Plot and save the frame
# plt.figure(figsize=(9, 6))
# plt.title(f"frame {frame_idx}")
# plt.imshow(frame)
# output_plot_path = "saved_plot.png"
# plt.savefig(output_plot_path, bbox_inches="tight")  # Save the plot

# with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
#     with torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA
#         ],
#         record_shapes=True,
#         with_stack=True,
#         profile_memory=True,   # Enables memory profiling
#         with_flops=True,
#         with_modules=True,
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
#     ) as prof:
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    # with record_function("SAM2_inference"):
    # log_memory_usage("Before Init State")
    inference_state = predictor.init_state(video_path=video_dir)
    # log_memory_usage("After Init State")
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array([[210, 350], [250, 220]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array([1, 1], np.int32)

    # with prof:
        # log_memory_usage("Before Add New Points")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
        # log_memory_usage("After Add New Points")
        # Create the figure
    # plt.figure(figsize=(9, 6))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))

    # # Add points and mask to the figure
    # show_points(points, labels, plt.gca())
    # show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    # # Save the figure
    # output_path =  f"frame_{ann_frame_idx}.png"
    # plt.savefig(output_path, bbox_inches="tight")  # Adjust bbox_inches for saving the complete figure
    # plt.close()  # Close the figure to release memory
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    # with prof:
        # log_memory_usage("Before Video Propagation")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    torch.cuda.synchronize()
        # log_memory_usage("After Video Propagation")
    # prof.step()
    # prof.export_chrome_trace("./log/profiler_trace.json")  # Save raw logs
    # render the segmentation results every few frames
    # vis_frame_stride = 30
    # plt.close("all")
    # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    #         # Save the figure
    #         output_path =  f"out/frame_{out_frame_idx}.png"
    #         plt.savefig(output_path, bbox_inches="tight")  # Adjust bbox_inches for saving the complete figure
    #         plt.close()  # Close the figure to release memory
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
# Export trace for visualization in Chrome Tracing (`chrome://tracing/`)
prof.export_chrome_trace("sam2_profile.json")

# print("max_allocated:", max_allocated)

# After running the model, print memory consumption results
print("\nMemory Usage per Sub-Layer:")
for idx, record in enumerate(memory_records):
    max_bytes = record['max']
    max_kb = max_bytes / 1024
    max_mb = max_kb / 1024
    print(f"Sub-layer {idx} ({record['layer'][0]}: {record['layer'][1]} -> {record['layer'][2]}): Max allocated memory = {max_bytes} bytes ({max_kb:.2f} KB, {max_mb:.2f} MB)")

# Clean up
gc.collect()
torch.cuda.empty_cache()

