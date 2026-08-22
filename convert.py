import torch
from safetensors.torch import save_file


src="../wavlm-base-plus/pytorch_model.bin"

dst="../wavlm-base-plus/model.safetensors"



state=torch.load(
    src,
    map_location="cpu",
    weights_only=True
)


save_file(
    state,
    dst
)

print("done")