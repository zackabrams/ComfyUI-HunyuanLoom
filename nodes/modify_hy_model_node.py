from ..modules.hy_layers import inject_blocks
from ..modules.hy_model import inject_model


class ConfigureModifiedHYNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "hunyuanloom"
    FUNCTION = "apply"

    def apply(self, model):
        inject_model(model.model.diffusion_model)
        inject_blocks(model.model.diffusion_model)
        return (model,)

