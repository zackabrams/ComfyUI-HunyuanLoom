DEFAULT_ATTN = {
    'double': [i for i in range(0, 100, 1)],#[0,1,2,3,4,5,6,7,9,11,13,15,17,19,21,23,25],
    'single': [i for i in range(0, 100, 2)]
}

class HYFetaEnhanceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "feta_weight": ("FLOAT", {"default": 2, "min": -100.0, "max": 100.0, "step":0.01}),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",)
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "hunyuanloom"
    FUNCTION = "apply"

    def apply(self, model, feta_weight, attn_override=DEFAULT_ATTN):
        model = model.clone()

        model_options = model.model_options.copy()
        transformer_options = model_options['transformer_options'].copy()

        transformer_options['feta_weight'] = feta_weight
        transformer_options['feta_layers'] = attn_override
        model_options['transformer_options'] = transformer_options

        model.model_options = model_options
        return (model,)

