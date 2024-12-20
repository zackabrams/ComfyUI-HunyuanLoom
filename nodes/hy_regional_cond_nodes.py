import torch

import comfy.model_sampling

from ..utils.mask_utils import consolidate_masks


DEFAULT_REGIONAL_ATTN = {
    'double': [i for i in range(1, 100, 1)],#[0,1,2,3,4,5,6,7,9,11,13,15,17,19,21,23,25],
    'single': [i for i in range(1, 100, 2)]
}


class RegionalMask(torch.nn.Module):
    def __init__(self, mask: torch.Tensor, start_percent: float, end_percent: float) -> None:
        super().__init__()
        self.register_buffer('mask', mask)
        self.start_percent = start_percent
        self.end_percent = end_percent

    def __call__(self, q, transformer_options, *args, **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent:
            return self.mask
        
        return None
    

class RegionalConditioning(torch.nn.Module):
    def __init__(self, region_cond: torch.Tensor, start_percent: float, end_percent: float, main_cond_strength: float, always_included: bool) -> None:
        super().__init__()
        self.register_buffer('region_cond', region_cond)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.main_cond_strength = main_cond_strength
        self.always_included = always_included

    def __call__(self, context, transformer_options, *args,  **kwargs):
        if self.start_percent <= 1 - transformer_options['sigmas'][0] < self.end_percent or self.always_included:
            return torch.cat([context*self.main_cond_strength, self.region_cond.to(context.dtype)], dim=1)
        return context


class HYCreateRegionalCondNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "cond": ("CONDITIONING",),
            "mask": ("MASK",),
            "cond_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01}),
            "mask_consolidation": (["first_only", "select_first", "select_last", "union"], { "tooltip": "How to choose the masking when they are compressed for the latents temporally."}),
        }, "optional": {
            "prev_regions": ("REGION_COND",),
        }}

    RETURN_TYPES = ("REGION_COND",)
    FUNCTION = "create"

    CATEGORY = "hunyuanloom"

    def create(self, cond, mask, cond_strength, mask_consolidation, prev_regions=[]):
        prev_regions = [*prev_regions]
        prev_regions.append({
            'mask': mask,
            'cond': cond[0][0] * cond_strength,
            'mask_consolidation': mask_consolidation,
        })

        return (prev_regions,)


class HYApplyRegionalCondsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "cond": ("CONDITIONING",),
            "region_conds": ("REGION_COND",),
            "latent": ("LATENT",),
            "start_percent": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
            "main_cond_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01}),
            "always_included": ("BOOLEAN", { "default": False, "tooltip": "Whether to keep the prompts always but allow them to affect the entire video (unmasked) outside of end/start percents." }),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",)
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "hunyuanloom"

    def patch(self, model, cond, region_conds, latent, start_percent, end_percent, main_cond_strength, always_included, attn_override=DEFAULT_REGIONAL_ATTN):
        model = model.clone()

        latent = latent['samples']
        b, c, f, h, w = latent.shape
        h //=2
        w //=2

        img_len = h*w*f

        main_cond = cond[0][0]
        main_cond_size = main_cond.shape[1]

        regional_conditioning = torch.cat([region_cond['cond'] for region_cond in region_conds], dim=1)
        text_len = main_cond_size + regional_conditioning.shape[1]

        regional_mask = torch.zeros((text_len + img_len, text_len + img_len), dtype=torch.bool)

        self_attend_masks = torch.zeros((img_len, img_len), dtype=torch.bool)
        union_masks = torch.zeros((img_len, img_len), dtype=torch.bool)

        main_mask = torch.zeros((f, h, w), dtype=torch.float16)
        if main_cond_strength == 0:
            main_mask = torch.ones((f, h, w), dtype=torch.float16)

        region_conds = [
            { 
                'mask': main_mask,
                'cond': torch.ones((1, main_cond_size, 4096), dtype=torch.float16),
                'mask_consolidation': 'first_only',
            },
            *region_conds
        ]

        current_seq_len = 0
        for region_cond_dict in region_conds:
            region_cond = region_cond_dict['cond']
            region_mask = region_cond_dict['mask']
            region_mask = consolidate_masks(region_mask, f, region_cond_dict['mask_consolidation'])
            region_mask = 1 - region_mask
            region_mask = torch.nn.functional.interpolate(region_mask[None, :, :, :], (h, w), mode='nearest-exact').flatten().unsqueeze(1).repeat(1, region_cond.size(1))
            next_seq_len = current_seq_len + region_cond.shape[1]

            # txt attends to itself
            regional_mask[current_seq_len:next_seq_len, current_seq_len:next_seq_len] = True

            # txt attends to corresponding regional img
            regional_mask[current_seq_len:next_seq_len, text_len:] = region_mask.transpose(-1, -2)

            # regional img attends to corresponding txt
            regional_mask[text_len:, current_seq_len:next_seq_len] = region_mask

            # regional img attends to corresponding regional img
            img_size_masks = region_mask[:, :1].repeat(1, img_len)
            img_size_masks_transpose = img_size_masks.transpose(-1, -2)
            self_attend_masks = torch.logical_or(self_attend_masks, 
                                                    torch.logical_and(img_size_masks, img_size_masks_transpose))

            # update union
            union_masks = torch.logical_or(union_masks, 
                                            torch.logical_or(img_size_masks, img_size_masks_transpose))
            
            current_seq_len = next_seq_len

        background_masks = torch.logical_not(union_masks)
        background_and_self_attend_masks = torch.logical_or(background_masks, self_attend_masks)
        regional_mask[text_len:, text_len:] = background_and_self_attend_masks

        # Split rows into text and image
        text_rows, img_rows = regional_mask.split([text_len, img_len], dim=0)

        # Now split each part into text and image columns
        text_rows_text_cols, text_rows_img_cols = text_rows.split([text_len, img_len], dim=1)
        img_rows_text_cols, img_rows_img_cols = img_rows.split([text_len, img_len], dim=1)

        double_regional_mask = torch.cat([
            torch.cat([img_rows_img_cols, img_rows_text_cols], dim=1),
            torch.cat([text_rows_img_cols, text_rows_text_cols], dim=1)
        ], dim=0)

        # Patch
        double_regional_mask = RegionalMask(double_regional_mask, start_percent, end_percent)
        single_regional_mask = RegionalMask(regional_mask, start_percent, end_percent)
        regional_conditioning = RegionalConditioning(regional_conditioning, start_percent, end_percent, main_cond_strength, always_included)


        model.set_model_patch(regional_conditioning, 'regional_conditioning')

        for block_idx in attn_override['double']:
            model.set_model_patch_replace(double_regional_mask, f"double", "mask_fn", int(block_idx))

        for block_idx in attn_override['single']:
            model.set_model_patch_replace(single_regional_mask, f"single", "mask_fn", int(block_idx))

        return (model,)
