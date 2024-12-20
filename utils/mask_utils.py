import torch


def consolidate_masks(
    masks: torch.Tensor,
    num_latents: int,
    method: str
) -> torch.Tensor:
    n_frames = masks.shape[0]
    frames_per_latent = 4
    expected_frames = (num_latents * frames_per_latent) - 3

    if n_frames == expected_frames:
        return masks
    elif n_frames == 1 or method == "first_only":
        return masks[:1].repeat(num_latents, 1, 1)
    

    
    if n_frames != expected_frames:
        raise ValueError(
            f"For {num_latents} latents, expected {expected_frames} frames "
            f"((num_latents * 4) - 3, but got {n_frames} frames"
        )
    
    # Add padding frame to make the reshape work
    padding_frame = masks[-1:].clone()  # Use last frame as padding
    padded_masks = torch.cat([masks, padding_frame.repeat(3, 1, 1)], dim=0)
    
    # Now reshape including the padding frame
    grouped_masks = padded_masks.reshape(num_latents, frames_per_latent, *masks.shape[1:])
    
    if method == "select_first":
        return grouped_masks[:, 0]  
    elif method == "select_last":
        return grouped_masks[:, -1]  
    elif method == "union":
        return grouped_masks.any(dim=1)
    else:
        raise ValueError(f"Unknown consolidation method: {method}")