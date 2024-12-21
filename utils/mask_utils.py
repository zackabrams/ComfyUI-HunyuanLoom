import torch

def consolidate_masks(masks: torch.Tensor, num_latents: int, method: str) -> torch.Tensor:
    # masks are floats (e.g., 0.0 or 1.0), do not convert to boolean

    n_frames = masks.shape[0]
    frames_per_latent = 4
    expected_frames = (num_latents * frames_per_latent) - 3

    if n_frames == expected_frames:
        # Perfect match
        pass
    elif n_frames == 1 or method == "first_only":
        # If only one frame, repeat it for all latents
        return masks[:1].repeat(num_latents, 1, 1)
    elif n_frames > expected_frames:
        # Truncate if there are more frames than expected
        masks = masks[:expected_frames]
        n_frames = expected_frames
    else:
        # Not enough frames
        raise ValueError(
            f"For {num_latents} latents, expected {expected_frames} frames "
            f"((num_latents * 4) - 3), but got {n_frames} frames"
        )

    # Add padding frames to form a multiple of frames_per_latent
    padding_frame = masks[-1:].clone()
    padded_masks = torch.cat([masks, padding_frame.repeat(3, 1, 1)], dim=0)

    # Reshape into (num_latents, frames_per_latent, H, W)
    grouped_masks = padded_masks.reshape(num_latents, frames_per_latent, *masks.shape[1:])

    if method == "select_first":
        return grouped_masks[:, 0]
    elif method == "select_last":
        return grouped_masks[:, -1]
    elif method == "union":
        return 1 - torch.clamp(grouped_masks.sum(dim=1), max=1)
    else:
        raise ValueError(f"Unknown consolidation method: {method}")
