import torch
import gc


from diffusers.utils.torch_utils import randn_tensor
import comfy.model_management as mm
from ..utils.rope_utils import get_rotary_pos_embed
from ..utils.latent_preview import prepare_callback


class HyVideoFlowEditSamplerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HYVIDEOMODEL",),
                "source_embeds": ("HYVIDEMBEDS", ),
                "target_embeds": ("HYVIDEMBEDS", ),
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"} ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "skip_steps": ("INT", {"default": 4, "min": 0}),
                "drift_steps": ("INT", {"default": 0, "min": 0}),
                "source_guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "target_guidance_scale": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "hunyuanloom"

    def process(self, 
                model, 
                source_embeds, 
                target_embeds,
                flow_shift, 
                steps, 
                skip_steps,
                drift_steps,
                source_guidance_scale, 
                target_guidance_scale,
                seed, 
                samples, 
                force_offload):
        model = model.model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = model["dtype"]
        transformer = model["pipe"].transformer
        pipeline = model["pipe"]
        
        generator = torch.Generator(device=torch.device("cpu")).manual_seed(seed)

        latents = samples["samples"] if samples is not None else None
        batch_size, num_channels_latents, latent_num_frames, latent_height, latent_width = latents.shape
        height = latent_height * pipeline.vae_scale_factor
        width = latent_width * pipeline.vae_scale_factor
        num_frames = (latent_num_frames - 1) * 4 + 1

        if width <= 0 or height <= 0 or num_frames <= 0:
            raise ValueError(
                f"`height` and `width` and `video_length` must be positive integers, got height={height}, width={width}, video_length={num_frames}"
            )
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"`video_length-1` must be a multiple of 4, got {num_frames}"
            )

        freqs_cos, freqs_sin = get_rotary_pos_embed(transformer, num_frames, height, width)

        pipeline.scheduler.shift = flow_shift
  
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                #print(name, param.data.device)
                if "single" not in name and "double" not in name:
                    param.data = param.data.to(device)
                
            transformer.block_swap(
                model["block_swap_args"]["double_blocks_to_swap"] - 1 , 
                model["block_swap_args"]["single_blocks_to_swap"] - 1,
                offload_txt_in = model["block_swap_args"]["offload_txt_in"],
                offload_img_in = model["block_swap_args"]["offload_img_in"],
            )
        elif model["manual_offloading"]:
            transformer.to(device)

        mm.soft_empty_cache()
        gc.collect()
        
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        pipeline.scheduler.set_timesteps(steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        timesteps = torch.cat([timesteps, torch.tensor([0]).to(timesteps.device)]).to(timesteps.device)

        latent_video_length = (num_frames - 1) // 4 + 1

        # 5. Prepare latent variables
        num_channels_latents = transformer.config.in_channels
        
        latents = latents.to(device)

        shape = (
            1,
            num_channels_latents,
            latent_video_length,
            int(height) // pipeline.vae_scale_factor,
            int(width) // pipeline.vae_scale_factor,
        )
        noise = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        
        frames_needed = noise.shape[1]
        current_frames = latents.shape[1]
        
        if frames_needed > current_frames:
            repeat_factor = frames_needed - current_frames
            additional_frame = torch.randn((latents.size(0), repeat_factor, latents.size(2), latents.size(3), latents.size(4)), dtype=latents.dtype, device=latents.device)
            latents = torch.cat((additional_frame, latents), dim=1)
            self.additional_frames = repeat_factor
        elif frames_needed < current_frames:
            latents = latents[:, :frames_needed, :, :, :]
            

        # 7. Denoising loop
        self._num_timesteps = len(timesteps)

        callback = prepare_callback(transformer, steps)

        x_init = latents.clone()

        from comfy.utils import ProgressBar
        from tqdm import tqdm
        comfy_pbar = ProgressBar(len(timesteps))
        N = len(timesteps)

        x_tgt = latents

        with tqdm(total=len(timesteps)) as progress_bar:
            for idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
                if idx < skip_steps:
                    continue
                t_expand = t.repeat(x_init.shape[0])
                source_guidance_expand = (
                    torch.tensor(
                        [source_guidance_scale] * x_init.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(pipeline.base_dtype)
                    * 1000.0
                    if source_guidance_scale is not None
                    else None
                )

                target_guidance_expand = (
                    torch.tensor(
                        [target_guidance_scale] * x_init.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(pipeline.base_dtype)
                    * 1000.0
                    if target_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=pipeline.base_dtype, enabled=True
                ):
                    
                    noise = torch.randn(x_init.shape, generator=generator).to(x_init.device)

                    sigma = t / 1000.0
                    sigma_prev = t_prev / 1000.0

                    zt_src = (1-sigma)*x_init + sigma*noise

                    if idx < N-drift_steps:
                        zt_tgt = x_tgt + zt_src - x_init
                        vt_src = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                            zt_src,  # [2, 16, 33, 24, 42]
                            t_expand,  # [2]
                            text_states=source_embeds["prompt_embeds"],  # [2, 256, 4096]
                            text_mask=source_embeds["attention_mask"],  # [2, 256]
                            text_states_2=source_embeds["prompt_embeds_2"],  # [2, 768]
                            freqs_cos=freqs_cos,  # [seqlen, head_dim]
                            freqs_sin=freqs_sin,  # [seqlen, head_dim]
                            guidance=source_guidance_expand,
                            stg_block_idx=-1,
                            stg_mode=None,
                            return_dict=True,
                        )["x"]
                    else:
                        if idx == N - drift_steps:
                            x_tgt = x_tgt + zt_src - x_init
                        zt_tgt = x_tgt
                        vt_src = 0

                    vt_tgt = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        zt_tgt,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=target_embeds["prompt_embeds"],  # [2, 256, 4096]
                        text_mask=target_embeds["attention_mask"],  # [2, 256]
                        text_states_2=target_embeds["prompt_embeds_2"],  # [2, 768]
                        freqs_cos=freqs_cos,  # [seqlen, head_dim]
                        freqs_sin=freqs_sin,  # [seqlen, head_dim]
                        guidance=target_guidance_expand,
                        stg_block_idx=-1,
                        stg_mode=None,
                        return_dict=True,
                    )["x"]

                    v_delta = vt_tgt - vt_src
                
                x_tgt = x_tgt.to(torch.float32)
                v_delta = v_delta.to(torch.float32)


                x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                x_tgt = x_tgt.to(torch.bfloat16)

                # compute the previous noisy sample x_t -> x_t-1
                #latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                
                progress_bar.update()
                if callback is not None:
                        callback(idx, latents.detach()[-1].permute(1,0,2,3), None, steps)
                else:
                    comfy_pbar.update(1)
                  
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        return ({
            "samples": x_tgt
            },)


