# ComfyUI-HunyuanLoom (WIP)
A set of nodes to edit videos using the Hunyuan Video model

## Installation
These nodes are for the Hunyuan Video model. If you haven't installed it yet, you can follow [these examples](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/).


## FlowEdit

This is an implementation of [FlowEdit](https://github.com/fallenshock/FlowEdit).

Use [this workflow](https://github.com/logtd/ComfyUI-HunyuanLoom/blob/main/example_workflows/example_hy_flowedit.json) to get started.

https://github.com/user-attachments/assets/91e38df0-1725-4a6f-9d7d-214baa0758ab

### Settings Overview
<img width="921" alt="flowedit_settings" src="https://github.com/user-attachments/assets/69d7ba35-4d56-441f-b13f-bc44e845b996" />

Note I took some liberties in naming these settings as the original research called them `n_min`, `n_max`, etc.
There are three kinds of steps in FlowEdit:
1. skip steps -- these come at the beginning and are literally skipping steps
2. diff steps -- in these settings they are unnamed and are the remaining steps from `total - skip_steps - drift_steps`
3. drift steps -- these steps come at the end. They can essentially be thought of as normal steps that you'd use in sampling. They are named `drift` because there's nothing controlling them from drifting away from the input video.

#### Skip Steps
The more steps you skip, the less overall power sampling has from changing the source input. Increasing these will significantly impact how much can be changed.

#### Diff Steps (the middle unnamed steps)
These steps do a comparison between the source prompt and target prompt and attempt to only take the smallest difference needed to make the input video go to the target video. Admittedly, these have very little affect on Hunyuan compared to Flux and LTX, but are still needed.

When using these steps in Hunyuan not a lot changes, but it does help the sampling move towards the target without going too far from the source. These steps help seed the direction of the generation so that it incorporates the input and the target prompt change.

#### Drift Steps
These steps come at the end and are essentially normal steps in a sampler. They help remove blur that can happen and "refine" the video. Also in Hunyuan cause the biggest changes to the input video.

## Regional Prompting (Experimental)

> [!IMPORTANT]  
> These nodes are experimental in their current state. They require some tuning to make work right and you should not expect quality results on your first try.

Regional prompting allows you to prompt specific areas of the video over time. Due to this, you can also give different prompts over time (more nodes to make this easier coming).
The root implementation is based on [InstantX's Regional Prompting for Flux](https://github.com/instantX-research/Regional-Prompting-FLUX/tree/main).

Use [this workflow](https://github.com/logtd/ComfyUI-HunyuanLoom/blob/main/example_workflows/example_hy_regional_prompting_t2v.json) to get started.

Example using regional prompts on left and right sides.

https://github.com/user-attachments/assets/37e34c3e-b85b-416a-a0a6-107a238a1783

Example using regional prompts for first half of frames and second half of frames.

https://github.com/user-attachments/assets/f24415b4-5c99-4bef-a5bc-7589b5f8a606


## Acknowledgements

FlowEdit
```
@article{kulikov2024flowedit,
	title = {FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
	author = {Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
	journal = {arXiv preprint arXiv:2412.08629},
	year = {2024}
	}
```

Regional Prompting
```
@article{chen2024training,
  title={Training-free Regional Prompting for Diffusion Transformers},
  author={Chen, Anthony and Xu, Jianjin and Zheng, Wenzhao and Dai, Gaole and Wang, Yida and Zhang, Renrui and Wang, Haofan and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2411.02395},
  year={2024}
}
```
