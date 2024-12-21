import torch
from torch import Tensor

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock
from comfy.ldm.modules.attention import optimized_attention

from ..utils.feta_enhance_utils import get_feta_scores


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, skip_rope=False) -> Tensor:
    if not skip_rope:
        q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask)
    return x


class ModifiedDoubleStreamBlock(DoubleStreamBlock):
    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, transformer_options={}):
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        mask_fn = transformer_options.get('patches_replace', {}).get(f'double', {}).get(('mask_fn', self.idx), None) 
        if mask_fn is not None:
            attn_mask = mask_fn(None, transformer_options, txt.shape[1])

        skip_rope = False
        q = torch.cat((img_q, txt_q), dim=2)
        k = torch.cat((img_k, txt_k), dim=2)
        feta_scores = None
        if transformer_options.get('feta_weight', 0) > 0 and self.idx in transformer_options['feta_layers']['double']:
            skip_rope = True
            q, k = apply_rope(q, k, pe)
            txt_size = transformer_options['txt_size']
            img_q = q[:,:,:-txt_size]
            img_k = k[:,:,:-txt_size]
            feta_scores = get_feta_scores(img_q, img_k, transformer_options)

        # run actual attention
        attn = attention(q, k, torch.cat((img_v, txt_v), dim=2),pe=pe, mask=attn_mask, skip_rope=skip_rope)
        
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1]:]

        if feta_scores is not None:
            img_attn *= feta_scores

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt += txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)

        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

        return img, txt


class ModifiedSingleStreamBlock(SingleStreamBlock):
    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, attn_mask=None, transformer_options={}) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)

        mask_fn = transformer_options.get('patches_replace', {}).get(f'single', {}).get(('mask_fn', self.idx), None) 
        if mask_fn is not None:
            attn_mask = mask_fn(q, transformer_options, None)
            
        skip_rope = False
        feta_scores = None
        txt_size = transformer_options['txt_size']
        if transformer_options.get('feta_weight', 0) > 0 and self.idx in transformer_options['feta_layers']['single']:
            skip_rope = True
            q, k = apply_rope(q, k, pe)
            img_q = q[:,:,:-txt_size]
            img_k = k[:,:,:-txt_size]
            feta_scores = get_feta_scores(img_q, img_k, transformer_options)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=attn_mask, skip_rope=skip_rope)

        if feta_scores is not None:
            attn[:,:-txt_size] *= feta_scores

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        x += mod.gate * output
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)

        return x



def inject_blocks(diffusion_model):
    for i, block in enumerate(diffusion_model.double_blocks):
        block.__class__ = ModifiedDoubleStreamBlock
        block.idx = i

    for i, block in enumerate(diffusion_model.single_blocks):
        block.__class__ = ModifiedSingleStreamBlock
        block.idx = i

    return diffusion_model