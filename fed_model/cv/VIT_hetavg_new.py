'''
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

'''

import pdb
import torch
import torch.nn as nn
import math

# __all__ = ['ResNet']
from tkinter import X
import torch.nn.functional as F

# import torchvision.models.vision_transformer as vit
from torchvision.models.vision_transformer import _vision_transformer
from typing import Any



from typing import Any, Callable, List, NamedTuple, Optional
from torch.hub import load_state_dict_from_url
from timm.models.vision_transformer import vit_small_patch16_224, vit_tiny_patch16_224
from timm.models.vision_transformer import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from timm.models.vision_transformer import PatchEmbed, Block, partial, DropPath, trunc_normal_, named_apply, get_init_weights_vit
from timm.models.vision_transformer import init_weights_vit_timm, _load_weights, checkpoint_seq

model_urls = {
    "vit_b_16": "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
    "vit_b_32": "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
    "vit_l_16": "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
    "vit_l_32": "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
}




class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    
    def forward_moon(self, x):
        x = self.forward_features(x)
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        encoded_features = x
        x = self.fc_norm(encoded_features)
        # x = self.forward_head(encoded_features)
        return encoded_features, self.head(x)




def _vision_transformer(
    arch: str,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    pretrained: bool,
    progress: bool,
    image_size: int = 224,
    **kwargs: Any,
) -> VisionTransformer:
    # image_size = kwargs["image_size"]

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if pretrained:
        if arch not in model_urls:
            raise ValueError(f"No checkpoint is available for model type '{arch}'!")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

    return model


def vit_b_16_10(pretrained: bool = True, progress: bool = True, **kwargs):
    """
    Constructs a vit_b_16 (10 layers) architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_model = _vision_transformer(
                            arch="vit_b_16",
                            patch_size=16,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            num_classes=1000,
                            pretrained=pretrained,
                            progress=progress,
                            **kwargs,
                        )
    
    vit_b_16_10 = _vision_transformer(
                      arch="vit_b_16_10",
                      patch_size=4,
                      num_layers=10,
                      num_heads=12,
                      hidden_dim=768,
                      mlp_dim=3072,
                      num_classes=10,
                      image_size=32,
                      pretrained=False,
                      progress=progress,
                      **kwargs,
                  )
    
    pretrained_parameters = pretrained_model.cpu().state_dict()
    new_vit_parameters = vit_b_16_10.cpu().state_dict()
    for key, val in pretrained_parameters.items():
        # ignore these three types of layers
        if 'pos_embedding' in key:
            continue
        elif 'head' in key:
            continue
        elif 'conv_proj' in key:
            continue
        elif key in new_vit_parameters:
            new_vit_parameters[key] = val
    
    vit_b_16_10.load_state_dict(new_vit_parameters)
    
    return vit_b_16_10



def vit_b_16_12(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """
    Constructs a vit_b_16 (12 layers) architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_model = _vision_transformer(
                            arch="vit_b_16",
                            patch_size=16,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            num_classes=1000,
                            pretrained=pretrained,
                            progress=progress,
                            **kwargs,
                        )
    
    vit_b_16_12 = _vision_transformer(
                      arch="vit_b_16_12",
                      patch_size=4,
                      num_layers=12,
                      num_heads=12,
                      hidden_dim=768,
                      mlp_dim=3072,
                      num_classes=10,
                      image_size=32,
                      pretrained=False,
                      progress=progress,
                      **kwargs,
                  )
    
    pretrained_parameters = pretrained_model.cpu().state_dict()
    new_vit_parameters = vit_b_16_12.cpu().state_dict()
    for key, val in pretrained_parameters.items():
        # ignore these three types of layers
        if 'pos_embedding' in key:
            continue
        elif 'head' in key:
            continue
        elif 'conv_proj' in key:
            continue
        elif key in new_vit_parameters:
            new_vit_parameters[key] = val
    
    vit_b_16_12.load_state_dict(new_vit_parameters)
    
    return vit_b_16_12




def vit_b_16_8(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """
    Constructs a vit_b_16 (8 layers) architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_model = _vision_transformer(
                            arch="vit_b_16",
                            patch_size=16,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            num_classes=1000,
                            pretrained=pretrained,
                            progress=progress,
                            **kwargs,
                        )
    
    vit_b_16_8 = _vision_transformer(
                      arch="vit_b_16_8",
                      patch_size=4,
                      num_layers=8,
                      num_heads=12,
                      hidden_dim=768,
                      mlp_dim=3072,
                      num_classes=10,
                      image_size=32,
                      pretrained=False,
                      progress=progress,
                      **kwargs,
                  )
    
    pretrained_parameters = pretrained_model.cpu().state_dict()
    new_vit_parameters = vit_b_16_8.cpu().state_dict()
    for key, val in pretrained_parameters.items():
        # ignore these three types of layers
        if 'pos_embedding' in key:
            continue
        elif 'head' in key:
            continue
        elif 'conv_proj' in key:
            continue
        elif key in new_vit_parameters:
            new_vit_parameters[key] = val
    
    vit_b_16_8.load_state_dict(new_vit_parameters)
    
    return vit_b_16_8

def vit_b_16_6(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """
    Constructs a vit_b_16 (10 layers) architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_model = _vision_transformer(
                            arch="vit_b_16",
                            patch_size=16,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            num_classes=1000,
                            pretrained=pretrained,
                            progress=progress,
                            **kwargs,
                        )
    
    vit_b_16_6 = _vision_transformer(
                      arch="vit_b_16_6",
                      patch_size=4,
                      num_layers=6,
                      num_heads=12,
                      hidden_dim=768,
                      mlp_dim=3072,
                      num_classes=10,
                      image_size=32,
                      pretrained=False,
                      progress=progress,
                      **kwargs,
                  )
    
    pretrained_parameters = pretrained_model.cpu().state_dict()
    new_vit_parameters = vit_b_16_6.cpu().state_dict()
    for key, val in pretrained_parameters.items():
        # ignore these three types of layers
        if 'pos_embedding' in key:
            continue
        elif 'head' in key:
            continue
        elif 'conv_proj' in key:
            continue
        elif key in new_vit_parameters:
            new_vit_parameters[key] = val
    
    vit_b_16_6.load_state_dict(new_vit_parameters)
    
    return vit_b_16_6



def vit_b_16_4(pretrained: bool = True, progress: bool = True, **kwargs: Any):
    """
    Constructs a vit_b_16 (10 layers) architecture from
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained_model = _vision_transformer(
                            arch="vit_b_16",
                            patch_size=16,
                            num_layers=12,
                            num_heads=12,
                            hidden_dim=768,
                            mlp_dim=3072,
                            num_classes=1000,
                            pretrained=pretrained,
                            progress=progress,
                            **kwargs,
                        )
    
    vit_b_16_4 = _vision_transformer(
                      arch="vit_b_16_4",
                      patch_size=4,
                      num_layers=4,
                      num_heads=12,
                      hidden_dim=768,
                      mlp_dim=3072,
                      num_classes=10,
                      image_size=32,
                      pretrained=False,
                      progress=progress,
                      **kwargs,
                  )
    
    pretrained_parameters = pretrained_model.cpu().state_dict()
    new_vit_parameters = vit_b_16_4.cpu().state_dict()
    for key, val in pretrained_parameters.items():
        # ignore these three types of layers
        if 'pos_embedding' in key:
            continue
        elif 'head' in key:
            continue
        elif 'conv_proj' in key:
            continue
        elif key in new_vit_parameters:
            new_vit_parameters[key] = val
    
    vit_b_16_4.load_state_dict(new_vit_parameters)
    
    return vit_b_16_4


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model



def vit_small_patch16_224_customLayers(layer_num, **kwargs):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=layer_num, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=False, **model_kwargs)
    return model

def vit_small_16_Layers(layer_num, pretrained=True):
    new_vit_small_model = vit_small_patch16_224_customLayers(layer_num=layer_num)
    new_vit_small_model_params = new_vit_small_model.state_dict()

    if pretrained:
        pretrained_model = vit_small_patch16_224(pretrained=pretrained)
        pretrained_model_params = pretrained_model.state_dict()
        for key, val in pretrained_model_params.items():
            if 'head' in key:
                continue
            elif key in new_vit_small_model_params:
                new_vit_small_model_params[key] = val

        new_vit_small_model.load_state_dict(new_vit_small_model_params)
    new_vit_small_model.head = nn.Linear(new_vit_small_model.head.weight.shape[1], 10)
    
    return new_vit_small_model
