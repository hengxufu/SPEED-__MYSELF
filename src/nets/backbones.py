from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import os.path as osp
from torch.hub import load_state_dict_from_url
import logging
import re


logger = logging.getLogger(__name__)


class BackboneOutput(object):
    def __init__(self, feat_map=None, feat_vec=None, aux_map=None, feat_maps=None):
        # feat_map: BxCxHxW feature map (for keypoint regression / heatmap tasks)
        # feat_vec: BxC pooled feature vector (for classification/regression heads)
        # aux_map: optional intermediate feature map for legacy CNN feature fusion
        self.feat_map = feat_map
        self.feat_vec = feat_vec
        self.aux_map = aux_map
        self.feat_maps = feat_maps


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True, debug_shapes=False):
        super().__init__()
        from torchvision import models

        try:
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        except Exception:
            m = models.mobilenet_v2(pretrained=pretrained)

        self.features = nn.ModuleList(list(m.features.children())[:-1])
        self.out_channels = 320
        self.aux_channels = 96
        self.debug_shapes = debug_shapes
        self._printed = False

    def forward(self, x):
        aux = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 13:
                aux = x
        if self.debug_shapes and (not self._printed):
            print(f'[backbone:mobilenet_v2] feat_map={tuple(x.shape)} aux_map={tuple(aux.shape) if aux is not None else None}')
            self._printed = True
        return BackboneOutput(feat_map=x, aux_map=aux)


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name,
        pretrained=True,
        weights_path=None,
        img_size=224,
        out_type='map',
        out_index=3,
        out_indices=None,
        debug_shapes=False,
    ):
        super().__init__()
        self.name = name
        self.pretrained = pretrained
        self.weights_path = weights_path
        self.img_size = img_size
        self.out_type = out_type
        self.out_index = int(out_index)
        self.out_indices = tuple(int(x) for x in out_indices) if out_indices is not None and len(out_indices) > 0 else None
        self.debug_shapes = debug_shapes
        self._printed = False
        self.pretrained_loaded = False
        self.pretrained_load_info = None

        try:
            import timm
        except Exception as e:
            raise ImportError('timm is required for transformer backbones. Please install: pip install timm') from e

        self._timm = timm

        self.is_features_only = False
        self.has_cls_token = False
        self.grid_size = None
        self.downsample_to_7 = None
        self._weights_is_url = isinstance(self.weights_path, str) and (
            self.weights_path.startswith('http://') or self.weights_path.startswith('https://')
        )
        if self.weights_path is not None and self.weights_path != '':
            if (not self._weights_is_url) and (not osp.exists(self.weights_path)):
                raise FileNotFoundError(f'Backbone weights not found: {self.weights_path}')
            self.pretrained = False

        if out_type == 'map':
            try:
                out_indices_tuple = (self.out_index,) if self.out_indices is None else tuple(self.out_indices)
                self.model = timm.create_model(
                    name,
                    pretrained=self.pretrained,
                    features_only=True,
                    out_indices=out_indices_tuple,
                    img_size=img_size,
                )
                self.is_features_only = True
                self.out_channels_list = list(self.model.feature_info.channels())
                self.out_channels = self.out_channels_list[-1]
            except Exception as e:
                if bool(self.pretrained):
                    raise RuntimeError(
                        'Failed to load timm backbone with pretrained=True. '
                        'This usually happens when the environment cannot download weights (offline / blocked network). '
                        'Provide local weights with --backbone_pretrained_path <file>, or disable with --no_backbone_pretrained. '
                        f'backbone={name} out_type={out_type} out_index={self.out_index} original_error={repr(e)}'
                    ) from e
                self.model = timm.create_model(name, pretrained=self.pretrained, img_size=img_size)
                self.is_features_only = False
                self.out_channels = getattr(self.model, 'num_features', None) or getattr(self.model, 'embed_dim', None)
                self.has_cls_token = bool(getattr(self.model, 'cls_token', None) is not None)

                patch = getattr(self.model, 'patch_embed', None)
                patch_size = getattr(patch, 'patch_size', None)
                patch_size = patch_size[0] if isinstance(patch_size, (tuple, list)) else patch_size
                if patch_size is None:
                    raise ValueError(f'Backbone {name} does not support feature maps and patch size is unknown.')
                g = img_size // patch_size
                self.grid_size = (g, g)
                if g == 14:
                    # 14x14 (patch16@224) -> 7x7 to match the original KRN head expectations
                    self.downsample_to_7 = nn.AvgPool2d(2, 2)
                elif g == 7:
                    self.downsample_to_7 = nn.Identity()
                else:
                    raise ValueError(f'Unsupported token grid {g}x{g} for backbone={name} img_size={img_size}')
        else:
            try:
                self.model = timm.create_model(name, pretrained=self.pretrained, num_classes=0, global_pool='avg', img_size=img_size)
            except Exception as e:
                if bool(self.pretrained):
                    raise RuntimeError(
                        'Failed to load timm backbone with pretrained=True. '
                        'This usually happens when the environment cannot download weights (offline / blocked network). '
                        'Provide local weights with --backbone_pretrained_path <file>, or disable with --no_backbone_pretrained. '
                        f'backbone={name} out_type={out_type} original_error={repr(e)}'
                    ) from e
                self.model = timm.create_model(name, pretrained=self.pretrained, num_classes=0, global_pool='avg', img_size=img_size)
            self.out_channels = getattr(self.model, 'num_features', None) or getattr(self.model, 'embed_dim', None)

        if self.weights_path is not None and self.weights_path != '':
            if self._weights_is_url:
                state = load_state_dict_from_url(self.weights_path, map_location='cpu', progress=True, check_hash=False)
            else:
                try:
                    if (not osp.exists(self.weights_path)) or (osp.getsize(self.weights_path) <= 0):
                        raise RuntimeError(
                            'Backbone pretrained weights file is missing or empty. '
                            f'path={self.weights_path} size_bytes={osp.getsize(self.weights_path) if osp.exists(self.weights_path) else -1}. '
                            'Provide a valid weights file via --backbone_pretrained_path, or disable pretrained/freeze.'
                        )
                    state = torch.load(self.weights_path, map_location='cpu')
                except Exception as e:
                    size_b = osp.getsize(self.weights_path) if osp.exists(self.weights_path) else -1
                    raise RuntimeError(
                        'Failed to load backbone pretrained weights. '
                        f'path={self.weights_path} size_bytes={size_b} original_error={repr(e)}. '
                        'If size_bytes is 0 or very small, the file is likely truncated/corrupted. '
                        'Re-download/re-copy the .pth and retry.'
                    ) from e
            if isinstance(state, dict):
                if 'state_dict' in state and isinstance(state['state_dict'], dict):
                    state = state['state_dict']
                elif 'model' in state and isinstance(state['model'], dict):
                    state = state['model']
            if isinstance(state, dict) and len(state) > 0:
                keys = list(state.keys())
                for pref in ('backbone.model.', 'model.', 'backbone.'):
                    if sum(1 for k in keys if isinstance(k, str) and k.startswith(pref)) >= max(int(0.8 * len(keys)), 1):
                        state = {k[len(pref):]: v for k, v in state.items() if isinstance(k, str) and k.startswith(pref)}
                        break

                model_keys_now = list(self.model.state_dict().keys())
                if any(isinstance(k, str) and k.startswith('layers_') for k in model_keys_now) and any(
                    isinstance(k, str) and k.startswith('layers.') for k in state.keys()
                ):
                    def _remap_layer_key(k):
                        k = re.sub(
                            r'^layers\.(\d+)\.downsample\.',
                            lambda m: f'layers_{int(m.group(1)) + 1}.downsample.',
                            k,
                        )
                        k = re.sub(r'^layers\.(\d+)\.', r'layers_\1.', k)
                        return k

                    state = {_remap_layer_key(k): v for k, v in state.items()}
            target_model = self.model
            inner = getattr(self.model, 'model', None)
            if isinstance(inner, nn.Module):
                target_model = inner

            model_state = target_model.state_dict()
            state_keys = set(state.keys()) if isinstance(state, dict) else set()
            model_keys = set(model_state.keys())
            matched = len(state_keys & model_keys)
            load_res = target_model.load_state_dict(state, strict=False)
            missing = list(getattr(load_res, 'missing_keys', []))
            unexpected = list(getattr(load_res, 'unexpected_keys', []))

            self.pretrained_loaded = True
            self.pretrained_load_info = {
                'backbone': str(self.name),
                'weights_path': str(self.weights_path),
                'weights_is_url': bool(self._weights_is_url),
                'ckpt_num_keys': int(len(state_keys)),
                'model_num_keys': int(len(model_keys)),
                'matched_num_keys': int(matched),
                'missing_num_keys': int(len(missing)),
                'unexpected_num_keys': int(len(unexpected)),
                'missing_keys_head': missing[:50],
                'unexpected_keys_head': unexpected[:50],
            }

            logger.info(
                '[timm_backbone] loaded weights: backbone=%s path=%s is_url=%s matched=%d/%d missing=%d unexpected=%d',
                self.name,
                self.weights_path,
                self._weights_is_url,
                matched,
                len(model_keys),
                len(missing),
                len(unexpected),
            )

    def forward(self, x):
        if self.out_type == 'vec':
            feat = self.model(x)
            if self.debug_shapes and (not self._printed):
                print(f'[backbone:{self.name}] feat_vec={tuple(feat.shape)}')
                self._printed = True
            return BackboneOutput(feat_vec=feat)

        if self.is_features_only:
            feats = self.model(x)
            if isinstance(feats, (tuple, list)):
                feat_maps = list(feats)
            else:
                feat_maps = [feats]

            exp_ch = getattr(self, 'out_channels_list', None)
            if isinstance(exp_ch, (tuple, list)) and len(exp_ch) == len(feat_maps):
                fixed = []
                for t, c in zip(feat_maps, exp_ch):
                    if t.dim() == 4 and int(c) > 0 and t.shape[1] != int(c) and t.shape[-1] == int(c):
                        fixed.append(t.permute(0, 3, 1, 2).contiguous())
                    else:
                        fixed.append(t)
                feat_maps = fixed
            feat_map = feat_maps[-1]
            if feat_map.dim() == 4:
                c = int(getattr(self, 'out_channels', 0) or 0)
                if c > 0 and feat_map.shape[1] != c and feat_map.shape[-1] == c:
                    feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
            if self.debug_shapes and (not self._printed):
                print(f'[backbone:{self.name}] feat_map={tuple(feat_map.shape)}')
                self._printed = True
            return BackboneOutput(feat_map=feat_map, feat_maps=feat_maps)

        tokens = self.model.forward_features(x)
        if tokens.dim() == 3:
            if self.has_cls_token:
                tokens = tokens[:, 1:, :]
            B, N, C = tokens.shape
            H, W = self.grid_size
            assert N == H * W, f'token count mismatch: N={N} H*W={H*W}'
            feat_map = tokens.transpose(1, 2).contiguous().view(B, C, H, W)
        elif tokens.dim() == 4:
            feat_map = tokens
            c = int(getattr(self, 'out_channels', 0) or 0)
            if c > 0 and feat_map.shape[1] != c and feat_map.shape[-1] == c:
                feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f'Unexpected forward_features output shape: {tuple(tokens.shape)}')

        feat_map = self.downsample_to_7(feat_map) if self.downsample_to_7 is not None else feat_map
        if self.debug_shapes and (not self._printed):
            print(f'[backbone:{self.name}] feat_map={tuple(feat_map.shape)}')
            self._printed = True
        return BackboneOutput(feat_map=feat_map)


def build_backbone(name, pretrained=True, weights_path=None, img_size=224, out_type='map', out_index=3, out_indices=None, debug_shapes=False):
    if name in (None, '', 'mobilenet_v2'):
        return MobileNetV2Backbone(pretrained=pretrained, debug_shapes=debug_shapes)
    return TimmBackbone(
        name=name,
        pretrained=pretrained,
        weights_path=weights_path,
        img_size=img_size,
        out_type=out_type,
        out_index=out_index,
        out_indices=out_indices,
        debug_shapes=debug_shapes,
    )
