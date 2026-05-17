from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from src.utils.heatmaps import keypoint_rmse_pixels, inside_image_percentage
from src.utils.heatmaps import per_keypoint_rmse_pixels, heatmap_peaks, heatmap_entropy, collapsed_keypoint_distance, keypoint_spread_min_px
from src.utils.heatmaps import heatmap_to_keypoints_dsnt
from src.utils.heatmap_pipeline import build_gt_heatmaps, heatmap_to_keypoints, heatmap_loss


class TransformerHeatmapPoseNet(nn.Module):
    def __init__(self, num_keypoints, cfg=None):
        super().__init__()
        self.nK = int(num_keypoints)
        self.cfg = cfg

        backbone_name = getattr(cfg, 'backbone', 'swin_tiny_patch4_window7_224') if cfg is not None else 'swin_tiny_patch4_window7_224'
        backbone_pretrained = getattr(cfg, 'backbone_pretrained', True) if cfg is not None else True
        backbone_pretrained_path = getattr(cfg, 'backbone_pretrained_path', None) if cfg is not None else None
        input_shape = getattr(cfg, 'input_shape', (224, 224)) if cfg is not None else (224, 224)
        img_size = int(input_shape[0])

        self.debug_shapes = bool(getattr(cfg, 'debug_shapes', False)) if cfg is not None else False
        self._printed = False

        self.input_normalize = getattr(cfg, 'input_normalize', 'imagenet') if cfg is not None else 'imagenet'
        self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone_name = backbone_name
        self.use_backbone_fpn = bool(getattr(cfg, 'backbone_fpn', False)) if cfg is not None else False
        out_indices = getattr(cfg, 'backbone_out_indices', []) if cfg is not None else []
        out_indices = list(out_indices) if isinstance(out_indices, (tuple, list)) else []
        self.backbone = build_backbone(
            name=backbone_name,
            pretrained=backbone_pretrained,
            weights_path=backbone_pretrained_path,
            img_size=img_size,
            out_type='map',
            out_index=int(getattr(cfg, 'backbone_out_index', 2)) if cfg is not None else 2,
            out_indices=out_indices if self.use_backbone_fpn else None,
            debug_shapes=self.debug_shapes,
        )

        if self.use_backbone_fpn and hasattr(self.backbone, 'out_channels_list') and isinstance(self.backbone.out_channels_list, (list, tuple)):
            fpn_chs = list(self.backbone.out_channels_list)
            self.fpn_lateral = nn.ModuleList([nn.Conv2d(int(c), 256, 1, bias=False) for c in fpn_chs])
            self.fpn_out = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            in_ch = 256
        else:
            self.fpn_lateral = None
            self.fpn_out = None
            in_ch = int(getattr(self.backbone, 'out_channels', 0) or 0)
        assert in_ch > 0, f'Invalid backbone out_channels for {backbone_name}: {in_ch}'

        self.neck = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        hm_size = getattr(cfg, 'heatmap_size', (56, 56)) if cfg is not None else (56, 56)
        self.heatmap_size = (int(hm_size[0]), int(hm_size[1]))
        self.heatmap_sigma = float(getattr(cfg, 'heatmap_sigma', 2.0)) if cfg is not None else 2.0
        self.heatmap_decode = getattr(cfg, 'heatmap_decode', 'argmax') if cfg is not None else 'argmax'
        self.heatmap_beta = float(getattr(cfg, 'heatmap_beta', 100.0)) if cfg is not None else 100.0
        self.heatmap_activation = getattr(cfg, 'heatmap_activation', 'none') if cfg is not None else 'none'
        self.heatmap_loss_type = getattr(cfg, 'heatmap_loss', 'bce') if cfg is not None else 'bce'
        self.coord_aux_weight = float(getattr(cfg, 'coord_aux_weight', 0.1)) if cfg is not None else 0.1
        self.heatmap_pos_thr = float(getattr(cfg, 'heatmap_pos_thr', 0.1)) if cfg is not None else 0.1
        self.heatmap_neg_weight = float(getattr(cfg, 'heatmap_neg_weight', 0.01)) if cfg is not None else 0.01

        hard = getattr(cfg, 'heatmap_hard_kpts', '3,4,5,6,10') if cfg is not None else '3,4,5,6,10'
        hard_idx = []
        if isinstance(hard, str) and hard.strip() != '':
            try:
                hard_idx = [int(x) for x in hard.replace(' ', '').split(',') if x != '']
            except Exception:
                hard_idx = []
        hw = float(getattr(cfg, 'heatmap_hard_kpt_weight', 1.5)) if cfg is not None else 1.5
        ow = float(getattr(cfg, 'heatmap_other_kpt_weight', 1.0)) if cfg is not None else 1.0
        w = torch.full((self.nK,), float(ow), dtype=torch.float32)
        for i in hard_idx:
            if 0 <= int(i) < self.nK:
                w[int(i)] = float(hw)
        self.register_buffer('kpt_loss_weights', w.view(1, self.nK), persistent=False)

        # Keep the decoder spatially agnostic: always resize to heatmap_size then refine with convs.
        # This avoids hard-coding assumptions about backbone feature map resolution (7x7 vs 14x14, etc.).
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, self.nK, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.bias)

        self.loss = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        self.last_pred_heatmaps = None
        self.last_pred_heatmaps_logits = None
        self.last_gt_heatmaps = None
        self.last_valid_mask = None

    def set_backbone_trainable(self, trainable=True):
        for p in self.backbone.parameters():
            p.requires_grad = bool(trainable)

    def _maybe_normalize(self, x):
        if self.input_normalize == 'imagenet':
            return (x - self._mean) / self._std
        return x

    def forward(self, x, y=None):
        x = self._maybe_normalize(x)

        out = self.backbone(x)
        if self.fpn_lateral is not None and getattr(out, 'feat_maps', None) is not None:
            feats = list(out.feat_maps)
            lat = [m(f) for m, f in zip(self.fpn_lateral, feats)]
            p = lat[-1]
            for i in range(len(lat) - 2, -1, -1):
                p = F.interpolate(p, size=lat[i].shape[-2:], mode='bilinear', align_corners=False) + lat[i]
            feat = self.fpn_out(p)
        else:
            feat = out.feat_map
        feat = self.neck(feat)

        feat_up = F.interpolate(feat, size=self.heatmap_size, mode='bilinear', align_corners=False)
        dec = self.decoder(feat_up)
        logits = self.head(dec)
        self.last_pred_heatmaps_logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        if str(self.heatmap_loss_type).lower() in ('heatmap_ce_coord_aux', 'ce_coord_aux'):
            B, K, Hh, Wh = self.last_pred_heatmaps_logits.shape
            pred_hm = torch.softmax(self.last_pred_heatmaps_logits.view(B, K, Hh * Wh), dim=-1).view(B, K, Hh, Wh)
        else:
            pred_hm = torch.sigmoid(self.last_pred_heatmaps_logits)
        pred_hm = torch.nan_to_num(pred_hm, nan=0.0, posinf=0.0, neginf=0.0)

        self.last_pred_heatmaps = pred_hm

        if self.debug_shapes and (not self._printed):
            print(f'[thpn] backbone={self.backbone_name} feat={tuple(feat.shape)} heatmap={tuple(pred_hm.shape)}')
            self._printed = True

        if (y is not None) and (not getattr(self, '_printed_train', False)):
            print(f'[thpn] model={self.__class__.__name__} head=heatmap loss={self.heatmap_loss_type} pred_hm={tuple(pred_hm.shape)}')
            self._printed_train = True

        if y is None:
            if str(self.heatmap_loss_type).lower() in ('heatmap_ce_coord_aux', 'ce_coord_aux') and str(self.heatmap_decode).lower() == 'softargmax':
                keypts_norm, _ = heatmap_to_keypoints_dsnt(pred_hm)
            else:
                keypts_norm, _ = heatmap_to_keypoints(pred_hm, decode=self.heatmap_decode, beta=self.heatmap_beta)
            xc = keypts_norm[:, 0, :].detach().cpu()
            yc = keypts_norm[:, 1, :].detach().cpu()
            return xc, yc

        gt_hm, valid = build_gt_heatmaps(y, self.heatmap_size, self.heatmap_sigma)
        self.last_gt_heatmaps = gt_hm
        self.last_valid_mask = valid

        loss, pos_loss_m, neg_loss_m = heatmap_loss(
            self.last_pred_heatmaps_logits,
            gt_hm,
            valid,
            loss_type=self.heatmap_loss_type,
            pos_thr=self.heatmap_pos_thr,
            neg_weight=self.heatmap_neg_weight,
            coord_aux_weight=self.coord_aux_weight,
            channel_weights=self.kpt_loss_weights,
        )
        if (logits.ndim != 4) or (logits.shape[1] != self.nK):
            raise RuntimeError('Heatmap head must output [B, K, Hh, Wh] logits')
        if gt_hm.ndim != 4 or gt_hm.shape[1] != self.nK:
            raise RuntimeError('Heatmap target must be [B, K, Hh, Wh]')
        if logits.shape[-2:] != gt_hm.shape[-2:]:
            raise RuntimeError('Heatmap pred/gt spatial shape mismatch')

        pred_kp, _ = heatmap_to_keypoints(pred_hm.detach(), decode=self.heatmap_decode, beta=self.heatmap_beta)
        rmse = keypoint_rmse_pixels(pred_kp, y.detach(), image_size=getattr(self.cfg, 'input_shape', (224, 224)), valid_mask=valid)
        inside_pct = inside_image_percentage(pred_kp, valid)
        valid_pct = valid.to(dtype=torch.float32).mean() * 100.0
        per_kp_rmse = per_keypoint_rmse_pixels(pred_kp, y.detach(), image_size=getattr(self.cfg, 'input_shape', (224, 224)), valid_mask=valid)
        per_kp_rmse_mean = per_kp_rmse.mean()
        per_kp_rmse_max = per_kp_rmse.max()

        peaks = heatmap_peaks(pred_hm.detach())
        ent = heatmap_entropy(self.last_pred_heatmaps_logits.detach(), beta=1.0)
        ent_beta = heatmap_entropy(self.last_pred_heatmaps_logits.detach(), beta=float(self.heatmap_beta))
        m = valid.to(dtype=peaks.dtype)
        denom = torch.clamp(m.sum(dim=1), min=1.0)
        peak_mean = (peaks * m).sum(dim=1) / denom
        peak_std = torch.sqrt(torch.clamp(((peaks - peak_mean.unsqueeze(1)) ** 2) * m, min=0.0).sum(dim=1) / denom)
        ent_mean = (ent * m).sum(dim=1) / denom
        ent_std = torch.sqrt(torch.clamp(((ent - ent_mean.unsqueeze(1)) ** 2) * m, min=0.0).sum(dim=1) / denom)
        ent_beta_mean = (ent_beta * m).sum(dim=1) / denom
        ent_beta_std = torch.sqrt(torch.clamp(((ent_beta - ent_beta_mean.unsqueeze(1)) ** 2) * m, min=0.0).sum(dim=1) / denom)

        coll_dist = collapsed_keypoint_distance(pred_kp, image_size=getattr(self.cfg, 'input_shape', (224, 224)), valid_mask=valid)
        spread_min = keypoint_spread_min_px(pred_kp, image_size=getattr(self.cfg, 'input_shape', (224, 224)), valid_mask=valid)

        summary = {
            'loss_x': float(loss.detach().cpu()),
            'loss_y': float(loss.detach().cpu()),
            'loss_hm': float(loss.detach().cpu()),
            'pos_loss': float(pos_loss_m.detach().cpu()),
            'neg_loss': float(neg_loss_m.detach().cpu()),
            'rmse_px': float(rmse.detach().cpu()),
            'inside_pct': float(inside_pct.detach().cpu()),
            'valid_kpt_pct': float(valid_pct.detach().cpu()),
            'pred_logits_min': float(self.last_pred_heatmaps_logits.min().detach().cpu()),
            'pred_logits_max': float(self.last_pred_heatmaps_logits.max().detach().cpu()),
            'pred_logits_mean': float(self.last_pred_heatmaps_logits.mean().detach().cpu()),
            'pred_logits_std': float(self.last_pred_heatmaps_logits.std(unbiased=False).detach().cpu()),
            'pred_hm_min': float(pred_hm.min().detach().cpu()),
            'pred_hm_max': float(pred_hm.max().detach().cpu()),
            'pred_hm_mean': float(pred_hm.mean().detach().cpu()),
            'pred_hm_std': float(pred_hm.std(unbiased=False).detach().cpu()),
            'gt_hm_min': float(gt_hm.min().detach().cpu()),
            'gt_hm_max': float(gt_hm.max().detach().cpu()),
            'gt_hm_mean': float(gt_hm.mean().detach().cpu()),
            'gt_hm_std': float(gt_hm.std(unbiased=False).detach().cpu()),
            'heatmap_peak_mean': float(peak_mean.mean().detach().cpu()),
            'heatmap_peak_std': float(peak_std.mean().detach().cpu()),
            'heatmap_entropy': float(ent_mean.mean().detach().cpu()),
            'heatmap_entropy_std': float(ent_std.mean().detach().cpu()),
            'heatmap_entropy_beta': float(ent_beta_mean.mean().detach().cpu()),
            'heatmap_entropy_beta_std': float(ent_beta_std.mean().detach().cpu()),
            'per_kpt_rmse_px_mean': float(per_kp_rmse_mean.detach().cpu()),
            'per_kpt_rmse_px_max': float(per_kp_rmse_max.detach().cpu()),
            'collapsed_keypoint_distance': float(coll_dist.mean().detach().cpu()),
            'collapsed_keypoint_spread_min_px': float(spread_min.mean().detach().cpu()),
        }
        return loss, summary
