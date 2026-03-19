"""Backbone SegFormer para classificação e segmentação (multi-task)."""

import torch
import torch.nn as nn
from transformers import SegformerForImageClassification, SegformerForSemanticSegmentation

from src.helpers import TransformerGradCAM


class SegformerMultiTask(nn.Module):
    """SegFormer para classificação e segmentação com geração de CAMs."""
    def __init__(self, num_classes, model_name='nvidia/mit-b0', image_size=448, shared_backbone=True):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.shared_backbone = shared_backbone
        model_cls = SegformerForImageClassification.from_pretrained(model_name)
        self.backbone = model_cls.segformer.to(device)
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
            dummy_output = self.backbone(dummy_input)
            in_features = dummy_output.last_hidden_state.shape[1]
        self.classifier = nn.Linear(in_features, num_classes).to(device)
        model_seg = SegformerForSemanticSegmentation.from_pretrained(model_name)
        model_seg.decode_head.classifier = nn.Conv2d(
            model_seg.decode_head.classifier.in_channels,
            num_classes,
            kernel_size=1
        )
        self.decode_head = model_seg.decode_head.to(device)
        self.seg_backbone = self.backbone
        self.cam_extractor = TransformerGradCAM(self)
        self.image_size = image_size
        self.num_classes = num_classes

    def classification_forward(self, x):
        features = self.backbone(x, output_hidden_states=True).hidden_states
        pooled = torch.mean(features[-1], dim=(2, 3))
        logits_cls = self.classifier(pooled)
        return logits_cls

    def generate_cams_original(self, x, logits_cls, target_labels, test=False):
        cams = torch.zeros((x.shape[0], self.num_classes, x.shape[2], x.shape[3]), device=x.device)
        for idx in range(x.shape[0]):
            img = x[idx].unsqueeze(0)
            if target_labels is not None and len(target_labels) > 0:
                tgt_with_bg = target_labels[idx]
                if tgt_with_bg.shape[0] == self.num_classes:
                    pass
                elif tgt_with_bg.shape[0] == self.num_classes - 1:
                    tgt_fg = tgt_with_bg
                    tgt_with_bg = torch.zeros(self.num_classes, device=tgt_fg.device)
                    tgt_with_bg[1:] = tgt_fg
                    tgt_with_bg[0] = 1.0
            else:
                pred = (torch.sigmoid(logits_cls[idx]) > 0.5).int()
                tgt_with_bg = pred
            present_classes = torch.nonzero(tgt_with_bg, as_tuple=False).squeeze(1).flatten()
            fg_classes = present_classes[present_classes > 0].tolist()
            if len(fg_classes) > 0:
                img.requires_grad = True
                cam = self.cam_extractor.generate_cam(img, fg_classes, test=test)
                cams[idx, fg_classes] = cam
            if len(fg_classes) > 0:
                max_fg_cam = torch.max(cams[idx, fg_classes], dim=0)[0]
                cams[idx, 0] = torch.clamp(1.0 - max_fg_cam, min=0.0)
            else:
                cams[idx, 0] = 1.0
        return cams

    def generate_cams_for_predicted_classes(self, x, logits_cls, existing_cams, gt_classes, threshold=0.5):
        cams = existing_cams.clone()
        cls_probs = torch.sigmoid(logits_cls)
        for idx in range(x.shape[0]):
            img = x[idx].unsqueeze(0)
            pred_probs_fg = cls_probs[idx, 1:]
            predicted_classes = torch.where(pred_probs_fg > threshold)[0]
            gt_set = set(gt_classes[idx]) if idx < len(gt_classes) else set()
            predicted_not_gt = [cls.item() for cls in predicted_classes if cls.item() not in gt_set]
            if len(predicted_not_gt) > 0:
                predicted_not_gt_model = [c + 1 for c in predicted_not_gt]
                classes_to_generate = [c for c in predicted_not_gt_model if cams[idx, c].sum() == 0]
                if len(classes_to_generate) > 0:
                    img.requires_grad = True
                    cam = self.cam_extractor.generate_cam(img, classes_to_generate, test=True)
                    cams[idx, classes_to_generate] = cam
                    fg_cams = cams[idx, 1:]
                    fg_classes_with_cam = torch.nonzero(fg_cams.sum(dim=(1, 2)) > 0, as_tuple=False).squeeze(1)
                    if len(fg_classes_with_cam) > 0:
                        fg_classes_model = fg_classes_with_cam + 1
                        max_fg_cam = torch.max(cams[idx, fg_classes_model], dim=0)[0]
                        cams[idx, 0] = torch.clamp(1.0 - max_fg_cam, min=0.0)
        return cams

    def forward(self, x, target_labels=None, test=False):
        self.backbone.to(x.device)
        features = self.backbone(x, output_hidden_states=True).hidden_states
        pooled = torch.mean(features[-1], dim=(2, 3))
        logits_cls = self.classifier(pooled)
        if not self.shared_backbone:
            features = self.seg_backbone(x, output_hidden_states=True).hidden_states
        spatial_features = list(features)
        logits_seg = self.decode_head(spatial_features)
        if target_labels is not None and len(target_labels) > 0:
            cams = self.generate_cams_original(x, logits_cls, target_labels, test)
        else:
            cams = torch.zeros((x.shape[0], self.num_classes, x.shape[2], x.shape[3]), device=x.device)
        return {
            "classification": logits_cls,
            "segmentation": logits_seg,
            "cams": cams,
            "cams_refined": cams,
            "features_map": features[-1]
        }
