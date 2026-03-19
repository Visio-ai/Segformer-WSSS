import torch.nn.functional as F
import torch
import numpy as np

class TransformerGradCAM:
    def __init__(self, model, detach=False):
        self.model = model
        self.activations = None
        self.model.eval()
        self.hook_layer()
        self.device = next(model.parameters()).device

    def hook_layer(self):
        def forward_hook(module, input, output):
            self.activations = output
        
        target_layer = self.model.backbone.encoder.layer_norm[3]
        target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, target_labels=None, test=False, normalize=True):
        input_tensor = input_image.to(self.device)
        orig_size = input_image.shape[2:]
        self.model = self.model.to(self.device)
        
        logits = self.model.classification_forward(input_tensor)
        
        if target_labels is None:
            target_labels = [logits.argmax(dim=1).item()]
        elif not isinstance(target_labels, list):
            target_labels = [target_labels]
        
        activations = self.activations[0]
        num_tokens = activations.size(0)
        map_size = int(np.sqrt(num_tokens))
        
        cams = []
        
        for _, class_idx  in enumerate(target_labels):
            grad_outputs = torch.zeros_like(logits)
            grad_outputs[0, class_idx] = 1
            
            gradients = torch.autograd.grad(
                outputs=logits,
                inputs=self.activations,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=not test
            )[0][0]
            
            weights = gradients.mean(dim=0)  # [hidden_dim]
            
            cam = torch.matmul(activations, weights)  # [num_tokens]
            # cam = F.relu(cam)
            cam = cam.reshape(map_size, map_size)
            cams.append(cam)
        
        if len(cams) == 0:
            return torch.zeros((1, orig_size[0], orig_size[1]), device=self.device)

        cams = torch.stack(cams)  # [num_classes, map_size, map_size]

        if normalize:
            cams_min = cams.amin(dim=(1, 2), keepdim=True)
            cams_max = cams.amax(dim=(1, 2), keepdim=True)
            cams = (cams - cams_min) / (cams_max - cams_min + 1e-8)
        
        cams = F.interpolate(
            cams.unsqueeze(1),
            size=orig_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return cams
