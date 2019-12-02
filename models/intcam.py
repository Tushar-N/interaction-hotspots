import torch
import torch.nn as nn
import torch.nn.functional as F

class IntCAM(nn.Module):
    def __init__(self, model, out_sz=224):
        super().__init__()
        self.out_sz = out_sz
        self.num_classes = model.num_classes
        self.model = model

    def hook_grad(self, grad):
        self.gradients = grad

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        return self

    def backward_for_target_grad(self, output, activations, target_grad):
        self.model.zero_grad()
        output.backward(gradient=target_grad, retain_graph=True)
        cam = F.relu(self.gradients*activations).sum(2) # (B, T, C, 7, 7) --> (B, T, 7, 7)

        delattr(self, 'gradients')
        return cam.detach()

    def generate_output_activation(self, inputs, **kwargs):

        # generate activations if that's not what the input is
        with torch.no_grad():
            activations = self.model.ic_features(inputs, **kwargs) # (B, T, C, 7, 7)

        # hook gradients
        activations.requires_grad_()
        activations.register_hook(self.hook_grad)

        # generate outputs
        outputs = self.model.ic_classifier(activations, **kwargs)

        return activations, outputs

    # Generate cams for multiple classes
    def generate_cams(self, inputs, classes, **kwargs):

        # If it's a static image, let the model decide how to generate activations
        if inputs.dim()==4:
            kwargs['static'] = True
            kwargs['c_id'] = classes[0]

        activations, outputs = self.generate_output_activation(inputs, **kwargs)

        # iterate over classes and generate cams
        cams = []
        for c_id in classes:

            if isinstance(c_id, int):
                targets = torch.zeros(outputs.shape[0]).fill_(c_id).long()
            elif isinstance(c_id, torch.Tensor):
                targets = c_id

            # set an entire channel to 1
            target_grad = torch.zeros(outputs.shape).to(outputs.device)
            target_grad[torch.arange(outputs.shape[0]), targets] = 1

            # perform the backward pass and weight the activations by the gradient
            cam = self.backward_for_target_grad(outputs, activations, target_grad) # (B, T, 7, 7) for a single class
            cams.append(cam.cpu())

        cams = torch.stack(cams, 2) # (B, T, C, 7, 7)

        return cams