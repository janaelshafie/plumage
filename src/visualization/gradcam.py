"""Grad-CAM (Selvaraju et al., 2017): which parts of the image made the model choose a species.

For one image and one class:
  1. Run the image forward and keep the feature maps A of the Grad-CAM layer, shape [C, h, w].
  2. Backpropagate the class score and keep the gradients dA of those maps.
  3. Average each gradient map into one number: how much channel c matters for this class.
  4. Weighted sum of the feature maps, then ReLU (keep only what raises the class score).
  5. Resize the h x w map to the image size and scale it to 0..1.

Works with any model that has get_gradcam_layer() (see encoders.py).
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import colormaps


class GradCAM:
    """Usage:
        gradcam = GradCAM(model)
        cam, pred, probs = gradcam(image)   # image: one normalized [3, H, W] tensor
        gradcam.remove()
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        # The forward hook runs every time the layer produces its output.
        self.hook = model.get_gradcam_layer().register_forward_hook(self.save_activations)

    def save_activations(self, module, inputs, output):
        self.activations = output.detach()
        # A hook on the output tensor runs during backward and receives its gradient.
        if output.requires_grad:
            output.register_hook(self.save_gradients)

    def save_gradients(self, grad):
        self.gradients = grad.detach()

    def __call__(self, image, class_idx=None):
        """Explain one image. class_idx is the species to explain (default: the predicted one).

        Returns (cam, pred, probs):
            cam    [H, W] tensor in 0..1, high where the model looked
            pred   index of the predicted species
            probs  [num_classes] softmax probabilities
        """
        self.model.eval()
        # A frozen encoder has requires_grad=False on its weights, so without this no
        # gradient would flow back to the Grad-CAM layer and the hook would never run.
        x = image.unsqueeze(0).clone().requires_grad_(True)

        with torch.enable_grad():
            scores = self.model(x)
            pred = scores.argmax(dim=1).item()
            if class_idx is None:
                class_idx = pred
            self.model.zero_grad()
            scores[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)                # [1, C, 1, 1]
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))    # [1, 1, h, w]
        cam = F.interpolate(cam, size=image.shape[1:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # + 1e-8 so an all-zero map doesn't divide by 0

        probs = scores.softmax(dim=1)[0].detach()
        return cam.cpu(), pred, probs.cpu()

    def remove(self):
        """Detach the hook so the model runs normally again."""
        self.hook.remove()


def overlay(image, cam, alpha=0.5):
    """Draw the heatmap over the image.

    image: [3, H, W] tensor in 0..1 (use denormalize() from transforms.py first)
    cam:   [H, W] tensor in 0..1
    Returns an [H, W, 3] numpy array that plt.imshow() can show.
    """
    heatmap = colormaps["jet"](cam.numpy())[..., :3]  # red = high, blue = low; drop the alpha channel
    return (1 - alpha) * image.permute(1, 2, 0).numpy() + alpha * heatmap


def save_gradcam_figure(image, cam, title, path):
    """Save the image and its Grad-CAM overlay side by side. image is [3, H, W] in 0..1."""
    fig, (image_ax, cam_ax) = plt.subplots(1, 2, figsize=(8, 4.3))
    image_ax.imshow(image.permute(1, 2, 0).numpy())
    image_ax.set_title("input")
    cam_ax.imshow(overlay(image, cam))
    cam_ax.set_title("Grad-CAM")
    for ax in (image_ax, cam_ax):
        ax.axis("off")
    fig.suptitle(title)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_image_grid(panels, path, columns=3):
    """Save several images in a grid, filled row by row.

    panels: list of (image, title), each image an [H, W, 3] numpy array
            (e.g. from overlay(), or an input image with .permute(1, 2, 0).numpy())
    """
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 3.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")  # also hides empty cells when panels don't fill the last row
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
