import torch
import matplotlib.pyplot as plt
from torchvision.utils import draw_segmentation_masks

def debug_evaluate(model, data_loader, device, num_classes, class_names=None):
    """
    Debug version of evaluate function.
    It will:
        - Print shapes and values of key variables
        - Visualize one image, target, and prediction
        - Check for common issues (e.g. mask is not long, pred is all background)
    """
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            print("==" * 40)
            print("Debugging a batch...")

            images = images.to(device)
            targets = targets.to(device)

            # Step 1: Model Inference
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']

            print("Image shape:", images.shape)
            print("Target shape:", targets.shape)
            print("Output shape:", outputs.shape)
            print("Unique values in target:", torch.unique(targets))

            # Step 2: Predictions
            pred_logits = outputs
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_labels = pred_probs.argmax(dim=1)

            print("Pred labels shape:", pred_labels.shape)
            print("Unique predicted labels:", torch.unique(pred_labels))
            print("Pred labels sum:", pred_labels.sum().item())

            # Step 3: Visualization (first sample in the batch)
            img = images[0].cpu()
            target = targets[0].cpu()
            pred_label = pred_labels[0].cpu()

            print("First image unique pixels:", torch.unique(img))
            print("First target unique pixels:", torch.unique(target))
            print("First prediction unique pixels:", torch.unique(pred_label))

            # Convert tensors to displayable format
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
            img = img.permute(1, 2, 0).numpy()  # HWC

            # One-hot decode target and prediction for visualization
            def one_hot_decode(mask, num_classes):
                return torch.eye(num_classes)[mask.long()].permute(2, 0, 1).bool()

            target_mask = one_hot_decode(target, num_classes)
            pred_mask = one_hot_decode(pred_label, num_classes)

            # Draw masks on image
            target_img = draw_segmentation_masks(torch.tensor(img * 255).byte().permute(2, 0, 1), target_mask, alpha=0.6, colors=["red"] * num_classes)
            pred_img = draw_segmentation_masks(torch.tensor(img * 255).byte().permute(2, 0, 1), pred_mask, alpha=0.6, colors=["green"] * num_classes)

            target_img = target_img.permute(1, 2, 0).numpy()
            pred_img = pred_img.permute(1, 2, 0).numpy()

            # Plotting
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(target_img)
            plt.title("True Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_img)
            plt.title("Predicted Mask")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

            break  # Only visualize one batch