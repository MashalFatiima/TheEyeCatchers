from torchvision import transforms

def preprocess_image(image):
    """
    Preprocess the uploaded image for the model.
    Args:
        image (PIL.Image): Input image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(          # Normalize with ImageNet means and stds
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension
