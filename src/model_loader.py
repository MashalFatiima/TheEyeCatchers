import os
import torch

def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    full_model_path = os.path.join(model_path, "trained_model.pth")
    model = torch.load(full_model_path, map_location=torch.device('cpu'))  # Load full model
    model.eval()  # Set the model to evaluation mode
    return model
