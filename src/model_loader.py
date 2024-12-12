import torch
from torchvision.models import resnet50

def load_model(model_path="model/trained_model.pth"):
    """
    Loads a pre-trained model from the specified path.
    """
    model = resnet50(pretrained=False)  # Define the model architecture
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model
