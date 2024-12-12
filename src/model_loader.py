import torch
from torchvision.models import resnet50

def load_model(model_path):
    """
    Load the pre-trained model from the given path.
    """
    model = resnet50(pretrained=False)  # Define the architecture
    full_model_path = os.path.join(model_path, "trained_model.pth")
    model.load_state_dict(torch.load(full_model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model
