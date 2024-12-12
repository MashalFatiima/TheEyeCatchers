import torch

def predict(model, input_tensor):
    """
    Makes a prediction using the loaded model and input tensor.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()