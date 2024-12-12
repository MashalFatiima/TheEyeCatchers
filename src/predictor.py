import torch

def predict(model, input_tensor):
    """
    Predict whether the input image has diabetic retinopathy.
    Returns:
        0: No DR
        1: DR detected
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
    return predicted.item()
