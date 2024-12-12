import torch

def predict(model, input_tensor):
    """
    Make a prediction using the pre-trained model.
    Args:
        model (torch.nn.Module): Loaded model.
        input_tensor (torch.Tensor): Preprocessed input tensor.
    Returns:
        int: Prediction (0 for No DR, 1 for DR).
    """
    with torch.no_grad():  # Disable gradient calculations
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score
        return predicted.item()  # Return as an integer
