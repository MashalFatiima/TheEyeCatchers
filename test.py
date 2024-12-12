import torch

model = torch.load(r"C:\Users\Abdullah\Desktop\TheEyeCatchers\model\trained_model.pth", map_location=torch.device('cpu'))
print(type(model))  # Should output <class 'torchvision.models.resnet.ResNet'>
