import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image 
import io
import os


class RSNAModelService:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.model = self._load_model()
        self.transform = self._build_transform()

    #Model building
    def _build_model(self):
        model = resnet50(weights=None)

        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.fc = nn.Linear(2048, 2)

        return model
    
    def _load_model(self):
        model_path = os.environ.get("RSNA_MODEL_LOCATION")

        if not model_path:
            raise RuntimeError(
                "RSNA_MODEL_LOCATION environment variable is not set"
            )
        
        if not os.path.exists(model_path):
            raise FileExistsError(
                f"Model file not found at: {model_path}"
            )
        
        model = self._build_model()
        model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        model.to(self.device)
        model.eval()

        return model
    
    def _build_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((256, 256))
        tensor = self.transform(image)
        return tensor.unsqueeze(0)
    
    #inference
    def predict(self, image_tensor):
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
        return pred_class
    
    def predict_from_bytes(self, image_bytes):
        tensor = self.preprocess(image_bytes)
        return self.predict(tensor)
        




























































#FUNCTIONAL PROGRAMMING
# Set model location in environment variable
# load model
# device = torch.device("cpu")

# #build model
# def build_model():
#   # Import resnet50
#   # model= torch.hub.load('pytorch/vision', 'resnet50', pretrained = False)

#   model = resnet50(weights=None)

#   #freezing layers in resnet50 except batchnorm so as ot to cause mismatch and loss of useful signal in new data
#   for name, param in model.named_parameters():
#     if("bn" not in name):
#       param.requires_grad = False

#   #Replacing first layer to match greyscale image and top layers in resnet50 with binary classification
#   model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   model.fc = nn.Linear(2048, 2)

#   return model


# def load_model():
#   model_path = os.environ.get("RSNA_MODEL_LOCATION")
  
#   if not model_path:
#      raise RuntimeError(
#         "RSNA_MODEL_LOCATION environment variable is not set"
#      )
  
#   if not os.path.exists(model_path):
#      raise FileNotFoundError(
#         f"Model file not found at: {model_path}"
#      )
  
#   model = build_model()
#   model.load_state_dict(
#      torch.load(model_path, map_location=device)
#   )
#   model.to(device)
#   model.eval()
#   return model

# # image -> tensor
# def transform_image(image_bytes):
#     image = Image.open(io.BytesIO(image_bytes))
#     image = image.convert("L")
#     image = image.resize((256,256))
#     transform = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.5], std=[0.5])
#     ])
#     tensor = transform(image)
#     return tensor.unsqueeze(0)

# model = load_model()
# # predict
# def get_prediction(image_tensor):
#    with torch.no_grad():
#         #images = images.reshape(-1, 256*256)
#         outputs = model(image_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         pred_class = torch.argmax(probs, dim=1)
#    return pred_class