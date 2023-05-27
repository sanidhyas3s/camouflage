import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from labels.imagenet import labels

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
relative_path = "images/flower.jpg"
image_path = os.path.join(script_dir, relative_path)

# Load the pretrained ResNet model
model = models.resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Preprocess the input image
image = Image.open(image_path)
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Run the image through the model
with torch.no_grad():
    output = model(input_batch)

# Calculate the predicted probabilities
probabilities = torch.nn.functional.softmax(output[0], dim=0)

# Print the top k predicted classes and their probabilities
num_classes = 1000  # Number of classes in the pretrained model (ImageNet)
top_probabilities, top_classes = torch.topk(probabilities, k=5)
for prob, cls in zip(top_probabilities, top_classes):
    class_index = cls.item()
    class_name = labels[class_index]
    print("Class: {}, Probability: {:.5f}".format(class_name, prob.item()))
