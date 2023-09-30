import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

image_path = ''

# Load the pre-trained model (you'll need to replace 'trained_model.pt' with the actual path to your model file)
model = torch.load('trained_model.pt', map_location="cpu")
model.eval()

# Define class names for the possible image classifications
class_names = ["infected", "normal"]

# Define a series of image transformations to be applied to each loaded image (same as done for training images)
loader = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.252, 0.293, 0.288], [0.146, 0.191, 0.193])
])

# Define a function to load an image, preprocess it, and convert it to a tensor
def image_loader(image):
    """Load an image and return it as a CUDA tensor (assumes GPU usage)"""
    image = Image.open(image)  # Open the image file
    image = loader(image).float()  # Apply the defined transformations to the image
    image = Variable(image, requires_grad=True)  # Create a PyTorch variable with gradients enabled
    image = image.unsqueeze(0)  # Reshape the image tensor (not necessary for ResNet)
    return image  # Return the preprocessed image tensor

# Create a Streamlit web app
st.title("COVID-19 Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform predictions when a button is clicked
    if st.button("Classify"):
        image_path = "temp.jpg"  # Temporary image path

        # Save the uploaded image to a temporary file
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        # Load and preprocess the uploaded image
        image = image_loader(image_path)

        # Perform a prediction using the loaded model
        output = model(image)
        index = output.data.cpu().numpy().argmax()
        predicted_class = class_names[index]

        # Determine the color for the classification label (red for infected, green for normal)
        color = "red" if predicted_class == "infected" else "green"

        # Display the result in text form with the determined color
        st.markdown(
            f"<p style='color:{color};font-size:20px;text-align:center;'>Classified as {predicted_class}</p>",
            unsafe_allow_html=True,
        )

# Clean up the temporary image file
if os.path.exists(image_path):
    os.remove(image_path)
