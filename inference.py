import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

image_path = ''

# Das vortrainierte Modell laden (Man muss "trained_model.pt" durch den tatsächlichen Pfad zu der Modelldatei ersetzen)
model = torch.load('trained_model.pt', map_location="cpu")
model.eval()

# Klassenbezeichnungen für mögliche Bildklassifikationen definieren
class_names = ["infected", "normal"]

# Reihe von Bildtransformationen definieren, die auf jedes geladene Bild angewendet werden sollen (genauso wie bei den Trainingsbildern durchgeführt)
loader = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.252, 0.293, 0.288], [0.146, 0.191, 0.193])
])

#  Funktion definieren, um ein Bild zu laden, es vorzubereiten und in einen Tensor umzuwandeln
def image_loader(image):
  """Laden Sie ein Bild und geben Sie es als CUDA-Tensor zurück (setzt GPU-Nutzung voraus)  """

    image = Image.open(image)  # Bilddatei öffnen
    image = loader(image).float()  # definierte Transformationen auf das Bild auftragen
    image = Variable(image, requires_grad=True)  # PyTorch-Variable mit aktivierten Gradienten erstellen
    image = image.unsqueeze(0)  # Bildtensor umformen (für ResNet nicht notwendig)
    return image  # Die vorverarbeitete Bildmatrix zurückgeben

# Streamlit web app erstellen
st.title("COVID-19 Classifier")

# Bild hochladen
uploaded_image = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    # Hochgeladenes Bild anzeigen
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Predictions ausführen, wenn auf eine Schaltfläche geklickt wird
    if st.button("Classify"):
        image_path = "temp.jpg"  # Temporary image path

        # Das hochgeladene Bild in einer temporären Datei speichern
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

        #Das hochgeladene Bild laden und vorverarbeiten
        image = image_loader(image_path)

        # Eine Vorhersage unter Verwendung des geladenen Modells durchführen
        output = model(image)
        index = output.data.cpu().numpy().argmax()
        predicted_class = class_names[index]

        #Die Farbe für das Klassifikationslabel bestimmen (Rot für infiziert, Grün für normal)
        color = "red" if predicted_class == "infected" else "green"

        #  Ergebnis in Textform mit der festgelegten Farbe anzeigen
        st.markdown(
            f"<p style='color:{color};font-size:20px;text-align:center;'>Classified as {predicted_class}</p>",
            unsafe_allow_html=True,
        )

# Temporäre Bilddatei aufräumen
if os.path.exists(image_path):
    os.remove(image_path)
