import io
import argparse
import torch
from PIL import Image
from torchvision import transforms
from eye_disease_classification import CNNModel  # Ensure this is the correct import path

def load_model():
    model_path = 'trained_model.pth'  # Hardcoded model path
    model = torch.load(model_path)
    model.eval()
    return model

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img

def predict_class(image_path, model, class_names):
    img = Image.open(image_path)  # Open the image
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        predicted_class_index = torch.argmax(output).item()
        predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

def main():
    parser = argparse.ArgumentParser(description='Predict eye disease from an image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
    model = load_model()

    predicted_class = predict_class(args.image_path, model, class_names)
    print(f"Predicted class for the uploaded image: {predicted_class}")

if __name__ == '__main__':
    main()
