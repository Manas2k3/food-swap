from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms
import json
import re
from torchvision.models import ResNet50_Weights

# Flask app
app = Flask(__name__)

# Load Model and Classes
num_classes = 101
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_food101_model.pth', map_location='cpu'))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load class names
with open('food-101/meta/classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load nutrition info
with open('food_nutrition_data.json', 'r') as f:
    nutrition_data = json.load(f)["foods"]

# Load alternatives
with open('alternatives.json', 'r') as f:
    alternatives_data = json.load(f)

# Load alternative nutrition info
with open('alternative_food_nutrition_data.json', 'r') as f:
    alt_nutrition_raw = json.load(f)
    # Build a dict with snake_case name as key
    alt_nutrition_data = {
        item["name"]: item["nutritional_info"] for item in alt_nutrition_raw
    }


# Format helpers
def format_snake_to_title(snake_str):
    return re.sub(r'_', ' ', snake_str).title()

def format_title_to_snake(title):
    return re.sub(r'\s+', '_', title.strip().lower())

# Lookup functions
def get_nutrition_info(food_name):
    for item in nutrition_data:
        if item["name"].lower() == food_name.lower():
            info = item["nutritional_info"]
            return {
                "Calories": info.get("Calories", "N/A"),
                "Carbohydrates": info.get("Carbohydrates", "N/A"),
                "Protein": info.get("Protein", "N/A"),
                "Total Fat": info.get("Fats", {}).get("Total", "N/A"),
                "Sugars": info.get("Sugars", "N/A")
            }
    return None

def get_alternatives_with_nutrition(food_snake_case):
    formatted_name = format_snake_to_title(food_snake_case)
    for entry in alternatives_data.values():
        if entry["original"].strip().lower() == formatted_name.strip().lower():
            result = []
            for alt in entry["alternatives"]:
                alt_name_snake = format_title_to_snake(alt["name"])
                nutrition = alt_nutrition_data.get(alt_name_snake, {})
                result.append({
                    "name": alt["name"],
                    "image": alt["image"],
                    "nutritional_info": {
                        "Calories": nutrition.get("Calories", "N/A"),
                        "Carbohydrates": nutrition.get("Carbohydrates", "N/A"),
                        "Protein": nutrition.get("Protein", "N/A"),
                        "Total Fat": nutrition.get("Fats", {}).get("Total", "N/A"),
                        "Sugars": nutrition.get("Sugars", "N/A"),
                    }
                })
            return result
    return []

# API Route
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_class = class_names[predicted_idx.item()]

    # Results
    nutrition = get_nutrition_info(predicted_class)
    alternatives = get_alternatives_with_nutrition(predicted_class)

    response = {
        "predicted_food": format_snake_to_title(predicted_class),
        "nutritional_info": nutrition,
        "alternatives": alternatives
    }

    return jsonify(response), 200

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)



