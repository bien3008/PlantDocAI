import argparse
import json
import ast
import sys
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

# Ensure the root directory is in sys.path so we can import 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.modelFactory import buildModel
from src.training.checkpoint import loadCheckpoint

def parseArgs():
    parser = argparse.ArgumentParser(description="Inference using a trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to predict")
    parser.add_argument("--modelDir", type=str, required=True, help="Directory containing config.json and checkpoint (e.g. checkpoints/best.pt)")
    return parser.parse_args()

def main():
    args = parseArgs()
    
    modelDir = Path(args.modelDir)
    configPath = modelDir / "config.json"
    
    # Priority: checkpoints/best.pt -> best.pt -> checkpoints/last.pt
    weightsPath = None
    for p in ["checkpoints/best.pt", "best.pt", "checkpoints/last.pt"]:
        if (modelDir / p).exists():
            weightsPath = modelDir / p
            break
            
    if not configPath.exists():
        raise FileNotFoundError(f"Cannot find config at {configPath}")
    if weightsPath is None:
        raise FileNotFoundError(f"Cannot find checkpoint (best.pt) in {modelDir}")

    # 1. Load configuration
    with open(configPath, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Some versions of python/json might save lists as strings, safe-eval it if needed
    classNames = config.get("classNames", [])
    if isinstance(classNames, str):
        classNames = ast.literal_eval(classNames)
        
    numClasses = config.get("numClasses", len(classNames))
    modelName = config.get("modelName", "mobilenetV2")
    imageSize = config.get("imageSize", 224)

    # Get normalization constants from config or fallback to ImageNet
    mean = config.get("mean", [0.485, 0.456, 0.406])
    std = config.get("std", [0.229, 0.224, 0.225])

    # 2. Build model architecture
    print(f"[INFO] Building model: {modelName} for {numClasses} classes")
    model = buildModel(
        modelName=modelName, 
        numClasses=numClasses, 
        usePretrained=False, # We will load our own weights
        freezeBackbone=False
    )

    # 3. Load the trained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading weights onto {device} from {weightsPath}")
    loadCheckpoint(checkpointPath=str(weightsPath), model=model, mapLocation=str(device))
    model.to(device)
    model.eval() # Set model to evaluation mode (important for BatchNorm/Dropout)

    # 4. Prepare image transforms
    evalTransform = transforms.Compose([
        transforms.Resize((imageSize, imageSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 5. Load and preprocess image
    try:
        with Image.open(args.image) as img:
            img = img.convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image: {e}")
        return

    tensorImg = evalTransform(img).unsqueeze(0).to(device) # Add batch dimension: [1, 3, H, W]

    # 6. Predict
    print(f"[INFO] Predicting on {args.image}...")
    with torch.no_grad():
        outputs = model(tensorImg)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get Top-3 predictions
    topProb, topClassIdx = torch.topk(probabilities, k=min(3, numClasses))
    
    print("\n--- KẾT QUẢ DỰ ĐOÁN (Top 3) ---")
    for i in range(len(topClassIdx)):
        idx = topClassIdx[i].item()
        prob = topProb[i].item()
        
        # If classNames is available in config, use it, else just print the ID
        name = classNames[idx] if len(classNames) > idx else f"Class_{idx}"
        print(f"{i+1}. {name}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
