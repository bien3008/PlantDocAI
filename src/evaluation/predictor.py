import json
import ast
from pathlib import Path
from typing import Dict, Tuple, List, Union

import torch
from torchvision import transforms
from PIL import Image

from src.models.modelFactory import buildModel
from src.training.checkpoint import loadCheckpoint


class InferencePipeline:
    """
    A reusable pipeline to run inference on single or multiple images using a trained PlantDocAI model.
    It encapsulates configuration loading, model building, weight loading, and image preprocessing.
    """
    
    def __init__(self, modelDir: Union[str, Path], device: str = "auto") -> None:
        """
        Initializes the InferencePipeline.
        
        Args:
            modelDir (str or Path): The directory containing `config.json` and checkpoint (`best.pt`).
            device (str): Device to run inference on ('cpu', 'cuda', or 'auto').
        """
        self.modelDir = Path(modelDir)
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self._loadConfig()
        self._initModel()
        self._initTransforms()

    def _loadConfig(self) -> None:
        configPath = self.modelDir / "config.json"
        if not configPath.exists():
            raise FileNotFoundError(f"Cannot find config at {configPath}")

        with open(configPath, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Parse class names
        classNames = self.config.get("classNames", [])
        if isinstance(classNames, str):
            classNames = ast.literal_eval(classNames)
        self.classNames = classNames
        self.numClasses = self.config.get("numClasses", len(self.classNames))
        
        # Parse model params
        self.modelName = self.config.get("modelName", "mobilenetv2_100")
        self.imageSize = self.config.get("imageSize", 224)
        
        # Parse normalization params
        self.mean = self.config.get("mean", [0.485, 0.456, 0.406])
        self.std = self.config.get("std", [0.229, 0.224, 0.225])

    def _initModel(self) -> None:
        # Determine weights path
        weightsPath = None
        for p in ["checkpoints/best.pt", "best.pt", "checkpoints/last.pt"]:
            if (self.modelDir / p).exists():
                weightsPath = self.modelDir / p
                break
                
        if weightsPath is None:
            raise FileNotFoundError(f"Cannot find checkpoint (e.g., best.pt) in {self.modelDir}")

        # Build model via abstraction
        self.model = buildModel(
            modelName=self.modelName, 
            numClasses=self.numClasses, 
            usePretrained=False,
            freezeBackbone=False
        )

        # Load weights
        loadCheckpoint(
            checkpointPath=str(weightsPath), 
            model=self.model, 
            mapLocation=str(self.device)
        )
        
        self.model.to(self.device)
        self.model.eval()

    def _initTransforms(self) -> None:
        self.transform = transforms.Compose([
            transforms.Resize((self.imageSize, self.imageSize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def predict(self, imagePath: Union[str, Path], topK: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Predicts classes for a single image.
        
        Args:
            imagePath (str or Path): Path to the image file.
            topK (int): Number of top predictions to return.
            
        Returns:
            List[Dict]: A list of length topK containing dictionaries with keys 'className' and 'probability'.
        """
        try:
            with Image.open(imagePath) as img:
                img = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image for prediction at {imagePath}. Error: {e}")

        # Preprocess and prepare tensor
        tensorImg = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensorImg)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top-k results
        actualK = min(topK, self.numClasses)
        topProb, topClassIdx = torch.topk(probabilities, k=actualK)
        
        results = []
        for i in range(actualK):
            idx = topClassIdx[i].item()
            prob = topProb[i].item()
            
            name = self.classNames[idx] if len(self.classNames) > idx else f"Class_{idx}"
            results.append({
                "className": name,
                "probability": prob
            })
            
        return results
