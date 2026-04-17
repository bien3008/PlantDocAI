import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.data.dataTransforms import AugConfig

@dataclass
class TrainingConfig:
    experimentName: str = "baseline"
    dataDir: str = "data/PlantVillage/train"
    splitDir: str = "data/splits"
    outputDir: str = "artifacts/baseline"
    modelName: str = "mobilenetV2"
    imageSize: int = 224
    freezeBackbone: bool = False
    batchSize: int = 32
    numEpochs: int = 5
    learningRate: float = 0.001
    weightDecay: float = 0.0001
    numWorkers: int = 2
    topK: int = 3
    device: str = "auto"
    augConfig: AugConfig = field(default_factory=AugConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys and k != "augConfig"}

        # Parse nested `augmentation:` block from YAML into AugConfig
        aug_raw = d.get("augmentation", {})
        if isinstance(aug_raw, dict):
            aug_fields = {f for f in AugConfig.__dataclass_fields__}
            aug_filtered = {k: v for k, v in aug_raw.items() if k in aug_fields}
            filtered_dict["augConfig"] = AugConfig(**aug_filtered)
        else:
            filtered_dict["augConfig"] = AugConfig()

        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "augConfig"}
        # Serialise AugConfig as a nested dict under 'augmentation'
        result["augmentation"] = {k: getattr(self.augConfig, k) for k in self.augConfig.__dataclass_fields__}
        return result

def loadYamlConfig(configPath: str) -> dict:
    """Load a YAML configuration file as raw dictionary."""
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {configPath}")
        
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        
    if config is None:
        config = {}
        
    return config

def loadTrainingConfig(configPath: str) -> TrainingConfig:
    """Load a YAML configuration file specifically into TrainingConfig."""
    raw_dict = loadYamlConfig(configPath)
    return TrainingConfig.from_dict(raw_dict)
