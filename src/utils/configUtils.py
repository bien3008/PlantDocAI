import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

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
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_dict = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

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
