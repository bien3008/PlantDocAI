import argparse
from pathlib import Path

import torch
from torch import nn
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders
from src.models.modelFactory import buildModel
from src.training.checkpoint import saveJson
from src.training.trainer import Trainer
from src.utils.configUtils import loadYamlConfig

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    return parser.parse_args()

def resolveDevice(deviceArg: str) -> torch.device:
    if deviceArg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if deviceArg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Fallback to CPU.")
        return torch.device("cpu")

    if deviceArg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def countParameters(model: torch.nn.Module) -> tuple[int, int]:
    totalParams = sum(param.numel() for param in model.parameters())
    trainableParams = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return totalParams, trainableParams

def main() -> None:
    args = parseArgs()
    
    # Load config from YAML
    config = loadYamlConfig(args.config)
    
    deviceArg = config.get("device", "auto")
    device = resolveDevice(deviceArg)

    loaders, classToId = buildDataLoaders(
        dataDir=config.get("dataDir", "data/Plantvillage"),
        splitDir=config.get("splitDir", "data/splits"),
        inputSize=config.get("imageSize", 224),
        batchSize=config.get("batchSize", 32),
        numWorkers=config.get("numWorkers", 2),
    )

    trainLoader = loaders["train"]
    valLoader = loaders["val"]

    idToClass = {v: k for k, v in classToId.items()}
    classNames = [idToClass[i] for i in range(len(classToId))]
    numClasses = len(classNames)
    
    modelName = config.get("modelName", "mobilenetV2")
    freezeBackbone = config.get("freezeBackbone", False)

    model = buildModel(
        modelName=modelName,
        numClasses=numClasses,
        usePretrained=True,
        freezeBackbone=freezeBackbone,
    ).to(device)

    totalParams, trainableParams = countParameters(model)
    trainableRatio = (trainableParams / totalParams) if totalParams > 0 else 0.0

    print(f"[INFO] modelName={modelName}")
    print(f"[INFO] freezeBackbone={freezeBackbone}")
    print(f"[INFO] totalParams={totalParams:,}")
    print(f"[INFO] trainableParams={trainableParams:,}")
    print(f"[INFO] trainableRatio={trainableRatio:.4f}")

    criterion = nn.CrossEntropyLoss()

    trainableModelParams = [param for param in model.parameters() if param.requires_grad]
    if len(trainableModelParams) == 0:
        raise RuntimeError("No trainable parameters found. Check freezeBackbone logic.")

    optimizer = torch.optim.Adam(
        trainableModelParams,
        lr=config.get("learningRate", 1e-3),
        weight_decay=config.get("weightDecay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    outputDir = Path(config.get("outputDir", "artifacts/baseline"))
    outputDir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the actual runtime configuration
    config["numClasses"] = numClasses
    config["classNames"] = classNames
    saveJson(str(outputDir / "config.json"), config)

    topK = config.get("topK", 3)
    numEpochs = config.get("numEpochs", 5)

    trainer = Trainer(
        model=model,
        device=device,
        trainLoader=trainLoader,
        valLoader=valLoader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        classNames=classNames,
        outputDir=str(outputDir),
        topK=topK,
        saveBestMetric="valTop1",
    )

    trainer.fit(numEpochs=numEpochs, config=config)


if __name__ == "__main__":
    main()