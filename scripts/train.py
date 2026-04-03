import argparse
from pathlib import Path

import torch
from torch import nn
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataLoader import buildDataLoaders
from src.data.dataSplit import createSplits, SplitConfig
from src.models.modelFactory import buildModel
from src.training.checkpoint import saveJson, loadCheckpoint
from src.training.trainer import Trainer
from src.utils.configUtils import loadYamlConfig
from src.utils.colabUtils import mountGoogleDrive, isColabEnvironment
import os

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if it exists")
    parser.add_argument("--useGdrive", action="store_true", help="Mount Google Drive and save artifacts there if running in Colab")
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

    dataDir = config.get("dataDir", "data/Plantvillage")
    splitDir = config.get("splitDir", "data/splits")

    requiredSplits = ["train.csv", "val.csv", "test.csv", "classes.csv"]
    hasAllSplits = all((Path(splitDir) / f).exists() for f in requiredSplits)

    if not hasAllSplits:
        print(f"[INFO] Missing or incomplete split files in '{splitDir}'. Auto-generating splits...")
        splitConfig = SplitConfig()
        createSplits(dataDir=dataDir, outDir=splitDir, splitConfig=splitConfig)
        print("[INFO] Split generation completed.")
    else:
        print(f"[INFO] Found existing split files in '{splitDir}'.")

    loaders, classToId = buildDataLoaders(
        dataDir=dataDir,
        splitDir=splitDir,
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

    print(f"[INFO] Model: {modelName} | Freeze Backbone: {freezeBackbone}")
    print(f"[INFO] Params: {totalParams:,} (Total) | {trainableParams:,} (Trainable) | {trainableRatio:.1%} Ratio")

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

    outputDirStr = config.get("outputDir", "artifacts/baseline")
    
    if args.useGdrive and isColabEnvironment():
        driveRoot = mountGoogleDrive()
        if driveRoot:
            # Output will be redirected to Google Drive
            gdrivePrefix = os.path.join(driveRoot, "MyDrive", "PlantDocAI_Outputs")
            # Determine folder name instead of deep nesting
            baseDirName = os.path.basename(outputDirStr.rstrip("/"))
            if not baseDirName:
                baseDirName = "runs"
            outputDirStr = os.path.join(gdrivePrefix, baseDirName)
            print(f"[INFO] useGdrive flag enabled. Redirecting output to: {outputDirStr}")

    outputDir = Path(outputDirStr)
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

    startEpoch = 1
    if args.resume:
        lastCheckpointPath = outputDir / "checkpoints" / "last.pt"
        if lastCheckpointPath.exists():
            print(f"[INFO] Resuming from checkpoint: {lastCheckpointPath}")
            checkpoint = loadCheckpoint(
                checkpointPath=str(lastCheckpointPath),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                mapLocation=str(device)
            )
            startEpoch = checkpoint.get("epoch", 0) + 1
            bestMetric = checkpoint.get("bestMetric", float("-inf"))
            trainer.bestMetric = bestMetric
            print(f"[INFO] Restored state. Resuming from epoch {startEpoch}...")
        else:
            print(f"[INFO] --resume flag is set but {lastCheckpointPath} not found. Starting from scratch.")

    trainer.fit(numEpochs=numEpochs, config=config, startEpoch=startEpoch)


if __name__ == "__main__":
    main()