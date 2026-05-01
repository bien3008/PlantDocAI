import argparse
import os
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
from src.utils.configUtils import loadTrainingConfig
from src.utils.colabUtils import setupColabOutput

def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint if it exists")
    parser.add_argument("--useGdrive", action="store_true", help="Mount Google Drive and save artifacts there if running in Colab")
    parser.add_argument(
        "--pretrainedCheckpoint", type=str, default=None,
        help="Path to a pretrained checkpoint (e.g., Stage 1 best.pt). "
             "Loads model weights only (not optimizer/scheduler). "
             "Overrides pretrainedCheckpoint in config YAML if both are set."
    )
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
    
    # Load typed config object, no more hidden fallbacks
    config = loadTrainingConfig(args.config)
    device = resolveDevice(config.device)

    # Validate or prepare splits
    requiredSplits = ["train.csv", "val.csv", "test.csv", "classes.csv"]
    hasAllSplits = all((Path(config.splitDir) / f).exists() for f in requiredSplits)

    if not hasAllSplits:
        print(f"[INFO] Missing or incomplete split files in '{config.splitDir}'. Auto-generating splits...")
        createSplits(dataDir=config.dataDir, outDir=config.splitDir, splitConfig=SplitConfig())
        print("[INFO] Split generation completed.")
    else:
        print(f"[INFO] Found existing split files in '{config.splitDir}'.")

    # Build Dataloaders
    loaders, classToId = buildDataLoaders(
        dataDir=config.dataDir,
        splitDir=config.splitDir,
        inputSize=config.imageSize,
        batchSize=config.batchSize,
        numWorkers=config.numWorkers,
        aug=config.augConfig,
        useWeightedSampler=config.useWeightedSampler,
    )
    
    idToClass = {v: k for k, v in classToId.items()}
    classNames = [idToClass[i] for i in range(len(classToId))]
    numClasses = len(classNames)

    # ── Resolve pretrained checkpoint (CLI > config > ImageNet) ─────────────
    pretrainedCkpt = args.pretrainedCheckpoint or config.pretrainedCheckpoint
    useImageNetPretrained = (pretrainedCkpt is None)  # dùng ImageNet chỉ khi không có checkpoint

    # Build Model
    model = buildModel(
        modelName=config.modelName,
        numClasses=numClasses,
        usePretrained=useImageNetPretrained,
        freezeBackbone=config.freezeBackbone,
    ).to(device)

    # ── Load pretrained checkpoint nếu có (chỉ weights, KHÔNG load optimizer) ─
    if pretrainedCkpt is not None:
        ckptPath = Path(pretrainedCkpt)
        if not ckptPath.exists():
            print(f"[ERROR] Pretrained checkpoint not found: {ckptPath}")
            sys.exit(1)
        print(f"[INFO] Loading pretrained weights from: {ckptPath}")
        ckptData = torch.load(str(ckptPath), map_location=str(device))
        # Kiểm tra tương thích numClasses
        ckptClassNames = ckptData.get("classNames", [])
        if len(ckptClassNames) != numClasses:
            print(f"[WARN] Checkpoint has {len(ckptClassNames)} classes, current model has {numClasses} classes.")
            print(f"[WARN] Loading weights with strict=False (classifier head sẽ được init lại).")
            # Load non-strict: bỏ qua classifier head nếu size không khớp
            model.load_state_dict(ckptData["modelStateDict"], strict=False)
        else:
            model.load_state_dict(ckptData["modelStateDict"])
        print(f"[INFO] Pretrained weights loaded successfully.")

    totalParams, trainableParams = countParameters(model)
    print(f"[INFO] Model: {config.modelName} | Freeze Backbone: {config.freezeBackbone}")
    print(f"[INFO] Params: {totalParams:,} (Total) | {trainableParams:,} (Trainable)")
    if pretrainedCkpt:
        print(f"[INFO] Pretrained from: {pretrainedCkpt} (NOT ImageNet)")

    # Loss and Optimizer
    if config.useClassWeights:
        from collections import Counter
        from src.data.dataSplit import loadSplitCsv
        
        trainSamples = loadSplitCsv(f"{config.splitDir}/train.csv")
        labelCounts = Counter(s.labelId for s in trainSamples)
        total = sum(labelCounts.values())
        
        # Calculate inverse frequency weights to balance classes
        weights = [total / (numClasses * labelCounts[i]) for i in range(numClasses)]
        classWeightsTensor = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=classWeightsTensor)
        print(f"[INFO] CrossEntropyLoss uses calculated class weights.")
    else:
        criterion = nn.CrossEntropyLoss()
        
    if config.freezeBackbone:
        trainableModelParams = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(
            trainableModelParams,
            lr=config.learningRate,
            weight_decay=config.weightDecay,
        )
        print("[INFO] Using single learning rate for the newly initialized head.")
    else:
        classifier_params = []
        try:
            # Check if model has get_classifier (timm models usually do)
            if hasattr(model, 'get_classifier'):
                classifier_params = list(model.get_classifier().parameters())
            else:
                print("[WARN] Model does not have get_classifier(). Falling back to single learning rate.")
        except Exception as e:
             print(f"[WARN] Failed to get classifier for LLR: {e}")
             
        if classifier_params:
            classifier_ids = set(id(p) for p in classifier_params)
            backbone_params = [p for p in model.parameters() if id(p) not in classifier_ids and p.requires_grad]
            
            # Backbone gets 10x smaller learning rate to prevent catastrophic forgetting
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': config.learningRate * 0.1},
                {'params': classifier_params, 'lr': config.learningRate}
            ], weight_decay=config.weightDecay)
            print(f"[INFO] Using Layer-wise Learning Rate (LLR). Classifier LR: {config.learningRate}, Backbone LR: {config.learningRate * 0.1}")
        else:
            trainableModelParams = [param for param in model.parameters() if param.requires_grad]
            optimizer = torch.optim.Adam(
                trainableModelParams,
                lr=config.learningRate,
                weight_decay=config.weightDecay,
            )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Prepare logic isolation for output paths (Google Drive aware)
    outputDirStr = setupColabOutput(config.outputDir, args.useGdrive)
    outputDir = Path(outputDirStr)
    outputDir.mkdir(parents=True, exist_ok=True)

    # Prepare runtime config dictionary for saving
    runtimeConfig = config.to_dict()
    runtimeConfig["numClasses"] = numClasses
    runtimeConfig["classNames"] = classNames
    saveJson(str(outputDir / "config.json"), runtimeConfig)

    # Build Trainer
    trainer = Trainer(
        model=model,
        device=device,
        trainLoader=loaders["train"],
        valLoader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        classNames=classNames,
        outputDir=str(outputDir),
        topK=config.topK,
        saveBestMetric="valTop1",
        earlyStoppingPatience=config.earlyStoppingPatience,
        earlyStoppingMinDelta=config.earlyStoppingMinDelta,
    )

    # Handle Checkpoint Resume
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
            trainer.bestMetric = checkpoint.get("bestMetric", float("-inf"))
            print(f"[INFO] Restored state. Resuming from epoch {startEpoch}...")
        else:
            print(f"[INFO] --resume flag given but {lastCheckpointPath} not found. Starting fresh.")

    # Start Training
    trainer.fit(numEpochs=config.numEpochs, config=runtimeConfig, startEpoch=startEpoch)

if __name__ == "__main__":
    main()