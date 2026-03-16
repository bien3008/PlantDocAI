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


def parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PlantDoc AI baseline model")
    parser.add_argument("--dataDir", type=str, default="data/Plantvillage", help="Root dir containing images")
    parser.add_argument("--splitDir", type=str, default="data/splits", help="Dir containing train.csv, val.csv, test.csv")
    parser.add_argument("--outputDir", type=str, default="artifacts/mobilenetV2Baseline", help="Output directory")
    parser.add_argument("--modelName", type=str, default="mobilenetV2", choices=["mobilenetV2", "efficientnetB0"])
    parser.add_argument("--imageSize", type=int, default=224)
    parser.add_argument("--batchSize", type=int, default=32)
    parser.add_argument("--numEpochs", type=int, default=5)
    parser.add_argument("--learningRate", type=float, default=1e-3)
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--numWorkers", type=int, default=2)
    parser.add_argument("--topK", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--freezeBackbone", action="store_true", help="Freeze backbone and train classifier head only")
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
    device = resolveDevice(args.device)

    loaders, classToId = buildDataLoaders(
        dataDir=args.dataDir,
        splitDir=args.splitDir,
        inputSize=args.imageSize,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
    )

    trainLoader = loaders["train"]
    valLoader = loaders["val"]

    idToClass = {v: k for k, v in classToId.items()}
    classNames = [idToClass[i] for i in range(len(classToId))]
    numClasses = len(classNames)

    model = buildModel(
        modelName=args.modelName,
        numClasses=numClasses,
        usePretrained=True,
        freezeBackbone=args.freezeBackbone,
    ).to(device)

    totalParams, trainableParams = countParameters(model)
    trainableRatio = (trainableParams / totalParams) if totalParams > 0 else 0.0

    print(f"[INFO] modelName={args.modelName}")
    print(f"[INFO] freezeBackbone={args.freezeBackbone}")
    print(f"[INFO] totalParams={totalParams:,}")
    print(f"[INFO] trainableParams={trainableParams:,}")
    print(f"[INFO] trainableRatio={trainableRatio:.4f}")

    criterion = nn.CrossEntropyLoss()

    trainableModelParams = [param for param in model.parameters() if param.requires_grad]
    if len(trainableModelParams) == 0:
        raise RuntimeError("No trainable parameters found. Check freezeBackbone logic.")

    optimizer = torch.optim.Adam(
        trainableModelParams,
        lr=args.learningRate,
        weight_decay=args.weightDecay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    outputDir = Path(args.outputDir)
    outputDir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataDir": args.dataDir,
        "splitDir": args.splitDir,
        "outputDir": args.outputDir,
        "modelName": args.modelName,
        "imageSize": args.imageSize,
        "batchSize": args.batchSize,
        "numEpochs": args.numEpochs,
        "learningRate": args.learningRate,
        "weightDecay": args.weightDecay,
        "numWorkers": args.numWorkers,
        "topK": args.topK,
        "device": str(device),
        "numClasses": numClasses,
        "classNames": classNames,
        "freezeBackbone": args.freezeBackbone,
    }

    saveJson(str(outputDir / "config.json"), config)

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
        topK=args.topK,
        saveBestMetric="valTop1",
    )

    trainer.fit(numEpochs=args.numEpochs, config=config)


if __name__ == "__main__":
    main()