import csv
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.training.checkpoint import saveCheckpoint
from src.training.metrics import AverageMeter, accuracyAtK


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        trainLoader: DataLoader,
        valLoader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        classNames: list[str],
        outputDir: str,
        topK: int = 3,
        saveBestMetric: str = "valTop1",
    ):
        self.model = model
        self.device = device
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.classNames = classNames
        self.outputDir = Path(outputDir)
        self.topK = topK
        self.saveBestMetric = saveBestMetric

        self.checkpointDir = self.outputDir / "checkpoints"
        self.logDir = self.outputDir / "logs"
        self.checkpointDir.mkdir(parents=True, exist_ok=True)
        self.logDir.mkdir(parents=True, exist_ok=True)

        self.csvLogPath = self.logDir / "trainMetrics.csv"
        self.bestMetric = float("-inf")

        self._initCsvLogger()

    def _initCsvLogger(self):
        if not self.csvLogPath.exists():
            with open(self.csvLogPath, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "epoch",
                    "trainLoss",
                    "trainTop1",
                    "trainTopK",
                    "valLoss",
                    "valTop1",
                    "valTopK",
                    "lr",
                    "epochSeconds",
                ])

    def _writeCsvRow(self, rowData: Dict[str, Any]):
        with open(self.csvLogPath, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                rowData["epoch"],
                f"{rowData['trainLoss']:.6f}",
                f"{rowData['trainTop1']:.6f}",
                f"{rowData['trainTopK']:.6f}",
                f"{rowData['valLoss']:.6f}",
                f"{rowData['valTop1']:.6f}",
                f"{rowData['valTopK']:.6f}",
                f"{rowData['lr']:.8f}",
                f"{rowData['epochSeconds']:.2f}",
            ])

    def _runOneEpoch(self, loader: DataLoader, training: bool):
        if training:
            self.model.train()
        else:
            self.model.eval()

        lossMeter = AverageMeter()
        top1Meter = AverageMeter()
        topKMeter = AverageMeter()

        validBatchCount = 0

        for batch in loader:
            if batch is None:
                continue

            images, targets = batch
            if images.size(0) == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(training):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            batchMetrics = accuracyAtK(logits, targets, topK=self.topK)
            batchSize = images.size(0)

            lossMeter.update(loss.item(), batchSize)
            top1Meter.update(batchMetrics["top1"], batchSize)
            topKMeter.update(batchMetrics["topK"], batchSize)
            validBatchCount += 1

        if validBatchCount == 0:
            raise RuntimeError(
                "No valid batch found. Check dataset path, corrupt images, or collateFn filtering."
            )

        return {
            "loss": lossMeter.avg,
            "top1": top1Meter.avg,
            "topK": topKMeter.avg,
        }

    def fit(self, numEpochs: int, config: Dict[str, Any], startEpoch: int = 1):
        print("[INFO] Start training loop")

        for epoch in range(startEpoch, numEpochs + 1):
            startTime = time.time()

            trainMetrics = self._runOneEpoch(self.trainLoader, training=True)
            valMetrics = self._runOneEpoch(self.valLoader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            currentLr = self.optimizer.param_groups[0]["lr"]
            epochSeconds = time.time() - startTime

            rowData = {
                "epoch": epoch,
                "trainLoss": trainMetrics["loss"],
                "trainTop1": trainMetrics["top1"],
                "trainTopK": trainMetrics["topK"],
                "valLoss": valMetrics["loss"],
                "valTop1": valMetrics["top1"],
                "valTopK": valMetrics["topK"],
                "lr": currentLr,
                "epochSeconds": epochSeconds,
            }
            self._writeCsvRow(rowData)

            print(
                f"[Epoch {epoch}/{numEpochs}] "
                f"trainLoss={trainMetrics['loss']:.4f} "
                f"trainTop1={trainMetrics['top1']:.4f} "
                f"trainTop{self.topK}={trainMetrics['topK']:.4f} | "
                f"valLoss={valMetrics['loss']:.4f} "
                f"valTop1={valMetrics['top1']:.4f} "
                f"valTop{self.topK}={valMetrics['topK']:.4f} | "
                f"lr={currentLr:.8f}"
            )

            metricValue = rowData[self.saveBestMetric]
            if metricValue > self.bestMetric:
                self.bestMetric = metricValue
                saveCheckpoint(
                    savePath=str(self.checkpointDir / "best.pt"),
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    bestMetric=self.bestMetric,
                    classNames=self.classNames,
                    config=config,
                )
                print(f"[INFO] Saved new best checkpoint: {self.checkpointDir / 'best.pt'}")

            saveCheckpoint(
                savePath=str(self.checkpointDir / "last.pt"),
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                bestMetric=self.bestMetric,
                classNames=self.classNames,
                config=config,
            )