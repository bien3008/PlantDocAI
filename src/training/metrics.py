from typing import Dict, Iterable, List

import torch


class AverageMeter:
    """
    Dùng để cộng dồn metric qua nhiều batch rồi tính trung bình cuối epoch.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def accuracyAtK(logits: torch.Tensor, targets: torch.Tensor, topK: int = 3) -> Dict[str, float]:
    """
    Tính top1 và topK accuracy cho 1 batch.

    Args:
        logits: Tensor shape [batchSize, numClasses]
        targets: Tensor shape [batchSize]
        topK: Giá trị K cho topK accuracy

    Returns:
        dict: {"top1": ..., "topK": ...}
    """
    with torch.no_grad():
        maxK = min(topK, logits.size(1))
        batchSize = targets.size(0)

        _, pred = logits.topk(maxK, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1Correct = correct[:1].reshape(-1).float().sum(0).item()
        topKCorrect = correct[:maxK].reshape(-1).float().sum(0).item()

        return {
            "top1": top1Correct / batchSize,
            "topK": topKCorrect / batchSize,
        }


def computePerClassAccuracy(
    allPreds: Iterable[int],
    allTargets: Iterable[int],
    classNames: List[str],
) -> Dict[str, float]:
    """
    Tính accuracy theo từng class.

    Args:
        allPreds: Danh sách prediction index
        allTargets: Danh sách target index
        classNames: Danh sách tên class

    Returns:
        dict: {className: accuracy}
    """
    perClassCorrect = {name: 0 for name in classNames}
    perClassTotal = {name: 0 for name in classNames}

    for pred, target in zip(allPreds, allTargets):
        className = classNames[target]
        perClassTotal[className] += 1
        if pred == target:
            perClassCorrect[className] += 1

    result = {}
    for className in classNames:
        total = perClassTotal[className]
        result[className] = perClassCorrect[className] / total if total > 0 else 0.0

    return result