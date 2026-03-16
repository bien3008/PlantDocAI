from typing import Any, Optional, Tuple

from torchvision.datasets import ImageFolder


class SafeImageFolder(ImageFolder): 

    def __getitem__(self, index: int) -> Optional[Tuple[Any, int]]:
        try:
            return super().__getitem__(index)
        except Exception as exc:
            path, _ = self.samples[index]
            print(f"[WARN] Skip corrupt image: {path} | error={exc}")
            return None