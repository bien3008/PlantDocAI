# src/evaluation/predictor.py
"""
Inference Pipeline — PlantDocAI.

Module này đóng vai trò cầu nối giữa model đã train và ứng dụng thực tế.
Tách biệt hoàn toàn logic suy luận khỏi notebook/script, cung cấp API ổn định
để Streamlit app hoặc bất kỳ consumer nào có thể gọi.

Thiết kế:
  - Tái sử dụng buildModel() từ modelFactory
  - Tái sử dụng loadCheckpoint() từ checkpoint module
  - Tái sử dụng buildInferenceTransform() từ dataTransforms (single source of truth)
  - Class names đọc từ config.json artifact (được tạo khi train)
  - Output chuẩn hóa: classId, className, confidence (post-softmax)
"""

import json
import ast
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image

from src.models.modelFactory import buildModel
from src.training.checkpoint import loadCheckpoint
from src.data.dataTransforms import buildInferenceTransform


class InferencePipeline:
    """
    Pipeline suy luận tái sử dụng cho PlantDocAI.

    Đóng gói toàn bộ: load config → build model → load weights → preprocess → predict.
    App chỉ cần khởi tạo một lần, rồi gọi predict() hoặc predictFromPil() nhiều lần.

    Usage::

        pipeline = InferencePipeline(modelDir="artifacts/mobilenetV2_colab_artifacts")

        # Predict từ file path
        results = pipeline.predict("path/to/leaf.jpg", topK=3)

        # Predict từ PIL Image (Streamlit)
        results = pipeline.predictFromPil(pil_image, topK=3)

        # Top-1 tiện lợi
        result = pipeline.predictTop1FromPil(pil_image)
    """

    def __init__(self, modelDir: Union[str, Path], device: str = "auto") -> None:
        """
        Khởi tạo InferencePipeline.

        Args:
            modelDir: Thư mục chứa ``config.json`` và checkpoint
                      (e.g. ``artifacts/mobilenetV2_colab_artifacts``).
            device: Device chạy inference ('cpu', 'cuda', hoặc 'auto').
        """
        self.modelDir = Path(modelDir)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._loadConfig()
        self._initModel()
        self._initTransforms()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization helpers (private)
    # ─────────────────────────────────────────────────────────────────────────

    def _loadConfig(self) -> None:
        """Load config.json artifact được tạo khi train."""
        configPath = self.modelDir / "config.json"
        if not configPath.exists():
            raise FileNotFoundError(f"Cannot find config at {configPath}")

        with open(configPath, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Parse class names — có thể là list hoặc string representation của list
        classNames = self.config.get("classNames", [])
        if isinstance(classNames, str):
            classNames = ast.literal_eval(classNames)
        self.classNames = classNames
        self.numClasses = self.config.get("numClasses", len(self.classNames))

        # Model params
        self.modelName = self.config.get("modelName", "mobilenetv2_100")
        self.imageSize = self.config.get("imageSize", 224)

    def _initModel(self) -> None:
        """Build model via timm factory và load checkpoint weights."""
        # Tìm checkpoint theo thứ tự ưu tiên
        weightsPath = None
        for p in ["checkpoints/best.pt", "best.pt", "checkpoints/last.pt"]:
            if (self.modelDir / p).exists():
                weightsPath = self.modelDir / p
                break

        if weightsPath is None:
            raise FileNotFoundError(
                f"Cannot find checkpoint (e.g., best.pt) in {self.modelDir}"
            )

        # Build model qua modelFactory — usePretrained=False vì sẽ load weights từ checkpoint
        self.model = buildModel(
            modelName=self.modelName,
            numClasses=self.numClasses,
            usePretrained=False,
            freezeBackbone=False,
        )

        # Load weights từ checkpoint (dict format: { modelStateDict, ... })
        loadCheckpoint(
            checkpointPath=str(weightsPath),
            model=self.model,
            mapLocation=str(self.device),
        )

        self.model.to(self.device)
        self.model.eval()

    def _initTransforms(self) -> None:
        """
        Build preprocessing transform cho inference.

        Dùng buildInferenceTransform() từ dataTransforms.py — đây là single source of truth,
        đảm bảo preprocessing khớp chính xác với eval pipeline dùng khi train:
        Resize(inputSize × 1.14) → CenterCrop(inputSize) → ToTensor → Normalize(ImageNet).
        """
        self.transform = buildInferenceTransform(self.imageSize)

    # ─────────────────────────────────────────────────────────────────────────
    # Public prediction API
    # ─────────────────────────────────────────────────────────────────────────

    def predictFromPil(
        self, image: Image.Image, topK: int = 3
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Predict trên một PIL Image.

        Đây là method cốt lõi — tất cả các predict methods khác đều delegate về đây.
        Phù hợp nhất cho Streamlit (nhận ảnh upload trực tiếp dưới dạng PIL).

        Args:
            image: PIL Image (sẽ được convert sang RGB nếu cần).
            topK: Số lượng dự đoán top-k trả về (mặc định 3).

        Returns:
            List[Dict]: Danh sách top-k predictions, sắp xếp giảm dần theo confidence.
                Mỗi item gồm: ``classId`` (int), ``className`` (str), ``confidence`` (float).
        """
        # Đảm bảo RGB — ảnh có thể là RGBA, L, P,...
        image = image.convert("RGB")

        # Preprocess: Resize → CenterCrop → ToTensor → Normalize
        tensorImg = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensorImg)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        # Top-k results, đã sắp xếp giảm dần theo confidence
        actualK = min(topK, self.numClasses)
        topProb, topClassIdx = torch.topk(probabilities, k=actualK)

        results = []
        for i in range(actualK):
            idx = topClassIdx[i].item()
            prob = topProb[i].item()
            name = self.classNames[idx] if idx < len(self.classNames) else f"Class_{idx}"
            results.append({
                "classId": idx,
                "className": name,
                "confidence": prob,
            })

        return results

    def predict(
        self, imagePath: Union[str, Path], topK: int = 3
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Predict trên một ảnh từ đường dẫn file.

        Args:
            imagePath: Đường dẫn tới file ảnh.
            topK: Số lượng dự đoán top-k trả về.

        Returns:
            List[Dict]: Danh sách top-k predictions (xem ``predictFromPil``).
        """
        try:
            with Image.open(imagePath) as img:
                # Copy ra ngoài context manager để tránh file handle issue
                img = img.copy()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load image for prediction at {imagePath}. Error: {e}"
            )

        return self.predictFromPil(img, topK=topK)

    def predictTop1FromPil(
        self, image: Image.Image
    ) -> Dict[str, Union[str, int, float]]:
        """
        Predict top-1 trên PIL Image. Trả về single dict thay vì list.

        Args:
            image: PIL Image.

        Returns:
            Dict: ``{classId, className, confidence}``
        """
        return self.predictFromPil(image, topK=1)[0]

    def predictTop1(
        self, imagePath: Union[str, Path]
    ) -> Dict[str, Union[str, int, float]]:
        """
        Predict top-1 từ đường dẫn file. Trả về single dict thay vì list.

        Args:
            imagePath: Đường dẫn tới file ảnh.

        Returns:
            Dict: ``{classId, className, confidence}``
        """
        return self.predict(imagePath, topK=1)[0]
