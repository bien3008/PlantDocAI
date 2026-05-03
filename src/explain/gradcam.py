# src/explain/gradcam.py
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) — PlantDocAI.

Module trực quan hóa vùng ảnh mà mô hình tập trung khi đưa ra dự đoán.
Thiết kế tách biệt khỏi inference pipeline, predictor chỉ đóng vai trò gọi.

Hỗ trợ:
  - Mọi model timm đã tích hợp trong project (MobileNetV2, EfficientNet, …)
  - Tự động phát hiện target layer nếu không chỉ định
  - Output PIL.Image sẵn sàng cho Streamlit

Giới hạn (XAI Disclaimer):
  Grad-CAM là công cụ trực quan hóa hậu nghiệm (post-hoc visualization).
  Heatmap cho thấy vùng mô hình "chú ý" để ra quyết định, KHÔNG phải bằng
  chứng nhân quả rằng mô hình "hiểu" bệnh theo cách chuyên gia nông nghiệp.
  - Confidence cao không đồng nghĩa heatmap đáng tin tuyệt đối.
  - Với ảnh ngoài phân phối dữ liệu train, heatmap có thể gây hiểu lầm.
  - Kết quả nhạy với việc chọn target layer và preprocessing.

References:
  Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
  via Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Registry: target layer names theo kiến trúc timm
# ─────────────────────────────────────────────────────────────────────────────
# Mỗi entry là tên attribute trên model timm tương ứng với lớp conv cuối cùng
# trước global pooling.  Nếu cần thêm kiến trúc, chỉ cần bổ sung entry mới.

_TARGET_LAYER_REGISTRY: dict[str, str] = {
    "mobilenetv2":  "conv_head",       # MobileNetV2: 1×1 conv trước pooling
    "mobilenetv3":  "conv_head",       # MobileNetV3: tương tự v2
    "efficientnet": "conv_head",       # EfficientNet-B*: cùng kiến trúc conv_head
    "convnext":     "stages[-1]",      # ConvNeXt: stage cuối
    "resnet":       "layer4",          # ResNet variants
    "densenet":     "features",        # DenseNet
    "regnet":       "s4",              # RegNet: stage 4
}


def _resolveTargetLayer(model: torch.nn.Module, modelName: str = "") -> torch.nn.Module:
    """
    Tự động tìm target layer cho Grad-CAM dựa trên tên model.

    Chiến lược (theo thứ tự ưu tiên):
    1. Tra registry theo prefix modelName.
    2. Fallback: tìm Conv2d cuối cùng trong model (heuristic tổng quát).

    Args:
        model: Model PyTorch (timm).
        modelName: Tên model (e.g. 'mobilenetv2_100', 'efficientnet_b0').

    Returns:
        torch.nn.Module: Layer được chọn làm target cho Grad-CAM.

    Raises:
        RuntimeError: Nếu không tìm được layer phù hợp.
    """
    # 1) Tra registry theo prefix
    nameLower = modelName.lower()
    for prefix, attrPath in _TARGET_LAYER_REGISTRY.items():
        if prefix in nameLower:
            # Giải quyết path đơn giản (không chứa '[-1]' phức tạp)
            if "[-1]" not in attrPath and hasattr(model, attrPath):
                return getattr(model, attrPath)
            # Giải quyết path có index (ví dụ: stages[-1])
            parts = attrPath.replace("[-1]", "").strip()
            if hasattr(model, parts):
                container = getattr(model, parts)
                if hasattr(container, '__getitem__') and len(container) > 0:
                    return container[-1]

    # 2) Fallback: tìm Conv2d cuối cùng
    lastConv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            lastConv = module
    if lastConv is not None:
        return lastConv

    raise RuntimeError(
        f"Cannot auto-detect target layer for Grad-CAM on model '{modelName}'. "
        "Please provide target_layer explicitly."
    )


class GradCAM:
    """
    Grad-CAM cho model phân loại ảnh.

    Sử dụng PyTorch hooks để capture activations và gradients từ target layer,
    sau đó tính trọng số kênh và tạo heatmap.

    Usage::

        cam = GradCAM(model, model_name="mobilenetv2_100")
        heatmap = cam.generate(input_tensor, class_idx=predicted_class)
        overlay = cam.overlay_on_image(original_pil, heatmap)
        cam.close()  # cleanup hooks

    Hỗ trợ context manager::

        with GradCAM(model, model_name="mobilenetv2_100") as cam:
            heatmap = cam.generate(input_tensor)
            overlay = cam.overlay_on_image(original_pil, heatmap)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        model_name: str = "",
    ) -> None:
        """
        Khởi tạo GradCAM.

        Args:
            model: Model PyTorch đã load weights, ở eval mode.
            target_layer: Layer cụ thể để hook. Nếu None, tự động detect.
            device: Device của model. Nếu None, suy ra từ parameters.
            model_name: Tên model timm (để auto-detect target layer).
        """
        self.model = model
        self.model_name = model_name

        # Suy ra device
        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        # Xác định target layer
        if target_layer is not None:
            self._target_layer = target_layer
        else:
            self._target_layer = _resolveTargetLayer(model, model_name)

        # Storage cho hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Đăng ký hooks
        self._fwd_hook = self._target_layer.register_forward_hook(self._forwardHook)
        self._bwd_hook = self._target_layer.register_full_backward_hook(self._backwardHook)

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── Hooks ────────────────────────────────────────────────────────────────

    def _forwardHook(self, module, input, output) -> None:
        """Capture activations (feature maps) từ target layer."""
        self._activations = output.detach()

    def _backwardHook(self, module, grad_input, grad_output) -> None:
        """Capture gradients của output target layer."""
        self._gradients = grad_output[0].detach()

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Tạo Grad-CAM heatmap cho input tensor.

        Quy trình:
        1. Forward pass để lấy activations + logits
        2. Backward pass từ logit của target class
        3. Global average pooling gradients → trọng số kênh
        4. Weighted combination activations → CAM
        5. ReLU + normalize → heatmap [0, 1]

        Args:
            input_tensor: Tensor đã preprocessed, shape (1, C, H, W) hoặc (C, H, W).
                          PHẢI dùng cùng preprocessing với inference pipeline.
            class_idx: Index class cần giải thích. Nếu None, dùng predicted class (argmax).

        Returns:
            np.ndarray: Heatmap 2D, shape (H_feature, W_feature), giá trị [0, 1].
        """
        # Đảm bảo batch dimension
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Bật gradient tạm thời (model có thể ở eval mode)
        input_tensor.requires_grad_(True)

        # Forward
        self.model.zero_grad()
        logits = self.model(input_tensor)

        # Xác định target class
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # One-hot target rồi backward
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=False)

        # Gradient & Activations: (1, C, h, w)
        gradients = self._gradients    # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture data. "
                "Ensure the model performed a forward+backward pass through the target layer."
            )

        # Global average pooling trên spatial dims → trọng số mỗi kênh: (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination: sum(weight_c * activation_c) → (1, 1, h, w)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: chỉ giữ vùng có đóng góp dương
        cam = F.relu(cam)

        # Squeeze về 2D
        cam = cam.squeeze().cpu().numpy()

        # Normalize về [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay_on_image(
        self,
        pil_image: Image.Image,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = 2,  # cv2.COLORMAP_JET = 2
    ) -> Image.Image:
        """
        Overlay heatmap lên ảnh gốc, trả về PIL.Image RGB.

        Dùng matplotlib colormap thay vì OpenCV để giảm dependency.
        Output sẵn sàng cho Streamlit st.image().

        Args:
            pil_image: Ảnh gốc (PIL). Sẽ convert sang RGB.
            cam: Heatmap 2D [0, 1] từ generate().
            alpha: Độ trong suốt của heatmap (0=chỉ ảnh, 1=chỉ heatmap).
            colormap: Không sử dụng (giữ để tương thích API). Dùng matplotlib 'jet'.

        Returns:
            PIL.Image.Image: Ảnh overlay RGB, cùng kích thước với ảnh gốc.
        """
        # Đảm bảo RGB
        pil_image = pil_image.convert("RGB")
        img_w, img_h = pil_image.size

        # Resize heatmap về kích thước ảnh gốc
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = cam_resized.resize((img_w, img_h), Image.BILINEAR)
        cam_array = np.array(cam_resized).astype(np.float32) / 255.0

        # Áp dụng colormap (matplotlib 'jet')
        try:
            from matplotlib import cm
            colored = cm.jet(cam_array)[:, :, :3]  # (H, W, 3), float [0, 1]
        except ImportError:
            # Fallback: nếu không có matplotlib, tạo colormap đơn giản (red-heat)
            colored = np.stack([cam_array, np.zeros_like(cam_array), np.zeros_like(cam_array)], axis=-1)

        colored = (colored * 255).astype(np.uint8)

        # Overlay: blend ảnh gốc + heatmap
        img_array = np.array(pil_image).astype(np.float32)
        heatmap_array = colored.astype(np.float32)
        blended = (1.0 - alpha) * img_array + alpha * heatmap_array
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)

    def close(self) -> None:
        """
        Gỡ hooks và giải phóng tài nguyên.

        BẮT BUỘC gọi sau khi dùng xong để tránh memory leak,
        đặc biệt quan trọng khi chạy nhiều lần trong Streamlit app.
        """
        if self._fwd_hook is not None:
            self._fwd_hook.remove()
            self._fwd_hook = None
        if self._bwd_hook is not None:
            self._bwd_hook.remove()
            self._bwd_hook = None
        self._activations = None
        self._gradients = None
