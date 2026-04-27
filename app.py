# app.py
"""
PlantDocAI — Streamlit Demo App.

Demo phân loại bệnh lá cây sử dụng mô hình đã train.
Tái sử dụng hoàn toàn InferencePipeline từ src/evaluation/predictor.py.

Chạy: streamlit run app.py
"""

import sys
from pathlib import Path

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
MIN_IMAGE_SIZE = 32
MAX_FILE_SIZE_MB = 10

# ─────────────────────────────────────────────────────────────────────────────
# Disease Recommendations
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATIONS = {
    "Apple___Apple_scab": "Bệnh ghẻ táo — Cần kiểm tra và loại bỏ lá bị nhiễm. Tham khảo chuyên gia nông nghiệp về biện pháp phòng trừ phù hợp.",
    "Apple___Black_rot": "Bệnh thối đen táo — Loại bỏ quả và cành bị nhiễm. Đảm bảo vệ sinh vườn tốt.",
    "Apple___Cedar_apple_rust": "Bệnh rỉ sắt táo — Tránh trồng gần cây tuyết tùng. Tham khảo chuyên gia về thuốc bảo vệ thực vật phù hợp.",
    "Apple___healthy": "Lá táo khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Blueberry___healthy": "Lá việt quất khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Cherry_(including_sour)___Powdery_mildew": "Bệnh phấn trắng anh đào — Đảm bảo thông thoáng gió. Tham khảo chuyên gia về biện pháp xử lý.",
    "Cherry_(including_sour)___healthy": "Lá anh đào khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Bệnh đốm xám lá ngô — Luân canh cây trồng và sử dụng giống kháng bệnh nếu có.",
    "Corn_(maize)___Common_rust_": "Bệnh rỉ sắt ngô — Theo dõi mức độ lây lan. Tham khảo chuyên gia về giống kháng bệnh.",
    "Corn_(maize)___Northern_Leaf_Blight": "Bệnh cháy lá phía bắc — Sử dụng giống kháng bệnh và luân canh cây trồng.",
    "Corn_(maize)___healthy": "Lá ngô khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Grape___Black_rot": "Bệnh thối đen nho — Loại bỏ quả và lá bị nhiễm. Đảm bảo vệ sinh vườn.",
    "Grape___Esca_(Black_Measles)": "Bệnh Esca nho — Bệnh phức tạp, cần tham khảo chuyên gia nông nghiệp.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Bệnh cháy lá nho — Loại bỏ lá bị nhiễm và cải thiện thông gió.",
    "Grape___healthy": "Lá nho khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Orange___Haunglongbing_(Citrus_greening)": "Bệnh vàng lá gân xanh cam — Bệnh nghiêm trọng, cần báo cáo cơ quan nông nghiệp địa phương.",
    "Peach___Bacterial_spot": "Bệnh đốm vi khuẩn đào — Sử dụng giống kháng bệnh. Tham khảo chuyên gia về biện pháp phòng trừ.",
    "Peach___healthy": "Lá đào khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Pepper,_bell___Bacterial_spot": "Bệnh đốm vi khuẩn ớt chuông — Sử dụng hạt giống sạch bệnh và luân canh cây trồng.",
    "Pepper,_bell___healthy": "Lá ớt chuông khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Potato___Early_blight": "Bệnh cháy sớm khoai tây — Luân canh cây trồng và loại bỏ tàn dư thực vật.",
    "Potato___Late_blight": "Bệnh mốc sương khoai tây — Bệnh nguy hiểm, cần xử lý kịp thời. Tham khảo chuyên gia.",
    "Potato___healthy": "Lá khoai tây khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Raspberry___healthy": "Lá mâm xôi khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Soybean___healthy": "Lá đậu nành khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Squash___Powdery_mildew": "Bệnh phấn trắng bí — Cải thiện thông gió và tránh tưới lên lá.",
    "Strawberry___Leaf_scorch": "Bệnh cháy lá dâu tây — Loại bỏ lá bị nhiễm và đảm bảo thoát nước tốt.",
    "Strawberry___healthy": "Lá dâu tây khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
    "Tomato___Bacterial_spot": "Bệnh đốm vi khuẩn cà chua — Sử dụng hạt giống sạch bệnh và tránh tưới trên lá.",
    "Tomato___Early_blight": "Bệnh cháy sớm cà chua — Luân canh cây trồng và loại bỏ tàn dư thực vật.",
    "Tomato___Late_blight": "Bệnh mốc sương cà chua — Bệnh nguy hiểm, cần xử lý kịp thời.",
    "Tomato___Leaf_Mold": "Bệnh mốc lá cà chua — Cải thiện thông gió trong nhà kính.",
    "Tomato___Septoria_leaf_spot": "Bệnh đốm lá Septoria cà chua — Loại bỏ lá bị nhiễm từ dưới lên.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Nhện đỏ hai chấm trên cà chua — Kiểm tra mặt dưới lá. Tham khảo biện pháp sinh học.",
    "Tomato___Target_Spot": "Bệnh đốm vòng cà chua — Luân canh và cải thiện thoát nước.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Virus xoăn vàng lá cà chua — Kiểm soát bọ phấn trắng, sử dụng giống kháng.",
    "Tomato___Tomato_mosaic_virus": "Virus khảm cà chua — Vệ sinh dụng cụ, loại bỏ cây bệnh.",
    "Tomato___healthy": "Lá cà chua khỏe mạnh — Tiếp tục chăm sóc và theo dõi định kỳ.",
}


def _formatClassName(rawName: str) -> str:
    """Chuyển 'Plant___Condition' thành 'Plant — Condition' dễ đọc."""
    parts = rawName.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ").strip()
        return f"{plant} — {condition}"
    return rawName.replace("_", " ")


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def loadPipeline(modelDir: str):
    """Load InferencePipeline — cached, chỉ chạy 1 lần."""
    from src.evaluation.predictor import InferencePipeline
    return InferencePipeline(modelDir=modelDir, device="cpu")


def _discoverArtifactDirs() -> list:
    """Tìm tất cả artifact dirs có config.json + checkpoint."""
    dirs = []
    if not ARTIFACTS_DIR.exists():
        return dirs
    for d in sorted(ARTIFACTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        hasConfig = (d / "config.json").exists()
        hasCheckpoint = any([
            (d / "checkpoints" / "best.pt").exists(),
            (d / "best.pt").exists(),
            (d / "checkpoints" / "last.pt").exists(),
        ])
        if hasConfig and hasCheckpoint:
            dirs.append(d.name)
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Image Validation
# ─────────────────────────────────────────────────────────────────────────────

def validateImage(uploadedFile) -> tuple:
    """
    Validate ảnh upload. Returns (image: PIL.Image | None, error: str | None).
    """
    if uploadedFile is None:
        return None, "Chưa có ảnh nào được tải lên."

    filename = uploadedFile.name.lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return None, f"Định dạng file không được hỗ trợ (.{ext}). Vui lòng dùng: {', '.join(ALLOWED_EXTENSIONS)}."

    fileSize = uploadedFile.size
    if fileSize > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, f"File quá lớn ({fileSize / 1024 / 1024:.1f} MB). Giới hạn: {MAX_FILE_SIZE_MB} MB."

    try:
        image = Image.open(uploadedFile)
        image.load()
    except Exception:
        return None, "Không thể đọc file ảnh. File có thể bị hỏng hoặc không phải định dạng ảnh hợp lệ."

    w, h = image.size
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return None, f"Ảnh quá nhỏ ({w}×{h} px). Kích thước tối thiểu: {MIN_IMAGE_SIZE}×{MIN_IMAGE_SIZE} px."

    try:
        image = image.convert("RGB")
    except Exception:
        return None, "Không thể chuyển đổi ảnh sang RGB. Vui lòng thử ảnh khác."

    return image, None


# ─────────────────────────────────────────────────────────────────────────────
# Inference (with session_state caching to survive reruns)
# ─────────────────────────────────────────────────────────────────────────────

def _runAnalysis(pipeline, image, topK, showGradCAM, gradcamAlpha):
    """Chạy inference và lưu kết quả vào session_state."""
    gradcamOverlay = None
    predictions = None
    gradcamError = False

    try:
        if showGradCAM:
            result = pipeline.explainFromPil(image, topK=topK, alpha=gradcamAlpha)
            predictions = result["predictions"]
            gradcamOverlay = result["gradcamOverlay"]
        else:
            predictions = pipeline.predictFromPil(image, topK=topK)
    except Exception:
        if showGradCAM:
            gradcamError = True
            try:
                predictions = pipeline.predictFromPil(image, topK=topK)
            except Exception as e2:
                return None, None, str(e2)
        else:
            import traceback
            return None, None, traceback.format_exc()

    return predictions, gradcamOverlay, "gradcam_fallback" if gradcamError else None


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PlantDocAI — Phân loại bệnh lá cây",
        page_icon="🌿",
        layout="wide",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Cài đặt")

        artifactDirs = _discoverArtifactDirs()
        if not artifactDirs:
            st.error("Không tìm thấy artifact nào trong thư mục `artifacts/`.")
            st.stop()

        defaultIdx = 0
        for i, name in enumerate(artifactDirs):
            if "extended" in name.lower():
                defaultIdx = i
                break

        selectedDir = st.selectbox(
            "Mô hình", artifactDirs, index=defaultIdx,
            help="Chọn artifact directory chứa model checkpoint.",
        )
        topK = st.slider("Số lượng dự đoán (Top-K)", 1, 10, 3)
        showGradCAM = st.checkbox("Hiển thị Grad-CAM", value=True,
                                   help="Trực quan hóa vùng ảnh mô hình tập trung.")
        gradcamAlpha = st.slider("Độ đậm Grad-CAM", 0.2, 0.8, 0.5, 0.05,
                                  disabled=not showGradCAM)

        st.divider()
        st.caption(
            "⚠️ **Lưu ý:** Đây là công cụ hỗ trợ nghiên cứu, "
            "không thay thế ý kiến chuyên gia nông nghiệp. "
            "Mô hình được train trên tập dữ liệu PlantVillage (38 loại)."
        )

    # ── Header ────────────────────────────────────────────────────────────
    st.title("🌿 PlantDocAI")
    st.markdown("**Hệ thống nhận diện bệnh lá cây bằng Deep Learning** — "
                "Upload ảnh lá để nhận kết quả phân tích.")

    # ── Load model ────────────────────────────────────────────────────────
    modelDirPath = str(ARTIFACTS_DIR / selectedDir)
    try:
        with st.spinner("Đang tải mô hình..."):
            pipeline = loadPipeline(modelDirPath)
    except FileNotFoundError as e:
        st.error(f"Không tìm thấy file cần thiết: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

    # ── Upload ────────────────────────────────────────────────────────────
    st.subheader("📤 Tải ảnh lên")
    uploadedFile = st.file_uploader(
        "Chọn ảnh lá cây (JPG, PNG, WebP)",
        type=list(ALLOWED_EXTENSIONS),
        help="Chọn ảnh chụp rõ nét lá cây cần phân tích.",
    )

    if uploadedFile is None:
        # Reset results khi không có ảnh
        st.session_state.pop("results", None)
        st.info("👆 Vui lòng tải lên ảnh lá cây để bắt đầu phân tích.")
        st.stop()

    # ── Validate ──────────────────────────────────────────────────────────
    image, error = validateImage(uploadedFile)
    if error:
        st.error(f"❌ {error}")
        st.stop()

    # ── Preview ───────────────────────────────────────────────────────────
    st.subheader("🖼️ Ảnh tải lên")
    st.image(image, caption=f"{uploadedFile.name} ({image.size[0]}×{image.size[1]} px)",
             width=350)

    # ── Analyze ───────────────────────────────────────────────────────────
    analyzeClicked = st.button("🔍 Phân tích", type="primary",
                                use_container_width=True)

    if analyzeClicked:
        with st.spinner("Đang phân tích ảnh..."):
            predictions, gradcamOverlay, errorMsg = _runAnalysis(
                pipeline, image, topK, showGradCAM, gradcamAlpha
            )
        if errorMsg and errorMsg != "gradcam_fallback":
            st.error(f"❌ Lỗi khi phân tích ảnh. Vui lòng thử lại hoặc dùng ảnh khác.")
            st.stop()

        # Lưu kết quả vào session_state để survive reruns
        st.session_state["results"] = {
            "predictions": predictions,
            "gradcamOverlay": gradcamOverlay,
            "image": image,
            "filename": uploadedFile.name,
            "gradcamFallback": errorMsg == "gradcam_fallback",
            "showGradCAM": showGradCAM,
        }

    # ── Display results (from session_state) ──────────────────────────────
    if "results" not in st.session_state:
        st.stop()

    res = st.session_state["results"]
    predictions = res["predictions"]
    gradcamOverlay = res["gradcamOverlay"]

    if not predictions:
        st.stop()

    st.divider()
    st.subheader("📊 Kết quả phân tích")

    # ── Grad-CAM side-by-side ─────────────────────────────────────────────
    if gradcamOverlay is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", width="stretch")
        with col2:
            st.image(gradcamOverlay,
                     caption="Grad-CAM — Vùng mô hình tập trung",
                     width="stretch")
        st.caption(
            "ℹ️ *Grad-CAM cho thấy vùng ảnh mô hình \"chú ý\" khi ra quyết định. "
            "Vùng đỏ/vàng = chú ý cao, vùng xanh = chú ý thấp. "
            "Đây là trực quan hóa hậu nghiệm, không phải bằng chứng nhân quả.*"
        )
    elif res.get("gradcamFallback"):
        st.warning("⚠️ Grad-CAM gặp lỗi, hiển thị kết quả dự đoán không có heatmap.")
    elif res.get("showGradCAM"):
        st.info("ℹ️ Grad-CAM không khả dụng cho lần phân tích này.")

    # ── Top-1 Prediction ──────────────────────────────────────────────────
    top1 = predictions[0]
    confidence_pct = top1["confidence"] * 100
    displayName = _formatClassName(top1["className"])

    if confidence_pct >= 80:
        confColor = "green"
    elif confidence_pct >= 50:
        confColor = "orange"
    else:
        confColor = "red"

    st.markdown(f"""
### 🏷️ Dự đoán chính

| | |
|---|---|
| **Kết quả** | **{displayName}** |
| **Độ tin cậy** | :{confColor}[**{confidence_pct:.1f}%**] |
""")

    if confidence_pct < 50:
        st.warning("⚠️ Độ tin cậy thấp — kết quả có thể không chính xác. "
                   "Hãy thử chụp ảnh rõ hơn hoặc tham khảo chuyên gia.")

    # ── Top-K Results ─────────────────────────────────────────────────────
    if len(predictions) > 1:
        st.markdown("#### Các dự đoán (Top-K)")
        for i, pred in enumerate(predictions):
            conf = pred["confidence"] * 100
            name = _formatClassName(pred["className"])
            prefix = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"{i+1}."))
            st.progress(pred["confidence"], text=f"{prefix} {name} — {conf:.1f}%")

    # ── Recommendation ────────────────────────────────────────────────────
    recommendation = RECOMMENDATIONS.get(top1["className"])
    if recommendation:
        st.divider()
        st.subheader("💡 Khuyến nghị")
        isHealthy = "healthy" in top1["className"].lower()
        if isHealthy:
            st.success(f"✅ {recommendation}")
        else:
            st.warning(f"🔍 {recommendation}")
        st.caption("*Khuyến nghị mang tính tham khảo. Luôn tham vấn chuyên gia "
                   "nông nghiệp trước khi áp dụng biện pháp xử lý.*")

    # ── Footer ────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("ℹ️ Về mô hình và giới hạn"):
        st.markdown(f"""
- **Kiến trúc:** MobileNetV2 (timm)
- **Tập dữ liệu:** PlantVillage Extended ({pipeline.numClasses} lớp)
- **Tiền xử lý:** Resize → CenterCrop(224) → Normalize(ImageNet)
- **Giới hạn:**
  - Chỉ nhận diện {pipeline.numClasses} loại bệnh/trạng thái đã được train
  - Ảnh ngoài phân phối dữ liệu train có thể cho kết quả sai
  - Không thay thế chẩn đoán chuyên gia
  - Grad-CAM là trực quan hóa hậu nghiệm, không phải bằng chứng nhân quả
""")


if __name__ == "__main__":
    main()
