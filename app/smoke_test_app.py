"""Quick smoke test for the inference pipeline used by app.py."""
import sys
sys.path.insert(0, ".")

from src.evaluation.predictor import InferencePipeline
from PIL import Image

print("[1] Loading pipeline...")
p = InferencePipeline("artifacts/mobilenetV2_extended_artifacts", device="cpu")
print(f"    Model: {p.modelName}, Classes: {p.numClasses}")

print("[2] Testing predict...")
r = p.predict("imageTest/Tomato bacterial_spot.png", topK=3)
for x in r:
    print(f"    {x['className']}: {x['confidence']*100:.1f}%")

print("[3] Testing explainFromPil...")
img = Image.open("imageTest/Tomato bacterial_spot.png").convert("RGB")
result = p.explainFromPil(img, topK=3)
print(f"    Predictions: {len(result['predictions'])}")
print(f"    Overlay type: {type(result['gradcamOverlay'])}")
print(f"    Overlay size: {result['gradcamOverlay'].size}")
print(f"    Target class idx: {result['targetClassIdx']}")

print("[4] Testing RGBA image...")
rgba = Image.open("imageTest/pic1.png")
print(f"    Mode: {rgba.mode}")
r2 = p.predictFromPil(rgba, topK=1)
print(f"    Result: {r2[0]['className']}: {r2[0]['confidence']*100:.1f}%")

print("\n[OK] All tests passed.")
