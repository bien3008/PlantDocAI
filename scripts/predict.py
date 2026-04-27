import argparse
from pathlib import Path
import sys

# Ensure the root directory is in sys.path so we can import 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.predictor import InferencePipeline

def parseArgs():
    parser = argparse.ArgumentParser(description="Inference using a trained model with optional Grad-CAM explainability")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to predict")
    parser.add_argument("--modelDir", type=str, required=True, help="Directory containing config.json and checkpoint")
    parser.add_argument("--topK", type=int, default=3, help="Number of top predictions to display")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM overlay")
    parser.add_argument("--output", type=str, default="prediction_explain.png", help="Path to save Grad-CAM overlay if --explain is used")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap transparency (0.0 to 1.0)")
    return parser.parse_args()

def main():
    args = parseArgs()
    
    print(f"[INFO] Initializing Inference Pipeline from {args.modelDir}...")
    try:
        pipeline = InferencePipeline(modelDir=args.modelDir)
    except Exception as e:
        print(f"[ERROR] Pipeline initialization failed: {e}")
        sys.exit(1)

    print(f"[INFO] Processing {args.image}...")
    
    try:
        if args.explain:
            # Chạy kết hợp dự đoán và giải thích
            result = pipeline.explain(imagePath=args.image, topK=args.topK, alpha=args.alpha)
            predictions = result["predictions"]
            overlay = result["gradcamOverlay"]
            
            # Lưu ảnh overlay
            overlay.save(args.output)
            print(f"[INFO] Grad-CAM overlay saved to: {args.output}")
        else:
            # Chạy dự đoán thuần túy
            predictions = pipeline.predict(imagePath=args.image, topK=args.topK)
            
    except Exception as e:
        print(f"[ERROR] Operation failed: {e}")
        sys.exit(1)

    print("\n--- PREDICTION RESULTS ---")
    for i, result in enumerate(predictions):
        classId = result["classId"]
        className = result["className"]
        confidence = result["confidence"]
        print(f"{i+1}. [{classId:>2d}] {className}: {confidence * 100:.2f}%")

if __name__ == "__main__":
    main()
