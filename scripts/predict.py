import argparse
from pathlib import Path
import sys

# Ensure the root directory is in sys.path so we can import 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.evaluation.predictor import InferencePipeline

def parseArgs():
    parser = argparse.ArgumentParser(description="Inference using a trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to predict")
    parser.add_argument("--modelDir", type=str, required=True, help="Directory containing config.json and checkpoint (e.g. checkpoints/best.pt)")
    parser.add_argument("--topK", type=int, default=3, help="Number of top predictions to display")
    return parser.parse_args()

def main():
    args = parseArgs()
    
    print(f"[INFO] Initializing Inference Pipeline from {args.modelDir}...")
    try:
        pipeline = InferencePipeline(modelDir=args.modelDir)
    except Exception as e:
        print(f"[ERROR] Pipeline initialization failed: {e}")
        sys.exit(1)

    print(f"[INFO] Predicting on {args.image}...")
    try:
        results = pipeline.predict(imagePath=args.image, topK=args.topK)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        sys.exit(1)

    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    for i, result in enumerate(results):
        className = result["className"]
        prob = result["probability"]
        print(f"{i+1}. {className}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
