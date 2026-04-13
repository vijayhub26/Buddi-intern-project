import argparse
import os
import time
import sys

# Import our custom pipeline modules safely
from pipeline.post_corrector import PostCorrector
from evaluate_performance import compute_accuracy

def main():
    parser = argparse.ArgumentParser(description="Standalone LLM OCR Post-Corrector.")
    parser.add_argument("--input", "-i", required=True, help="Path to raw OCR .txt file.")
    parser.add_argument("--model", "-m", default="llama3.2:3b", help="Ollama model name to use.")
    parser.add_argument("--ground-truth", "-gt", default=None, help="Optional: Path to ground truth .txt file to measure accuracy.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        sys.exit(1)

    print(f"Loading '{args.input}'...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Starting standalone LLM Post-Corrector (Model: {args.model})...")
    print("This will process with 100% of your GPU assigned to the LLM.")
    
    corrector = PostCorrector(model=args.model, enabled=True)
    
    t0 = time.time()
    corrected_text = corrector.correct(text, verbose=True)
    elapsed = time.time() - t0

    out_path = args.input.replace(".txt", "_corrected.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corrected_text)

    print(f"\n✅ Done! Corrected text saved to: {out_path}")
    print(f"⏱  Total LLM inference time: {elapsed:.1f}s\n")
    
    if args.ground_truth:
        if not os.path.exists(args.ground_truth):
             print(f"Warning: Ground truth file '{args.ground_truth}' not found. Skipping accuracy check.")
             sys.exit(0)
             
        with open(args.ground_truth, "r", encoding="utf-8") as f:
            gt_text = f.read()
            
        print("Evaluating accuracy metrics...")
        
        # Calculate raw vs corrected metrics
        raw_acc = compute_accuracy(text, gt_text)
        corr_acc = compute_accuracy(corrected_text, gt_text)
        
        # Simple line counts
        gt_l = len(gt_text.splitlines())
        raw_l = len(text.splitlines())
        corr_l = len(corrected_text.splitlines())
        
        print("\n══════════════════════════════════════════════════════════════")
        print(" AI POST-CORRECTION METRICS")
        print("══════════════════════════════════════════════════════════════")
        print(f"                   Raw OCR          →    LLM Corrected")
        print(f"  Word Error (WER): {raw_acc['wer']*100:6.2f}%         →    {corr_acc['wer']*100:6.2f}%")
        print(f"  Char Error (CER): {raw_acc['cer']*100:6.2f}%         →    {corr_acc['cer']*100:6.2f}%")
        print(f"  Line Δ vs GT:      {raw_l - gt_l:+5d}             →    {corr_l - gt_l:+5d}")
        print("══════════════════════════════════════════════════════════════\n")

if __name__ == "__main__":
    main()
