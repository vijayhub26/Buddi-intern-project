import os
import sys

# Import core evaluating functions from your existing evaluate_performance
from evaluate_performance import run_pipeline_timed, compute_accuracy, compute_layout_metrics

datasets = [
    {"pdf": "samples/batch1-0001.pdf", "gt": "ground_truth/groundtruth_b1.txt"},
    {"pdf": "samples/batch2.pdf", "gt": "ground_truth/batch2.txt"},
    {"pdf": "samples/test_dataset.pdf", "gt": "ground_truth/groundtruth_t1.txt"}
]

total_wer = 0.0
total_cer = 0.0
valid_wer_count = 0

print("Running batch evaluation...")

for ds in datasets:
    pdf_path = ds["pdf"]
    gt_path = ds["gt"]
    
    print(f"\nEvaluating: {pdf_path}")
    
    # 1. Run pipeline
    result = run_pipeline_timed(
        pdf_path=pdf_path,
        dpi=300,
        min_confidence=0.0,
        pages=None,
        exclude_patterns=None,
        post_correct=False
    )
    
    # 2. Read Ground Truth
    gt_text = None
    if os.path.exists(gt_path):
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
    else:
        print(f"  Missing GT: {gt_path}")
        continue
        
    # 3. Compute metrics
    acc = compute_accuracy(gt_text, result["full_text"], ignore_symbols=True)
    
    if acc.get("error"):
        print(f"  Error calculating accuracy: {acc['error']}")
    else:
        wer_pct = acc['wer'] * 100
        cer_pct = acc['cer'] * 100
        total_wer += wer_pct
        total_cer += cer_pct
        valid_wer_count += 1
        
        print(f"  WER: {wer_pct:.2f}%")
        print(f"  CER: {cer_pct:.2f}%")
        
    layout = compute_layout_metrics(result["full_text"], ground_truth=gt_text)
    print(f"  Line Δ vs GT: {layout.get('line_count_delta', 'N/A')}")
    print(f"  Latency: {result['latency_s']:.2f}s")
    
print("\n" + "="*40)
print("AVERAGE METRICS")
print("="*40)
if valid_wer_count > 0:
    print(f"Average WER: {total_wer / valid_wer_count:.2f}%")
    print(f"Average CER: {total_cer / valid_wer_count:.2f}%")
else:
    print("Could not compute averages.")
