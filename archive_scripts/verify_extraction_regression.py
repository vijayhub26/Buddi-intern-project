import os
import sys

def verify_extraction(text_file):
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found.")
        return False

    with open(text_file, 'r', encoding='utf-8') as f:
        content = f.read().lower()

    # List of critical lines/fragments that MUST be present on Page 2
    critical_fragments = [
        "prices are subject to change",
        "account for gst under the reverse charge procedure",
        "as a goods and services tax registered non resident company",
        "we are charging you india gst of o% as required under india tax laws",
        "if you are a registered gst business in india",
        "reverse charge may apply",
        "have questions or need help",
        "please visit our help center"
    ]

    print(f"\n--- Verifying Extraction: {text_file} ---")
    missing = []
    for fragment in critical_fragments:
        if fragment in content.replace('\n', ' '):
            print(f"[OK] Found: '{fragment}'")
        else:
            print(f"[FAIL] Missing: '{fragment}'")
            missing.append(fragment)

    if not missing:
        print("\nSUCCESS: All critical fragments found!")
        return True
    else:
        print(f"\nFAILURE: {len(missing)} fragments missing.")
        return False

if __name__ == "__main__":
    target = "results/test_dataset_fixed_final_tiled.txt"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    verify_extraction(target)
