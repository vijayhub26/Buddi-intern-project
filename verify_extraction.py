import os
import sys
import re

def verify_and_report(text_file):
    if not os.path.exists(text_file):
        print(f"Error: {text_file} not found.")
        return

    with open(text_file, 'r', encoding='utf-8') as f:
        # Load and normalize (remove extra internal spaces for easier checking, but keep line info)
        lines = f.readlines()
        content = " ".join([l.strip() for l in lines]).lower()
        content = re.sub(r'\s+', ' ', content)

    # Dictionary of critical fragments to their "success" condition
    # Key: Label, Value: String fragment to look for
    checks = {
        "Boilerplate: Subject to Change": "prices are subject to change",
        "Boilerplate: Reverse Charge": "account for gst under the reverse charge procedure",
        "Tax Law: Resident Company": "as a goods and services tax registered non resident company",
        "Tax Law: India GST 0%": "we are charging you india gst of o% as required under india tax laws",
        "Tax Law: Registered Business": "if you are a registered gst business in india",
        "Tax Law: Apply": "reverse charge may apply",
        "Footer: Help Center": "visit our help center"
    }

    print(f"\n{'='*60}")
    print(f" OCR EXTRACTION AUDIT: {os.path.basename(text_file)}")
    print(f"{'='*60}")
    
    success_count = 0
    for label, fragment in checks.items():
        # We use a fuzzy check that ignores spaces to differentiate "Detection" from "Formatting"
        clean_fragment = fragment.replace(" ", "")
        clean_content = content.replace(" ", "")
        
        detected = clean_fragment in clean_content
        formatted = fragment in content
        
        status = ""
        if detected and formatted:
            status = "✅ PASS"
            success_count += 1
        elif detected and not formatted:
            status = "⚠️  DETECTION OK, FORMATTING FAILED (Clumping)"
        else:
            status = "❌ MISSING"
            
        print(f"{label:<30} : {status}")

    print(f"{'='*60}")
    print(f" TOTAL SUCCESS: {success_count}/{len(checks)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_extraction.py <extracted_text_file>")
    else:
        verify_and_report(sys.argv[1])
