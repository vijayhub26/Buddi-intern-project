import sys
import os

def test_paddle_setup():
    print("--- PaddleOCR Environment Smoke Test ---")
    
    try:
        import paddle
        print(f" PaddlePaddle version: {paddle.__version__}")
        print(f"   GPU available: {paddle.is_compiled_with_cuda()}")
    except ImportError:
        print(" PaddlePaddle NOT found.")
        return False

    try:
        from paddleocr import PaddleOCR
        # Initialize OCR (this will download models on first run)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        print("PaddleOCR initialized successfully.")
    except ImportError:
        print(" PaddleOCR NOT found.")
        return False
    except Exception as e:
        print(f" PaddleOCR initialization failed: {e}")
        return False

    print("--- Test Complete ---")
    return True

if __name__ == "__main__":
    if not test_paddle_setup():
        sys.exit(1)
