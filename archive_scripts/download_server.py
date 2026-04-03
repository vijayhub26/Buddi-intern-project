import urllib.request
import os

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

urls = {
    "ch_PP-OCRv4_server_rec_infer.onnx": "https://huggingface.co/RapidAI/paddleocr-models/resolve/main/ch_PP-OCRv4_server_rec_infer.onnx",
    "ch_PP-OCRv4_det_server_infer.onnx": "https://huggingface.co/RapidAI/paddleocr-models/resolve/main/ch_PP-OCRv4_det_server_infer.onnx"
}

for name, url in urls.items():
    print(f"Downloading {name}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(os.path.join(model_dir, name), 'wb') as f:
                f.write(response.read())
        print(f"Downloaded {name} successfully.")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
