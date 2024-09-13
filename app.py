import os
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# ルートエンドポイント
@app.route('/')
def hello_world():
    return "Flask API is running!"

# 画像アップロードと処理のエンドポイント
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # 画像を取得
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # ファイルの内容をメモリ上で読み込み
    np_img = np.frombuffer(file.read(), np.uint8)

    # OpenCVで画像をデコード
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Unable to read the image'}), 400

    # OpenCVで画像処理 (例: グレースケールに変換)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 処理後の画像をメモリに保存し、レスポンスとして返す
    _, buffer = cv2.imencode('.jpg', gray_img)
    gray_img_bytes = buffer.tobytes()

    return jsonify({'message': 'Image processed successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
