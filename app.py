import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ルートエンドポイント
@app.route('/')
def hello_world():
    return "Flask API is running!"

# 画像アップロードと処理のエンドポイント
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files or 'points' not in request.form:
        return jsonify({'error': 'No image or points provided'}), 400

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

    # クライアントから送られた4点のスケール用ポイントと2点の計測用ポイントを取得
    points = request.form.get('points')
    points = eval(points)  # 文字列をPythonのリストに変換

    if len(points) < 6:
        return jsonify({'error': 'Not enough points provided'}), 400

    scale_points = points[:4]  # 最初の4点はスケール設定用
    measurement_points = points[4:6]  # 次の2点は実際の計測用

    # ホモグラフィー変換のためのポイント (千円札の実際のサイズに対応する正しい座標)
    # 千円札の長辺が15.5cm、短辺が7.6cmと仮定
    pts_dst = np.array([[0, 0], [155, 0], [155, 76], [0, 76]], dtype=float)  # 単位はミリメートル

    # スケールポイントをOpenCV形式に変換
    pts_src = np.array([[p['x'], p['y']] for p in scale_points], dtype=float)

    # ホモグラフィー行列を計算
    h, status = cv2.findHomography(pts_src, pts_dst)

    # 画像をホモグラフィー変換（パース補正）
    height, width = img.shape[:2]
    corrected_img = cv2.warpPerspective(img, h, (width, height))

    # 補正された画像上での計測する2点の距離を計算
    def calculate_real_length(measurement_points, h):
        # 計測ポイントをホモグラフィー変換
        pts_measure = np.array([[p['x'], p['y']] for p in measurement_points], dtype=float).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(pts_measure, h)

        # 2点間の距離をピクセル単位で計算
        point1 = transformed_points[0][0]
        point2 = transformed_points[1][0]
        pixel_distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

        # 実際の長さを計算 (ピクセルからmmへの変換)
        real_length_mm = pixel_distance  # ここでは1ピクセルが1mm相当と仮定
        real_length_cm = real_length_mm / 10  # cmに変換

        return real_length_cm

    # 実際の長さを計算
    real_length = calculate_real_length(measurement_points, h)

    return jsonify({
        'message': 'Image processed successfully',
        'measured_length': real_length  # 計測結果を返す
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
