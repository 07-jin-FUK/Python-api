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

    # 千円札の長さを基準とした実際の距離を計算する関数
    def calculate_real_length(scale_points, measurement_points, real_length_of_bill=15.5):
        # スケールポイント（千円札の長辺）の距離をピクセル単位で計算
        scale_pixel_distance = np.sqrt((scale_points[1]['x'] - scale_points[0]['x']) ** 2 +
                                       (scale_points[1]['y'] - scale_points[0]['y']) ** 2)

        # 計測する2点間の距離（ピクセル単位）
        measurement_pixel_distance = np.sqrt((measurement_points[1]['x'] - measurement_points[0]['x']) ** 2 +
                                             (measurement_points[1]['y'] - measurement_points[0]['y']) ** 2)

        # 実際の長さを換算
        real_distance = (measurement_pixel_distance / scale_pixel_distance) * real_length_of_bill
        return real_distance

    # 実際の長さを計算
    real_length = calculate_real_length(scale_points, measurement_points)

    return jsonify({
        'message': 'Image processed successfully',
        'measured_length': real_length  # 計測結果を返す
    }), 200

if __name__ == '__main__':
    app.run(debug=True)
