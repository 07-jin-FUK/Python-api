import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import time
import json

app = Flask(__name__)
CORS(app)

# ルートエンドポイント
@app.route('/')
def hello_world():
    return "Flask API is running!"


# 画像アップロードと処理のエンドポイント
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
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
        points = json.loads(points)  # 安全な方法で文字列をリストに変換

        if len(points) < 6:
            return jsonify({'error': 'Not enough points provided'}), 400

        scale_points = points[:4]  # 最初の4点はスケール設定用
        measurement_points = points[4:6]  # 次の2点は実際の計測用

        # ホモグラフィー変換のためのポイント (千円札の実際のサイズに対応する正しい座標)
        pts_dst = np.array([[0, 0], [155, 0], [155, 76], [0, 76]], dtype=float)  # 単位はミリメートル

        # スケールポイントをOpenCV形式に変換
        pts_src = np.array([[p['x'], p['y']] for p in scale_points], dtype=float)

        # ホモグラフィー行列を計算
        h, status = cv2.findHomography(pts_src, pts_dst)

        if h is None:
            return jsonify({'error': 'Homography calculation failed'}), 400

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

            return round(real_length_cm, 2)

        # 実際の長さを計算
        real_length = calculate_real_length(measurement_points, h)

        return jsonify({
            'message': 'Image processed successfully',
            'measured_length': real_length  # 計測結果を返す
        }), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # エラーログをサーバーに表示
        return jsonify({'error': 'Internal server error occurred'}), 500


# 3D画像の処理
@app.route('/process-3d-image', methods=['POST'])
def process_3d_image():
    try:
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

        # クライアントから送られた12点のポイントを取得
        points = request.form.get('points')
        points = json.loads(points)  # 安全に文字列をリストに変換

        if len(points) < 12:
            return jsonify({'error': 'Not enough points provided'}), 400

        red_points = points[:6]  # 最初の6点は千円札の基準用
        blue_points = points[6:]  # 次の6点は目的物の計測用

        # ホモグラフィー変換のためのポイント (千円札の実際のサイズに対応する正しい座標)
        pts_dst_half = np.array([[0, 0], [155 / 2, 0], [155 / 2, 76], [0, 76]], dtype=float)  # 千円札の半分

        # 天面計算用ホモグラフィー行列を計算
        red_pts_src_top = np.array([[p['x'], p['y']] for p in [red_points[0], red_points[1], red_points[4], red_points[5]]], dtype=float)
        h_top, _ = cv2.findHomography(red_pts_src_top, pts_dst_half)

        if h_top is None:
            return jsonify({'error': 'Homography calculation failed (top)'}), 400

        # 側面計算用ホモグラフィー行列を計算
        red_pts_src_side = np.array([[p['x'], p['y']] for p in [red_points[1], red_points[2], red_points[3], red_points[4]]], dtype=float)
        h_side, _ = cv2.findHomography(red_pts_src_side, pts_dst_half)

        if h_side is None:
            return jsonify({'error': 'Homography calculation failed (side)'}), 400

        # 天面のサイズ計算
        def calculate_size(points, h):
            pts_measure = np.array([[p['x'], p['y']] for p in points], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h)

            # 縦と横の距離を計算
            vertical_distance = np.sqrt((transformed_points[3][0][0] - transformed_points[0][0][0]) ** 2 +
                                        (transformed_points[3][0][1] - transformed_points[0][0][1]) ** 2)
            horizontal_distance = np.sqrt((transformed_points[1][0][0] - transformed_points[0][0][0]) ** 2 +
                                          (transformed_points[1][0][1] - transformed_points[0][0][1]) ** 2)

            # 面積を計算
            area = vertical_distance * horizontal_distance / 100  # cm²に変換
            return round(vertical_distance / 10, 2), round(horizontal_distance / 10, 2), round(area, 2)

        # 天面の縦、横サイズと面積
        top_vertical, top_horizontal, top_area = calculate_size(blue_points[:4], h_top)

        # 側面の高さと面積
        side_vertical, side_horizontal, side_area = calculate_size([blue_points[3], blue_points[4], blue_points[5], blue_points[0]], h_side)

        # 立体体積を計算
        volume = top_area * side_vertical  # 天面積 × 高さ

        return jsonify({
            'message': '3D object size calculated successfully',
            'top_vertical': f"{top_vertical} cm",
            'top_horizontal': f"{top_horizontal} cm",
            'side_height': f"{side_vertical} cm",
            'top_area': f"{top_area} cm²",
            'side_area': f"{side_area} cm²",
            'volume': f"{round(volume, 2)} cm³"
        }), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # エラーログをサーバーに表示
        return jsonify({'error': 'Internal server error occurred'}), 500


# 円柱サイズを測定するためのエンドポイント
@app.route('/process-cylinder-image', methods=['POST'])
def process_cylinder_image():
    try:
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

        # クライアントから送られた9点のポイントを取得
        points = request.form.get('points')
        points = json.loads(points)  # 安全に文字列をリストに変換

        if len(points) < 9:
            return jsonify({'error': 'Not enough points provided'}), 400

        red_points = points[:6]  # 最初の6点は千円札の基準用
        blue_points = points[6:]  # 次の3点は円柱の計測用

        # ホモグラフィー変換のためのポイント (千円札の実際のサイズに対応する正しい座標)
        pts_dst_half = np.array([[0, 0], [155 / 2, 0], [155 / 2, 76], [0, 76]], dtype=float)  # 千円札の半分

        # 上半分（直径計算用）ホモグラフィー行列を計算
        red_pts_src_upper = np.array([[p['x'], p['y']] for p in [red_points[0], red_points[1], red_points[4], red_points[5]]], dtype=float)
        h_cylinder_upper, _ = cv2.findHomography(red_pts_src_upper, pts_dst_half)

        if h_cylinder_upper is None:
            return jsonify({'error': 'Homography calculation failed (upper)'}), 400

        # 下半分（高さ計算用）ホモグラフィー行列を計算
        red_pts_src_lower = np.array([[p['x'], p['y']] for p in [red_points[1], red_points[2], red_points[3], red_points[4]]], dtype=float)
        h_cylinder_lower, _ = cv2.findHomography(red_pts_src_lower, pts_dst_half)

        if h_cylinder_lower is None:
            return jsonify({'error': 'Homography calculation failed (lower)'}), 400

        # 直径の計算には上半分のホモグラフィー行列を使用
        def calculate_cylinder_diameter(points, h_upper):
            pts_measure = np.array([[p['x'], p['y']] for p in points[:2]], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h_upper)

            # 直径を計算 (blue_points[0]とblue_points[1]の距離)
            point1 = transformed_points[0][0]
            point2 = transformed_points[1][0]
            diameter = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) / 10  # cmに変換
            return diameter

        # パース補正後のポイントを可視化するための関数
        def visualize_transformed_points(img, points, h, color=(0, 255, 0)):
            transformed_points = cv2.perspectiveTransform(np.array(points).reshape(-1, 1, 2), h)
            for point in transformed_points:
                x, y = point[0]
                cv2.circle(img, (int(x), int(y)), 5, color, -1)

        # 円柱の高さの計算
        def calculate_cylinder_height(points, h_lower, img=None):
            pts_measure = np.array([[p['x'], p['y']] for p in points], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h_lower)

            # Optional: 補正後のポイントを可視化（デバッグ用）
            if img is not None:
                visualize_transformed_points(img, points, h_lower, color=(255, 0, 0))

            # 高さを計算 (bluePoints[0]とbluePoints[2]の垂直距離)
            point1 = transformed_points[0][0]  # bluePoints[0] に対応
            point3 = transformed_points[2][0]  # bluePoints[2] に対応
            # 2点間のユークリッド距離を計算
            height = np.sqrt((point3[0] - point1[0]) ** 2 + (point3[1] - point1[1]) ** 2) / 10  # cmに変換

            print(f"Calculated height between bluePoint[0] and bluePoint[2]: {height} cm")
            return height

        # 直径と高さをそれぞれ異なるホモグラフィー行列で計算
        diameter = calculate_cylinder_diameter(blue_points, h_cylinder_upper)
        height = calculate_cylinder_height(blue_points, h_cylinder_lower)

        # 天面積と側面積を計算
        top_area = (np.pi * (diameter / 2) ** 2)  # 円の面積 (cm²)
        side_area = np.pi * diameter * height  # 側面積 (cm²)

        # 体積を計算
        volume = top_area * height  # 体積 (cm³)

        return jsonify({
            'message': 'Cylinder size calculated successfully',
            'diameter': f"{round(diameter, 2)} cm",
            'height': f"{round(height, 2)} cm",
            'top_area': f"{round(top_area, 2)} cm²",
            'side_area': f"{round(side_area, 2)} cm²",
            'volume': f"{round(volume, 2)} cm³"
        }), 200

    except Exception as e:
        print(f"Error occurred: {e}")  # エラーログをサーバーに表示
        return jsonify({'error': 'Internal server error occurred'}), 500


# ウォームアップエンドポイント
@app.route('/warmup', methods=['POST'])
def warmup():
    start_time = time.time()

    # 他のウォームアップ処理をここに追加
    elapsed_time = time.time() - start_time
    print(f"Warmup completed in {elapsed_time} seconds")

    return {'status': 'Server is ready', 'time_taken': elapsed_time}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
