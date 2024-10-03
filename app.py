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

# 画像アップロードと処理のエンドポイント（長さおよび平面モード）
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files or 'points' not in request.form:
            return jsonify({'error': 'No image or points provided'}), 400

        mode = request.form.get('mode')
        if not mode:
            return jsonify({'error': 'Measurement mode not specified'}), 400

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

        # ポイントを取得
        points = request.form.get('points')
        points = json.loads(points)

        # スケールポイントを取得
        scale_points = points[:4]
        pts_src = np.array([[p['x'], p['y']] for p in scale_points], dtype=float)

        # ホモグラフィー変換のためのポイント (千円札の実際のサイズに対応する正しい座標)
        pts_dst = np.array([[0, 0], [155, 0], [155, 76], [0, 76]], dtype=float)  # 単位はミリメートル

        # ホモグラフィー行列を計算
        h, status = cv2.findHomography(pts_src, pts_dst)
        if h is None:
            return jsonify({'error': 'Homography calculation failed'}), 400

        if mode == 'length':
            if len(points) < 6:
                return jsonify({'error': 'Not enough points provided for length measurement'}), 400
            measurement_points = points[4:6]
            real_length = calculate_real_length(measurement_points, h)
            return jsonify({
                'message': 'Length measurement successful',
                'measured_length': real_length
            }), 200

        elif mode == 'plane':
            if len(points) < 8:
                return jsonify({'error': 'Not enough points provided for plane measurement'}), 400
            plane_points = points[4:8]
            plane_edges = calculate_plane_edges(plane_points, h)
            area = calculate_area(plane_edges)
            return jsonify({
                'message': 'Plane measurement successful',
                'measured_area': area,
                'plane_edges': plane_edges
            }), 200

        else:
            return jsonify({'error': 'Unknown measurement mode'}), 400

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500

# 必要な関数の定義
def calculate_real_length(measurement_points, h):
    # 計測ポイントをホモグラフィー変換
    pts_measure = np.array([[p['x'], p['y']] for p in measurement_points], dtype=float).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(pts_measure, h)

    # 2点間の距離を計算
    point1 = transformed_points[0][0]
    point2 = transformed_points[1][0]
    distance_mm = np.linalg.norm(point2 - point1)  # 距離をミリメートルで計算

    real_length_cm = distance_mm / 10  # cmに変換

    return round(real_length_cm, 2)

def calculate_plane_edges(plane_points, h):
    # 平面ポイントをホモグラフィー変換
    pts_plane = np.array([[p['x'], p['y']] for p in plane_points], dtype=float).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(pts_plane, h)

    # 各辺の長さを計算
    edges = {}
    edges['top_edge'] = np.linalg.norm(transformed_points[1][0] - transformed_points[0][0]) / 10  # cmに変換
    edges['right_edge'] = np.linalg.norm(transformed_points[2][0] - transformed_points[1][0]) / 10
    edges['bottom_edge'] = np.linalg.norm(transformed_points[3][0] - transformed_points[2][0]) / 10
    edges['left_edge'] = np.linalg.norm(transformed_points[0][0] - transformed_points[3][0]) / 10

    # 小数点以下2桁に丸める
    for key in edges:
        edges[key] = round(edges[key], 2)

    return edges

def calculate_area(edges):
    # 平面の面積を計算（長方形として計算）
    area_cm2 = edges['top_edge'] * edges['left_edge']  # 面積 = 縦 × 横
    return round(area_cm2, 2)

# サーバーの起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


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
            return round(vertical_distance / 10, 2), round(horizontal_distance / 10, 2), round(area, 2), transformed_points

        # 天面の各辺の長さを計算する関数
        def calculate_edges(transformed_points):
            top_edge = np.sqrt((transformed_points[1][0][0] - transformed_points[0][0][0]) ** 2 +
                               (transformed_points[1][0][1] - transformed_points[0][0][1]) ** 2)
            right_edge = np.sqrt((transformed_points[2][0][0] - transformed_points[1][0][0]) ** 2 +
                                 (transformed_points[2][0][1] - transformed_points[1][0][1]) ** 2)
            bottom_edge = np.sqrt((transformed_points[3][0][0] - transformed_points[2][0][0]) ** 2 +
                                  (transformed_points[3][0][1] - transformed_points[2][0][1]) ** 2)
            left_edge = np.sqrt((transformed_points[0][0][0] - transformed_points[3][0][0]) ** 2 +
                                (transformed_points[0][0][1] - transformed_points[3][0][1]) ** 2)

            return {
                'top_edge': round(top_edge / 10, 2),
                'right_edge': round(right_edge / 10, 2),
                'bottom_edge': round(bottom_edge / 10, 2),
                'left_edge': round(left_edge / 10, 2)
            }

        # 天面の縦、横サイズと面積を計算し、各辺の長さも計算
        top_vertical, top_horizontal, top_area, transformed_top_points = calculate_size(blue_points[:4], h_top)
        top_edges = calculate_edges(transformed_top_points)

        # 側面の高さと面積を計算
        side_vertical, side_horizontal, side_area, _ = calculate_size([blue_points[3], blue_points[4], blue_points[5], blue_points[0]], h_side)

        # 立体体積を計算
        volume = top_area * side_vertical  # 天面積 × 高さ

        # 側面の辺を計算
        def calculate_real_length(measurement_points, h):
            # 計測ポイントをホモグラフィー変換
            pts_measure = np.array([[p['x'], p['y']] for p in measurement_points], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h)

            # 2点間の距離を計算
            point1 = transformed_points[0][0]
            point2 = transformed_points[1][0]
            distance_mm = np.linalg.norm(point2 - point1)  # 距離をミリメートルで計算

            real_length_cm = distance_mm / 10  # cmに変換

            return round(real_length_cm, 2)

        side_right_edge = calculate_real_length([blue_points[2], blue_points[5]], h_side)  # 側面右辺
        side_bottom_edge = calculate_real_length([blue_points[4], blue_points[5]], h_side)  # 側面下辺
        side_left_edge = calculate_real_length([blue_points[3], blue_points[4]], h_side)  # 側面左辺

        return jsonify({
            'message': '3D object size calculated successfully',
            'top_vertical': f"{top_vertical} cm",
            'top_horizontal': f"{top_horizontal} cm",
            'top_area': f"{top_area} cm²",
            'top_edges': top_edges,  # ここで天面の各辺の長さを返す
            'side_height': f"{side_vertical} cm",
            'side_area': f"{side_area} cm²",
            'volume': f"{round(volume, 2)} cm³",
            'side_right_edge': f"{side_right_edge} cm",
            'side_bottom_edge': f"{side_bottom_edge} cm",
            'side_left_edge': f"{side_left_edge} cm"
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

        # 円柱の高さの計算
        def calculate_cylinder_height(points, h_lower):
            pts_measure = np.array([[p['x'], p['y']] for p in points], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h_lower)

            # 高さを計算 (bluePoints[0]とbluePoints[2]の垂直距離)
            point1 = transformed_points[0][0]  # bluePoints[0] に対応
            point3 = transformed_points[2][0]  # bluePoints[2] に対応
            height = np.sqrt((point3[0] - point1[0]) ** 2 + (point3[1] - point1[1]) ** 2) / 10  # cmに変換
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
    elapsed_time = time.time() - start_time
    print(f"Warmup completed in {elapsed_time} seconds")

    return {'status': 'Server is ready', 'time_taken': elapsed_time}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
