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
        pts_dst = np.array([[0, 0], [150, 0], [150, 76], [0, 76]], dtype=float)  # 単位はミリメートル

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

    return round(real_length_cm, 1)

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
    return round(area_cm2, 1)

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

        # 天面用の実世界の座標（mm）
        pts_dst_half_top = np.array([
            [75, 0],          # D: 左上（RedPoint[5]）
            [75, 76],         # C: 右上（RedPoint[0]）
            [0, 76],          # B: 右下（RedPoint[1]）
            [0, 0]            # A: 左下（RedPoint[4]）
        ], dtype=float)

        # 側面用の実世界の座標（mm）
        pts_dst_half_side = np.array([
            [0, 0],            # D: 左上（RedPoint[1]）
            [75, 0],           # C: 右上（RedPoint[2]）
            [75, 76],          # B: 右下（RedPoint[3]）
            [0, 76]            # A: 左下（RedPoint[4]）
        ], dtype=float)

        # 天面計算用ホモグラフィー行列を計算
        red_pts_src_top = np.array([
            [red_points[5]['x'], red_points[5]['y']],  # D: redPoints[5]
            [red_points[0]['x'], red_points[0]['y']],  # C: redPoints[0]
            [red_points[1]['x'], red_points[1]['y']],  # B: redPoints[1]
            [red_points[4]['x'], red_points[4]['y']]   # A: redPoints[4]
        ], dtype=float)
        h_top, _ = cv2.findHomography(red_pts_src_top, pts_dst_half_top)

        if h_top is None:
            return jsonify({'error': 'Homography calculation failed (top)'}), 400

        # 側面計算用ホモグラフィー行列を計算
        red_pts_src_side = np.array([
            [red_points[1]['x'], red_points[1]['y']],  # D: redPoints[1]
            [red_points[2]['x'], red_points[2]['y']],  # C: redPoints[2]
            [red_points[3]['x'], red_points[3]['y']],  # B: redPoints[3]
            [red_points[4]['x'], red_points[4]['y']]   # A: redPoints[4]
        ], dtype=float)
        h_side, _ = cv2.findHomography(red_pts_src_side, pts_dst_half_side)

        if h_side is None:
            return jsonify({'error': 'Homography calculation failed (side)'}), 400

        # サイズ計算関数の定義
        def calculate_size(points, h):
            pts_measure = np.array([
                [p['x'], p['y']] for p in points
            ], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h)

            # 縦と横の距離を計算（mm単位）
            vertical_distance = np.linalg.norm(transformed_points[3][0] - transformed_points[0][0])
            horizontal_distance = np.linalg.norm(transformed_points[1][0] - transformed_points[0][0])

            # 面積を計算（mm²単位）
            area_mm2 = vertical_distance * horizontal_distance

            # 長さをcmに変換
            vertical_cm = vertical_distance / 10
            horizontal_cm = horizontal_distance / 10

            return (
                round(vertical_cm, 1),
                round(horizontal_cm, 1),
                area_mm2,
                transformed_points
            )

        # 各辺の長さを計算する関数
        def calculate_edges(transformed_points):
            top_edge = np.linalg.norm(transformed_points[1][0] - transformed_points[0][0])
            right_edge = np.linalg.norm(transformed_points[2][0] - transformed_points[1][0])
            bottom_edge = np.linalg.norm(transformed_points[3][0] - transformed_points[2][0])
            left_edge = np.linalg.norm(transformed_points[0][0] - transformed_points[3][0])

            return {
                'top_edge': round(top_edge / 10, 1),       # mmをcmに変換
                'right_edge': round(right_edge / 10, 1),
                'bottom_edge': round(bottom_edge / 10, 1),
                'left_edge': round(left_edge / 10, 1)
            }

        # 天面のサイズと各辺の長さを計算
        blue_points_top = [blue_points[0], blue_points[1], blue_points[2], blue_points[3]]
        top_vertical, top_horizontal, top_area_mm2, transformed_top_points = calculate_size(blue_points_top, h_top)
        top_edges = calculate_edges(transformed_top_points)

        # 側面のサイズと各辺の長さを計算
        blue_points_side = [blue_points[2], blue_points[5], blue_points[4], blue_points[3]]
        side_vertical, side_horizontal, side_area_mm2, transformed_side_points = calculate_size(blue_points_side, h_side)
        side_edges = calculate_edges(transformed_side_points)

        # 面積の単位を決定（1 m²以上なら m²、そうでなければ cm²）
        def format_area(area_mm2):
            area_m2 = area_mm2 / 1e6  # mm²からm²へ
            if area_m2 >= 1:
                return f"{round(area_m2, 4)} m²"
            else:
                area_cm2 = area_mm2 / 100  # mm²からcm²へ
                return f"{round(area_cm2, 2)} cm²"

        # 体積の計算
        volume_m3 = (top_area_mm2 / 1e6) * (side_vertical / 100)  # m³単位
        if volume_m3 >= 1:
            volume_str = f"{round(volume_m3, 4)} m³"
        else:
            volume_cm3 = (top_area_mm2 / 100) * side_vertical  # cm³単位
            volume_str = f"{round(volume_cm3, 2)} cm³"

        return jsonify({
            'message': '3D object size calculated successfully',
            'top_vertical': f"{top_vertical} cm",
            'top_horizontal': f"{top_horizontal} cm",
            'top_area': format_area(top_area_mm2),
            'top_edges': top_edges,
            'side_height': f"{side_vertical} cm",
            'side_area': format_area(side_area_mm2),
            'volume': volume_str,
            'side_edges': side_edges
        }), 200

    except Exception as e:
        print(f"Error occurred: {e}")
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

        # 千円札の実際のサイズに対応する正しい座標（mm単位）
        pts_dst_half = np.array([
            [0, 0],
            [75, 0],     # 150mmの半分 = 75mm
            [75, 76],
            [0, 76]
        ], dtype=float)

        # 上半分（直径計算用）ホモグラフィー行列を計算
        red_pts_src_upper = np.array([
            [red_points[0]['x'], red_points[0]['y']],  # 右上
            [red_points[1]['x'], red_points[1]['y']],  # 右中央
            [red_points[4]['x'], red_points[4]['y']],  # 左中央
            [red_points[5]['x'], red_points[5]['y']]   # 左上
        ], dtype=float)
        h_cylinder_upper, _ = cv2.findHomography(red_pts_src_upper, pts_dst_half)

        if h_cylinder_upper is None:
            return jsonify({'error': 'Homography calculation failed (upper)'}), 400

        # 下半分（高さ計算用）ホモグラフィー行列を計算
        red_pts_src_lower = np.array([
            [red_points[1]['x'], red_points[1]['y']],  # 右中央
            [red_points[2]['x'], red_points[2]['y']],  # 右下
            [red_points[3]['x'], red_points[3]['y']],  # 左下
            [red_points[4]['x'], red_points[4]['y']]   # 左中央
        ], dtype=float)
        h_cylinder_lower, _ = cv2.findHomography(red_pts_src_lower, pts_dst_half)

        if h_cylinder_lower is None:
            return jsonify({'error': 'Homography calculation failed (lower)'}), 400

        # 直径の計算には上半分のホモグラフィー行列を使用
        def calculate_cylinder_diameter(points, h_upper):
            pts_measure = np.array([[p['x'], p['y']] for p in points[:2]], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h_upper)

            # 直径を計算 (mm単位)
            point1 = transformed_points[0][0]
            point2 = transformed_points[1][0]
            diameter_mm = np.linalg.norm(point2 - point1)

            # cmに変換
            diameter_cm = diameter_mm / 10
            return diameter_cm

        # 円柱の高さの計算
        def calculate_cylinder_height(points, h_lower):
            pts_measure = np.array([[p['x'], p['y']] for p in [points[0], points[2]]], dtype=float).reshape(-1, 1, 2)
            transformed_points = cv2.perspectiveTransform(pts_measure, h_lower)

            # 高さを計算 (mm単位)
            point1 = transformed_points[0][0]
            point3 = transformed_points[1][0]
            height_mm = np.linalg.norm(point3 - point1)

            # cmに変換
            height_cm = height_mm / 10
            return height_cm

        # 直径と高さをそれぞれ異なるホモグラフィー行列で計算
        diameter = calculate_cylinder_diameter(blue_points, h_cylinder_upper)
        height = calculate_cylinder_height(blue_points, h_cylinder_lower)

        # 天面積と側面積を計算
        top_area_cm2 = np.pi * (diameter / 2) ** 2  # cm²単位
        side_area_cm2 = np.pi * diameter * height   # cm²単位

        # 面積の単位を決定
        if top_area_cm2 >= 10000:  # 1 m² = 10000 cm²
            top_area_m2 = top_area_cm2 / 10000
            top_area_str = f"{round(top_area_m2, 4)} m²"
        else:
            top_area_str = f"{round(top_area_cm2, 2)} cm²"

        if side_area_cm2 >= 10000:
            side_area_m2 = side_area_cm2 / 10000
            side_area_str = f"{round(side_area_m2, 4)} m²"
        else:
            side_area_str = f"{round(side_area_cm2, 2)} cm²"

        # 体積を計算
        volume_cm3 = top_area_cm2 * height  # cm³単位

        if volume_cm3 >= 1e6:  # 1 m³ = 1,000,000 cm³
            volume_m3 = volume_cm3 / 1e6
            volume_str = f"{round(volume_m3, 4)} m³"
        else:
            volume_str = f"{round(volume_cm3, 2)} cm³"

        return jsonify({
            'message': 'Cylinder size calculated successfully',
            'diameter': f"{round(diameter, 2)} cm",
            'height': f"{round(height, 2)} cm",
            'top_area': top_area_str,
            'side_area': side_area_str,
            'volume': volume_str
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
