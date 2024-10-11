import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
from flask_cors import CORS
import time
import json
import mysql.connector
import bcrypt
import jwt
import datetime
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://objectsize-fine.vercel.app"}})

# シークレットキーの設定（JWT用）
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")  # .envファイルから取得

def create_connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            port=os.getenv("DB_PORT")
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None





@app.route('/login', methods=['POST'])
def login():
    email = request.json.get('email')
    password = request.json.get('password')

    if not email or not password:
        return jsonify({'error': 'メールアドレスとパスワードを提供してください'}), 400

    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        # データベースからユーザーを検索
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            # パスワードが正しいか確認
            if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                # JWTトークンを生成
                token = jwt.encode({
                    'user_id': user['id'],
                    'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
                }, app.config['SECRET_KEY'], algorithm="HS256")

                # IDと名前を含むレスポンスを返す
                return jsonify({'token': token, 'message': 'ログイン成功', 'name': user['name'], 'id': user['id']}), 200
            else:
                return jsonify({'error': '無効なメールアドレスまたはパスワード'}), 401
        else:
            return jsonify({'error': 'ユーザーが見つかりませんでした'}), 404

    except Exception as e:
        print(f"Error occurred during login: {e}")
        return jsonify({'error': 'ログイン中にエラーが発生しました'}), 500

    finally:
        if conn:
            conn.close()



@app.route('/register', methods=['POST'])
def register():
    name = request.json.get('name')
    email = request.json.get('email')
    password = request.json.get('password')

    if not name or not email or not password:
        return jsonify({'error': '名前、メールアドレス、パスワードを提供してください'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    conn = None
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password.decode('utf-8')))
        conn.commit()
        return jsonify({'message': 'ユーザー登録に成功しました'}), 201
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'ユーザー登録に失敗しました'}), 500
    finally:
        conn.close()



# ルートエンドポイント
@app.route('/')
def hello_world():
    return "Flask API is running!"



@app.route('/spaces', methods=['GET'])
def get_spaces():
    user_id = request.args.get('user_id')
    try:
        conn = create_connection()
        cursor = conn.cursor()
        
        # user_id に基づいてスペースを取得
        cursor.execute("SELECT * FROM spaces WHERE user_id = %s", (user_id,))
        spaces = cursor.fetchall()

        if not spaces:
            return jsonify({'error': 'No spaces found for this user'}), 404

        space_list = []
        for space in spaces:
            space_data = {
                'id': space[0],
                'user_id': space[1],
                'dimensions': space[6],
                'background_color': space[7],
                'floor_color': space[8],
                'back_color': space[10],
                'left_side_color': space[11],
                'right_side_color': space[12],
                'is_single_sided': space[9],
            }
            space_list.append(space_data)

        return jsonify({'spaces': space_list}), 200

    except Exception as e:
        print(f"Error occurred while fetching spaces: {e}")
        return jsonify({'error': 'Failed to fetch spaces'}), 500
    finally:
        conn.close()

@app.route('/spaces/<int:user_id>/save-objects', methods=['POST'])
def save_objects(user_id):
    data = request.json
    objects = data.get('objects', [])

    if not objects:
        return jsonify({'error': 'オブジェクト情報が不足しています'}), 400

    try:
        conn = create_connection()
        cursor = conn.cursor()

        # user_id に基づいて既存のオブジェクトを削除
        cursor.execute("DELETE FROM objects WHERE user_id = %s", (user_id,))

        # 新しいオブジェクト情報を保存
        for obj in objects:
            cursor.execute(
                """
                INSERT INTO objects (user_id, object_type, position, size, color, is_wireframe)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, obj['object_type'], json.dumps(obj['position']), json.dumps(obj['size']), obj['color'], obj.get('isWireframe', False))
            )

        conn.commit()
        return jsonify({'message': 'オブジェクトが保存されました'}), 200
    except Exception as e:
        print(f"Error saving objects: {e}")
        return jsonify({'error': 'オブジェクト保存中にエラーが発生しました'}), 500
    finally:
        conn.close()



@app.route('/spaces/<int:user_id>/objects', methods=['GET'])
def get_objects(user_id):
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)

        # user_id に基づいてオブジェクトを取得
        cursor.execute("SELECT * FROM objects WHERE user_id = %s", (user_id,))
        objects = cursor.fetchall()

        return jsonify({'objects': objects}), 200
    except Exception as e:
        print(f"Error fetching objects: {e}")
        return jsonify({'error': 'オブジェクト取得中にエラーが発生しました'}), 500
    finally:
        conn.close()


@app.route('/spaces/<int:user_id>', methods=['GET'])
def get_space(user_id):
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True, buffered=True)  # buffered=Trueを追加

        cursor.execute("SELECT * FROM spaces WHERE user_id = %s", (user_id,))
        space = cursor.fetchone()

        if not space:
            return jsonify({'error': '空間が見つからないか、権限がありません'}), 404

        return jsonify({'space': space}), 200

    except Exception as e:
        print(f"Error occurred while retrieving space: {e}")  
        return jsonify({'error': '空間の取得に失敗しました'}), 500

    finally:
        cursor.close()  
        conn.close()



@app.route('/spaces/<int:user_id>/save', methods=['POST'])
def save_space(user_id):
    data = request.json
    try:
        conn = create_connection()
        cursor = conn.cursor(buffered=True)  # buffered=Trueを追加

        # user_id に基づいて空間の存在を確認して削除
        cursor.execute("DELETE FROM spaces WHERE user_id = %s", (user_id,))

        # 新しい空間の情報を挿入
        cursor.execute(
            """INSERT INTO spaces (user_id, dimensions, background_color, floor_color, 
                                    back_color, left_side_color, right_side_color, is_single_sided) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                user_id,
                json.dumps(data['dimensions']),
                data['backgroundColor'],
                data['floorColor'],
                data['backColor'],
                data['leftSideColor'],
                data['rightSideColor'],
                data['isSingleSided']
            )
        )

        conn.commit()
        return jsonify({'message': 'Space saved successfully'}), 201

    except Exception as e:
        print(f"Error occurred while saving: {e}", exc_info=True)
        return jsonify({'error': 'Failed to save space'}), 500

    finally:
        cursor.close()
        conn.close()



@app.route('/users/<int:user_id>/spaces', methods=['GET'])
def get_user_spaces(user_id):
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM spaces WHERE user_id = %s", (user_id,))
        spaces = cursor.fetchall()
        return jsonify(spaces), 200
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to retrieve spaces'}), 500
    finally:
        conn.close()



@app.route('/users', methods=['GET'])
def get_users():
    try:
        conn = create_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
        return jsonify(users), 200
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Internal server error occurred'}), 500


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
            max_width, max_height = calculate_max_dimensions(plane_edges)
            return jsonify({
                'message': 'Plane measurement successful',
                'measured_area': area,
                'plane_edges': plane_edges,
                'max_width': max_width,
                'max_height': max_height
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

def calculate_max_dimensions(edges):
    # 最長横幅と最長縦幅を計算
    max_width = max(edges['top_edge'], edges['bottom_edge'])
    max_height = max(edges['left_edge'], edges['right_edge'])
    return round(max_width, 2), round(max_height, 2)

# サーバーの起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


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
    app.run(host='0.0.0.0', port=5000 , debug=True)
