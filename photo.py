import cv2
import numpy as np
import openai
from PIL import Image
from skimage import io, color, transform
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import deque, defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
# カメラデバイスの初期化
camera = cv2.VideoCapture(2)  # 0は通常デフォルトのカメラを指定

if not camera.isOpened():
    print("カメラが開けませんでした。")
    exit()

# 画像を1回取得
ret, frame = camera.read()
if ret:
    filename = "1_season.jpg"
    cv2.imwrite(filename, frame)
    print(f"{filename} を保存しました。")
else:
    print("撮影に失敗しました。")

# カメラデバイスの解放
camera.release()
cv2.destroyAllWindows()

print("撮影が完了しました。")

# 画像の読み込み
image_path = "1_season.jpg"  # 画像のパスを指定
output_path = "output.jpg"  # 切り取った画像の保存先
image = Image.open(image_path)

# 画像情報の出力
print(f"画像のサイズ: {image.size}")  # (幅, 高さ)
print(f"画像のフォーマット: {image.format}")
print(f"画像のモード: {image.mode}")

# 左右を一定距離切り取る（例: 左右それぞれ30ピクセルずつ）
left_trim =46#を切り取る距離
right_trim =0  # 右側を切り取る距離

width, height = image.size
crop_area = (left_trim, 1, width - right_trim, height)  # (左, 上, 右, 下)
cropped_image = image.crop(crop_area)

# 画像をRGBモードに変換
cropped_image = cropped_image.convert('RGB')  # 必要に応じて変換

# 切り取った画像を保存
cropped_image.save(output_path)
print(f"切り取った画像を保存しました: {output_path}")




def board1():
    # 画像ファイルのパス
    image_path = 'output.jpg'

    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("画像が読み込めませんでした。ファイルパスを確認してください。")

    # 画像をリサイズ（7×6のマス目に合わせる）
    resized_image = cv2.resize(image, (700, 600))

    # 色空間をHSVに変換
    hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    # 赤色と黄色の閾値を設定
    # 赤色の閾値を狭める
    # 赤色の閾値（さらに厳密に）
    lower_red1 = np.array([0, 150, 100])   # Hueを0〜10に限定し、彩度と明るさを高める
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([175, 150, 100])  # もう一つの赤の範囲を175〜180に狭める
    upper_red2 = np.array([180, 255, 255])

    # 黄色の閾値（さらに厳密に）
    lower_yellow = np.array([20, 150, 100])  # Hueを20〜30に限定
    upper_yellow = np.array([30, 255, 255])


    # 赤色と黄色のマスク
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # ボードのマスを縦6、横7のグリッドに分割
    grid_rows, grid_cols = 6, 7
    cell_height = resized_image.shape[0] // grid_rows
    cell_width = resized_image.shape[1] // grid_cols

    # 結果画像と出力フォーマット
    result_image = resized_image.copy()
    board_state = []

    # 判定の処理
    for row in range(grid_rows):
        row_state = []
        for col in range(grid_cols):
            # 各セルの範囲を計算
            y_start, y_end = row * cell_height, (row + 1) * cell_height
            x_start, x_end = col * cell_width, (col + 1) * cell_width

            # 各マスクのピクセル値を集計
            red_density = np.mean(mask_red[y_start:y_end, x_start:x_end])
            yellow_density = np.mean(mask_yellow[y_start:y_end, x_start:x_end])

            # 色の割合で判定
            if red_density > yellow_density:
                row_state.append('R')
                result = 'R'
                color = (0, 0, 255)  # 赤色
            elif yellow_density > red_density:
                row_state.append('Y')
                result = 'Y'
                color = (0, 255, 255)  # 黄色
            else:
                row_state.append('-')
                result = '-'
                color = (255, 255, 255)  # 白色

            # 結果を描画
            cv2.putText(result_image, result, (x_start + 10, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            # マス目の境界線を描画
            cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        board_state.append(row_state)

    # 結果をフォーマットして表示
    output = "\n".join(" ".join(row) for row in board_state)
    print(output)


    
    return board_state


    # ボードの状態を返す
    return board_state

# 実行して結果を確認
board = board1()
print("Board State:", board)

# 画像ファイルのパス
image_path = 'output.jpg'

# 画像を読み込む
image = io.imread(image_path)
if image is None:
    raise ValueError("画像が読み込めませんでした。ファイルパスを確認してください。")

# 画像をリサイズ（高速化のため小さくする場合もあり）
image = transform.resize(image, (600, 700), anti_aliasing=True)

# Check if the image has an alpha channel (4 channels)
if image.shape[2] == 4:
    # If so, remove the alpha channel by selecting only the first 3 channels (RGB)
    image = image[:, :, :3]

# 色空間をRGBからHSVに変換
hsv_image = color.rgb2hsv(image)

# 赤色と黄色の閾値を設定（色相に基づく範囲調整）
lower_red1 = (0.0, 0.5, 0.5)  # SとVの閾値を上げて、より純粋な赤を検出
upper_red1 = (0.03, 1.0, 1.0)  # Hueの上限を下げる（範囲を狭める）
lower_red2 = (0.9, 0.5, 0.5)   # こちらもSとVの下限を引き上げ
upper_red2 = (1.0, 1.0, 1.0)   # Hueの下限を引き上げ


lower_yellow = (0.10, 0.3, 0.3)  
upper_yellow = (0.25, 1.0, 1.0)  # Hue範囲を広げる

# マスクを作成
mask_red1 = ((hsv_image[:, :, 0] >= lower_red1[0]) & (hsv_image[:, :, 0] <= upper_red1[0]) &
             (hsv_image[:, :, 1] >= lower_red1[1]) & (hsv_image[:, :, 2] >= lower_red1[2]))
mask_red2 = ((hsv_image[:, :, 0] >= lower_red2[0]) & (hsv_image[:, :, 0] <= upper_red2[0]) &
             (hsv_image[:, :, 1] >= lower_red2[1]) & (hsv_image[:, :, 2] >= lower_red2[2]))
mask_red = mask_red1 | mask_red2

mask_yellow = ((hsv_image[:, :, 0] >= lower_yellow[0]) & (hsv_image[:, :, 0] <= upper_yellow[0]) &
               (hsv_image[:, :, 1] >= lower_yellow[1]) & (hsv_image[:, :, 2] >= lower_yellow[2]))



# グリッドに分割して駒を認識
grid_rows, grid_cols = 6, 7
cell_height = image.shape[0] // grid_rows
cell_width = image.shape[1] // grid_cols

board_state = []
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image)


for row in range(grid_rows):
    row_state = []
    for col in range(grid_cols):
        y_start, y_end = row * cell_height, (row + 1) * cell_height
        x_start, x_end = col * cell_width, (col + 1) * cell_width

        red_density = np.mean(mask_red[y_start:y_end, x_start:x_end])
        yellow_density = np.mean(mask_yellow[y_start:y_end, x_start:x_end])

        if red_density > yellow_density and red_density > 0.15:  # 赤が支配的
            row_state.append('R')
            color = 'red'
        elif yellow_density > red_density and yellow_density > 0.15:  # 黄が支配的
            row_state.append('Y')
            color = 'yellow'
        else:  # 空白
            row_state.append('-')
            color = 'white'

        # 描画
        rect = Rectangle((x_start, y_start), cell_width, cell_height, linewidth=1, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_start + cell_width // 3, y_start + cell_height // 2, row_state[-1], fontsize=12, color=color)
    board_state.append(row_state)

# 盤面の状態をboard2に格納
board2 = [row.copy() for row in board_state]

# board2の内容を出力して確認
for row in board2:
    print(row)
# board2の出力
print("board2 =")
for row in board2:
    print(row)
output_path = "output.jpg"

# 画像から駒の色を識別する関数
def detect_pieces(output_path):
    image = cv2.imread(output_path)
    if image is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません: {output_path}")

    # 画像のサイズ取得
    height, width, _ = image.shape
    cell_height = height // 6
    cell_width = width // 7

    # HSV変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色の範囲
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 黄色の範囲
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 赤色のマスク
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 | mask_red2

    # 黄色のマスク
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 盤面の初期化
    board3 = [['-' for _ in range(7)] for _ in range(6)]

    # 各セルを調査
    for row in range(6):
        for col in range(7):
            x_start, y_start = col * cell_width, row * cell_height
            x_end, y_end = x_start + cell_width, y_start + cell_height

            # セルごとの色の平均を計算
            cell_red = np.sum(mask_red[y_start:y_end, x_start:x_end])
            cell_yellow = np.sum(mask_yellow[y_start:y_end, x_start:x_end])

            if cell_red > cell_yellow and cell_red > 5000:  # 赤色のピクセルが多い
                board3[row][col] = 'R'
            elif cell_yellow > cell_red and cell_yellow > 5000:  # 黄色のピクセルが多い
                board3[row][col] = 'Y'

    return board3

# ChatGPT APIとの連携
def send_to_chatgpt(board3):
    board_string = "\n".join(["".join(row) for row in board3])

    openai.api_key = ""  # OpenAI APIキー（適切なものに変更）

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは重力四目並べの盤面を解析するアシスタントです。"},
            {"role": "user", "content": f"次の盤面の駒配置を確認してください。\n"
                                        f"「R」は赤の駒、「Y」は黄色の駒、「-」は空白を意味します。\n"
                                        f"説明は不要で、6行7列のリスト形式で出力してください。\n"
                                        f"盤面:\n{board_string}"}
        ]
    )

    return response["choices"][0]["message"]["content"]

# メイン関数
if __name__ == "__main__":
    image_path = "output.jpg"

    try:
        board3 = detect_pieces(image_path)  # 画像解析して直接 board3 を作成
        chatgpt_response = send_to_chatgpt(board3)  # ChatGPT に盤面を送る

        print("解析された盤面:")
        for row in board3:
            print(row)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
def find_changed_position(previous_board, current_board):
    """
    前回の盤面と現在の盤面を比較して変わった位置を特定する。
    :param previous_board: 前回の盤面 (6x7のリスト)
    :param current_board: 現在の盤面 (6x7のリスト)
    :return: 変わった駒の位置 (行, 列) または None (変化がない場合)
    """
    rows, cols = len(previous_board), len(previous_board[0])
    for r in range(rows):
        for c in range(cols):
            if previous_board[r][c] != current_board[r][c]:
                return (r, c)
    return None


from collections import Counter


def majority_vote(boards):
    rows, cols = len(boards[0]), len(boards[0][0])
    result_board = [[None] * cols for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            values = [board[r][c] for board in boards]
            most_common_value, _ = Counter(values).most_common(1)[0]
            result_board[r][c] = most_common_value
    
    return result_board

# board1, board2, board3 は関数の場合、() をつけて呼び出す
boards = [board1(), board2, board3]  

# 多数決で現在の盤面を生成
now_board = majority_vote(boards)

# 最終盤面の出力
print("最終盤面:")
for row in now_board:
    print(" ".join(map(str, row)))  # 数値を文字列に変換



# 勝敗判定
def check_winner(board, player):
    """4連続があるかを確認する"""
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            # 横方向
            if c + 3 < cols and all(board[r][c+i] == player for i in range(4)):
                return True
            # 縦方向
            if r + 3 < rows and all(board[r+i][c] == player for i in range(4)):
                return True
            # 右下方向（斜め）
            if r + 3 < rows and c + 3 < cols and all(board[r+i][c+i] == player for i in range(4)):
                return True
            # 左下方向（斜め）
            if r + 3 < rows and c - 3 >= 0 and all(board[r+i][c-i] == player for i in range(4)):
                return True
    return False

def read_board(filename):
    """ファイルから盤面を読み込む"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            board = [list(line.strip()) for line in f.readlines()]
        return board
    except FileNotFoundError:
        print(f"{filename} が見つかりません。")
        return None
def find_changed_position(previous_board, now_board):
    """前回の盤面と現在の盤面を比較し、変化した位置を探す"""
    for row in range(len(now_board)):
        for col in range(len(now_board[0])):
            if previous_board[row][col] != now_board[row][col]:
                return row, col  # 変更があった行と列を返す
    return None  # 変化なし

# 現在の盤面を `now_board` に変更
if check_winner(now_board, 'R'):
    print("赤の勝利！")
    
elif check_winner(now_board, 'Y'):
    print("黄色の勝利！")
    
else:
    print("勝者なし。ゲーム継続可能。")
# 前回の盤面を読み込み
previous_board = read_board("board.txt")
# 変化を確認
if previous_board:
    # 変化を確認
    changed_position = find_changed_position(previous_board, now_board)
    if changed_position is not None:
        changed_row, changed_col = changed_position
        with open("next_number.txt", "w", encoding="utf-8") as f:
            f.write(f"{changed_col}")  # 行・列の番号を1から始める
        print(f"変わった駒の位置: 行 {changed_row + 1}, 列 {changed_col + 1}")
    else:
        print("盤面に変化はありません。")

