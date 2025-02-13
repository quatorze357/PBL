import threading
import numpy as np
import random
import DobotDllType as dType
from collections import defaultdict
import subprocess
# Dobotの設定
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}

api = dType.load()
state = dType.ConnectDobot(api, "COM5", 115200)[0]
print("Connect status:", CON_STR[state])

if state != dType.DobotConnect.DobotConnect_NoError:
    print("Dobotの接続に失敗しました。")
    exit()

# Dobotの初期設定
dType.SetQueuedCmdClear(api)
dType.SetHOMEParams(api, 229.3698, -10.6802, 40.9014, -2.6660, isQueued=1)
dType.SetPTPCommonParams(api, 100, 100, isQueued=1)
dType.SetHOMECmd(api, temp=0, isQueued=1)
dType.SetQueuedCmdStartExec(api)

# Dobotの座標設定（各列に対応する座標）
positions = {
    0: (267.6974, -116.0947, -25.9729, 0.8906),
    1: (270.8838, -78.9317, -24.5473, 2.2826),
    2: (274.0009, -44.0094, -27.6715, 1.6512),
    3: (273.5045, -5.9036, -32.3533, -1.2365),
    4: (275.2483, 32.7436, -21.4395, -2.3840),
    5: (273.2720, 71.9551, -21.5785, -5.6963),
    6: (269.9661, 105.0571, -27.5011, -6.6486)
}

# Connect Four のゲームクラス
class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 人間が先手

    def reset(self):
        self.board.fill(0)
        self.current_player = 1  # 人間が先手
        return self.board

    def step(self, action):
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        winner = self.check_winner()
        done = winner is not None or np.all(self.board != 0)

        reward = 1 if winner == self.current_player else -1 if winner is not None else 0
        self.current_player *= -1
        return self.board.copy(), reward, done

    def valid_moves(self):
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def check_winner(self):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[row, col] == 0:
                    continue
                for dr, dc in directions:
                    if self.check_direction(row, col, dr, dc):
                        return self.board[row, col]
        return None

    def check_direction(self, row, col, dr, dc):
        player = self.board[row, col]
        for i in range(1, 4):
            r, c = row + dr * i, col + dc * i
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols or self.board[r, c] != player:
                return False
        return True

    def render(self):
        print(self.board)

# MuZeroエージェント
class MuZeroAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.policy = defaultdict(lambda: np.ones(action_size) / action_size)

    def select_action(self, state, valid_moves):
        probabilities = self.policy[tuple(map(tuple, state))]
        probabilities = probabilities[valid_moves]
        probabilities /= probabilities.sum()
        return np.random.choice(valid_moves, p=probabilities)

# AIと人間の対戦
env = ConnectFour()
agent = MuZeroAgent(action_size=env.cols)

state = env.reset()
done = False
env.render()
def initialize_board():
    with open("board.txt", "w") as file:
        for _ in range(6):
            file.write("- " * 6 + "-\n")  # 7列にするために "- " * 6 + "-" にする

initialize_board()  # ゲーム開始時に初期化
def get_human_move():
    # `phot.py` の実行でエラーが発生しても続行できるようにする
    try:
        result = subprocess.run(["python", "phot.py"], check=True, capture_output=True, text=True)
        print("phot.py の出力:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("phot.py の実行に失敗しました:", e)
        return get_human_move()

    # `next_number.txt` の読み取り
    try:
        with open("next_number.txt", "r") as file:
            action = int(file.read().strip())
            if action in range(7):  # 0-6 の範囲チェック
                return action
            else:
                print("無効な手が検出されました。再入力してください。")
    except (ValueError, FileNotFoundError) as e:
        print(f"エラー: {e}。再入力してください。")

    return get_human_move()

while not done:
    if env.current_player == 1:  # 人間のターン
        try:
            user_input = int(input("駒を置き終わったら1を入力してください: "))
            if user_input == 1:
                action = get_human_move()
                print(f"あなたの選択: 列 {action}")
                state, reward, done = env.step(action)
                env.render()
                continue  # 次のターンへ進む
            else:
                print("無効な入力です。もう一度入力してください。")
        except ValueError:
            print("数字を入力してください。")
    else:  # AIのターン
        valid_moves = env.valid_moves()
        action = agent.select_action(state, valid_moves)
        print(f"AIの選択: 列 {action}")
        
        # 盤面を board.txt に保存
        with open("board.txt", "w") as file:
            for row in state:
                file.write(" ".join(map(str, row)) + "\n")

        if action in positions:
            x, y, z, r = positions[action]
            print(f"Dobotが列 {action} に移動します。")
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r, isQueued=1)
            dType.dSleep(3000)

            print("グリッパーを開きます...")
            dType.SetEndEffectorGripper(api, enableCtrl=True, on=False, isQueued=1)[0]
            dType.dSleep(2000)

            # Gripper close (grasp object)
            print("グリッパーを閉じます...")
            lastIndex =dType.SetEndEffectorGripper(api, enableCtrl=True, on=True, isQueued=1)[0]
            dType.dSleep(2000)

            # Move to the specified coordinates
            x1, y1, z1, r1 = 33.1957, -271.5589, 60.5876, -92.1987   # コマを掴む上座標
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

            print("グリッパーを開きます...")
            dType.SetEndEffectorGripper(api, enableCtrl=True, on=False, isQueued=1)[0]
            dType.dSleep(2000)

            # Move to the specified coordinates
            x1, y1, z1, r1 = 27.7156, -257.2699, 29.1780, -93.0193   # コマを掴む下座標
            print("コマを補充しています！")
            lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

            # Wait for the robot to reach the target position
            while dType.GetQueuedCmdCurrentIndex(api)[0] < lastIndex:
                dType.dSleep(100)

            # Gripper close (grasp object)
            print("グリッパーを閉じます...")
            lastIndex =dType.SetEndEffectorGripper(api, enableCtrl=True, on=True, isQueued=1)[0]
            dType.dSleep(2000)

            #ポンプ吸引切断
            dType.SetEndEffectorSuctionCup(api, True, False, isQueued=1)
            print("吸引切断")

    state, reward, done = env.step(action)
    env.render()

    if done:
        if reward == 1:
            print("AIの勝ち！" if env.current_player == 1 else "あなたの勝ち！")
        else:
            print("引き分け！")
        break

dType.SetQueuedCmdStopExec(api)
dType.DisconnectDobot(api)
print("ゲーム終了！Dobotを切断しました。")
