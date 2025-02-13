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
    0: (264.6974, -116.0947, -25.9729, 0.8906),
    1: (268.8838, -78.9317, -24.5473, 2.2826),
    2: (270.0009, -44.0094, -27.6715, 1.6512),
    3: (267.5045, -5.9036, -32.3533, -1.2365),
    4: (273.2483, 32.7436, -21.4395, -2.3840),
    5: (269.2720, 71.9551, -21.5785, -5.6963),
    6: (266.9661, 105.0571, -27.5011, -6.6486)
}

# Dobotの座標設定（各列に対応する座標）
positions2 = {
    0: (260.1360,-119.2503,40.6070,-6.4781),
    1: (262.4670,-79.3871,40.3508,-6.4781),
    2: (261.6420,-36.4297,40.2099,3.6175),
    3: (262.8202,260.8202,40.2102,0.6895),
    4: (260.9515,39.7704,40.2110,3.0655),
    5: (265.9533,75.6747,40.2115,-2.2865),
    6: (262.5227,112.6349,40.2112,-2.2865)
}

# Connect Four のゲームクラス
class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = -1  # AI starts

    def reset(self):
        self.board.fill(0)
        self.current_player = -1  # AIが先手
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
        with open("board.txt", "w") as file:
            for row in self.board:
                file.write("".join(['Y' if cell == -1 else 'R' if cell == 1 else '-' for cell in row]) + "\n")
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

# Gripper open (release object)
print("グリッパーを開きます...")
lastIndex = dType.SetEndEffectorGripper(api, enableCtrl=True, on=False, isQueued=1)[0]
dType.dSleep(1000)

# Move to the specified coordinates
x1, y1, z1, r1 = 13.2638,-272.4662,28.4801,-7.7490 #コマを掴む上座標    
print("コマを補充しています！")
lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

# Wait for the robot to reach the target position
while dType.GetQueuedCmdCurrentIndex(api)[0] < lastIndex:
    dType.dSleep(100)

# Move to the specified coordinates
x1, y1, z1, r1 = 7.5092,-283.3722,-12.6227,-5.6169 #コマを掴む下座標    
print("コマを補充しています！")
lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

# Wait for the robot to reach the target position
while dType.GetQueuedCmdCurrentIndex(api)[0] < lastIndex:
    dType.dSleep(100)

print("コマを掴みました！")

# Gripper close (grasp object)
print("グリッパーを閉じます...")
lastIndex =dType.SetEndEffectorGripper(api, enableCtrl=True, on=True, isQueued=1)[0]
dType.dSleep(2000)

# Move to the specified coordinates
x1, y1, z1, r1 = 12.0141,-279.1613,65.2446,-0.4345 #コマを掴む上２座標    
print("コマを補充しています！")
lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

# Wait for the robot to reach the target position
while dType.GetQueuedCmdCurrentIndex(api)[0] < lastIndex:
    dType.dSleep(100)

def get_human_move():
    # photo.py を実行
    subprocess.run(["python", "phot2.py"], check=True)
    
    # next_number.txt から数字を読み取る
    try:
        with open("next_number.txt", "r") as file:
            action = int(file.read().strip())
            if action in range(7):  # 0-6 の範囲チェック
                return action
            else:
                print("無効な手が検出されました。再入力してください。")
                return get_human_move()
    except (ValueError, FileNotFoundError):
        print("無効な入力またはファイルが見つかりません。再入力してください。")
        return get_human_move()



while not done:
    if env.current_player == 1:  # AIのターン
        valid_moves = env.valid_moves()
        action = agent.select_action(state, valid_moves)
        print(f"AIの選択: 列 {action}")
        


        # Dobotを動かす
        if action in positions:
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 229.3698, -10.6802, 40.9014, -2.6660, isQueued=1)
            dType.dSleep(3000)  # Dobotの動作を待つ
            
            x, y, z, r = positions[action]
            print(f"Dobotが列 {action} に移動します。")
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, r, isQueued=1)
            dType.dSleep(3000)  # Dobotの動作を待つ

            # Gripper open (release object)
            print("グリッパーを開きます...")
            dType.SetEndEffectorGripper(api, enableCtrl=True, on=False, isQueued=1)[0]
            dType.dSleep(2000)

             # Gripper close (grasp object)
            print("グリッパーを閉じます...")
            lastIndex =dType.SetEndEffectorGripper(api, enableCtrl=True, on=True, isQueued=1)[0]
            dType.dSleep(2000)

            # Move to the specified coordinates
            x1, y1, z1, r1 = 13.2638,-272.4662,28.4801,-7.7490   # コマを掴む上座標
            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

            print("グリッパーを開きます...")
            dType.SetEndEffectorGripper(api, enableCtrl=True, on=False, isQueued=1)[0]
            dType.dSleep(4000)

            # Move to the specified coordinates
            x1, y1, z1, r1 = 7.5092,-283.3722,-12.6227,-5.6169   # コマを掴む座標
            print("コマを補充しています！")
            lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]

            # Gripper close (grasp object)
            print("グリッパーを閉じます...")
            lastIndex =dType.SetEndEffectorGripper(api, enableCtrl=True, on=True, isQueued=1)[0]
            dType.dSleep(3000)
            
            x1, y1, z1, r1 = 12.0141,-279.1613,65.2446,-0.4345 #コマを掴む上２座標    
            print("コマを補充しています！")
            lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x1, y1, z1, r1, isQueued=1)[0]
            
            # Wait for the robot to reach the target position
            while dType.GetQueuedCmdCurrentIndex(api)[0] < lastIndex:
                dType.dSleep(100)

            #ポンプ吸引切断
            dType.SetEndEffectorSuctionCup(api, True, False, isQueued=1)
            print("吸引切断")

    else:  # 人間のターン
        while True:
            try:
                user_input = int(input("駒を置き終わったら1を入力してください: "))
                if user_input == 1:
                    action = get_human_move()
                    print(f"あなたの選択: 列 {action}")
                    break
                else:
                    print("無効な入力です。もう一度入力してください。")
            except ValueError:
                print("数字を入力してください。")

    # ゲームの更新
    state, reward, done = env.step(action)
    env.render()

    # 終了判定
    if done:
        if reward == 1:
            print("あなたの勝ち！" if env.current_player == 1 else "ＡＩの勝ち！")
        else:
            print("引き分け！")
        break

# Dobotの停止
dType.SetQueuedCmdStopExec(api)
dType.DisconnectDobot(api)
print("ゲーム終了！Dobotを切断しました。")