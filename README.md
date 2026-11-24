# Air Hockey Reinforcement Learning

這是一個以 **Pygame Air Hockey** 為遊戲環境、並使用 **PPO（Proximal Policy Optimization）** 訓練 AI 球拍的強化學習專案。專案包含基礎對戰環境、Gym-compatible 訓練環境、PPO 訓練流程，以及以既有模型為基礎進行 DDA（Dynamic Difficulty Adjustment / 動態難度調整）微調的訓練與測試流程。

本專案的核心概念是：先使用一個已訓練好的基礎模型 A 控制上方球拍，再訓練另一個補償模型 B。B 模型不直接取代 A，而是輸出額外動作，與 A 的動作相加後控制球拍，使 AI 能根據比賽狀態進行更動態的調整。

---

## 專案特色

- 使用 Pygame 實作 Air Hockey 遊戲畫面、球體運動、球拍碰撞、牆面反彈與得分邏輯。
- 使用 Gym 風格的 `Env` 包裝遊戲，方便強化學習訓練。
- 使用 PyTorch 實作 Actor-Critic 網路。
- 使用 PPO 搭配 GAE 進行策略更新。
- 支援模型訓練、模型測試、玩家對 AI，以及 DDA 微調流程。
- 支援 TensorBoard 記錄訓練 reward 與 loss。
- 部分環境支援無視窗渲染模式，方便長時間訓練。

---

## 專案架構

```text
.
├── air_hockey_game.py          # 基礎 Air Hockey 遊戲邏輯
├── air_hockey_game_mouse.py    # 支援滑鼠控制與 render 開關的遊戲版本
├── air_hockey_env.py           # 基礎 Gym 訓練環境
├── no_rulebased_game.py        # 無 rule-based 對手版本，用於模型推論或玩家對 AI
├── ppo.py                      # 初始 PPO 訓練流程
├── run_model.py                # 載入已訓練模型，讓玩家與 AI 對戰
├── DDA_game.py                 # DDA 用遊戲核心，載入基礎模型 A
├── DDA_train_env.py            # DDA Gym 環境，訓練補償模型 B
├── DDA_train.py                # DDA / PPO 訓練腳本
└── DDA_test.py                 # DDA 模型測試腳本
```

---

## 核心模組說明

### 1. 遊戲核心

`air_hockey_game.py` 與 `air_hockey_game_mouse.py` 負責 Air Hockey 的基礎遊戲機制，包括畫面初始化、球與球拍位置、速度更新、得分、碰撞處理、球門判斷與 Pygame 渲染。

`air_hockey_game_mouse.py` 額外支援 `render=False` 的模式，會使用 SDL dummy driver 避免跳出視窗，較適合訓練環境使用。

`DDA_game.py` 是 DDA 流程使用的遊戲核心。它在初始化時會載入一個已訓練好的基礎模型 A，並提供：
- `get_observation()`：產生模型輸入狀態。
- `control_bottom()`：控制下方 rule-based 球拍。
- `move_ball()`：更新球的物理狀態。
- `check_collisions()`：處理球、牆、球拍與障礙物的碰撞。
- `_will_score_with_one_bounce()`：判斷目前球路是否可能直接或經一次側牆反彈後得分。

### 2. Gym 訓練環境

`air_hockey_env.py` 是基礎訓練環境，觀測空間為 18 維，動作空間為 2 維連續控制，適合訓練一般 Air Hockey AI。

`DDA_train_env.py` 是 DDA 訓練環境。它會先讓基礎模型 A 根據目前狀態輸出動作，再將訓練中的 B 模型輸出動作加到 A 的動作上：

```python
combined = base_act + action
```

其中：
- `base_act`：基礎模型 A 的動作。
- `action`：DDA 補償模型 B 的動作。
- `combined`：最終套用到上方球拍的速度控制。

DDA 環境的 reward 主要根據分差變化計算。當分差縮小時給正 reward，分差擴大時給負 reward，分差不變則 reward 為 0。這使 B 模型的訓練目標更偏向「調整遊戲平衡」，而不是單純追求最大化得分。

### 3. Actor-Critic 網路

專案中的主要策略模型採用 Actor-Critic 架構。網路包含共用特徵層，並分成 actor 與 critic 兩個輸出頭：

- Actor：輸出 2 維連續動作，代表球拍在 x、y 方向的速度控制。
- Critic：輸出狀態價值，用於計算 advantage 與 PPO loss。
- DDA 訓練版本中，模型額外使用 `log_std` 來建構連續動作的高斯分佈。

---

## 安裝需求

建議使用 Python 3.9 以上版本。

```bash
pip install numpy pygame gym torch tensorboard
```

若使用 GPU 訓練，請依照你的 CUDA 版本安裝對應的 PyTorch 版本。

---

## 快速開始

### 1. 執行基礎遊戲

```bash
python air_hockey_game.py
```

或執行滑鼠/改良版本：

```bash
python air_hockey_game_mouse.py
```

---

### 2. 訓練基礎 PPO 模型

```bash
python ppo.py
```

訓練過程會：
- 建立 PPO agent，環境是 `air_hockey_env.AirHockeyEnv`（遊戲物理來自 `air_hockey_game_mouse.py`）。
- 與 Air Hockey Gym 環境互動。
- 收集 trajectory，並用 GAE 計算 advantage。
- 依照 PPO loss 更新 actor-critic。
- 週期性儲存模型。
- 將訓練資訊寫入 TensorBoard。

`ppo.py` 預設不開視窗（`AirHockeyEnv(render_mode=None)`）；想看訓練畫面時改成 `render_mode="human"`。

模型預設會儲存在類似以下資料夾：

```text
saved_models_YYYYMMDD_HHMMSS/
```

> **2026-11 修正後需要重新訓練基礎模型。** 這次修正了：PPO 取樣和更新用了不同的動作標準差、自己得分沒有獎勵、距離獎勵沒有和上一步比較、1000 步逾時沒有作用，以及一次反彈射門在擊球前就判斷。舊模型（例如 `saved_models_20250518_151900/`）是在這些錯誤下訓練出來的。
>
> 另外，訓練環境的球拍最高速是 25（`air_hockey_game_mouse.py`），但 `run_model.py` 使用的 `no_rulebased_game.py` 和 DDA 使用的 `DDA_game.py` 是 50。拿新模型對戰或做 DDA 之前，請先統一這幾個檔案的 `PADDLE_MAX_SPEED`。

---

### 3. 與已訓練 AI 對戰

請先確認 `run_model.py` 內的模型路徑存在：

```python
model_path = "saved_models_20250518_151900/model_episode_10500.pth"
```

執行：

```bash
python run_model.py
```

玩家使用滑鼠控制下方球拍，AI 控制上方球拍。若模型是以上方球拍視角控制，通常需要設定：

```python
play_against_ai(model_path, is_upper_paddle=True)
```

---

## DDA 訓練流程

DDA 流程的目標是保留基礎模型 A 的行為能力，並訓練一個補償模型 B 來調整最終動作。

### 1. 準備基礎模型 A

`DDA_train.py` 預設會載入：

```python
A_MODEL_PATH = "saved_models_20250518_151900/model_episode_10500.pth"
```

請確認該檔案存在，或將路徑改成你自己的基礎模型。

### 2. 開始訓練 DDA 補償模型 B

```bash
python DDA_train.py
```

訓練過程會：
- 建立 `DDA_train_env.AirHockeyEnv`。
- 載入基礎模型 A。
- 建立新的 Actor-Critic 作為補償模型 B。
- 使用 PPO 與 GAE 訓練 B。
- 將 reward、policy loss、value loss 與 entropy loss 寫入 TensorBoard。
- 每 10 個 episode 儲存一次 DDA 模型。

輸出資料夾格式：

```text
DDA_saved_models_YYYYMMDD_HHMMSS/
runs/ppo_YYYYMMDD_HHMMSS/
```

---

## DDA 模型測試

請先確認 `DDA_test.py` 中的模型路徑存在：

```python
A_MODEL_PATH = "saved_models_20250518_151900/model_episode_10500.pth"
```

以及 `find_latest_b_model()` 內的 B 模型路徑：

```python
DDA_saved_models_20250719_192533/DDA_agent_ep4470.pth
```

執行：

```bash
python DDA_test.py
```

測試腳本會：
- 載入基礎模型 A。
- 載入 DDA 補償模型 B。
- 進行多個 episode 評估。
- 輸出每回合 total reward。
- 統計平均 reward 與標準差。

---

## TensorBoard

訓練時會將資訊寫入 `runs/` 目錄。可使用以下指令查看訓練曲線：

```bash
tensorboard --logdir runs
```

常見觀察指標包括：

- `Reward/Episode`
- `Loss/Policy`
- `Loss/Value`
- `Loss/Entropy`

---

## 主要超參數

`DDA_train.py` 中定義了主要 PPO 超參數：

```python
LEARNING_RATE        = 3e-4
GAMMA                = 0.99
GAE_LAMBDA           = 0.95
CLIP_EPSILON         = 0.2
VALUE_COEF           = 0.5
ENTROPY_COEF         = 0.01
BATCH_SIZE           = 256
EPOCHS               = 10
TIMESTEPS_PER_UPDATE = 2048
MAX_STEPS_PER_EPISODE = 5000
SEED                 = 123
```

若要調整訓練穩定性，建議優先嘗試：
- 降低 `LEARNING_RATE`
- 調整 `ENTROPY_COEF`
- 增加或減少 `EPOCHS`
- 修改 reward 設計
- 調整 episode 長度或勝利條件

---

## Reward 設計

DDA 環境中的 reward 重點是分差變化：

```python
reward = int(diff_before - diff_after)
```

因此：
- 若 AI 行為讓雙方分差縮小，reward 為正。
- 若分差擴大，reward 為負。
- 若分差不變，reward 為 0。

這種設計適合訓練「動態調整難度」的補償策略，讓 AI 不只是追求勝利，而是根據場上狀態調整行為。

此外，環境中也保留了一些可擴充 reward 權重，例如：
- `one_bounce_goal`
- `edge_penalty`

後續可以根據訓練目標加入更多細緻獎勵，例如擊球獎勵、防守位置獎勵、避免貼邊懲罰、射門角度獎勵等。

---

## 常見問題與注意事項

### 1. 模型路徑目前是硬編碼

部分腳本直接寫死模型路徑，例如：

```python
saved_models_20250518_151900/model_episode_10500.pth
DDA_saved_models_20250719_192533/DDA_agent_ep4470.pth
```

如果檔案不存在，程式會報錯。建議後續改成 argparse 參數，例如：

```bash
python DDA_train.py --base-model path/to/model.pth
python DDA_test.py --base-model path/to/model.pth --dda-model path/to/dda.pth
```

### 2. Pygame 視窗未回應

長時間訓練或 render 時，需定期呼叫：

```python
pygame.event.pump()
```

目前 `DDA_train.py` 在 render 模式下已有處理，以避免視窗未回應。

### 3. 無畫面訓練

若不需要畫面，可以將 render 關閉，例如：

```python
RENDER = False
```

或在環境中使用：

```python
render_mode="none"
```

`air_hockey_env.AirHockeyEnv` 只要 `render_mode` 不是 `"human"` 就不開視窗；DDA 用的 `DDA_game.py` 目前仍會建立視窗，只是不繪圖。

### 4. checkpoint 格式不完全一致

基礎 PPO 模型可能儲存為：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "episode": ...
}
```

DDA 模型則可能直接儲存 `state_dict`。讀取模型時要確認 checkpoint 格式是否一致。

---

## 建議改進方向

- 將模型路徑、訓練回合數、render 開關改成命令列參數。
- 統一 checkpoint 儲存格式。
- 整理檔案命名，例如區分 `base_ppo_*` 與 `dda_*`。
- 將遊戲常數集中到 `config.py`。
- 增加 `requirements.txt`。
- 增加自動尋找最新 checkpoint 的通用工具函式。
- 將 reward 設計抽成獨立模組，方便比較不同訓練策略。
- 增加 evaluation script，輸出勝率、平均分差、平均 episode 長度等指標。

---

## 建議的 requirements.txt

```text
numpy
pygame
gym
torch
tensorboard
```

---

## License

本專案目前尚未指定授權條款。若要公開發布，建議補上 MIT、Apache-2.0 或其他合適的開源授權。
