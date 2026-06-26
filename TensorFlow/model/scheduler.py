import tensorflow as tf
import math

# =============================================================================
# 餘弦退火 + 週期性重啟 學習率排程器 (Cosine Annealing with Warm Restarts)
# -----------------------------------------------------------------------------
# 概念：學習率按照餘弦曲線從高到低衰減，衰減到底之後「重啟」回高點，
#       每次重啟後的週期長度會乘以 t_mul 倍（越來越長），
#       幫助模型跳出局部最小值、探索更好的解。
#
# 視覺化 (假設 t_mul=2)：
#   LR
#   ↑  ╭─╮
#   │  │  ╲       ╭────╮
#   │  │   ╲     ╱      ╲          ╭────────╮
#   │  │    ╲   ╱        ╲        ╱          ╲
#   │  │     ╰─╯          ╰──────╯            ╰─ ...
#   └──┼─────────────────────────────────────────→ step
#      第1週期   第2週期(2倍長)   第3週期(4倍長)
# =============================================================================


class CosineDecayWithRestartsLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    帶有週期性重啟的餘弦衰減學習率排程器。

    參考論文: "SGDR: Stochastic Gradient Descent with Warm Restarts"
              (Loshchilov & Hutter, 2017)
    """

    def __init__(self, initial_lr, min_lr, total_steps, first_decay_steps, t_mul=2.0, m_mul=1.0):
        """
        Args:
            initial_lr       (float): 初始（最大）學習率，例如 2e-4
            min_lr           (float): 最小學習率下限，例如 1e-6
            total_steps      (int)  : 訓練的總步數 = total_epochs × steps_per_epoch
            first_decay_steps(int)  : 第一個餘弦週期的步數（之後每個週期乘以 t_mul）
            t_mul            (float): 週期長度倍增因子，預設 2.0
                                      → 第1週期: first_decay_steps
                                      → 第2週期: first_decay_steps × 2
                                      → 第3週期: first_decay_steps × 4 ...
            m_mul            (float): 振幅衰減因子（目前程式碼未使用，保留欄位）
        """
        super(CosineDecayWithRestartsLearningRateSchedule, self).__init__()
        self.initial_lr = initial_lr              # 最大學習率
        self.min_lr = min_lr                      # 最小學習率
        self.total_steps = total_steps            # 訓練總步數
        self.first_decay_steps = first_decay_steps  # 第一週期的步數
        self.t_mul = t_mul                        # 每次重啟後，週期長度的倍數
        self.m_mul = m_mul                        # 振幅衰減倍數（未使用）
        self.alpha = min_lr / initial_lr          # 最小學習率佔初始學習率的比例
                                                  # 用來確保 LR 不會衰減到 0

    def __call__(self, step):
        """
        根據當前訓練步數計算對應的學習率。

        Args:
            step: 當前的全局訓練步數

        Returns:
            當前步數對應的學習率（若超過 total_steps 則回傳 min_lr）
        """

        # ── 步驟 1：計算當前進度（0.0 ~ 1.0）──
        # completed_fraction = 已完成的訓練比例
        completed_fraction = step / self.total_steps

        # ── 步驟 2：算出目前處於第幾次重啟（i_restart = 0, 1, 2, ...）──
        # 利用等比級數的反推公式：
        #   各週期長度: T₀, T₀·t, T₀·t², ...（等比數列）
        #   前 n 個週期的總長度 = T₀ · (1 - t^n) / (1 - t)
        #   解出 n（即 i_restart）就是取 log 後向下取整
        i_restart = tf.floor(
            tf.math.log(1 - completed_fraction * (1 - self.t_mul)) /
            tf.math.log(self.t_mul)
        )

        # ── 步驟 3：計算到當前重啟週期開始前，所有已完成週期佔總步數的比例 ──
        # sum_r = 前 i_restart 個週期的累計長度比例
        # 等比級數求和公式: S_n = (1 - t^n) / (1 - t)
        sum_r = (1 - self.t_mul ** i_restart) / (1 - self.t_mul)

        # ── 步驟 4：計算在當前週期內已完成的進度（0.0 ~ 1.0）──
        # = (總進度 - 之前週期的累計進度) / 當前週期的長度比例
        completed_fraction_since_restart = (completed_fraction - sum_r) / self.t_mul ** i_restart

        # ── 步驟 5：計算當前週期的步數（供參考，此變數未直接使用）──
        decay_steps = self.first_decay_steps * self.t_mul ** i_restart

        # ── 步驟 6：套用餘弦衰減公式 ──
        # cosine_decay 的值域為 [0, 1]：
        #   - 週期開始時 (進度=0) → cos(0) = 1   → LR 最高
        #   - 週期結束時 (進度=1) → cos(π) = -1  → LR 最低
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * completed_fraction_since_restart))

        # ── 步驟 7：映射到 [alpha, 1] 的範圍 ──
        # 確保 LR 不會降到 0，最低會是 min_lr
        decayed = (1 - self.alpha) * cosine_decay + self.alpha

        # ── 步驟 8：乘以初始學習率得到最終 LR ──
        new_lr = self.initial_lr * decayed

        # ── 步驟 9：超出總步數後，固定回傳最小學習率 ──
        return tf.where(step < self.total_steps, new_lr, self.min_lr)
