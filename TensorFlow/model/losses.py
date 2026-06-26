# =============================================================================
# LYT-Net 損失函數模組
# =============================================================================
# 本檔案定義了 6 種損失函數，並在最後的 loss() 中加權組合。
# 設計理念：結合「像素級」、「結構級」、「語義級」、「統計級」四個層次，
#           讓模型同時學習精確的像素重建、正確的結構、自然的視覺感知、
#           以及一致的色彩與亮度分布。
#
# 損失函數總覽：
# ┌─────────────────┬────────┬──────────────────────────────────┐
# │ 損失函數         │ 權重   │ 負責的面向                       │
# ├─────────────────┼────────┼──────────────────────────────────┤
# │ Smooth L1       │ 1.00   │ 像素級重建（主損失）              │
# │ MS-SSIM         │ 0.50   │ 多尺度結構相似度                  │
# │ Color           │ 0.25   │ 全局色彩一致性                    │
# │ Perceptual(VGG) │ 0.06   │ 高階語義/紋理特徵                 │
# │ Histogram       │ 0.05   │ 亮度分布對齊                      │
# │ PSNR            │ 0.0083 │ 峰值信噪比約束                    │
# └─────────────────┴────────┴──────────────────────────────────┘
# =============================================================================

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model


# =============================================================================
# 1. 色彩損失 (Color Loss)
# =============================================================================
# 目的：確保輸出圖片的「整體色調」與目標一致。
# 做法：分別算真實圖 & 預測圖在空間維度 (H, W) 的平均 RGB 值，
#       再取兩者差的絕對值 → 全局色彩偏差。
# 直覺：如果整張圖偏黃或偏藍，這個損失會懲罰。
def color_loss(y_true, y_pred):
    # reduce_mean(axis=[1,2]) → 每張圖每個通道的平均值 (shape: [B, C])
    return tf.reduce_mean(tf.abs(tf.reduce_mean(y_true, axis=[1, 2]) - tf.reduce_mean(y_pred, axis=[1, 2])))


# =============================================================================
# 2. PSNR 損失 (Peak Signal-to-Noise Ratio Loss)
# =============================================================================
# PSNR 越高越好（最高理論值 ≈ 48 dB for 8-bit）。
# 用 40 - PSNR 轉成「越低越好」的損失值。
# 40 dB 是經驗 anchor，超過 40 dB 的改善不會被過度懲罰。
def psnr_loss(y_true, y_pred):
    return 40.0 - tf.image.psnr(y_true, y_pred, max_val=1.0)


# =============================================================================
# 3. 感知損失 (Perceptual Loss) — 使用預訓練 VGG19
# =============================================================================
# 人眼感知的「相似度」不只是像素差異，還包含紋理、邊緣、物體結構。
# 透過預訓練 VGG19 提取中層特徵，比較特徵空間的差異。

def load_vgg():
    """
    載入預訓練 VGG19 並截取到 block3_conv3 層作為特徵提取器。

    為什麼選 block3_conv3？
    - 太淺 (block1)：只能捕捉邊緣，語義不夠
    - 太深 (block5)：太抽象，失去空間細節
    - block3 是平衡語義豐富度與空間解析度的甜蜜點
    """
    vgg = VGG19(include_top=False, weights='imagenet')  # ImageNet 預訓練權重
    vgg.trainable = False                                # 凍結，不參與訓練
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    return loss_model


def perceptual_loss(y_true, y_pred, loss_model):
    """感知損失 = VGG 特徵空間中的 MSE: mean((VGG(true) - VGG(pred))²)"""
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))


# =============================================================================
# 4. 平滑 L1 損失 (Smooth L1 / Huber Loss) — 主損失
# =============================================================================
# 公式：|diff| < 1 → 0.5 × diff²（平方，對小誤差敏感）
#       |diff| ≥ 1 → |diff| - 0.5（線性，對離群值穩定）
# 優勢：比 L2 對大誤差更穩定，比 L1 在接近 0 時更平滑。
def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)                                   # 逐像素絕對差
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)          # mask: diff < 1
    # 分段函數：小誤差用二次方，大誤差用線性
    smooth_l1_loss = (0.5 * diff**2) * less_than_one + (diff - 0.5) * (1.0 - less_than_one)
    return tf.reduce_mean(smooth_l1_loss)


# =============================================================================
# 5. 多尺度結構相似度損失 (Multi-Scale SSIM Loss)
# =============================================================================
# 在多個尺度上評估亮度、對比度、結構三個面向。
# 值域 [0, 1]，1 = 完全相同 → 取 1 - MS_SSIM 作為損失。
# power_factors 控制各尺度權重（[0.5, 0.5] = 兩個尺度等權）。
def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val, power_factors=power_factors))


# =============================================================================
# 6. 直方圖損失 (Histogram Loss)
# =============================================================================
# 目的：確保輸出圖片的「亮度分布」與目標一致。
# 傳統直方圖不可微 → 用高斯核做可微分的「軟直方圖」(soft histogram)。
# 做法：
#   1. 建立 256 個 bin (0~1 等分)
#   2. 每個像素用高斯核軟性投票到各 bin (sigma 控制平滑程度)
#   3. 正規化成概率分布
#   4. 計算兩個分布的 L1 距離
def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = tf.linspace(0.0, 1.0, bins)           # 256 個等距 bin 邊界

    def gaussian_kernel(x, mu, sigma):
        """高斯核：像素值 x 對 bin 中心 mu 的投票權重"""
        return tf.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # [..., tf.newaxis] 讓像素值廣播到所有 bin 進行計算
    y_true_hist = tf.reduce_sum(gaussian_kernel(y_true[..., tf.newaxis], bin_edges, sigma), axis=0)
    y_pred_hist = tf.reduce_sum(gaussian_kernel(y_pred[..., tf.newaxis], bin_edges, sigma), axis=0)

    # 正規化成概率分布（總和 = 1）
    y_true_hist /= tf.reduce_sum(y_true_hist)
    y_pred_hist /= tf.reduce_sum(y_pred_hist)

    # L1 距離
    hist_distance = tf.reduce_mean(tf.abs(y_true_hist - y_pred_hist))

    return hist_distance


# =============================================================================
# 總損失函數 — 加權組合所有損失
# =============================================================================
def loss(y_true, y_pred, loss_model):
    """
    計算最終的加權複合損失。

    Args:
        y_true:     真實目標圖片 ([-1, 1]，來自 data_loading 的正規化)
        y_pred:     模型預測圖片 ([-1, 1]，最終層用 tanh 激活)
        loss_model: 預訓練 VGG19 特徵提取器
    Returns:
        total_loss: 加權組合後的總損失值
    """

    # ── 將 [-1, 1] 還原為 [0, 1]（PSNR/SSIM/VGG 期望此範圍）──
    y_true = (y_true + 1.0) / 2.0
    y_pred = (y_pred + 1.0) / 2.0

    # ── 各損失的權重 ──
    alpha1 = 1.00    # Smooth L1  — 像素級主損失（權重最大，主導訓練）
    alpha2 = 0.06    # Perceptual — 語義特徵匹配
    alpha3 = 0.05    # Histogram  — 亮度分布對齊
    alpha4 = 0.5     # MS-SSIM    — 結構保持（第二大權重）
    alpha5 = 0.0083  # PSNR       — 信噪比（極小權重，輔助）
    alpha6 = 0.25    # Color      — 色彩一致性（防止色偏）

    # ── 分別計算各項損失 ──
    smooth_l1_l = smooth_l1_loss(y_true, y_pred)
    ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
    perc_l = perceptual_loss(y_true, y_pred, loss_model=loss_model)
    hist_l = histogram_loss(y_true, y_pred)
    psnr_l = psnr_loss(y_true, y_pred)
    color_l = color_loss(y_true, y_pred)

    # ── 加權求和 ──
    total_loss = alpha1 * smooth_l1_l + alpha2 * perc_l + alpha3*hist_l + alpha5*psnr_l + alpha6*color_l+ alpha4*ms_ssim_l
    total_loss = tf.reduce_mean(total_loss)
    return total_loss
