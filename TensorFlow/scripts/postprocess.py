"""
後處理模組 (Post-Processing Module)
===================================
對 LYT-Net 模型輸出進行後處理，進一步提升影像品質。

後處理流程：
  模型輸出 (RGB, float32, [0,1])
      │
      ▼
  RGB → YCrCb
      │
      ├── Y 通道 → CLAHE（局部對比度增強）→ Y'
      ├── Cr 通道（不動）
      └── Cb 通道（不動）
      │
      ▼
  YCrCb → RGB（合併）
      │
      ▼
  Guided Filter（邊緣保留平滑）
      │
      ▼
  最終輸出
"""

import cv2
import numpy as np


def apply_clahe_y_channel(image_np, clip_limit=1.5, tile_grid_size=(8, 8)):
    """
    對 Y（亮度）通道施加 CLAHE（對比度限制自適應直方圖均衡化）。

    Args:
        image_np:       RGB 影像，float32，值域 [0, 1]，shape (H, W, 3)
        clip_limit:     對比度限制閾值（越大增強越強，建議 1.0~2.0）
        tile_grid_size: 局部區域的網格大小

    Returns:
        處理後的 RGB 影像，float32，值域 [0, 1]
    """
    # float32 [0,1] → uint8 [0,255]（CLAHE 需要 uint8）
    img_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)

    # RGB → YCrCb
    ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    # 對 Y 通道做 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_enhanced = clahe.apply(y_channel)

    # 合併回 YCrCb → RGB
    ycrcb_enhanced = cv2.merge([y_enhanced, cr_channel, cb_channel])
    rgb_enhanced = cv2.cvtColor(ycrcb_enhanced, cv2.COLOR_YCrCb2RGB)

    # uint8 → float32 [0,1]
    return rgb_enhanced.astype(np.float32) / 255.0


def apply_guided_filter(image_np, radius=2, eps=0.006):
    """
    對整張影像施加 Guided Filter（導向濾波）進行邊緣保留平滑。

    使用影像自身作為引導圖（self-guided），在去除殘留雜訊的同時
    保持邊緣銳利度。

    Args:
        image_np: RGB 影像，float32，值域 [0, 1]，shape (H, W, 3)
        radius:   濾波視窗半徑（越大平滑越強，建議 2~8）
        eps:      正則化參數（越小越銳利，越大越平滑，建議 0.001~0.1）

    Returns:
        處理後的 RGB 影像，float32，值域 [0, 1]
    """
    # Guided Filter 使用自身作為引導圖
    filtered = cv2.ximgproc.guidedFilter(
        guide=image_np,
        src=image_np,
        radius=radius,
        eps=eps
    )
    return np.clip(filtered, 0.0, 1.0)


def postprocess(image_np, clahe_clip=1.5, guided_radius=2, guided_eps=0.001):
    """
    完整後處理流程：CLAHE (Y通道) → Guided Filter (全圖)

    Args:
        image_np:      RGB 影像，float32，值域 [0, 1]，shape (H, W, 3)
        clahe_clip:    CLAHE 對比度限制閾值
        guided_radius: Guided Filter 視窗半徑
        guided_eps:    Guided Filter 正則化參數

    Returns:
        後處理完成的 RGB 影像，float32，值域 [0, 1]
    """
    # Step 1: CLAHE 增強 Y 通道對比度
    # result = apply_clahe_y_channel(image_np, clip_limit=clahe_clip)

    # Step 2: Guided Filter 邊緣保留平滑
    result = apply_guided_filter(image_np, radius=guided_radius, eps=guided_eps)

    return result
