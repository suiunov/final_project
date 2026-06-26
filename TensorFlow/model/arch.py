# =============================================================================
# LYT-Net 模型架構 (Lightweight YUV Transformer-based Network)
# =============================================================================
# 論文: "LYT-Net: Lightweight YUV Transformer-based Network
#        for Low-Light Image Enhancement" (Brateanu et al., 2024)
#
# 核心設計思想：
#   1. 將 RGB 轉為 YUV (YCbCr) 色彩空間，分離「亮度 (Y)」與「色度 (Cb, Cr)」
#   2. 色度通道 (Cb, Cr) 各自經過獨立的 Denoiser 降噪
#   3. 亮度通道 (Y) 使用 Multi-Head Self-Attention 進行全局增強
#   4. 最後透過 MSEF 融合亮度與色度資訊，輸出增強的 RGB 圖片
#
# 整體架構流程圖：
#
#   輸入 RGB ──→ RGB_to_YUV ──┬── Y  ──→ process_y ──→ MHSA池化增強 ──┐
#                              ├── Cb ──→ Denoiser_cb ──→ process_cb ──┤
#                              └── Cr ──→ Denoiser_cr ──→ process_cr ──┤
#                                                                      ↓
#                                           亮度引導色度融合 (MSEF) ──→ 輸出 RGB
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras


# =============================================================================
# MSEFBlock (Multi-Scale Enhancement Fusion Block)
# =============================================================================
# 多尺度增強融合模組：結合「局部特徵」（DepthwiseConv）與「通道注意力」（SE）
#
# 流程：
#   input ──→ LayerNorm ──┬── DepthwiseConv (局部空間特徵) ──→ x1 ──┐
#                          └── SEBlock (通道注意力權重)     ──→ x2 ──┤
#                                                                    ↓
#                                                              x1 × x2 (逐元素相乘)
#                                                                    ↓
#                                                         + input (殘差連接) ──→ output
class MSEFBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(MSEFBlock, self).__init__(**kwargs)
        # LayerNorm：對最後一個軸（通道）做正規化，穩定訓練
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        # DepthwiseConv2D：每個通道獨立做 3×3 卷積，提取局部空間特徵（參數量極少）
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        # SEBlock：通道注意力，學習每個通道的重要性權重
        self.se_attn = SEBlock(filters)

    def call(self, inputs):
        x = self.layer_norm(inputs)          # 正規化輸入
        x1 = self.depthwise_conv(x)          # 局部空間特徵（「在哪裡」重要）
        x2 = self.se_attn(x)                 # 通道注意力（「哪個特徵」重要）
        x_fused = layers.Multiply()([x1, x2])  # 門控融合：空間 × 通道
        x_out = layers.Add()([x_fused, inputs])  # 殘差連接：保留原始資訊
        return x_out


# =============================================================================
# SEBlock (Squeeze-and-Excitation Block) — 通道注意力機制
# =============================================================================
# 論文: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
#
# 核心思想：不是所有通道都一樣重要，讓網路「自動學習」每個通道的權重。
#
# 流程：
#   input (B,H,W,C) ──→ GlobalAvgPool (B,C) ──→ FC₁ (C → C/r, ReLU)
#                                                ──→ FC₂ (C/r → C, Tanh)
#                                                ──→ Reshape (B,1,1,C)
#                                                ──→ input × scale ──→ output
#
# 注意：這裡用 tanh 而非原始論文的 sigmoid，允許負值縮放（範圍 [-1, 1]）
class SEBlock(layers.Layer):
    def __init__(self, input_channels, reduction_ratio=16, **kwargs):
        """
        Args:
            input_channels:   輸入通道數 C
            reduction_ratio:  壓縮比例 r，瓶頸層維度 = C // r（減少參數量）
        """
        super().__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling2D()   # 壓縮空間維度 (H,W) → 1
        self.fc1 = layers.Dense(input_channels // reduction_ratio, activation='relu')  # 瓶頸層
        self.fc2 = layers.Dense(input_channels, activation='tanh')   # 還原通道數

    def call(self, inputs):
        x = self.pool(inputs)                                   # (B,H,W,C) → (B,C)
        x = self.fc1(x)                                         # (B,C) → (B, C//r)
        x = self.fc2(x)                                         # (B, C//r) → (B,C)
        scale = tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])     # (B,C) → (B,1,1,C)
        return inputs * scale                                   # 逐通道縮放


# =============================================================================
# MultiHeadSelfAttention (MHSA) — 多頭自注意力機制
# =============================================================================
# 參考: "Attention Is All You Need" (Vaswani et al., 2017)
#
# 核心思想：讓特徵圖中的每個像素「關注」所有其他像素，捕捉全局依賴關係。
# 多頭：將注意力拆成多個獨立的「頭」，每個頭學習不同的關注模式。
#
# 流程：
#   input (B,H,W,C) ──→ Q,K,V 投影 ──→ 拆成 num_heads 個頭
#                  ──→ Scaled Dot-Product Attention
#                  ──→ 合併所有頭 ──→ 線性投影 ──→ output (B,H,W,C)
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        """
        Args:
            embed_size: 嵌入維度（= 通道數），必須能被 num_heads 整除
            num_heads:  注意力頭的數量（本模型用 4 個頭）
        """
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = embed_size // num_heads    # 每個頭的維度 = C / num_heads

        # Q, K, V 的線性投影層
        self.query_dense = layers.Dense(embed_size)    # 查詢 (Query)：「我在找什麼？」
        self.key_dense = layers.Dense(embed_size)      # 鍵 (Key)：「我有什麼？」
        self.value_dense = layers.Dense(embed_size)    # 值 (Value)：「我的內容是什麼？」
        self.combine_heads = layers.Dense(embed_size)  # 合併多頭後的輸出投影

    def split_heads(self, x, batch_size):
        """將最後一維拆成 (num_heads, head_dim)，並轉置方便矩陣乘法。
        (B, H*W, C) → (B, H*W, heads, head_dim) → (B, heads, H*W, head_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def attention(self, query, key, value):
        """
        Scaled Dot-Product Attention:
          Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V

        除以 √d_k 是為了避免 dot product 值過大導致 softmax 梯度消失。
        """
        matmul_qk = tf.matmul(query, key, transpose_b=True)     # Q·Kᵀ → (B, heads, N, N)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)           # d_k = head_dim
        logits = matmul_qk / tf.math.sqrt(depth)                # 縮放
        attention_weights = tf.nn.softmax(logits, axis=-1)       # 歸一化成權重
        output = tf.matmul(attention_weights, value)             # 加權聚合 Value
        return output

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # 計算 Q, K, V
        query = self.query_dense(inputs)    # (B,H,W,C)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # 拆分多頭：(B,H,W,C) → 內部展平為 (B,H*W,C) → (B,heads,H*W,head_dim)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 計算注意力
        attention = self.attention(query, key, value)            # (B,heads,H*W,head_dim)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (B,H*W,heads,head_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_size))  # 合併所有頭
        output = self.combine_heads(concat_attention)            # 最終線性投影
        output = tf.reshape(output, [batch_size, height, width, self.embed_size])  # 還原空間維度
        return output


# =============================================================================
# Denoiser — U-Net 式色度降噪器
# =============================================================================
# 用途：分別對 Cb 和 Cr 色度通道進行降噪（低光影像的色度雜訊特別嚴重）。
#
# 架構：Encoder-Bottleneck-Decoder + Skip Connections
#
#   input(1ch) ──→ conv1(s=1) ──→ conv2(s=2) ──→ conv3(s=2) ──→ conv4(s=2)
#                     │               │               │               ↓
#                     │               │               │         MHSA Bottleneck
#                     │               │               │               ↓
#                     │               │               └───(+)──→ up4 ──→
#                     │               └───────────(+)──→ up3 ──→
#                     └──────────────────(+)──→ up2 ──→ res_layer
#                                                          ↓
#                                                   output_layer(+input) ──→ 殘差輸出
#
# 設計要點：
# - 每層 stride=2 將解析度減半，直到最深層使用 MHSA 捕捉全局色彩關係
# - Skip Connections 保留每層細節，避免降噪過度模糊
# - 最終輸出 + 原始輸入 = 殘差學習（只學「要修正多少」）
# - 輸出用 tanh（範圍 [-1,1]），適合修正量的正負值
class Denoiser(Model):
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        """
        Args:
            num_filters: 每層的卷積濾波器數量（預設 16）
            kernel_size: 卷積核大小（預設 3×3）
            activation:  激活函數（預設 ReLU）
        """
        super(Denoiser, self).__init__()
        # ── Encoder（逐步降低解析度）──
        self.conv1 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=1, padding='same', activation=activation)  # 原始解析度
        self.conv2 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)  # 1/2 解析度
        self.conv3 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)  # 1/4 解析度
        self.conv4 = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same', activation=activation)  # 1/8 解析度

        # ── Bottleneck：在最低解析度使用全局注意力 ──
        self.bottleneck = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)

        # ── Decoder（逐步恢復解析度）──
        self.up2 = layers.UpSampling2D(2)   # 2倍上採樣
        self.up3 = layers.UpSampling2D(2)
        self.up4 = layers.UpSampling2D(2)

        # ── 輸出層（tanh 範圍 [-1,1]）──
        self.output_layer = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='tanh')
        self.res_layer = layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='tanh')

    def call(self, inputs):
        # ── Encoder ──
        x1 = self.conv1(inputs)    # 原始解析度特徵
        x2 = self.conv2(x1)       # 1/2 解析度
        x3 = self.conv3(x2)       # 1/4 解析度
        x4 = self.conv4(x3)       # 1/8 解析度

        # ── Bottleneck ──
        x = self.bottleneck(x4)    # 在最低解析度做全局注意力

        # ── Decoder + Skip Connections ──
        x = self.up4(x)            # 恢復到 1/4 解析度
        x = self.up3(x3 + x)      # 加上 encoder 的 1/4 特徵 → 恢復到 1/2
        x = self.up2(x2 + x)      # 加上 encoder 的 1/2 特徵 → 恢復到原始
        x = x + x1                # 加上 encoder 的原始特徵

        x = self.res_layer(x)     # 壓縮到 1 通道
        return self.output_layer(x + inputs)  # 殘差輸出：修正量 + 原始輸入


# =============================================================================
# LYT — 主模型 (Lightweight YUV Transformer)
# =============================================================================
# 這是整個網路的核心，將上面所有模組串接起來。
#
# 完整流程：
#   1. RGB → YUV 色彩空間轉換
#   2. Cb, Cr 各自通過 Denoiser 降噪（+ 殘差連接）
#   3. Y, Cb, Cr 各自通過獨立的 Conv2D 提取特徵（→ 32 通道）
#   4. Y 通道：下採樣 8x → MHSA 全局注意力 → 上採樣回原始大小
#   5. Cb + Cr 合併為色度參考 (ref)
#   6. 亮度資訊以 0.2 的比例注入色度參考 → MSEF 融合
#   7. 亮度 + 色度融合後 → 最終卷積輸出 3 通道 RGB
class LYT(Model):
    def __init__(self, filters=32, denoiser_cb=None, denoiser_cr=None):
        """
        Args:
            filters:      中間特徵通道數（預設 32）
            denoiser_cb:  Cb 通道的降噪器實例
            denoiser_cr:  Cr 通道的降噪器實例
        """
        super(LYT, self).__init__()

        # ── YUV 三通道各自的特徵提取層 ──
        # 每個通道用 1 層 3×3 Conv2D 將 1 通道映射到 filters 通道
        self.process_y = self._create_processing_layers(filters)    # Y  → (B,H,W,32)
        self.process_cb = self._create_processing_layers(filters)   # Cb → (B,H,W,32)
        self.process_cr = self._create_processing_layers(filters)   # Cr → (B,H,W,32)

        # ── 色度降噪器（外部傳入，因為要獨立 build）──
        self.denoiser_cb = denoiser_cb    # Cb 通道降噪
        self.denoiser_cr = denoiser_cr    # Cr 通道降噪

        # ── 亮度通道的全局注意力路徑 ──
        self.lum_pool = layers.MaxPooling2D(8)     # 8x 下採樣（大幅減少 MHSA 計算量）
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)  # 全局注意力
        self.lum_up = layers.UpSampling2D(8)       # 8x 上採樣回原始大小

        # ── 融合層 ──
        self.lum_conv = layers.Conv2D(filters, (1, 1), padding='same')   # 亮度 → 色度的 1×1 投影
        self.ref_conv = layers.Conv2D(filters, (1, 1), padding='same')   # 色度參考的 1×1 投影
        self.msef = MSEFBlock(filters)                                    # 多尺度增強融合

        # ── 重組合 & 最終輸出 ──
        self.recombine = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')  # 融合亮度+色度
        self.final_adjustments = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')  # 輸出 3 通道 RGB

    def _create_processing_layers(self, filters):
        """建立單通道特徵提取層：1 層 3×3 Conv2D + ReLU"""
        layerz = [layers.Conv2D(filters, (3, 3), activation='relu', padding='same') for _ in range(1)]
        return keras.Sequential(layerz)

    def call(self, inputs):
        """
        前向傳播。

        Args:
            inputs: RGB 圖片 (B, H, W, 3)，值域 [-1, 1]
        Returns:
            output: 增強後的 RGB 圖片 (B, H, W, 3)，值域 [-1, 1]
        """

        # ── 階段 1：RGB → YUV 色彩空間轉換 ──
        ycbcr = tf.image.rgb_to_yuv(inputs)       # (B,H,W,3) → YUV
        y, cb, cr = tf.split(ycbcr, 3, axis=-1)   # 拆成 Y, Cb, Cr 各 (B,H,W,1)

        # ── 階段 2：色度降噪（殘差學習）──
        # denoiser 學習噪聲修正量，加回原始值 = 降噪結果
        cb = self.denoiser_cb(cb) + cb    # Cb 降噪
        cr = self.denoiser_cr(cr) + cr    # Cr 降噪

        # ── 階段 3：各通道特徵提取 (1ch → 32ch) ──
        y_processed = self.process_y(y)      # Y  → (B,H,W,32)
        cb_processed = self.process_cb(cb)   # Cb → (B,H,W,32)
        cr_processed = self.process_cr(cr)   # Cr → (B,H,W,32)

        # ── 階段 4：色度參考合併 ──
        ref = tf.concat([cb_processed, cr_processed], axis=-1)  # (B,H,W,64)

        # ── 階段 5：亮度全局增強 ──
        # 先 8x 下採樣減少計算量 → MHSA 捕捉全局亮度關係 → 上採樣回來
        lum = y_processed
        lum_1 = self.lum_pool(lum)       # (B, H/8, W/8, 32)
        lum_1 = self.lum_mhsa(lum_1)     # 全局自注意力
        lum_1 = self.lum_up(lum_1)       # (B, H, W, 32) 恢復原始大小
        lum = lum + lum_1                # 殘差連接：原始 + 全局增強

        # ── 階段 6：亮度引導色度融合 ──
        ref = self.ref_conv(ref)          # (B,H,W,64) → (B,H,W,32) 通道對齊
        shortcut = ref                    # 保存捷徑用於殘差連接

        # 以 0.2 的比例將亮度資訊注入色度
        # 為什麼只用 0.2？→ 亮度是「引導」而非「取代」，避免亮度主導色度
        ref = ref + 0.2 * self.lum_conv(lum)

        ref = self.msef(ref)             # MSEF：多尺度增強融合
        ref = ref + shortcut             # 殘差連接

        # ── 階段 7：重組合輸出 ──
        # 將亮度特徵與色度特徵拼接 → 卷積融合 → 輸出 3 通道 RGB
        recombined = self.recombine(tf.concat([ref, lum], axis=-1))  # (B,H,W,64) → (B,H,W,32)
        output = self.final_adjustments(recombined)  # (B,H,W,32) → (B,H,W,3)，tanh 輸出
        return output
