import glob
import os
import tensorflow as tf


tf.random.set_seed(
    100
    )


def data_augmentation(raw_img, corrected_img):
    tf.random.set_seed(
    100
    )
    flip_lr = tf.random.uniform(shape=[]) > 0.5
    flip_ud = tf.random.uniform(shape=[]) > 0.5
    rot_k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)

    if flip_lr:
        raw_img = tf.image.flip_left_right(raw_img)
        corrected_img = tf.image.flip_left_right(corrected_img)

    if flip_ud:
        raw_img = tf.image.flip_up_down(raw_img)
        corrected_img = tf.image.flip_up_down(corrected_img)

    raw_img = tf.image.rot90(raw_img, k=rot_k)
    corrected_img = tf.image.rot90(corrected_img, k=rot_k)

    return raw_img, corrected_img


def load_image_test(image_path, crop_margin):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    
    original_shape = tf.shape(img)
    new_height = original_shape[0] - 2 * crop_margin
    new_width = original_shape[1] - 2 * crop_margin

    img = tf.image.crop_to_bounding_box(img, crop_margin, crop_margin, new_height, new_width)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def load_and_preprocess_image(raw_img_path, corrected_img_path):
    tf.random.set_seed(
    100
    )

    raw_img = tf.io.read_file(raw_img_path)
    raw_img = tf.image.decode_png(raw_img, channels=3)

    corrected_img = tf.io.read_file(corrected_img_path)
    corrected_img = tf.image.decode_png(corrected_img, channels=3)

    raw_img = tf.cast(raw_img, tf.float32)
    corrected_img = tf.cast(corrected_img, tf.float32)
    
    stacked_images = tf.stack([raw_img, corrected_img], axis=0) 
    cropped_images = tf.image.random_crop(stacked_images, size=[2, 256, 256, 3]) 
    raw_img, corrected_img = cropped_images[0], cropped_images[1] 

    raw_img = (raw_img / 255.0) * 2 - 1.0
    corrected_img = (corrected_img / 255.0) * 2 - 1.0

    return raw_img, corrected_img

def get_datasets(raw_image_path, corrected_image_path):
    tf.random.set_seed(
    100
    )
    
    train_raw_files = sorted(glob.glob(raw_image_path))
    train_corrected_files = sorted(glob.glob(corrected_image_path))

    train_dataset = tf.data.Dataset.from_tensor_slices((train_raw_files, train_corrected_files))
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    BATCH_SIZE = 1

    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

    return train_dataset
    

def get_datasets_metrics(raw_image_path, corrected_image_path, crop_margin):
    raw_image_files = sorted(glob.glob(raw_image_path))
    corrected_image_files = sorted(glob.glob(corrected_image_path))

    train_raw_files = raw_image_files
    train_corrected_files = corrected_image_files

    train_raw_dataset = tf.data.Dataset.from_tensor_slices(train_raw_files).map(lambda x: load_image_test(x, crop_margin), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_corrected_dataset = tf.data.Dataset.from_tensor_slices(train_corrected_files).map(lambda x: load_image_test(x, crop_margin), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_dataset = tf.data.Dataset.zip((train_raw_dataset, train_corrected_dataset))

    BATCH_SIZE = 1
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset


# =====================================================================
# LoLI-Street / general jpg+png support (added for transfer learning)
# =====================================================================

def _decode_any_image(raw_bytes):
    """Decode both jpg and png transparently."""
    img = tf.image.decode_image(raw_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return img


def _load_and_preprocess_loli(raw_path, gt_path, crop_size):
    """Load a low/high pair, random-crop, and normalise to [-1, 1]."""
    raw_img = _decode_any_image(tf.io.read_file(raw_path))
    gt_img  = _decode_any_image(tf.io.read_file(gt_path))

    raw_img = tf.cast(raw_img, tf.float32)
    gt_img  = tf.cast(gt_img,  tf.float32)

    # Paired random crop (same region for both)
    stacked = tf.stack([raw_img, gt_img], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, crop_size, crop_size, 3])
    raw_img, gt_img = cropped[0], cropped[1]

    # Normalise to [-1, 1]
    raw_img = (raw_img / 255.0) * 2 - 1.0
    gt_img  = (gt_img  / 255.0) * 2 - 1.0

    return raw_img, gt_img


def _load_test_any(image_path, crop_margin):
    """Load a single test image (jpg or png), apply crop margin, normalise."""
    img = _decode_any_image(tf.io.read_file(image_path))
    if crop_margin > 0:
        s = tf.shape(img)
        img = tf.image.crop_to_bounding_box(
            img, crop_margin, crop_margin,
            s[0] - 2 * crop_margin, s[1] - 2 * crop_margin)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


def get_loli_datasets(low_glob, high_glob, crop_size=256, batch_size=2):
    """
    Build a training tf.data.Dataset from LoLI-Street-style directories.

    Args:
        low_glob:   glob pattern for low-light images  (e.g. '.../Train/low/*.jpg')
        high_glob:  glob pattern for ground-truth images (e.g. '.../Train/high/*.jpg')
        crop_size:  random crop size (default 256)
        batch_size: batch size (default 2)
    """
    tf.random.set_seed(100)
    raw_files = sorted(glob.glob(low_glob))
    gt_files  = sorted(glob.glob(high_glob))
    assert len(raw_files) == len(gt_files), \
        f'Mismatch: {len(raw_files)} low vs {len(gt_files)} high images'
    print(f'[data] Loaded {len(raw_files)} training pairs.')

    ds = tf.data.Dataset.from_tensor_slices((raw_files, gt_files))
    # Shuffle file paths BEFORE decoding to avoid OOM
    ds = ds.shuffle(buffer_size=len(raw_files))
    ds = ds.map(lambda r, g: _load_and_preprocess_loli(r, g, crop_size),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_loli_datasets_metrics(low_glob, high_glob, crop_margin=0):
    """
    Build a test/val tf.data.Dataset (no crop, no shuffle, BS=1).
    """
    raw_files = sorted(glob.glob(low_glob))
    gt_files  = sorted(glob.glob(high_glob))
    assert len(raw_files) == len(gt_files), \
        f'Mismatch: {len(raw_files)} low vs {len(gt_files)} high images'
    print(f'[data] Loaded {len(raw_files)} test pairs.')

    raw_ds = tf.data.Dataset.from_tensor_slices(raw_files).map(
        lambda x: _load_test_any(x, crop_margin),
        num_parallel_calls=tf.data.AUTOTUNE)
    gt_ds = tf.data.Dataset.from_tensor_slices(gt_files).map(
        lambda x: _load_test_any(x, crop_margin),
        num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.zip((raw_ds, gt_ds)).batch(1)
    return ds