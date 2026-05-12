"""
Autoencoder Denoising — Dataset Test Script
============================================
Seedha chalao: python autoencoder_dataset_test.py
3 datasets support karta hai: MNIST, CIFAR-10, Custom images
"""
 
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
 
# ──────────────────────────────────────────────
# CONFIG — yahan se change karo
# ──────────────────────────────────────────────
DATASET     = "mnist"       # "mnist" | "cifar10" | "custom"
CUSTOM_PATH = "./my_images" # sirf DATASET="custom" ke liye
NOISE_FACTOR = 0.3          # 0.0 = no noise, 1.0 = full noise
EPOCHS      = 10            # jyada = better quality, slow training
BATCH_SIZE  = 128
LATENT_DIM  = 64            # bottleneck size — kam = more compression
 
 
# ──────────────────────────────────────────────
# DATASET LOADERS
# ──────────────────────────────────────────────
def load_mnist():
    print("[*] MNIST load ho raha hai (auto-download ~12MB pehli baar)...")
    from tensorflow.keras.datasets import mnist
    (x_train, _), (x_test, _) = mnist.load_data()
    # Normalize + reshape
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)  # (60000, 28, 28, 1)
    x_test  = np.expand_dims(x_test,  -1)  # (10000, 28, 28, 1)
    print(f"    Train: {x_train.shape} | Test: {x_test.shape}")
    return x_train, x_test, (28, 28, 1)
 
 
def load_cifar10():
    print("[*] CIFAR-10 load ho raha hai (auto-download ~170MB pehli baar)...")
    from tensorflow.keras.datasets import cifar10
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    print(f"    Train: {x_train.shape} | Test: {x_test.shape}")
    return x_train, x_test, (32, 32, 3)
 
 
def load_custom(folder_path):
    """
    Apni images load karo kisi bhi folder se.
    Supported: .jpg .jpeg .png .bmp
    """
    print(f"[*] Custom images load ho rahi hain from: {folder_path}")
    try:
        from PIL import Image
    except ImportError:
        print("    PIL nahi mila — pip install pillow karo")
        sys.exit(1)
 
    IMG_SIZE = (64, 64)
    images = []
    extensions = (".jpg", ".jpeg", ".png", ".bmp")
 
    if not os.path.exists(folder_path):
        print(f"    ERROR: Folder nahi mila: {folder_path}")
        print(f"    '{folder_path}' naam ka folder banao aur usme images daalo.")
        sys.exit(1)
 
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith(extensions):
            path = os.path.join(folder_path, fname)
            img  = Image.open(path).convert("RGB").resize(IMG_SIZE)
            images.append(np.array(img, dtype="float32") / 255.0)
 
    if len(images) == 0:
        print(f"    ERROR: Koi image nahi mili folder mein!")
        sys.exit(1)
 
    print(f"    {len(images)} images mili")
    images = np.array(images)
 
    # 80/20 train/test split
    split = int(len(images) * 0.8)
    x_train, x_test = images[:split], images[split:]
 
    if len(x_test) == 0:
        x_test = x_train[:5]  # fallback agar kam images hain
 
    print(f"    Train: {x_train.shape} | Test: {x_test.shape}")
    return x_train, x_test, (IMG_SIZE[0], IMG_SIZE[1], 3)
 
 
# ──────────────────────────────────────────────
# NOISE ADDER
# ──────────────────────────────────────────────
def add_noise(images, factor=0.3):
    noisy = images + factor * np.random.normal(
        loc=0.0, scale=1.0, size=images.shape
    )
    return np.clip(noisy, 0.0, 1.0)
 
 
# ──────────────────────────────────────────────
# AUTOENCODER MODEL
# ──────────────────────────────────────────────
def build_autoencoder(input_shape, latent_dim=64):
    from tensorflow.keras import layers, models
 
    # ── ENCODER ──
    encoder_input = layers.Input(shape=input_shape, name="encoder_input")
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    encoder = models.Model(encoder_input, x, name="encoder")
 
    # ── DECODER ──
    decoder_input = layers.Input(shape=x.shape[1:], name="decoder_input")
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", padding="same")(x)
    decoder_output = layers.Conv2DTranspose(input_shape[-1], 3, activation="sigmoid", padding="same")(x)
    decoder = models.Model(decoder_input, decoder_output, name="decoder")
 
    # ── FULL AUTOENCODER ──
    ae_input  = layers.Input(shape=input_shape)
    ae_output = decoder(encoder(ae_input))
    autoencoder = models.Model(ae_input, ae_output, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
 
    return autoencoder, encoder, decoder
 
 
# ──────────────────────────────────────────────
# VISUALIZE RESULTS
# ──────────────────────────────────────────────
def show_results(x_test, x_noisy, x_denoised, n=8):
    """
    3 rows dikhata hai:
      Row 1: Original clean images
      Row 2: Noisy images (input to model)
      Row 3: Denoised output (model ka output)
    """
    plt.figure(figsize=(n * 1.8, 5))
    plt.suptitle(
        f"Autoencoder Denoising Results  |  noise={NOISE_FACTOR}  |  dataset={DATASET.upper()}",
        fontsize=12, y=1.01
    )
 
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        img = x_test[i]
        ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Original", fontsize=9, loc="left")
 
        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        img = x_noisy[i]
        ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Noisy input", fontsize=9, loc="left")
 
        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        img = x_denoised[i]
        ax.imshow(img.squeeze(), cmap="gray" if img.shape[-1] == 1 else None)
        ax.axis("off")
        if i == 0:
            ax.set_title("Denoised output", fontsize=9, loc="left")
 
    plt.tight_layout()
    save_path = f"denoising_result_{DATASET}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[*] Result image save ho gayi: {save_path}")
    plt.show()
 
 
def print_metrics(x_test, x_denoised):
    """PSNR calculate karo — jyada = better"""
    mse  = np.mean((x_test - x_denoised) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    print(f"\n{'='*40}")
    print(f"  MSE  (lower is better) : {mse:.6f}")
    print(f"  PSNR (higher is better): {psnr:.2f} dB")
    print(f"  Rough guide: >30dB = good, >40dB = excellent")
    print(f"{'='*40}\n")
 
 
# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("\n" + "="*50)
    print("  Autoencoder Denoising Project")
    print("="*50)
 
    # 1. TensorFlow import check
    try:
        import tensorflow as tf
        print(f"[*] TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[*] GPU available: {'Yes — ' + str(len(gpus)) + ' device(s)' if gpus else 'No — CPU pe chalega'}")
    except ImportError:
        print("ERROR: TensorFlow nahi mila!")
        print("Run karo: pip install tensorflow")
        sys.exit(1)
 
    # 2. Dataset load
    if DATASET == "mnist":
        x_train, x_test, input_shape = load_mnist()
    elif DATASET == "cifar10":
        x_train, x_test, input_shape = load_cifar10()
    elif DATASET == "custom":
        x_train, x_test, input_shape = load_custom(CUSTOM_PATH)
    else:
        print(f"Unknown DATASET: {DATASET}")
        sys.exit(1)
 
    # 3. Noise add karo
    print(f"\n[*] Noise add ho rahi hai (factor={NOISE_FACTOR})...")
    x_train_noisy = add_noise(x_train, NOISE_FACTOR)
    x_test_noisy  = add_noise(x_test,  NOISE_FACTOR)
 
    # 4. Model build karo
    print(f"[*] Autoencoder build ho raha hai (latent_dim={LATENT_DIM})...")
    autoencoder, encoder, decoder = build_autoencoder(input_shape, LATENT_DIM)
    autoencoder.summary()
 
    # 5. Train karo
    print(f"\n[*] Training shuru — {EPOCHS} epochs, batch={BATCH_SIZE}...")
    history = autoencoder.fit(
        x_train_noisy, x_train,           # input: noisy | target: clean
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
        verbose=1
    )
 
    # 6. Predict karo
    print("\n[*] Test images pe denoising kar raha hai...")
    x_denoised = autoencoder.predict(x_test_noisy, batch_size=BATCH_SIZE)
 
    # 7. Metrics
    print_metrics(x_test, x_denoised)
 
    # 8. Visualize
    n_show = min(8, len(x_test))
    show_results(x_test, x_test_noisy, x_denoised, n=n_show)
 
    # 9. Model save karo
    autoencoder.save(f"autoencoder_{DATASET}.keras")
    print(f"[*] Model saved: autoencoder_{DATASET}.keras")
    print("\nDone! Agar aur test karna ho:")
    print("  - NOISE_FACTOR badhao (0.5, 0.8) aur dobara run karo")
    print("  - EPOCHS badhao better results ke liye")
    print("  - DATASET='cifar10' try karo color images ke liye")
 
 
if __name__ == "__main__":
    main()
