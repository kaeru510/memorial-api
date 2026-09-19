"""
Wav2Lip 高速アバター常駐サーバー (Google Colab 実行用)
ローカル側で更新・GitHubへプッシュすることで、Colab側は常に最新コードで起動されます。
"""

import os
import sys
import shutil
import subprocess
import time

print("=" * 60)
print("🚀 Wav2Lip 高速アバター常駐サーバー 起動シーケンス開始")
print("=" * 60)

WAV2LIP_DIR = "/content/Wav2Lip"
DRIVE_DIR = "/content/drive/MyDrive/lipsync_avatar"

# 1. 環境チェック & Wav2Lip のセットアップ
print("⏳ [1/4] 環境のチェック中...")
if not os.path.exists(os.path.join(WAV2LIP_DIR, "audio.py")):
    print("📥 Wav2Lip をクローン中...")
    subprocess.run(["rm", "-rf", WAV2LIP_DIR], check=False)
    subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", WAV2LIP_DIR], check=True)
    subprocess.run(["pip", "install", "gradio", "-q"], check=True)

# 2. audio.py の librosa 引数エラー置換
AUDIO_PY = os.path.join(WAV2LIP_DIR, "audio.py")
if os.path.exists(AUDIO_PY):
    subprocess.run([
        "sed", "-i",
        "s/librosa.filters.mel(hp.sample_rate, hp.n_fft/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft/g",
        AUDIO_PY
    ], check=False)

# 3. Google Drive からのファイル復元（存在チェック付き）
os.makedirs(f"{WAV2LIP_DIR}/checkpoints", exist_ok=True)

if not os.path.exists("/content/base_avatar_loop.mp4"):
    src = f"{DRIVE_DIR}/base_avatar_loop.mp4"
    if os.path.exists(src):
        shutil.copy(src, "/content/base_avatar_loop.mp4")
        print("✅ base_avatar_loop.mp4 を復元しました")
    else:
        print(f"⚠️ 警告: {src} が見つかりません")

if not os.path.exists("/content/base_avatar_cache.npz"):
    src = f"{DRIVE_DIR}/base_avatar_cache.npz"
    if os.path.exists(src):
        shutil.copy(src, "/content/base_avatar_cache.npz")
        print("✅ base_avatar_cache.npz を復元しました")
    else:
        print(f"⚠️ 警告: {src} が見つかりません")

if not os.path.exists(f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth"):
    src = f"{DRIVE_DIR}/checkpoints/wav2lip_gan.pth"
    if os.path.exists(src):
        shutil.copy(src, f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth")
        print("✅ wav2lip_gan.pth を復元しました")
    else:
        print(f"⚠️ 警告: {src} が見つかりません")

# 4. ディレクトリ移動 & パス設定
print("⚙️ [2/4] ディレクトリ移動 & ライブラリ設定...")
os.chdir(WAV2LIP_DIR)
if WAV2LIP_DIR not in sys.path:
    sys.path.insert(0, WAV2LIP_DIR)

import cv2
import torch
import numpy as np
import gradio as gr
import audio
from models import Wav2Lip

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️ 使用デバイス: {device}")

# 5. モデルロード
print("🧠 [3/4] モデル読み込み & キャッシュ展開中...")
CHECKPOINT_PATH = f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth"

def load_wav2lip_model(path):
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    return model.to(device).eval()

model = load_wav2lip_model(CHECKPOINT_PATH)

# 6. キャッシュ読み込み & GPUテンソル事前展開
BASE_LOOP_VIDEO = "/content/base_avatar_loop.mp4"
CACHE_FILE = "/content/base_avatar_cache.npz"

cache_data = np.load(CACHE_FILE)
cached_coords = cache_data["coords"]
cached_fps = float(cache_data["fps"])

cap = cv2.VideoCapture(BASE_LOOP_VIDEO)
cached_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cached_frames.append(frame)
cap.release()

img_size = 96
precomputed_imgs = []
for f, (y1, y2, x1, x2) in zip(cached_frames, cached_coords):
    face_crop = cv2.resize(f[y1:y2, x1:x2], (img_size, img_size))
    face_crop_masked = face_crop.copy()
    face_crop_masked[img_size // 2 :] = 0
    combined = np.concatenate((face_crop_masked, face_crop), axis=2) / 255.0
    precomputed_imgs.append(combined.transpose(2, 0, 1))

gpu_face_tensor = torch.FloatTensor(np.array(precomputed_imgs)).to(device)

print("✅ キャッシュのGPU展開完了！")

# 7. 高速推論関数
wav2lip_batch_size = 128

def fast_process_pipeline(face_img_path, audio_path):
    t_start = time.time()

    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if mel.shape[1] < 16:
        mel = np.pad(mel, ((0, 0), (0, 16 - mel.shape[1])), mode='constant')

    mel_chunks = []
    mel_idx_multiplier = 80.0 / cached_fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + 16 > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - 16:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + 16])
        i += 1

    num_frames = len(mel_chunks)
    total_cached = len(cached_frames)
    full_frames = [cached_frames[idx % total_cached].copy() for idx in range(num_frames)]
    coords = [cached_coords[idx % total_cached] for idx in range(num_frames)]

    t_gpu = time.time()
    for i in range(0, num_frames, wav2lip_batch_size):
        batch_mels = mel_chunks[i:i + wav2lip_batch_size]
        batch_c = coords[i:i + wav2lip_batch_size]
        cur_batch_len = len(batch_mels)

        indices = [(i + k) % total_cached for k in range(cur_batch_len)]
        img_tensor = gpu_face_tensor[indices]
        mel_tensor = torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                pred = model(mel_tensor, img_tensor)

        pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for j, (p, (y1, y2, x1, x2)) in enumerate(zip(pred, batch_c)):
            idx = i + j
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            full_frames[idx][y1:y2, x1:x2] = p

    gpu_sec = time.time() - t_gpu

    t_enc = time.time()
    orig_h, orig_w, _ = full_frames[0].shape
    max_dim = 480
    scale = min(max_dim / max(orig_h, orig_w), 1.0)
    target_w = int(orig_w * scale) // 2 * 2
    target_h = int(orig_h * scale) // 2 * 2

    final_mp4 = "/content/final_synced_output.mp4"
    if os.path.exists(final_mp4):
        os.remove(final_mp4)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(cached_fps),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "26",
        "-c:a", "aac",
        "-b:a", "96k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        final_mp4
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in full_frames:
        if scale < 1.0:
            resized_f = cv2.resize(frame, (target_w, target_h))
            proc.stdin.write(resized_f.tobytes())
        else:
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    enc_sec = time.time() - t_enc

    print(f"⚡ [推論: {gpu_sec:.2f}秒 | エンコード: {enc_sec:.2f}秒 ({target_w}x{target_h}) | 合計: {time.time() - t_start:.2f}秒]")
    return final_mp4

# 8. Gradio 起動
print("🚀 [4/4] Gradio サーバー起動中...")
with gr.Blocks() as demo:
    face_in = gr.Image(type="filepath", label="顔画像")
    audio_in = gr.Audio(type="filepath", label="音声")
    video_out = gr.Video(label="出力動画")
    btn = gr.Button("生成")
    btn.click(
        fn=fast_process_pipeline,
        inputs=[face_in, audio_in],
        outputs=video_out,
        api_name="process_pipeline"
    )

demo.launch(share=True, debug=True, allowed_paths=["/content"])
