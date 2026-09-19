"""
Wav2Lip + LivePortrait 高速アバター常駐サーバー (Google Colab 実行用)
静止画1枚から自動でアバター（ループ動画＆顔座標キャッシュ）を生成し、即座に対話可能にします。
"""

import os
import sys
import shutil
import subprocess
import time

print("=" * 60)
print("🚀 Wav2Lip + LivePortrait 高速アバターサーバー 起動シーケンス開始")
print("=" * 60)

WAV2LIP_DIR = "/content/Wav2Lip"
LIVEPORTRAIT_DIR = "/content/LivePortrait"
DRIVE_DIR = "/content/drive/MyDrive/lipsync_avatar"
MOTION_DIR = "/content/motions"
os.makedirs(MOTION_DIR, exist_ok=True)
IDLE_VIDEO_PATH = os.path.join(MOTION_DIR, "idle.mp4")

# ==============================================================================
# 1. 環境チェック & Wav2Lip / LivePortrait のセットアップ
# ==============================================================================
print("⏳ [1/4] 環境のチェック中...")

# (1) Wav2Lip の確認
if not os.path.exists(os.path.join(WAV2LIP_DIR, "audio.py")):
    print("📥 Wav2Lip をクローン中...")
    subprocess.run(["rm", "-rf", WAV2LIP_DIR], check=False)
    subprocess.run(["git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", WAV2LIP_DIR], check=True)
    subprocess.run(["pip", "install", "gradio", "-q"], check=True)

AUDIO_PY = os.path.join(WAV2LIP_DIR, "audio.py")
if os.path.exists(AUDIO_PY):
    subprocess.run([
        "sed", "-i",
        "s/librosa.filters.mel(hp.sample_rate, hp.n_fft/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft/g",
        AUDIO_PY
    ], check=False)

# (2) LivePortrait の確認（未導入ならクローン＆重み取得）
if not os.path.exists(os.path.join(LIVEPORTRAIT_DIR, "inference.py")):
    print("📥 LivePortrait をクローン中...")
    subprocess.run(["git", "clone", "https://github.com/KwaiVGI/LivePortrait.git", LIVEPORTRAIT_DIR], check=True)

# LivePortrait の依存パッケージ確認
try:
    import tyro
except ImportError:
    print("📥 LivePortrait 依存ライブラリ (tyro, pykalman 等) をインストール中...")
    subprocess.run(["pip", "install", "tyro", "pykalman", "pyyaml", "albumentations", "lmdb", "ffmpeg-python", "-q"], check=True)


if not os.path.exists(os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights")):
    print("📥 LivePortrait の重みファイルをダウンロード中...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="KwaiVGI/LivePortrait", local_dir=os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights"))
    except Exception as e:
        print(f"⚠️ LivePortrait 重み取得警告: {e}")

# (3) モーション動画 (idle.mp4) の準備
if not os.path.exists(IDLE_VIDEO_PATH):
    drive_idle = f"{DRIVE_DIR}/idle.mp4"
    lp_default_d0 = f"{LIVEPORTRAIT_DIR}/assets/examples/driving/d0.mp4"
    if os.path.exists(drive_idle):
        shutil.copy(drive_idle, IDLE_VIDEO_PATH)
        print("✅ Drive から idle.mp4 を配置しました")
    elif os.path.exists(lp_default_d0):
        shutil.copy(lp_default_d0, IDLE_VIDEO_PATH)
        print("✅ LivePortrait サンプルから idle.mp4 を初期配置しました")

# (4) Drive からの初期ファイル復元（存在すれば利用）
os.makedirs(f"{WAV2LIP_DIR}/checkpoints", exist_ok=True)
if not os.path.exists("/content/base_avatar_loop.mp4") and os.path.exists(f"{DRIVE_DIR}/base_avatar_loop.mp4"):
    shutil.copy(f"{DRIVE_DIR}/base_avatar_loop.mp4", "/content/base_avatar_loop.mp4")
if not os.path.exists("/content/base_avatar_cache.npz") and os.path.exists(f"{DRIVE_DIR}/base_avatar_cache.npz"):
    shutil.copy(f"{DRIVE_DIR}/base_avatar_cache.npz", "/content/base_avatar_cache.npz")
if not os.path.exists(f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth") and os.path.exists(f"{DRIVE_DIR}/checkpoints/wav2lip_gan.pth"):
    shutil.copy(f"{DRIVE_DIR}/checkpoints/wav2lip_gan.pth", f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth")

# (5) 顔検出モデル (s3fd) の確認 & PyTorch 2.x パッチ適用
sfd_dir = f"{WAV2LIP_DIR}/face_detection/detection/sfd"
os.makedirs(sfd_dir, exist_ok=True)
sfd_path = os.path.join(sfd_dir, "s3fd.pth")
if not os.path.exists(sfd_path) or os.path.getsize(sfd_path) < 1000000:
    print("📥 顔検出モデル (s3fd.pth) を取得中...")
    subprocess.run([
        "wget", "-q", "-O", sfd_path,
        "https://huggingface.co/camenduru/Wav2Lip/resolve/main/face_detection/detection/sfd/s3fd.pth"
    ], check=False)
    for alias in ["s3fd-619a316847.pth", "s3fd-619a316812.pth"]:
        shutil.copy(sfd_path, os.path.join(sfd_dir, alias))

# PyTorch 2.6+ 互換パッチ（weights_only=False）
sfd_file = os.path.join(sfd_dir, "sfd_detector.py")
if os.path.exists(sfd_file):
    with open(sfd_file, "r") as f:
        code = f.read()
    if "weights_only=False" not in code:
        code = code.replace("torch.load(path_to_detector)", "torch.load(path_to_detector, weights_only=False)")
        code = code.replace("model_weights = torch.load(path_to_detector)", "model_weights = torch.load(path_to_detector, weights_only=False)")
        with open(sfd_file, "w") as f:
            f.write(code)
        print("🔧 sfd_detector.py に PyTorch 互換パッチを適用しました")


# ==============================================================================
# 2. ディレクトリ移動 & パス設定
# ==============================================================================
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

# ==============================================================================
# 3. モデル読み込み & キャッシュ展開
# ==============================================================================
print("🧠 [3/4] モデル読み込み中...")
CHECKPOINT_PATH = f"{WAV2LIP_DIR}/checkpoints/wav2lip_gan.pth"

def load_wav2lip_model(path):
    model = Wav2Lip()
    checkpoint = torch.load(path, map_location=device)
    s = checkpoint["state_dict"]
    new_s = {k.replace('module.', ''): v for k, v in s.items()}
    model.load_state_dict(new_s)
    return model.to(device).eval()

model = load_wav2lip_model(CHECKPOINT_PATH)

# キャッシュ展開関数（アバター切替時にも再利用）
cached_frames = []
cached_coords = []
cached_fps = 25.0
gpu_face_tensor = None
img_size = 96
wav2lip_batch_size = 128

def reload_avatar_cache(loop_video_path="/content/base_avatar_loop.mp4", cache_file_path="/content/base_avatar_cache.npz"):
    global cached_frames, cached_coords, cached_fps, gpu_face_tensor
    if not (os.path.exists(loop_video_path) and os.path.exists(cache_file_path)):
        print("ℹ️ キャッシュファイルが未生成です。setup_avatar を実行して生成してください。")
        return False

    cache_data = np.load(cache_file_path)
    cached_coords = cache_data["coords"]
    cached_fps = float(cache_data["fps"])

    cap = cv2.VideoCapture(loop_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    cached_frames = frames

    precomputed_imgs = []
    for f, (y1, y2, x1, x2) in zip(cached_frames, cached_coords):
        face_crop = cv2.resize(f[y1:y2, x1:x2], (img_size, img_size))
        face_crop_masked = face_crop.copy()
        face_crop_masked[img_size // 2 :] = 0
        combined = np.concatenate((face_crop_masked, face_crop), axis=2) / 255.0
        precomputed_imgs.append(combined.transpose(2, 0, 1))

    gpu_face_tensor = torch.FloatTensor(np.array(precomputed_imgs)).to(device)
    print(f"✅ アバターキャッシュ展開完了 ({len(cached_frames)} フレーム, FPS: {cached_fps})")
    return True

# 既存キャッシュがあれば展開
reload_avatar_cache()

# ==============================================================================
# 4. アバター自動生成・切り替え関数 (setup_avatar)
# ==============================================================================
def setup_avatar(face_img_path):
    """
    静止画1枚から LivePortrait で待機ループ動画と顔座標キャッシュを全自動生成し、
    メモリ内のモデルキャッシュを即座に更新。ブラウザ待機用の avatar_idle.mp4 を返します。
    """
    t_start = time.time()
    print(f"\n🎨 [アバター自動セットアップ開始] 入力画像: {face_img_path}")

    try:
        if not os.path.exists(face_img_path):
            raise FileNotFoundError(f"入力顔画像が見つかりません: {face_img_path}")

        if not os.path.exists(IDLE_VIDEO_PATH):
            raise FileNotFoundError(f"モーション動画が見つかりません: {IDLE_VIDEO_PATH}")

        # 1. LivePortrait の実行（顔画像 + idleモーション動画）
        print("🚀 [1/4] LivePortrait でベース表情モーションを生成中...")
        lp_output_dir = os.path.join(LIVEPORTRAIT_DIR, "animations")
        os.makedirs(lp_output_dir, exist_ok=True)

        lp_cmd = [
            sys.executable, "inference.py",
            "-s", face_img_path,
            "-d", IDLE_VIDEO_PATH,
            "--flag_relative_motion",
            "--flag_do_crop"
        ]
        res = subprocess.run(lp_cmd, cwd=LIVEPORTRAIT_DIR, capture_output=True, text=True)
        if res.returncode != 0:
            print("❌ LivePortrait 実行エラー:")
            print(res.stderr)
            raise RuntimeError(f"LivePortraitエラー: {res.stderr[-500:] if res.stderr else res.stdout[-500:]}")

        generated_videos = [
            os.path.join(lp_output_dir, f) for f in os.listdir(lp_output_dir) if f.endswith(".mp4")
        ]
        if not generated_videos:
            raise RuntimeError("LivePortrait の動画生成結果 (.mp4) が見つかりませんでした。")

        latest_lp = max(generated_videos, key=os.path.getmtime)
        print(f"✅ LivePortrait 生成完了: {latest_lp}")


        # 2. 30秒ピンポンループ動画の作成
        print("🚀 [2/4] 30秒のシームレス往復ループ動画を作成中...")
        BASE_LOOP_VIDEO = "/content/base_avatar_loop.mp4"
        cap = cv2.VideoCapture(latest_lp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
        cap.release()

        if not raw_frames:
            raise RuntimeError("LivePortrait動画の読み込みに失敗しました。")

        pingpong_cycle = raw_frames + raw_frames[-2:0:-1] if len(raw_frames) > 1 else raw_frames
        total_frames = int(30 * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(BASE_LOOP_VIDEO, fourcc, fps, (width, height))
        loop_frames = []
        for i in range(total_frames):
            f = pingpong_cycle[i % len(pingpong_cycle)]
            out.write(f)
            loop_frames.append(f)
        out.release()

        # 3. Wav2Lipの顔検出器で全フレームの顔座標をキャッシュ
        print("🚀 [3/4] 顔座標を計算・キャッシュ中...")
        import face_detection
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        batch_size = 32
        coords = []
        pads = [0, 10, 0, 0]

        for i in range(0, len(loop_frames), batch_size):
            batch_f = loop_frames[i:i + batch_size]
            preds = detector.get_detections_for_batch(np.array(batch_f))
            for j, det in enumerate(preds):
                if det is None:
                    coords.append(coords[-1] if coords else [0, height, 0, width])
                    continue
                s = det
                y1 = max(0, s[1] - pads[0])
                y2 = min(height, s[3] + pads[1])
                x1 = max(0, s[0] - pads[2])
                x2 = min(width, s[2] + pads[3])
                coords.append([int(y1), int(y2), int(x1), int(x2)])

        CACHE_FILE = "/content/base_avatar_cache.npz"
        np.savez_compressed(CACHE_FILE, coords=np.array(coords), fps=fps)

        # 4. メモリ内キャッシュを即座に再読み込み
        print("🚀 [4/4] サーバーメモリのキャッシュを最新アバターに更新...")
        reload_avatar_cache(BASE_LOOP_VIDEO, CACHE_FILE)

        # 5. ブラウザ待機用の軽量 avatar_idle.mp4 を生成
        avatar_idle_path = "/content/avatar_idle.mp4"
        max_dim = 480
        scale = min(max_dim / max(height, width), 1.0)
        target_w = int(width * scale) // 2 * 2
        target_h = int(height * scale) // 2 * 2

        cmd = [
            "ffmpeg", "-y",
            "-i", BASE_LOOP_VIDEO,
            "-vf", f"scale={target_w}:{target_h}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-an",
            avatar_idle_path
        ]
        subprocess.run(cmd, check=True)

        # Google Drive にもバックアップ保存
        try:
            shutil.copy(BASE_LOOP_VIDEO, f"{DRIVE_DIR}/base_avatar_loop.mp4")
            shutil.copy(CACHE_FILE, f"{DRIVE_DIR}/base_avatar_cache.npz")
            shutil.copy(avatar_idle_path, f"{DRIVE_DIR}/avatar_idle.mp4")
        except Exception:
            pass

        print(f"✨ アバターセットアップ完了！ (所要時間: {time.time() - t_start:.2f}秒)\n")
        return avatar_idle_path
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ [アバターセットアップ失敗]: {e}\n")
        raise gr.Error(f"Colabアバター生成エラー: {e}")


# ==============================================================================
# 5. 高速対話推論関数 (fast_process_pipeline)
# ==============================================================================
def fast_process_pipeline(face_img_path, audio_path):
    t_start = time.time()

    if gpu_face_tensor is None or len(cached_frames) == 0:
        raise RuntimeError("アバターがセットアップされていません。まずアバター画像をセットアップしてください。")

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

    print(f"⚡ 推論: {gpu_sec:.2f}秒 | エンコード: {enc_sec:.2f}秒 ({target_w}x{target_h}) | 合計: {time.time() - t_start:.2f}秒")
    return final_mp4

# ==============================================================================
# 6. Gradio サーバー起動
# ==============================================================================
print("🚀 [4/4] Gradio サーバー起動中...")
with gr.Blocks() as demo:
    # 1. 対話動画生成 API
    with gr.Row():
        face_in = gr.Image(type="filepath", label="顔画像")
        audio_in = gr.Audio(type="filepath", label="音声")
        video_out = gr.Video(label="出力動画")
        btn_talk = gr.Button("対話生成")
        btn_talk.click(
            fn=fast_process_pipeline,
            inputs=[face_in, audio_in],
            outputs=video_out,
            api_name="process_pipeline"
        )

    # 2. アバター初期化・切替 API
    with gr.Row():
        avatar_img_in = gr.Image(type="filepath", label="新規アバター顔写真")
        avatar_video_out = gr.Video(label="生成された待機動画")
        btn_setup = gr.Button("アバター切替")
        btn_setup.click(
            fn=setup_avatar,
            inputs=[avatar_img_in],
            outputs=avatar_video_out,
            api_name="setup_avatar"
        )

# ==============================================================================
# 7. URL自動クラウド同期（ローカルPCと完全自動接続）
# ==============================================================================
import threading
import requests

def sync_url_worker():
    sync_endpoint = "https://api.cl1p.net/kaeru510-memorial"
    print("📡 URL自動同期ワーカー開始...")
    for _ in range(40):  # 最大80秒待機
        time.sleep(2)
        url = getattr(demo, "share_url", None)
        if url:
            try:
                requests.post(sync_endpoint, data=url.strip(), timeout=5)
                print("\n" + "=" * 60)
                print(f"📡 【URL自動同期完了】最新URLをクラウドに送信しました！")
                print(f"👉 URL: {url}")
                print(f"（ローカルPC側で自動検出されるため、コピペ不要です）")
                print("=" * 60 + "\n")
                break
            except Exception as e:
                print(f"⚠️ URL同期リトライ中: {e}")

threading.Thread(target=sync_url_worker, daemon=True).start()

demo.launch(share=True, debug=True, show_error=True, allowed_paths=["/content"])


