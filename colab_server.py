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

# ==============================================================================
# 0. 前回の実行の残骸を片付ける
#    セルを停止しても python / AivisSpeech / ngrok の子プロセスが残ることがあり、
#    残っているとポート 7860 / 10101 が使用中で起動に失敗し、GPU メモリも占有される
# ==============================================================================
def cleanup_previous_run():
    import signal
    me = os.getpid()
    ancestors = set()
    pid = me
    while pid > 1:  # 自分と親（ノートブックのシェル）は対象外
        ancestors.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    targets = ("colab_server.py", "run.py --use_gpu", "ngrok http", "cloudflared tunnel")
    killed = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) in ancestors:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore")
        except Exception:
            continue
        if any(t in cmd for t in targets):
            try:
                os.kill(int(entry), signal.SIGKILL)
                killed.append(cmd.strip()[:60])
            except Exception:
                pass
    if killed:
        print(f"🧹 前回の実行の残りを停止しました ({len(killed)} 件)", flush=True)
        time.sleep(2)  # ポートと GPU メモリの解放待ち

cleanup_previous_run()

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
    import onnx
    import onnxruntime
    import pykalman
    import lmdb
except ImportError:
    print("📥 LivePortrait 依存ライブラリ (onnx, onnxruntime, tyro, pykalman 等) をインストール中...")
    subprocess.run([
        "pip", "install",
        "onnx", "onnxruntime-gpu", "onnxruntime", "tyro", "pykalman", "pyyaml",
        "albumentations", "lmdb", "ffmpeg-python", "transformers", "-q"
    ], check=False)


target_weight = os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights", "liveportrait", "base_models", "appearance_feature_extractor.pth")
if not os.path.exists(target_weight):
    print("📥 LivePortrait の重みファイルを準備中...")
    drive_weights = os.path.join(DRIVE_DIR, "pretrained_weights")
    if os.path.exists(os.path.join(drive_weights, "liveportrait", "base_models", "appearance_feature_extractor.pth")):
        print("📁 Google Drive から LivePortrait の重みを復元中...")
        shutil.copytree(drive_weights, os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights"), dirs_exist_ok=True)
        print("✅ Drive から重みを復元しました")
    else:
        print("📥 Hugging Face から LivePortrait の重みをダウンロード中 (約1.5GB)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="KwaiVGI/LivePortrait", local_dir=os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights"))
            print("✅ LivePortrait 重みダウンロード完了")
            # 次回高速化のために Google Drive にバックアップ
            try:
                os.makedirs(drive_weights, exist_ok=True)
                shutil.copytree(os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights"), drive_weights, dirs_exist_ok=True)
                print("💾 次回起動高速化のため Google Drive に重みをバックアップしました")
            except Exception as be:
                print(f"⚠️ Drive バックアップスキップ: {be}")
        except Exception as e:
            print(f"❌ LivePortrait 重み取得エラー: {e}")


# (3) 自律モーション生成モジュール (procedural_motion) の準備
print("✅ 自律モーション生成モジュール (procedural_motion) 準備完了（外部参照動画は不要です）")


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

    gpu_face_tensor = precompute_face_tensor(cached_frames, cached_coords)
    emotion_cache.clear()  # 別アバターの感情ループが残らないように（必要なら load_emotion_caches で読み直す）
    print(f"✅ アバターキャッシュ展開完了 ({len(cached_frames)} フレーム, FPS: {cached_fps})")
    return True

def precompute_face_tensor(frames, coords):
    """Wav2Lip 入力用の顔切り抜き（下半分マスク＋参照）を全フレーム分 GPU に展開"""
    precomputed_imgs = []
    for f, (y1, y2, x1, x2) in zip(frames, coords):
        face_crop = cv2.resize(f[y1:y2, x1:x2], (img_size, img_size))
        face_crop_masked = face_crop.copy()
        face_crop_masked[img_size // 2 :] = 0
        combined = np.concatenate((face_crop_masked, face_crop), axis=2) / 255.0
        precomputed_imgs.append(combined.transpose(2, 0, 1))
    return torch.FloatTensor(np.array(precomputed_imgs)).to(device)

# ------------------------------------------------------------------------------
# 感情ループ: 頭の動き・まばたきは通常ループと完全に同じで、表情だけが違うループ
#   → コマ番号をそのまま共有でき、感情が切り替わっても頭の位置は飛ばず表情だけが変わる。
#   顔座標は通常ループのものを流用（頭の位置が同じため）。
# ------------------------------------------------------------------------------
import json
EMOTION_NAMES = ("happy", "calm", "sad")
# 目を細める等も表情キーポイント（expression の eyes）で指定する（eye_open は目の調整機能を使う場合のみ有効）
# 値は 2026-10-03 に試し撮り（/api/expression_preview）で調整したもの
DEFAULT_EMOTION_PRESETS = {
    "happy": {"expression": {"smile": 0.9, "eyebrow": 3.0, "eyes": -3.0}},   # 笑顔＋目を少し細める
    "calm": {"expression": {"smile": 0.4, "eyes": -2.0}},                    # 控えめな微笑み
    "sad": {"expression": {"smile": -0.45, "eyebrow": -8.0, "eyes": -6.0}},  # 伏し目＋口角を少し下げる（強いとすねた顔に見える）
}
EMOTION_PRESETS_FILE = f"{DRIVE_DIR}/emotion_presets.json"
emotion_presets = dict(DEFAULT_EMOTION_PRESETS)
try:
    if os.path.exists(EMOTION_PRESETS_FILE):
        emotion_presets.update(json.load(open(EMOTION_PRESETS_FILE, encoding="utf-8")))
except Exception as e:
    print(f"⚠️ 感情プリセットの読み込みをスキップ: {e}")
emotion_cache = {}  # emotion -> {"frames": [...], "gpu": tensor}

def emotion_loop_path(emo):
    return f"/content/base_avatar_loop_{emo}.mp4"

def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_emotion_caches():
    """保存済みの感情ループ（/content か Drive）を読み込む。通常ループとコマ数が違うもの（古いアバター用）は無視"""
    for emo in EMOTION_NAMES:
        path = emotion_loop_path(emo)
        drive_path = f"{DRIVE_DIR}/base_avatar_loop_{emo}.mp4"
        if not os.path.exists(path) and os.path.exists(drive_path):
            shutil.copy(drive_path, path)
        if not os.path.exists(path):
            continue
        frames = read_video_frames(path)
        if len(frames) != len(cached_frames) or len(cached_frames) == 0:
            print(f"ℹ️ 感情ループ [{emo}] は現在のアバターと一致しないため使いません")
            continue
        emotion_cache[emo] = {"frames": frames, "gpu": precompute_face_tensor(frames, cached_coords)}
    if emotion_cache:
        print(f"✅ 感情ループ展開完了: {', '.join(sorted(emotion_cache))}")

def build_emotion_loops(face_img_path, emotions=EMOTION_NAMES, progress=None):
    """現在のアバター写真から感情ループを生成（1つあたり LivePortrait 数分）"""
    for n, emo in enumerate(emotions, 1):
        if progress:
            progress(f"感情ループ生成中 {n}/{len(emotions)} ({emo})")
        preset = emotion_presets[emo]
        frames, fps = render_idle_loop_frames(face_img_path, expression=preset.get("expression"),
                                              eye_open=preset.get("eye_open", 1.0), tag=emo)
        if len(frames) != len(cached_frames):
            raise RuntimeError(f"感情ループ [{emo}] のコマ数 {len(frames)} が通常ループ {len(cached_frames)} と一致しません")
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(emotion_loop_path(emo), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()
        try:
            shutil.copy(emotion_loop_path(emo), f"{DRIVE_DIR}/base_avatar_loop_{emo}.mp4")
        except Exception:
            pass
        gpu = precompute_face_tensor(frames, cached_coords)
        with gpu_lock:
            emotion_cache[emo] = {"frames": frames, "gpu": gpu}
        print(f"✅ 感情ループ [{emo}] 準備完了", flush=True)

# 既存キャッシュがあれば展開
reload_avatar_cache()
load_emotion_caches()

# ==============================================================================
# 4. アバター自動生成・切り替え関数 (setup_avatar)
# ==============================================================================
IDLE_LOOP_SEC = 8.0 if device == 'cpu' else 32.0  # 8 の倍数秒: 全周期が揃い、順再生で継ぎ目のないループになる
IDLE_LOOP_PAD = 25  # 前後1秒の余白（LivePortrait の平滑化の端の影響を切り落とす）

# LivePortrait の --flag_eye_retargeting は相対モーション時に「元画像＋目の変化」だけを使い、
# テンプレートの頭の動き・呼吸・表情をすべて捨てる。既定ではオフにして、まばたきは表情キーポイントで作る。
# 値は試し撮りで調整できるよう Drive の motion_settings.json で上書き可能
MOTION_SETTINGS_FILE = "/content/drive/MyDrive/lipsync_avatar/motion_settings.json"
motion_settings = {"eye_retargeting": False, "blink_eyes": -20.0}
try:
    if os.path.exists(MOTION_SETTINGS_FILE):
        import json as _json
        motion_settings.update(_json.load(open(MOTION_SETTINGS_FILE, encoding="utf-8")))
except Exception as e:
    print(f"⚠️ モーション設定の読み込みをスキップ: {e}")

def render_idle_loop_frames(face_img_path, expression=None, eye_open=1.0, tag="idle", duration=None):
    """待機モーション（＋任意の表情）を LivePortrait で動画化し、余白を除いたループのフレーム列と fps を返す。
    expression 以外（頭の動き・まばたき）は常に同じなので、表情違いのループ同士はコマ単位で頭の位置が一致する"""
    try:
        from procedural_motion import save_procedural_motion_template
    except ImportError:
        sys.path.append("/content/memorial-api")
        from procedural_motion import save_procedural_motion_template

    duration = duration or IDLE_LOOP_SEC
    pkl = f"/content/procedural_{tag}.pkl"
    use_retarget = bool(motion_settings.get("eye_retargeting"))
    save_procedural_motion_template(pkl, motion_type="idle", duration_sec=duration, fps=25,
                                    wrap_pad_frames=IDLE_LOOP_PAD, expression=expression, eye_open=eye_open,
                                    blink_eyes=None if use_retarget else motion_settings.get("blink_eyes", -20.0))

    lp_output_dir = os.path.join(LIVEPORTRAIT_DIR, "animations")
    os.makedirs(lp_output_dir, exist_ok=True)
    lp_cmd = [
        sys.executable, "inference.py",
        "-s", face_img_path,
        "-d", pkl,
        "--flag_relative_motion",
        "--flag_do_crop",
        "--driving_option", "expression-friendly",
    ]
    if use_retarget:
        lp_cmd.append("--flag_eye_retargeting")
    if device == 'cpu':
        lp_cmd.append("--flag_force_cpu")
        print("⚠️ CPUモードのため LivePortrait に --flag_force_cpu を適用します", flush=True)

    t0 = time.time()
    proc = subprocess.Popen(lp_cmd, cwd=LIVEPORTRAIT_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    output_lines = [line for line in proc.stdout]
    proc.wait()
    if proc.returncode != 0:
        full_err = "".join(output_lines[-20:])
        print(f"❌ LivePortrait 実行エラー:\n{full_err}")
        raise RuntimeError(f"LivePortraitエラー: {full_err}")

    # 出力名は「<元画像名>--<テンプレート名>.mp4」（並べて表示する *_concat.mp4 は除外）
    stem = os.path.splitext(os.path.basename(face_img_path))[0]
    out_mp4 = os.path.join(lp_output_dir, f"{stem}--procedural_{tag}.mp4")
    if not os.path.exists(out_mp4):
        cands = [os.path.join(lp_output_dir, f) for f in os.listdir(lp_output_dir)
                 if f.endswith(".mp4") and "concat" not in f]
        if not cands:
            raise RuntimeError("LivePortrait の動画生成結果 (.mp4) が見つかりませんでした。")
        out_mp4 = max(cands, key=os.path.getmtime)

    cap = cv2.VideoCapture(out_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("LivePortrait動画の読み込みに失敗しました。")

    expected = int(duration * 25)
    if len(frames) >= expected + 2 * IDLE_LOOP_PAD:
        frames = frames[IDLE_LOOP_PAD:IDLE_LOOP_PAD + expected]
    print(f"✅ LivePortrait 生成完了 [{tag}] ({len(frames)} フレーム, {time.time() - t0:.0f}秒)", flush=True)
    return frames, fps

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

        # 1. 外部動画不要！数式アルゴリズムから自律モーションテンプレートを作り LivePortrait で動画化
        print("🧠 [1/4] 数式アルゴリズムから自律待機モーション（呼吸・ゆらぎ・まばたき）を生成中...", flush=True)
        raw_frames, fps = render_idle_loop_frames(face_img_path)
        BASE_LOOP_VIDEO = "/content/base_avatar_loop.mp4"
        height, width = raw_frames[0].shape[:2]
        total_frames = len(raw_frames)
        print(f"🚀 [2/4] {total_frames / fps:.0f}秒のループ動画を作成中 ({total_frames}フレーム)...", flush=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(BASE_LOOP_VIDEO, fourcc, fps, (width, height))
        loop_frames = []
        for f in raw_frames:
            out.write(f)
            loop_frames.append(f)
        out.release()

        # 3. Wav2Lipの顔検出器で全フレームの顔座標をキャッシュ
        print(f"🚀 [3/4] 顔座標を計算・キャッシュ中 (全 {len(loop_frames)} フレーム)...", flush=True)
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
            cur_end = min(i + batch_size, len(loop_frames))
            print(f"   🔍 顔検出進捗: {cur_end}/{len(loop_frames)} フレーム...", flush=True)
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
        print("🚀 [4/4] サーバーメモリのキャッシュを最新アバターに更新...", flush=True)
        with gpu_lock:  # 会話の口パク合成中にキャッシュを差し替えないよう、合成の合間に切り替える
            reload_avatar_cache(BASE_LOOP_VIDEO, CACHE_FILE)

        # 5. ブラウザ待機用の avatar_idle.mp4 を口パク動画と同じ画質で生成（切り替え時の見た目を一致させる）
        avatar_idle_path = make_idle_video("/content/avatar_idle.mp4")

        # 前のアバターの感情ループは新しい顔と合わないので片付ける（Drive 側は *_prev へ退避）
        for emo in EMOTION_NAMES:
            if os.path.exists(emotion_loop_path(emo)):
                os.remove(emotion_loop_path(emo))

        # Google Drive にもバックアップ保存（前のアバターは *_prev として1世代残す）
        try:
            emo_names = [f"base_avatar_loop_{emo}.mp4" for emo in EMOTION_NAMES]
            for name in ["base_avatar_loop.mp4", "base_avatar_cache.npz", "avatar_idle.mp4"] + emo_names:
                src = f"{DRIVE_DIR}/{name}"
                if os.path.exists(src):
                    base, ext = os.path.splitext(src)
                    shutil.copy(src, f"{base}_prev{ext}")
                    if name in emo_names:
                        os.remove(src)
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
TALK_MAX_DIM = 400  # 口パク動画・待機動画の長辺（両者を同じ見た目にするため共通）

def encode_frames_to_mp4(frames, out_path, audio_path=None):
    """フレーム列を FFmpeg の標準入力へ流し込み mp4 化（口パク動画と待機動画で同じ画質設定を使う）"""
    orig_h, orig_w, _ = frames[0].shape
    scale = min(TALK_MAX_DIM / max(orig_h, orig_w), 1.0)
    target_w = int(orig_w * scale) // 2 * 2
    target_h = int(orig_h * scale) // 2 * 2

    if os.path.exists(out_path):
        os.remove(out_path)

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{target_w}x{target_h}",
        "-pix_fmt", "bgr24",
        "-r", str(cached_fps),
        "-i", "-",
    ]
    if audio_path:
        ffmpeg_cmd += ["-i", audio_path]
    ffmpeg_cmd += [
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
    ]
    if audio_path:
        ffmpeg_cmd += ["-c:a", "aac", "-b:a", "96k", "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    ffmpeg_cmd.append(out_path)

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame in frames:
        if scale < 1.0:
            frame = cv2.resize(frame, (target_w, target_h))
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return target_w, target_h

def make_idle_video(out_path="/content/idle_for_browser.mp4", emotion=None):
    """口パク合成の元になっているループ動画を、口パク動画と同じ画質で書き出す（ブラウザの待機動画用）。
    emotion を指定するとその感情ループ（返事の後に感情の顔から通常の顔へゆっくり戻す演出に使う）"""
    if len(cached_frames) == 0:
        raise RuntimeError("アバターがセットアップされていません。")
    frames = emotion_cache[emotion]["frames"] if emotion else cached_frames
    encode_frames_to_mp4(frames, out_path)
    return out_path

_mouth_mask_cache = {}

def mouth_blend_mask(h, w):
    """顔ボックス内で口まわりだけを 1、目元より上を 0 とし、境目と左右・下端をぼかしたマスク (h, w, 1)"""
    key = (h, w)
    if key not in _mouth_mask_cache:
        m = np.zeros((h, w), np.float32)
        m[int(h * 0.55):, :] = 1.0  # 鼻の下あたりから下（Wav2Lip が生成するのは顔の下半分）
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=max(w * 0.06, 1), sigmaY=max(h * 0.06, 1))
        # ボックスの左右・下端は元の顔へ徐々に戻して貼り付けの継ぎ目を消す
        xs = np.minimum(np.arange(w), np.arange(w)[::-1]) / max(w * 0.08, 1)
        ys = np.arange(h)[::-1] / max(h * 0.04, 1)
        m *= np.clip(xs, 0, 1)[None, :] * np.clip(ys, 0, 1)[:, None]
        _mouth_mask_cache[key] = m[..., None]
    return _mouth_mask_cache[key]

def count_video_frames(audio_path):
    """fast_process_pipeline がこの音声から作る動画のコマ数（区間をつなぐ開始コマの計算用。下と同じ数え方）"""
    mel_len = max(audio.melspectrogram(audio.load_wav(audio_path, 16000)).shape[1], 16)
    mel_idx_multiplier = 80.0 / cached_fps
    i = 0
    while int(i * mel_idx_multiplier) + 16 <= mel_len:
        i += 1
    return i + 1

def fast_process_pipeline(face_img_path, audio_path, start_frame=0, out_path="/content/final_synced_output.mp4", emotion=None, ramp_in=0):
    """start_frame: ループ動画の何コマ目から合成を始めるか（区間をまたいで頭の動きを連続させる）
    emotion: 感情ループ（happy / calm / sad）を土台にする。未生成なら通常ループ
    ramp_in: 先頭の何コマで通常の顔から感情の顔へ徐々に移すか（返事の最初の区間で使い、表情の急変を防ぐ）"""
    t_start = time.time()

    if gpu_face_tensor is None or len(cached_frames) == 0:
        raise RuntimeError("アバターがセットアップされていません。まずアバター画像をセットアップしてください。")
    emo = emotion_cache.get(emotion) if emotion else None
    src_frames = emo["frames"] if emo else cached_frames
    src_tensor = emo["gpu"] if emo else gpu_face_tensor

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
    start_frame = int(start_frame) % total_cached
    full_frames = [src_frames[(start_frame + idx) % total_cached].copy() for idx in range(num_frames)]
    if emo and ramp_in > 0:
        # 頭の位置は通常ループと一致しているので、コマごとの重ね合わせで表情だけがなめらかに移る
        for idx in range(min(ramp_in, num_frames)):
            a = (idx + 1) / (ramp_in + 1)
            a = a * a * (3 - 2 * a)  # smoothstep（動き始めと終わりをゆっくり）
            neutral = cached_frames[(start_frame + idx) % total_cached]
            full_frames[idx] = cv2.addWeighted(neutral, 1.0 - a, full_frames[idx], a, 0)
    coords = [cached_coords[(start_frame + idx) % total_cached] for idx in range(num_frames)]

    t_gpu = time.time()
    for i in range(0, num_frames, wav2lip_batch_size):
        batch_mels = mel_chunks[i:i + wav2lip_batch_size]
        batch_c = coords[i:i + wav2lip_batch_size]
        cur_batch_len = len(batch_mels)

        indices = [(start_frame + i + k) % total_cached for k in range(cur_batch_len)]
        img_tensor = src_tensor[indices]
        mel_tensor = torch.FloatTensor(np.array(batch_mels)).unsqueeze(1).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                pred = model(mel_tensor, img_tensor)

        pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

        for j, (p, (y1, y2, x1, x2)) in enumerate(zip(pred, batch_c)):
            idx = i + j
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            # 顔全体を貼ると 96px 由来のボケで目元まで変わるため、口まわり（下半分）だけをなじませて合成
            m = mouth_blend_mask(y2 - y1, x2 - x1)
            orig = full_frames[idx][y1:y2, x1:x2].astype(np.float32)
            full_frames[idx][y1:y2, x1:x2] = (p.astype(np.float32) * m + orig * (1.0 - m)).astype(np.uint8)

    gpu_sec = time.time() - t_gpu

    t_enc = time.time()
    final_mp4 = out_path
    target_w, target_h = encode_frames_to_mp4(full_frames, final_mp4, audio_path)
    enc_sec = time.time() - t_enc

    print(f"⚡ 推論: {gpu_sec:.2f}秒 | エンコード: {enc_sec:.2f}秒 ({target_w}x{target_h}) | 合計: {time.time() - t_start:.2f}秒")
    return final_mp4

# ==============================================================================
# 5.5. ウォームアップ（初回リクエストだけ librosa/CUDA の初期化で十数秒かかるのを起動時に済ませる）
# ==============================================================================
def warmup_pipeline():
    import wave
    warm_wav = "/content/warmup.wav"
    sr = 16000
    noise = (np.random.randn(int(sr * 1.5)) * 300).astype(np.int16)
    with wave.open(warm_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(noise.tobytes())
    t0 = time.time()
    fast_process_pipeline(None, warm_wav)
    print(f"🔥 ウォームアップ完了 ({time.time() - t0:.2f}秒) - 初回の会話から高速に応答します", flush=True)

if gpu_face_tensor is not None:
    try:
        warmup_pipeline()
    except Exception as e:
        print(f"⚠️ ウォームアップスキップ: {e}")

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
# 6.2. AivisSpeech Engine を Colab GPU で起動（音声合成もGPU化し、ローカルCPUの約1.5秒を短縮）
#      失敗してもサーバー本体には影響せず、ローカル側は従来どおり自前の AivisSpeech を使う
# ==============================================================================
import glob
import threading
import requests

AIVIS_DIR = "/content/AivisSpeech-Engine"
AIVIS_PORT = 10101
AIVIS_LOCAL_URL = f"http://127.0.0.1:{AIVIS_PORT}"
AIVIS_LOG = "/content/aivis_engine.log"
# ローカルで使っているモデル（まお。スピーカーID 888753760 = まお・ノーマル）。AivisHub から直接ダウンロードする
AIVIS_MODEL_UUIDS = os.environ.get("AIVIS_MODEL_UUIDS", "a59cb814-0083-4369-8542-f51a29e72af7").split(",")
AIVIS_WARM_SPEAKER = int(os.environ.get("AIVIS_WARM_SPEAKER", "888753760"))
aivis_state = {"ready": False, "detail": "未起動", "warm_synth_sec": None}

def aivis_setup_worker():
    def step(msg):
        aivis_state["detail"] = msg
        print(f"🗣️ [GPU音声合成] {msg}", flush=True)
    try:
        t0 = time.time()
        step("uv をインストール中...")
        subprocess.run(["pip", "install", "-q", "uv"], check=True)
        if not os.path.exists(AIVIS_DIR):
            step("AivisSpeech Engine を取得中...")
            subprocess.run(["git", "clone", "-q", "--depth", "1",
                            "https://github.com/Aivis-Project/AivisSpeech-Engine.git", AIVIS_DIR], check=True)
        step("依存ライブラリをインストール中（Python 3.11 環境を作成）...")
        subprocess.run(["uv", "sync", "--no-default-groups", "-q"], cwd=AIVIS_DIR, check=True)

        # モデル配置（ソース実行時の保存先は AivisSpeech-Engine-Dev）
        model_dir = os.path.expanduser("~/.local/share/AivisSpeech-Engine-Dev/Models")
        os.makedirs(model_dir, exist_ok=True)
        for uuid in AIVIS_MODEL_UUIDS:
            dest = os.path.join(model_dir, f"{uuid.strip()}.aivmx")
            if not os.path.exists(dest):
                step(f"音声モデルをダウンロード中 ({uuid.strip()[:8]})...")
                subprocess.run(["curl", "-sL", "-o", dest,
                                f"https://api.aivis-project.com/v1/aivm-models/{uuid.strip()}/download?model_type=AIVMX"], check=True)

        # onnxruntime-gpu が CUDA/cuDNN を見つけられるよう、Colab の torch 同梱 NVIDIA ライブラリを渡す
        env = os.environ.copy()
        nv_libs = glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/*/lib")
        env["LD_LIBRARY_PATH"] = ":".join(nv_libs + [env.get("LD_LIBRARY_PATH", "")])
        step("エンジン起動中（初回は BERT モデル約650MBのダウンロードあり）...")
        log = open(AIVIS_LOG, "w")
        subprocess.Popen([os.path.join(AIVIS_DIR, ".venv/bin/python"), "run.py", "--use_gpu",
                          "--host", "127.0.0.1", "--port", str(AIVIS_PORT)],
                         cwd=AIVIS_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)

        for _ in range(300):
            try:
                if requests.get(f"{AIVIS_LOCAL_URL}/version", timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"エンジンが起動しませんでした（ログ: {AIVIS_LOG}）")

        # ウォームアップ（初回合成はモデル読み込みを含むため2回測る）
        for _ in range(2):
            ts = time.time()
            aivis_synthesize("こんにちは、よろしくね。", AIVIS_WARM_SPEAKER)
            warm = time.time() - ts
        aivis_state["warm_synth_sec"] = round(warm, 3)
        aivis_state["ready"] = True
        step(f"準備完了（起動 {time.time() - t0:.0f}秒, 合成 {warm:.2f}秒/文）")
    except Exception as e:
        aivis_state["ready"] = False
        step(f"起動失敗（ローカルのCPU合成を使います）: {e}")

def aivis_synthesize(text, speaker, speed=1.22, pre=None, post=None):
    q = requests.post(f"{AIVIS_LOCAL_URL}/audio_query", params={"text": text, "speaker": speaker}, timeout=10)
    q.raise_for_status()
    query = q.json()
    query["speedScale"] = speed
    if pre is not None:
        query["prePhonemeLength"] = pre
    if post is not None:
        query["postPhonemeLength"] = post
    s = requests.post(f"{AIVIS_LOCAL_URL}/synthesis", params={"speaker": speaker}, json=query, timeout=30)
    s.raise_for_status()
    return s.content

# ==============================================================================
# 6.5. ダイレクト超高速 API (FastAPI 直結ルート) の登録
# ==============================================================================
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

# demo.app に登録したルートは demo.launch() 時に作り直される App に引き継がれず 404 になる。
# 独自の FastAPI にルートを登録し、Gradio をその上にマウントして起動する（ngrok モード時）
api_app = FastAPI()

@api_app.get("/api/idle_video")
def api_idle_video(emotion: str = ""):
    """ブラウザ用の待機動画（口パク合成の元と同一のループ・同一画質）。?emotion=happy 等で感情ループ"""
    if emotion and emotion not in emotion_cache:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"感情ループ {emotion} は未生成です")
    path = make_idle_video(f"/content/idle_for_browser_{emotion or 'normal'}.mp4", emotion or None)
    return FileResponse(
        path, media_type="video/mp4", filename="avatar_idle.mp4",
        headers={"X-Total-Frames": str(len(cached_frames)), "X-FPS": str(cached_fps)}
    )

@api_app.get("/api/tts_status")
def api_tts_status():
    # turn_support: 区間の並行依頼に対応 / emotions: 生成済みの感情ループ
    return {**aivis_state, "turn_support": True, "emotions": sorted(emotion_cache)}

# 1回の返答（turn）の区間は並行してリクエストされる。音声合成だけは区間順に行い、
# 各区間の開始コマ = 前の区間の終了コマ として頭の動きを連続させる。口パク合成は GPU を1本ずつ使う。
turn_states = {}
turn_states_lock = threading.Lock()
gpu_lock = threading.Lock()

def get_turn_state(turn_id, start_frame):
    with turn_states_lock:
        if turn_id not in turn_states:
            if len(turn_states) > 50:
                turn_states.clear()
            turn_states[turn_id] = {"cond": threading.Condition(), "next_index": 0, "next_frame": start_frame}
        return turn_states[turn_id]

@api_app.post("/api/generate_from_text")
def api_generate_from_text(
    text: str = Form(...),
    speaker: int = Form(AIVIS_WARM_SPEAKER),
    speed: float = Form(1.22),
    pre_silence: float = Form(0.05),
    post_silence: float = Form(0.05),
    start_frame: int = Form(0),
    turn_id: str = Form(""),
    emotion: str = Form(""),
    ramp_in: int = Form(0),
    seg_index: int = Form(0),
):
    """テキスト → GPU音声合成 → 口パク動画 を Colab 内で一括実行（音声のアップロード往復も不要）
    turn_id を指定すると、同じ返答の区間を並行で受け付け、区間順にコマ位置を連結する（start_frame は区間0のみ使用）"""
    if not aivis_state["ready"]:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"GPU音声合成は未準備: {aivis_state['detail']}")
    import wave as _wave
    t0 = time.time()
    temp_wav = f"/content/req_{turn_id or 'x'}_{seg_index}_{int(time.time() * 1000)}.wav"

    def synth_to_file():
        wav_bytes = aivis_synthesize(text, speaker, speed, pre_silence, post_silence)
        with open(temp_wav, "wb") as f:
            f.write(wav_bytes)
        with _wave.open(temp_wav) as w:
            return w.getnframes() / w.getframerate()

    if turn_id:
        st = get_turn_state(turn_id, start_frame)
        with st["cond"]:
            st["cond"].wait_for(lambda: st["next_index"] >= seg_index, timeout=15)
            try:
                duration = synth_to_file()
                my_start = st["next_frame"]
                st["next_frame"] = my_start + count_video_frames(temp_wav)
            finally:
                st["next_index"] = max(st["next_index"], seg_index + 1)
                st["cond"].notify_all()
    else:
        duration = synth_to_file()
        my_start = start_frame
    tts_sec = time.time() - t0

    out_mp4 = temp_wav.replace(".wav", ".mp4")
    try:
        with gpu_lock:
            mp4_path = fast_process_pipeline(None, temp_wav, my_start, out_mp4, emotion or None, ramp_in)
        print(f"⚡ [テキスト→動画 完了] 合成 {tts_sec:.2f}秒 | 合計 {time.time() - t0:.2f}秒: {text}", flush=True)
        from starlette.background import BackgroundTask
        return FileResponse(
            mp4_path, media_type="video/mp4", filename="final_output.mp4",
            headers={"X-Audio-Duration": f"{duration:.3f}", "X-TTS-Sec": f"{tts_sec:.3f}"},
            background=BackgroundTask(lambda: os.path.exists(mp4_path) and os.remove(mp4_path))
        )
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

# 重い処理の窓口は async にしない（async 内で重い処理を直接実行すると、終わるまで
# サーバー全体が他のリクエストに応答しなくなる。通常の def なら別スレッドで実行される）
@api_app.post("/api/generate")
def api_generate(audio: UploadFile = File(...), start_frame: int = Form(0), emotion: str = Form(""), ramp_in: int = Form(0)):
    t0 = time.time()
    temp_wav = f"/content/req_{int(time.time() * 1000)}.wav"
    with open(temp_wav, "wb") as f:
        f.write(audio.file.read())
    try:
        out_mp4 = temp_wav.replace(".wav", ".mp4")
        with gpu_lock:
            mp4_path = fast_process_pipeline(None, temp_wav, start_frame, out_mp4, emotion or None, ramp_in)
        print(f"⚡ [ダイレクトAPI動画生成完了] ({time.time() - t0:.2f}秒)", flush=True)
        from starlette.background import BackgroundTask
        return FileResponse(mp4_path, media_type="video/mp4", filename="final_output.mp4",
                            background=BackgroundTask(lambda: os.path.exists(mp4_path) and os.remove(mp4_path)))
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception:
                pass

@api_app.post("/api/setup_avatar")
def api_setup_avatar(image: UploadFile = File(...)):
    t0 = time.time()
    temp_img = f"/content/face_{int(time.time() * 1000)}.jpg"
    with open(temp_img, "wb") as f:
        f.write(image.file.read())
    try:
        idle_path = setup_avatar(temp_img)
        print(f"🎉 [ダイレクトAPIアバター更新完了] ({time.time() - t0:.2f}秒)", flush=True)
        return FileResponse(idle_path, media_type="video/mp4", filename="avatar_idle.mp4")
    finally:
        if os.path.exists(temp_img):
            try:
                os.remove(temp_img)
            except Exception:
                pass

# アバター生成は約10分かかり、1回の HTTP リクエストで待つとトンネル経由で接続が切れることがある。
# 開始だけ受け付けて裏で実行し、進み具合は /api/setup_avatar_status で確認してもらう方式。
setup_job = {"state": "idle", "detail": "", "started": None, "elapsed": None}

def run_setup_job(temp_img, kind="full"):
    """kind: full = 通常ループ＋感情ループ / emotions = 感情ループだけ（現在のアバターに追加）"""
    def progress(msg):
        setup_job["detail"] = msg
        print(f"🎭 {msg}", flush=True)
    try:
        if kind == "full":
            progress("通常ループ生成中")
            setup_avatar(temp_img)
        build_emotion_loops(temp_img, progress=progress)
        setup_job.update(state="done", detail="アバター生成完了" if kind == "full" else "感情ループ生成完了")
    except Exception as e:
        setup_job.update(state="error", detail=f"{type(e).__name__}: {e}")
    finally:
        setup_job["elapsed"] = round(time.time() - setup_job["started"], 1)
        if os.path.exists(temp_img):
            try:
                os.remove(temp_img)
            except Exception:
                pass

def start_setup_job(image, kind):
    if setup_job["state"] == "running":
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="アバター生成がすでに実行中です")
    temp_img = f"/content/face_{int(time.time() * 1000)}.jpg"
    with open(temp_img, "wb") as f:
        f.write(image.file.read())
    setup_job.update(state="running", detail="生成中", started=time.time(), elapsed=None)
    threading.Thread(target=run_setup_job, args=(temp_img, kind), daemon=True).start()
    return {"state": "running", "kind": kind}

@api_app.post("/api/setup_avatar_start")
def api_setup_avatar_start(image: UploadFile = File(...)):
    return start_setup_job(image, "full")

@api_app.post("/api/setup_emotions_start")
def api_setup_emotions_start(image: UploadFile = File(...)):
    """現在のアバター（同じ写真）に感情ループだけを追加生成。通常ループは作り直さない"""
    if len(cached_frames) == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="先に通常のアバターを生成してください")
    return start_setup_job(image, "emotions")

@api_app.get("/api/emotion_presets")
def api_get_emotion_presets():
    return emotion_presets

@api_app.post("/api/emotion_presets")
def api_set_emotion_presets(presets: dict):
    """表情プリセットを更新（Colab 再起動なしで調整するため）。Drive にも保存し次回起動時に読み込む"""
    for emo, val in presets.items():
        if emo in EMOTION_NAMES:
            emotion_presets[emo] = val
    try:
        os.makedirs(DRIVE_DIR, exist_ok=True)
        json.dump(emotion_presets, open(EMOTION_PRESETS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"⚠️ 感情プリセットの保存に失敗: {e}")
    return emotion_presets

@api_app.get("/api/motion_settings")
def api_get_motion_settings():
    return motion_settings

@api_app.post("/api/motion_settings")
def api_set_motion_settings(settings: dict):
    """待機モーションの設定（eye_retargeting / blink_eyes）を更新。Drive に保存し次回起動時も使う"""
    motion_settings.update({k: v for k, v in settings.items() if k in ("eye_retargeting", "blink_eyes")})
    try:
        os.makedirs(os.path.dirname(MOTION_SETTINGS_FILE), exist_ok=True)
        json.dump(motion_settings, open(MOTION_SETTINGS_FILE, "w", encoding="utf-8"), indent=1)
    except Exception as e:
        print(f"⚠️ モーション設定の保存に失敗: {e}")
    return motion_settings

preview_lock = threading.Lock()

@api_app.post("/api/expression_preview")
def api_expression_preview(image: UploadFile = File(...), preset: str = Form("{}")):
    """表情の試し撮り: 2秒の短いループを生成し、通常ループの顔と並べた PNG を返す（約30秒）"""
    p = json.loads(preset)
    temp_img = f"/content/preview_{int(time.time() * 1000)}.jpg"
    with open(temp_img, "wb") as f:
        f.write(image.file.read())
    try:
        with preview_lock:
            frames, _ = render_idle_loop_frames(temp_img, expression=p.get("expression"),
                                                eye_open=p.get("eye_open", 1.0), tag="preview", duration=2.0)
        # 左から: 現在の通常ループ / 試し撮り（0.4秒時点） / 試し撮り（まばたきで目を閉じる 1.28秒時点） / 試し撮り（1.8秒時点）
        shots = [frames[i] for i in (10, 32, 45) if i < len(frames)]
        ref = cached_frames[10] if len(cached_frames) > 10 else shots[0]
        if ref.shape != shots[0].shape:
            ref = cv2.resize(ref, (shots[0].shape[1], shots[0].shape[0]))
        out_png = temp_img.replace(".jpg", ".png")
        cv2.imwrite(out_png, np.hstack([ref] + shots))
        from starlette.background import BackgroundTask
        return FileResponse(out_png, media_type="image/png",
                            background=BackgroundTask(lambda: os.path.exists(out_png) and os.remove(out_png)))
    finally:
        if os.path.exists(temp_img):
            os.remove(temp_img)

@api_app.get("/api/setup_avatar_status")
def api_setup_avatar_status():
    st = dict(setup_job)
    if st["state"] == "running":
        st["elapsed"] = round(time.time() - st["started"], 1)
    return st

# ==============================================================================
# 7. 高速トンネル (Cloudflare) ＆ URL自動クラウド同期
# ==============================================================================
import threading
import subprocess
import re
import requests

def cloudflare_tunnel_worker():
    print("📡 Cloudflare 高速トンネル (日本国内エッジ) をセットアップ中...", flush=True)
    cf_bin = "/usr/local/bin/cloudflared"
    if not os.path.exists(cf_bin):
        try:
            subprocess.run(["curl", "-sL", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-o", cf_bin], check=True)
            subprocess.run(["chmod", "+x", cf_bin], check=True)
        except Exception as e:
            print(f"⚠️ cloudflared インストールスキップ: {e}")
            return

    try:
        proc = subprocess.Popen(
            [cf_bin, "tunnel", "--url", "http://127.0.0.1:7860"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if m:
                    cf_url = m.group(0)
                    print("\n" + "=" * 60)
                    print(f"⚡ 【Cloudflare 高速トンネル起動完了】: {cf_url}")
                    print("（超低遅延な日本国内エッジサーバー経由で高速通信します）")
                    print("=" * 60 + "\n", flush=True)
                    sync_endpoints = [
                        "https://api.cl1p.net/kaeru510-memorial"
                    ]
                    for ep in sync_endpoints:
                        try:
                            requests.post(ep, data=cf_url.strip(), timeout=5)
                        except Exception:
                            pass
                    break
    except Exception as e:
        print(f"⚠️ Cloudflare 起動スキップ: {e}")

def ngrok_tunnel_worker(authtoken, domain):
    """ngrok の固定ドメインでトンネルを張る（起動のたびにURLが変わらない）"""
    print(f"📡 ngrok 固定トンネルをセットアップ中: https://{domain}", flush=True)
    ngrok_bin = "/usr/local/bin/ngrok"
    if not os.path.exists(ngrok_bin):
        try:
            subprocess.run(["curl", "-sL", "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", "-o", "/tmp/ngrok.tgz"], check=True)
            subprocess.run(["tar", "-xzf", "/tmp/ngrok.tgz", "-C", "/usr/local/bin"], check=True)
        except Exception as e:
            print(f"⚠️ ngrok インストール失敗: {e}")
            return

    # セルを止めて再実行した場合に前回の ngrok が残っていると固定ドメインが使用中で失敗するため片付ける
    subprocess.run(["pkill", "-f", "ngrok http"], check=False)
    time.sleep(1)

    try:
        proc = subprocess.Popen(
            [ngrok_bin, "http", "127.0.0.1:7860", "--url", f"https://{domain}",
             "--authtoken", authtoken, "--log", "stdout", "--log-format", "logfmt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            if "started tunnel" in line:
                print("\n" + "=" * 60)
                print(f"⚡ 【ngrok 固定トンネル起動完了】: https://{domain}")
                print("（URLは毎回同じなので、ローカル側の設定変更は不要です）")
                print("=" * 60 + "\n", flush=True)
            elif "lvl=eror" in line or "ERR_NGROK" in line:
                print(f"⚠️ ngrok エラー: {line.strip()}", flush=True)
    except Exception as e:
        print(f"⚠️ ngrok 起動失敗: {e}")

# ノートブック側で NGROK_AUTHTOKEN / NGROK_DOMAIN が設定されていれば固定URLの ngrok、なければ従来の Cloudflare
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "").strip()
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "").strip().removeprefix("https://").rstrip("/")
if NGROK_AUTHTOKEN and NGROK_DOMAIN:
    threading.Thread(target=ngrok_tunnel_worker, args=(NGROK_AUTHTOKEN, NGROK_DOMAIN), daemon=True).start()
else:
    threading.Thread(target=cloudflare_tunnel_worker, daemon=True).start()

def sync_url_worker():
    sync_endpoints = [
        "https://api.cl1p.net/kaeru510-memorial"
    ]
    print("📡 URL自動同期ワーカー開始...")
    for _ in range(40):  # 最大80秒待機
        time.sleep(2)
        url = getattr(demo, "share_url", None)
        if url:
            for ep in sync_endpoints:
                try:
                    requests.post(ep, data=url.strip(), timeout=5)
                except Exception:
                    pass
            print("\n" + "=" * 60)
            print(f"📡 【URL自動同期完了】最新URLをクラウドに送信しました！")
            print(f"👉 URL: {url}")
            print(f"（ローカルPC側で自動検出されるため、コピペ不要です）")
            print("=" * 60 + "\n")
            break
        else:
            time.sleep(1)

if os.environ.get("AIVIS_ON_COLAB", "1") == "1":
    threading.Thread(target=aivis_setup_worker, daemon=True).start()

if NGROK_AUTHTOKEN and NGROK_DOMAIN:
    # 固定URL運用: ダイレクトAPI(/api/*) と Gradio(/) を同じポート 7860 で提供（gradio.live 共有リンクは不要）
    import uvicorn
    app = gr.mount_gradio_app(api_app, demo, path="/", allowed_paths=["/content"], show_error=True)
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
else:
    # 従来運用: gradio.live 共有リンク + クラウド同期（この場合ダイレクトAPIは使えず Gradio 経由になる）
    threading.Thread(target=sync_url_worker, daemon=True).start()
    demo.launch(share=True, debug=True, show_error=True, allowed_paths=["/content"])


