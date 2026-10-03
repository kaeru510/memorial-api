import re
import os
import sys
import time
import requests
import traceback
import shutil
import secrets
import urllib.parse
import threading
import json
import base64
import io
import wave
from concurrent.futures import ThreadPoolExecutor

# Windows の cp932 環境での絵文字出力エラー対策
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from gradio_client import Client, handle_file

# Google Gemini API のインポート
from google import genai
from google.genai import types

# ====================================================
# 設定項目（現在の設定を保持しています）
# ====================================================

# 1. Basic認証
ADMIN_USERNAME = "a"
ADMIN_PASSWORD = "a"

# 2. Gemini API キー（コードには書かず、環境変数 or 同じフォルダの .env から読み込む）
def load_env_file(path):
    """KEY=VALUE 形式の .env を読み、未設定の環境変数だけ補完する（追加パッケージ不要）"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY が未設定です。main.py と同じフォルダの .env に GEMINI_API_KEY=... を書いてください。")

# 3. Colab サーバーの URL
#    .env に COLAB_FIXED_URL（ngrok の固定ドメイン）があれば常にそれを使い、クラウド同期は行わない
COLAB_FIXED_URL = os.environ.get("COLAB_FIXED_URL", "").strip().rstrip("/")
COLAB_GRADIO_URL = COLAB_FIXED_URL or "https://7e42be673fb0f0c312.gradio.live"
gradio_client = None
# ngrok 無料プランの警告ページを飛ばすヘッダー（ngrok 以外では無視される）
TUNNEL_HEADERS = {"ngrok-skip-browser-warning": "true"}

# AivisSpeech 設定
AIVIS_URL = "http://127.0.0.1:10101"
SPEAKER_ID = 888753760

# 感情ラベル → 声のスタイル（まお）。顔は Colab 側の同名の感情ループを使い、声と顔の感情を一致させる
EMOTION_STYLE_IDS = {
    "normal": 888753760,  # ノーマル
    "happy": 888753762,   # あまあま
    "calm": 888753763,    # おちつき
    "sad": 888753765,     # せつなめ
}
EMOTION_ALIASES = {"nod": "calm", "curious": "normal"}  # 旧タグや想定外のタグの読み替え

def normalize_emotion(emotion):
    emotion = EMOTION_ALIASES.get(emotion, emotion)
    return emotion if emotion in EMOTION_STYLE_IDS else "normal"

# ====================================================
# ファイルパス設定
# ====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
FACE_IMG_PATH = os.path.join(BASE_DIR, "face.jpg")
OUTPUT_AUDIO_PATH = os.path.join(BASE_DIR, "output.wav")
FINAL_VIDEO_PATH = os.path.join(BASE_DIR, "final_output.mp4")

# ====================================================
# 1. Gemini の初期化
# ====================================================
SYSTEM_INSTRUCTION = """
あなたは親しみやすい対話AIアシスタントです。
以下のルールを必ず守って返答してください：
1. 友達のように明るく親しみやすい口調で話してください。
2. 会話のテンポを最優先するため、返答全体は20文字〜45文字程度に短くまとめてください。
3. 返答は必ず、相手の発言に合った短いリアクション（2〜6文字程度）と読点で始め、そのあとに本文を1文続けてください。
   リアクションは内容に合わせて毎回変え、同じ言葉を続けて使わないでください。
   リアクションの例: えー、 / そっか、 / いいね、 / うーん、 / なるほど、 / わあ、 / えっ、 / うんうん、
4. 文末は「〜だよ」「〜ですね」など、自然に会話を完結させてください。
5. 返答の先頭に必ず感情タグ [normal], [happy], [calm], [sad] のいずれか1つを付与してください。
   声と表情がこの感情になるので、返答の内容と合うものを選んでください。
   ・嬉しい話題・楽しい話・感謝・挨拶: [happy]
   ・共感・励まし・労い・穏やかな相槌: [calm]
   ・悲しい話・残念な話・寂しさへの寄り添い: [sad]
   ・質問への答えや説明など、それ以外: [normal]
   例: [happy] わあ、今日も会えて嬉しいよ！
   例: [calm] そっか、今日は本当にお疲れさま。
   例: [sad] そうなんだ、それはさみしかったね。
   例: [normal] うーん、映画ならアニメがおすすめだよ。
6. 絵文字、記号、マークダウン記号（*や#）、括弧による注釈は一切含めないでください。
"""

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

chat_session = gemini_client.chats.create(
    model="gemini-3.5-flash-lite",
    config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
        max_output_tokens=150,
    )
)

print("⚡ AI エンジンをウォームアップ中...")
try:
    gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="ping",
        config=types.GenerateContentConfig(max_output_tokens=5)
    )
    print("✅ AI 思考エンジンの準備完了（即レスモード）")
except Exception as e:
    print("⚠️ ウォームアップスキップ:", e)

SYNC_ENDPOINTS = [
    "https://api.cl1p.net/kaeru510-memorial"
]
CACHE_URL_FILE = os.path.join(BASE_DIR, "colab_url.txt")

def fetch_latest_colab_url():
    """クラウド同期エンドポイントまたはローカルキャッシュから最新のColab URLを取得"""
    global COLAB_GRADIO_URL, gradio_client
    if COLAB_FIXED_URL:
        return COLAB_GRADIO_URL  # 固定URL運用（画面の鉛筆から一時的に変えた場合はその値）
    for ep in SYNC_ENDPOINTS:
        try:
            res = requests.get(ep, timeout=2)
            if res.ok:
                remote_url = res.text.strip()
                if remote_url.startswith("http") and ("gradio.live" in remote_url or "trycloudflare.com" in remote_url or "ngrok" in remote_url):
                    if remote_url != COLAB_GRADIO_URL:
                        print(f"🔄 最新のColab URLを自動同期しました ({ep.split('/')[2]}): {remote_url}")
                        COLAB_GRADIO_URL = remote_url
                        gradio_client = None  # 再接続を促す
                        try:
                            with open(CACHE_URL_FILE, "w", encoding="utf-8") as f:
                                f.write(remote_url)
                        except Exception:
                            pass
                    return remote_url
        except Exception:
            pass

    # クラウド同期が応答しない場合はローカルキャッシュを確認
    if os.path.exists(CACHE_URL_FILE):
        try:
            with open(CACHE_URL_FILE, "r", encoding="utf-8") as f:
                saved = f.read().strip()
                if saved.startswith("http") and ("gradio.live" in saved or "trycloudflare.com" in saved or "ngrok" in saved):
                    if saved != COLAB_GRADIO_URL:
                        COLAB_GRADIO_URL = saved
                        gradio_client = None
                    return saved
        except Exception:
            pass

    return COLAB_GRADIO_URL

def get_gradio_client():
    """リクエスト時に接続を試みる（最新URLを自動取得し、失敗時は再取得）"""
    global gradio_client, COLAB_GRADIO_URL

    # まず最新URLを同期
    fetch_latest_colab_url()

    if gradio_client is None:
        print(f"🌐 Colab サーバーへ接続中: {COLAB_GRADIO_URL}")
        try:
            gradio_client = Client(COLAB_GRADIO_URL, headers=TUNNEL_HEADERS)
            print("✅ Colab 接続完了！")
        except Exception as e:
            # 接続失敗時、もう一度最新URLの同期を試みる
            print(f"⚠️ 接続失敗、最新URLを再チェック中...: {e}")
            latest_url = fetch_latest_colab_url()
            if latest_url and latest_url != COLAB_GRADIO_URL:
                try:
                    COLAB_GRADIO_URL = latest_url
                    gradio_client = Client(COLAB_GRADIO_URL, headers=TUNNEL_HEADERS)
                    print("✅ 再取得URLでのColab接続完了！")
                    return gradio_client
                except Exception:
                    pass
            print(f"❌ Colab 接続失敗: {COLAB_GRADIO_URL}")
            raise HTTPException(
                status_code=503,
                detail=f"Colabサーバー（{COLAB_GRADIO_URL}）に接続できませんでした。Colabセルが実行中か確認してください。"
            )
    return gradio_client

# ====================================================
# AivisSpeech 音声合成（起動直後でエンジン未準備なら最大30秒待つ）
# ====================================================
def synthesize_speech(text, wait_sec=30, pre_silence=None, post_silence=None, speaker=SPEAKER_ID, intonation=None):
    """pre_silence / post_silence: 音声前後の無音（秒）。区間分割時はつなぎ目の間を詰めるため短くする"""
    deadline = time.time() + wait_sec
    while True:
        try:
            query_res = requests.post(
                f"{AIVIS_URL}/audio_query",
                params={"text": text, "speaker": speaker},
                timeout=10
            )
            break
        except requests.exceptions.ConnectionError:
            if time.time() > deadline:
                raise
            time.sleep(1)
    query_res.raise_for_status()
    query_data = query_res.json()

    # ★ 話速を 1.22倍 に設定（自然な早口で動画総フレーム数を大幅削減）
    query_data["speedScale"] = 1.22
    if intonation is not None:
        query_data["intonationScale"] = intonation  # AivisSpeech では感情表現の強さ（1.0→重み1、2.0→重み10）
    if pre_silence is not None:
        query_data["prePhonemeLength"] = pre_silence
    if post_silence is not None:
        query_data["postPhonemeLength"] = post_silence

    synth_res = requests.post(
        f"{AIVIS_URL}/synthesis",
        params={"speaker": speaker},
        json=query_data,
        timeout=30
    )
    synth_res.raise_for_status()
    return synth_res.content

def warmup_aivis():
    """初回合成だけ遅い（約3.6秒→2回目以降1.5秒）ので、起動時に裏で1回合成しておく"""
    try:
        t0 = time.time()
        synthesize_speech("こんにちは。", wait_sec=90)
        print(f"🔥 音声合成ウォームアップ完了 ({time.time() - t0:.2f}秒)")
    except Exception as e:
        print(f"⚠️ 音声合成ウォームアップスキップ: {e}")

threading.Thread(target=warmup_aivis, daemon=True).start()

# ====================================================
# FastAPI アプリ初期化
# ====================================================
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    is_user_ok = secrets.compare_digest(credentials.username.encode("utf8"), ADMIN_USERNAME.encode("utf8"))
    is_pass_ok = secrets.compare_digest(credentials.password.encode("utf8"), ADMIN_PASSWORD.encode("utf8"))
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="認証に失敗しました",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

app = FastAPI(title="LipSync AI Chat API", dependencies=[Depends(verify_credentials)])

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ChatRequest(BaseModel):
    message: str

# ====================================================
# エンドポイント定義
# ====================================================
@app.get("/")
def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "index.html が見つかりません。"}

# ====================================================
# Colab GPU による口パク動画生成（ダイレクトAPI優先、失敗時 Gradio Client）
# ====================================================
def generate_video(audio_path, out_path, start_frame=0, emotion="normal", ramp_in=0):
    """start_frame: ループ動画の何コマ目から合成するか（ダイレクトAPIのみ対応）"""
    global gradio_client
    t0 = time.time()
    colab_url = fetch_latest_colab_url()

    # 【超高速優先ルート】ダイレクトAPI (/api/generate) へ音声のみ一撃POST
    try:
        with open(audio_path, "rb") as f:
            res = requests.post(
                f"{colab_url}/api/generate",
                files={"audio": ("output.wav", f, "audio/wav")},
                data={"start_frame": str(int(start_frame)), "emotion": emotion, "ramp_in": str(ramp_in)},
                headers=TUNNEL_HEADERS,
                timeout=20
            )
        if res.ok and len(res.content) > 1000:
            with open(out_path, "wb") as out_f:
                out_f.write(res.content)
            print(f"🎬 [3. 高速ダイレクト動画合成+転送] ({time.time() - t0:.2f}秒)")
            return
        print(f"ℹ️ ダイレクトAPI応答異常 (HTTP {res.status_code})、フォールバックします")
    except Exception as dir_err:
        print(f"ℹ️ ダイレクトAPI待機/フォールバック: {dir_err}")

    # 【フォールバック】従来の Gradio Client 経由
    print("🔄 Gradio Client へフォールバックして動画生成中...")
    try:
        client = get_gradio_client()
        result_video_path = client.predict(
            face_img_path=handle_file(FACE_IMG_PATH),
            audio_path=handle_file(audio_path),
            api_name="/process_pipeline"
        )
    except HTTPException:
        raise
    except Exception:
        gradio_client = None
        raise
    shutil.copy(result_video_path, out_path)
    print(f"🎬 [3. Gradioフォールバック動画合成] ({time.time() - t0:.2f}秒)")

# ====================================================
# 対話＆動画生成エンドポイント
# ====================================================
@app.post("/chat")
def chat_and_generate_video(req: ChatRequest):
    total_start = time.time()
    print("\n" + "=" * 50)
    print(f"👤 ユーザー発言: {req.message}")
    print("=" * 50)

    # 1. Gemini で返答文生成（万が一のAPI制限時もフォールバックで落とさない）
    t0 = time.time()
    emotion = "nod"
    try:
        response = chat_session.send_message(req.message)
        raw_reply = response.text.strip()
        m = re.match(r"^\[(happy|nod|curious|normal|calm|sad)\]\s*(.*)", raw_reply)
        if m:
            emotion = m.group(1)
            reply_text = m.group(2).strip()
        else:
            reply_text = raw_reply
        print(f"🤖 [1. AI思考 ({emotion})] ({time.time() - t0:.2f}秒): {reply_text}")
    except Exception as e:
        print(f"⚠️ Gemini APIエラー検知: {e}")
        reply_text = "おはようございます！今日も一日楽しくお話ししましょうね。"
        emotion = "nod"
        print(f"🤖 [代替返答] ({time.time() - t0:.2f}秒): {reply_text}")

# 2. AivisSpeech で音声合成（話速1.15倍でテンポUP & フレーム数削減）
    t0 = time.time()
    try:
        wav_bytes = synthesize_speech(reply_text)
        with open(OUTPUT_AUDIO_PATH, "wb") as f:
            f.write(wav_bytes)
        print(f"🎉 [2. 音声合成] ({time.time() - t0:.2f}秒)")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"音声合成エラー: {e}")

    # 3. Colab GPU による動画生成
    try:
        print("🚀 Colab で口パク動画を合成中...")
        generate_video(OUTPUT_AUDIO_PATH, FINAL_VIDEO_PATH)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"動画生成エラー: {e}")

    print("-" * 50)
    print(f"🏁 【全体の合計待ち時間】: {time.time() - total_start:.2f}秒")
    print("=" * 50 + "\n")

    encoded_reply = urllib.parse.quote(reply_text)
    return FileResponse(
        FINAL_VIDEO_PATH,
        media_type="video/mp4",
        filename="final_output.mp4",
        headers={
            "X-Reply-Text": encoded_reply,
            "X-Emotion": emotion,
            "Access-Control-Expose-Headers": "X-Reply-Text, X-Emotion"
        }
    )


# ====================================================
# ストリーミング対話エンドポイント（流れ作業で話し始めを早める）
#   Gemini の返答をストリームで受け、区切り（最初のリアクションは「、」、以降は文末）ごとに
#   音声合成 → 動画生成をパイプライン実行し、できた区間から NDJSON で順に返す。
#   音声合成と動画生成は別スレッドなので、区間1の動画生成中に区間2の音声合成が進む。
# ====================================================
FIRST_BOUNDARY = re.compile(r"[、。！？!?]")
SENTENCE_BOUNDARY = re.compile(r"[。！？!?]")
EMOTION_TAG = re.compile(r"^\s*\[(happy|nod|curious|normal|calm|sad)\]\s*")
tts_executor = ThreadPoolExecutor(max_workers=1)
video_executor = ThreadPoolExecutor(max_workers=1)

def iter_reply_segments(message):
    """Gemini のストリームから (emotion, 区間テキスト) を区切りが確定した順に yield"""
    buf = ""
    emotion = None
    first = True
    try:
        for chunk in chat_session.send_message_stream(message):
            buf += chunk.text or ""
            if emotion is None:
                m = EMOTION_TAG.match(buf)
                if m:
                    emotion = normalize_emotion(m.group(1))
                    buf = buf[m.end():]
                elif len(buf) < 12 and buf.lstrip().startswith("["):
                    continue  # タグの途中まで届いた状態
                else:
                    emotion = "normal"
            while True:
                m = (FIRST_BOUNDARY if first else SENTENCE_BOUNDARY).search(buf)
                if not m:
                    break
                seg, buf = buf[:m.end()].strip(), buf[m.end():]
                if seg:
                    yield emotion, seg
                    first = False
    except Exception as e:
        print(f"⚠️ Gemini APIエラー検知: {e}")
        if first:
            yield "calm", "ごめんね、ちょっと考えがまとまらなかったよ。"
            return
    rest = EMOTION_TAG.sub("", buf).strip()
    if rest:
        yield emotion or "normal", rest

# ----------------------------------------------------
# 待機動画と口パク動画の「共通の時計」
#   ブラウザの待機動画は会話中も裏で再生し続けるので、その再生位置を時計とみなす。
#   各区間は「再生が始まる見込みの時刻」に待機動画が表示しているはずのコマから合成すれば、
#   待機⇄会話の切り替えや区間のつなぎ目で頭の位置が飛ばない。
# ----------------------------------------------------
IDLE_TOTAL_FRAMES = 750   # Colab のループ動画のコマ数（/api/sync_idle で更新）
IDLE_FPS = 25.0
render_sec_ema = 1.4      # 動画生成+転送の所要秒数の移動平均（再生開始時刻の見積もりに使う）
SEG_SILENCE = 0.05        # 区間の前後につける無音（秒）。標準は約0.1〜0.17秒
EMOTION_RAMP_FRAMES = 20  # 返事の最初の区間で、通常の顔から感情の顔へ移すコマ数（0.8秒）。急に変わると不気味に見える

def ramp_for(index, emotion):
    return EMOTION_RAMP_FRAMES if index == 0 and emotion != "normal" else 0

class ChatStreamRequest(BaseModel):
    message: str
    idle_time: float = 0.0  # 送信時点の待機動画の再生位置（秒）

def wav_duration(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()

def render_segment(index, wav_future, clock, emotion="normal"):
    """音声合成の完了を待ち、再生開始見込み時刻のコマから動画を生成して mp4 のバイト列を返す"""
    global render_sec_ema
    wav_bytes = wav_future.result()
    wav_path = os.path.join(BASE_DIR, f"seg_{index}.wav")
    mp4_path = os.path.join(BASE_DIR, f"seg_{index}.mp4")
    with open(wav_path, "wb") as f:
        f.write(wav_bytes)

    # 再生開始見込み = max(動画が届く見込み, 前の区間が終わる時刻)
    now = time.time()
    play_at = max(now + render_sec_ema, clock["prev_end"])
    clock["prev_end"] = play_at + wav_duration(wav_bytes)
    start_frame = round((clock["idle_time"] + (play_at - clock["t_send"])) * IDLE_FPS) % IDLE_TOTAL_FRAMES

    generate_video(wav_path, mp4_path, start_frame, emotion, ramp_for(index, emotion))
    render_sec_ema = 0.7 * render_sec_ema + 0.3 * (time.time() - now)
    with open(mp4_path, "rb") as f:
        return f.read()

# ----------------------------------------------------
# Colab GPU 音声合成（使えるときはテキストだけ送り、音声合成＋口パクを Colab で一括実行）
# ----------------------------------------------------
colab_tts_ready = False
colab_turn_support = False  # Colab 側が区間の並行依頼（turn_id）に対応しているか
remote_sec_ema = 1.0      # テキスト→動画（Colab 一括）の所要秒数の移動平均

def poll_colab_tts():
    """Colab 側の GPU 音声合成の準備状況を定期確認（未準備・失敗時はローカルの CPU 合成を使う）"""
    global colab_tts_ready, colab_turn_support
    last_detail = None
    while True:
        try:
            res = requests.get(f"{fetch_latest_colab_url()}/api/tts_status", headers=TUNNEL_HEADERS, timeout=5)
            data = res.json() if res.ok else {}
            ready = bool(data.get("ready"))
            colab_turn_support = bool(data.get("turn_support"))
            detail = data.get("detail", f"HTTP {res.status_code}")
        except Exception as e:
            ready, detail = False, f"接続不可: {type(e).__name__}"
        if ready != colab_tts_ready or detail != last_detail:
            print(f"🗣️ Colab GPU 音声合成: {'使用中' if ready else '未使用'} ({detail})")
        colab_tts_ready, last_detail = ready, detail
        time.sleep(15)

threading.Thread(target=poll_colab_tts, daemon=True).start()

remote_executor = ThreadPoolExecutor(max_workers=4)  # 区間を並行して Colab に依頼する

def render_segment_remote(index, text, clock, emotion="normal"):
    """Colab でテキスト→音声→口パク動画を一括生成（区間は並行依頼、コマ位置の連結は Colab 側が区間順に行う）。
    失敗時はローカル CPU 合成にフォールバック"""
    global remote_sec_ema
    now = time.time()
    # 開始コマは区間0のみ使われる（以降は Colab 側で前区間の終了コマから連結）
    play_at = now + remote_sec_ema
    start_frame = round((clock["idle_time"] + (play_at - clock["t_send"])) * IDLE_FPS) % IDLE_TOTAL_FRAMES
    try:
        res = requests.post(
            f"{fetch_latest_colab_url()}/api/generate_from_text",
            data={"text": text, "speaker": str(EMOTION_STYLE_IDS[emotion]), "emotion": emotion, "speed": "1.22",
                  "intonation": str(current_style_strength),
                  "pre_silence": str(SEG_SILENCE), "post_silence": str(SEG_SILENCE),
                  "start_frame": str(start_frame), "turn_id": clock["turn_id"], "seg_index": str(index),
                  "ramp_in": str(ramp_for(index, emotion))},
            headers=TUNNEL_HEADERS,
            timeout=20
        )
        res.raise_for_status()
        elapsed = time.time() - now
        if index == 0:
            remote_sec_ema = 0.7 * remote_sec_ema + 0.3 * elapsed
        print(f"🎬 [区間{index + 1} Colab一括生成] ({elapsed:.2f}秒, うち合成待ち含む {res.headers.get('X-TTS-Sec', '?')}秒)")
        return res.content
    except Exception as e:
        print(f"ℹ️ Colab GPU 合成に失敗、ローカル合成へ切り替え: {e}")
        wav_future = tts_executor.submit(synthesize_speech, text, 30, SEG_SILENCE, SEG_SILENCE, EMOTION_STYLE_IDS[emotion],
                                         current_style_strength)
        return render_segment(index, wav_future, clock, emotion)

@app.post("/chat_stream")
def chat_stream(req: ChatStreamRequest):
    t_send = time.time()

    def event_stream():
        total_start = time.time()
        print("\n" + "=" * 50)
        print(f"👤 ユーザー発言: {req.message}")
        print("=" * 50)

        clock = {"t_send": t_send, "idle_time": req.idle_time, "prev_end": 0.0,
                 "turn_id": secrets.token_hex(6)}
        jobs = []
        reply_emotion = None  # 1回の返事は1つの感情で統一（声と顔が途中で変わると分かりにくい）
        try:
            for i, (emotion, seg) in enumerate(iter_reply_segments(req.message)):
                reply_emotion = reply_emotion or normalize_emotion(emotion)
                emotion = reply_emotion
                print(f"🤖 [区間{i + 1} ({emotion})] ({time.time() - total_start:.2f}秒): {seg}")
                if colab_tts_ready:
                    # 並行依頼は Colab 側が対応している場合のみ（古い Colab では出力ファイルが衝突するため直列）
                    executor = remote_executor if colab_turn_support else video_executor
                    video_future = executor.submit(render_segment_remote, i, seg, clock, emotion)
                else:
                    wav_future = tts_executor.submit(synthesize_speech, seg, 30, SEG_SILENCE, SEG_SILENCE, EMOTION_STYLE_IDS[emotion],
                                                     current_style_strength)
                    video_future = video_executor.submit(render_segment, i, wav_future, clock, emotion)
                jobs.append((emotion, seg, video_future))

            for i, (emotion, seg, video_future) in enumerate(jobs):
                mp4 = video_future.result()
                print(f"📤 [区間{i + 1} 送信] ({time.time() - total_start:.2f}秒)")
                yield json.dumps({
                    "type": "segment",
                    "index": i,
                    "emotion": emotion,
                    "text": seg,
                    "video": base64.b64encode(mp4).decode("ascii"),
                }) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error", "detail": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n"
        print(f"🏁 【全体の合計時間】: {time.time() - total_start:.2f}秒\n")

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ====================================================
# 待機動画を Colab の口パク合成元ループと同期（顔の見た目・コマ位置を一致させる）
# ====================================================
def sync_idle_video():
    global IDLE_TOTAL_FRAMES, IDLE_FPS
    res = requests.get(f"{fetch_latest_colab_url()}/api/idle_video", headers=TUNNEL_HEADERS, timeout=30)
    res.raise_for_status()
    if len(res.content) < 1000:
        raise RuntimeError("待機動画が空です")
    dest = os.path.join(STATIC_DIR, "avatar_idle.mp4")
    backup = os.path.join(STATIC_DIR, "avatar_idle_backup.mp4")
    if os.path.exists(dest) and not os.path.exists(backup):
        shutil.copy(dest, backup)  # 初回のみ、元の待機動画を退避
    with open(dest, "wb") as f:
        f.write(res.content)
    IDLE_TOTAL_FRAMES = int(res.headers.get("X-Total-Frames", IDLE_TOTAL_FRAMES))
    IDLE_FPS = float(res.headers.get("X-FPS", IDLE_FPS))

    # 感情ごとの待機動画（返事の後、感情の顔から通常の顔へゆっくり戻す演出用）
    emotions = []
    for emo in ("happy", "calm", "sad"):
        try:
            r = requests.get(f"{fetch_latest_colab_url()}/api/idle_video", params={"emotion": emo},
                             headers=TUNNEL_HEADERS, timeout=60)
            if r.ok and len(r.content) > 1000:
                with open(os.path.join(STATIC_DIR, f"avatar_idle_{emo}.mp4"), "wb") as f:
                    f.write(r.content)
                emotions.append(emo)
        except Exception as e:
            print(f"ℹ️ 感情待機動画 {emo} の同期スキップ: {e}")
    print(f"🔄 待機動画を Colab と同期しました ({IDLE_TOTAL_FRAMES} コマ, {IDLE_FPS} fps, 感情: {emotions or 'なし'})")
    return emotions

@app.post("/api/sync_idle")
def api_sync_idle():
    try:
        emotions = sync_idle_video()
        t = int(time.time())
        return {"status": "success", "idle_video_url": f"/static/avatar_idle.mp4?t={t}",
                "emotion_video_urls": {e: f"/static/avatar_idle_{e}.mp4?t={t}" for e in emotions}}
    except Exception as e:
        print(f"⚠️ 待機動画の同期に失敗（既存の待機動画を使います）: {e}")
        return {"status": "skipped", "detail": str(e)}


# ====================================================
# アバター自動生成・切り替えエンドポイント
# ====================================================
def run_setup_job_on_colab(colab_url, face_path, t0, endpoint="/api/setup_avatar_start", max_sec=2700):
    """Colab に生成を開始させ、完了まで進み具合を確認する。
    戻り値: True=完了 / None=Colab が開始方式に未対応（古い Colab）。失敗時は例外"""
    with open(face_path, "rb") as f:
        res = requests.post(f"{colab_url}{endpoint}",
                            files={"image": ("face.jpg", f, "image/jpeg")},
                            headers=TUNNEL_HEADERS, timeout=30)
    if res.status_code == 404:
        return None
    if res.status_code == 409:
        raise HTTPException(status_code=409, detail="Colab でアバター生成がすでに実行中です。終わるまでお待ちください。")
    res.raise_for_status()
    last_detail = None
    while time.time() - t0 < max_sec:  # 通常ループ＋感情ループ3種で T4 だと約25分
        time.sleep(5)
        try:
            st = requests.get(f"{colab_url}/api/setup_avatar_status", headers=TUNNEL_HEADERS, timeout=15).json()
        except Exception as e:
            print(f"ℹ️ 進捗確認に失敗（再試行します）: {e}")
            continue
        if st.get("detail") != last_detail:
            last_detail = st.get("detail")
            print(f"⏳ Colab: {last_detail} ({st.get('elapsed')}秒経過)")
        if st.get("state") == "done":
            print(f"🎉 [アバター更新完了] ({st.get('elapsed')}秒)")
            return True
        if st.get("state") == "error":
            raise RuntimeError(st.get("detail"))
    raise HTTPException(status_code=504, detail=f"アバター生成が{max_sec // 60}分以内に終わりませんでした。")

# 長時間かかるので通常の def（async にすると完了までサーバー全体が他のリクエストに応答しなくなる）
@app.post("/upload_avatar")
def upload_avatar(file: UploadFile = File(...)):
    print("\n" + "=" * 50)
    print(f"📷 新しいアバター画像を受信: {file.filename}")
    print("=" * 50)

    # 1. ローカルの face.jpg を上書き保存
    temp_face_path = os.path.join(BASE_DIR, "face.jpg")
    content = file.file.read()
    with open(temp_face_path, "wb") as f:
        f.write(content)
    print(f"✅ ローカルに保存しました: {temp_face_path}")

    # 2. Colab GPU で setup_avatar を呼び出し
    t0 = time.time()
    try:
        print("🚀 Colab で新しいアバターのループ動画＆キャッシュを生成中...")
        colab_url = fetch_latest_colab_url()
        setup_success = False
        dest_idle_path = os.path.join(STATIC_DIR, "avatar_idle.mp4")

        # 【推奨ルート】開始だけ依頼して進み具合を確認（約10分の処理でも接続切れの影響を受けない）
        try:
            setup_success = bool(run_setup_job_on_colab(colab_url, temp_face_path, t0))
        except requests.exceptions.ConnectionError as e:
            print(f"ℹ️ 開始方式の依頼に失敗: {e}")

        # 【旧ルート】ダイレクトAPI (/api/setup_avatar) へ画像を一撃POST（古い Colab 用）
        if not setup_success:
            try:
                with open(temp_face_path, "rb") as f:
                    res = requests.post(
                        f"{colab_url}/api/setup_avatar",
                        files={"image": ("face.jpg", f, "image/jpeg")},
                        headers=TUNNEL_HEADERS,
                        timeout=900  # 32秒ループの LivePortrait 生成＋顔検出で T4 だと約10分かかる
                    )
                if res.ok and len(res.content) > 1000:
                    with open(dest_idle_path, "wb") as out_f:
                        out_f.write(res.content)
                    setup_success = True
                    print(f"🎉 [アバター更新完了 (ダイレクト)] ({time.time() - t0:.2f}秒): {dest_idle_path}")
            except requests.exceptions.ReadTimeout:
                # Colab 側では生成が続いている。フォールバックすると同じ生成を二重に始めてしまうため中断
                raise HTTPException(status_code=504, detail="アバター生成が時間内に終わりませんでした。Colab 側で処理が続いている可能性があるので、数分後にページを再読み込みしてください。")
            except Exception as dir_err:
                print(f"ℹ️ ダイレクトアバター更新待機/フォールバック: {dir_err}")

        # 【フォールバック】従来の Gradio Client 経由
        if not setup_success:
            print("🔄 Gradio Client へフォールバックしてアバター更新中...")
            client = get_gradio_client()
            result_idle_path = client.predict(
                handle_file(temp_face_path),
                api_name="/setup_avatar"
            )
            shutil.copy(result_idle_path, dest_idle_path)
            print(f"🎉 [アバター更新完了 (Gradio)] ({time.time() - t0:.2f}秒): {dest_idle_path}")

        # 口パク合成元と同じ画質・コマ数の待機動画に差し替え（失敗時は上で保存したものを使う）
        try:
            sync_idle_video()
        except Exception as e:
            print(f"ℹ️ 待機動画の同期スキップ: {e}")

        return {
            "status": "success",
            "message": "アバターを更新しました！",
            "idle_video_url": f"/static/avatar_idle.mp4?t={int(time.time())}"
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        global gradio_client
        gradio_client = None
        raise HTTPException(status_code=500, detail=f"アバター生成エラー: {e}")


@app.post("/api/setup_emotions")
def setup_emotions():
    """現在のアバター写真（face.jpg）から感情ループ（うれしい・穏やか・悲しい）だけを追加生成する"""
    t0 = time.time()
    face_path = os.path.join(BASE_DIR, "face.jpg")
    print("\n🎭 感情ループの生成を Colab に依頼します（約15分）...")
    try:
        done = run_setup_job_on_colab(fetch_latest_colab_url(), face_path, t0, "/api/setup_emotions_start", 1800)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"感情ループ生成エラー: {e}")
    if done is None:
        raise HTTPException(status_code=501, detail="Colab のコードが古いため感情ループを生成できません。Colab を再起動してください。")
    return {"status": "success", "elapsed": round(time.time() - t0)}


# ====================================================
# 声の切り替え（まお ⇔ 学習した声）
#   学習した声は Colab の /api/voice/models にスタイル名 → スタイルIDが登録される。
#   感情（normal/happy/calm/sad）と同じ名前のスタイルがあればそれを使い、なければ Neutral を使う
# ====================================================
MAO_STYLE_IDS = dict(EMOTION_STYLE_IDS)
VOICE_SETTING_FILE = os.path.join(BASE_DIR, "voice_setting.json")
current_voice = "mao"
# 感情表現の強さ（intonationScale）。まおは標準で十分差が出るが、少量の録音で学習した声は
# 標準（重み1）だと感情の差が合成の揺れに埋もれるため最大（2.0 = 重み10）にする（2026-10-03 聞き比べで確認）
MAO_STYLE_STRENGTH = 1.0
CUSTOM_STYLE_STRENGTH = 2.0
current_style_strength = MAO_STYLE_STRENGTH

def colab_voice_models():
    try:
        r = requests.get(f"{fetch_latest_colab_url()}/api/voice/models", headers=TUNNEL_HEADERS, timeout=10)
        return r.json() if r.ok else {}
    except Exception:
        return {}

def install_voice_locally(model_name):
    """ローカルの AivisSpeech にも学習した声を入れる（Colab の GPU 合成が使えないときの予備用）"""
    try:
        r = requests.get(f"{fetch_latest_colab_url()}/api/voice/aivmx", params={"model_name": model_name},
                         headers=TUNNEL_HEADERS, timeout=120)
        r.raise_for_status()
        res = requests.post(f"{AIVIS_URL}/aivm_models/install",
                            files={"file": (f"{model_name}.aivmx", r.content)}, timeout=300)
        print(f"🎙️ ローカル音声エンジンへの登録: HTTP {res.status_code}")
    except Exception as e:
        print(f"ℹ️ ローカル音声エンジンへの登録をスキップ（予備用のため会話には影響なし）: {e}")

def select_voice(voice):
    global current_voice, current_style_strength
    if voice == "mao":
        EMOTION_STYLE_IDS.update(MAO_STYLE_IDS)
    else:
        info = colab_voice_models().get(voice)
        if not info or not info.get("styles"):
            raise HTTPException(status_code=404, detail=f"学習済みの声 {voice} が Colab にありません")
        styles = info["styles"]
        fallback = styles.get("Neutral", next(iter(styles.values())))
        EMOTION_STYLE_IDS.update({emo: styles.get(emo, fallback) for emo in MAO_STYLE_IDS})
        threading.Thread(target=install_voice_locally, args=(voice,), daemon=True).start()
    current_voice = voice
    current_style_strength = MAO_STYLE_STRENGTH if voice == "mao" else CUSTOM_STYLE_STRENGTH
    try:
        json.dump({"voice": voice}, open(VOICE_SETTING_FILE, "w", encoding="utf-8"))
    except Exception:
        pass
    print(f"🎙️ 声を切り替えました: {voice} {EMOTION_STYLE_IDS}")

def restore_voice_setting():
    """前回選んだ声を復元（Colab 側の登録を待って再試行）"""
    try:
        voice = json.load(open(VOICE_SETTING_FILE, encoding="utf-8")).get("voice", "mao")
    except Exception:
        return
    if voice == "mao":
        return
    for _ in range(60):
        try:
            select_voice(voice)
            return
        except Exception:
            time.sleep(10)

threading.Thread(target=restore_voice_setting, daemon=True).start()

class VoiceSelectRequest(BaseModel):
    voice: str

@app.get("/api/voices")
def api_voices():
    return {"current": current_voice, "voices": ["mao"] + sorted(colab_voice_models().keys())}

@app.post("/api/voices/select")
def api_voices_select(req: VoiceSelectRequest):
    select_voice(req.voice)
    return {"current": current_voice, "styles": EMOTION_STYLE_IDS}


# ====================================================
# Colab URL 取得・手動設定エンドポイント
# ====================================================
class SetUrlRequest(BaseModel):
    url: str

@app.get("/api/colab_url")
def get_colab_url():
    current_url = fetch_latest_colab_url()
    return {"url": current_url}

@app.post("/api/colab_url")
def set_colab_url(req: SetUrlRequest):
    global COLAB_GRADIO_URL, gradio_client
    new_url = req.url.strip()
    COLAB_GRADIO_URL = new_url
    gradio_client = None
    # 次回の fetch_latest_colab_url() でクラウド側の古いURLに戻されないよう、全同期先とローカルキャッシュを更新
    for ep in SYNC_ENDPOINTS:
        try:
            requests.post(ep, data=new_url, timeout=3)
        except Exception as e:
            print(f"⚠️ URL同期失敗 ({ep.split('/')[2]}): {e}")
    try:
        with open(CACHE_URL_FILE, "w", encoding="utf-8") as f:
            f.write(new_url)
    except Exception:
        pass
    return {"status": "success", "url": COLAB_GRADIO_URL}

