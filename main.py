import os
import sys
import time
import requests
import traceback
import shutil
import secrets
import urllib.parse

# Windows の cp932 環境での絵文字出力エラー対策
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import FileResponse
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

# 2. Gemini API キー
GEMINI_API_KEY = "AQ.Ab8RN6LbxmRb_omIfiTr1Np-m-8euT6-lxyqiWnmHQg4MnA12g"

# 3. Colab 実行時に発行された gradio.live の URL
COLAB_GRADIO_URL = "https://e8535d860d194b0ee0.gradio.live"

# AivisSpeech 設定
AIVIS_URL = "http://127.0.0.1:10101"
SPEAKER_ID = 888753760

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
2. 会話のテンポを保つため、1文または2文（30文字〜60文字程度）で簡潔に答えてください。
3. 文末は「〜だよ」「〜ですね」など、自然に会話を完結させてください。
4. 絵文字、記号、マークダウン記号（*や#）、括弧による注釈は一切含めないでください。
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

SYNC_ENDPOINT = "https://api.cl1p.net/kaeru510-memorial"

def fetch_latest_colab_url():
    """クラウド同期エンドポイントから最新のColab URLを取得"""
    global COLAB_GRADIO_URL, gradio_client
    try:
        res = requests.get(SYNC_ENDPOINT, timeout=3)
        if res.ok:
            remote_url = res.text.strip()
            if remote_url.startswith("http") and ("gradio.live" in remote_url or "ngrok" in remote_url):
                if remote_url != COLAB_GRADIO_URL:
                    print(f"🔄 最新のColab URLを自動同期しました: {remote_url}")
                    COLAB_GRADIO_URL = remote_url
                    gradio_client = None  # 再接続を促す
                return remote_url
    except Exception as e:
        print(f"⚠️ URL自動取得スキップ: {e}")
    return COLAB_GRADIO_URL

def get_gradio_client():
    """リクエスト時に接続を試みる（最新URLを自動取得し、失敗時は再取得）"""
    global gradio_client, COLAB_GRADIO_URL

    # まず最新URLを同期
    fetch_latest_colab_url()

    if gradio_client is None:
        print(f"🌐 Colab サーバーへ接続中: {COLAB_GRADIO_URL}")
        try:
            gradio_client = Client(COLAB_GRADIO_URL)
            print("✅ Colab 接続完了！")
        except Exception as e:
            # 接続失敗時、もう一度最新URLの同期を試みる
            print(f"⚠️ 接続失敗、最新URLを再チェック中...: {e}")
            latest_url = fetch_latest_colab_url()
            if latest_url and latest_url != COLAB_GRADIO_URL:
                try:
                    COLAB_GRADIO_URL = latest_url
                    gradio_client = Client(COLAB_GRADIO_URL)
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
    try:
        response = chat_session.send_message(req.message)
        reply_text = response.text.strip()
        print(f"🤖 [1. AI思考] ({time.time() - t0:.2f}秒): {reply_text}")
    except Exception as e:
        print(f"⚠️ Gemini APIエラー検知: {e}")
        reply_text = "おはようございます！今日も一日楽しくお話ししましょうね。"
        print(f"🤖 [代替返答] ({time.time() - t0:.2f}秒): {reply_text}")

# 2. AivisSpeech で音声合成（話速1.15倍でテンポUP & フレーム数削減）
    t0 = time.time()
    try:
        query_res = requests.post(
            f"{AIVIS_URL}/audio_query",
            params={"text": reply_text, "speaker": SPEAKER_ID},
            timeout=10
        )
        query_res.raise_for_status()
        query_data = query_res.json()
        
        # ★ 話速を 1.15倍 に設定（自然な早口になり、動画の総フレーム数を約15%削減）
        query_data["speedScale"] = 1.15

        synth_res = requests.post(
            f"{AIVIS_URL}/synthesis",
            params={"speaker": SPEAKER_ID},
            json=query_data,
            timeout=30
        )
        synth_res.raise_for_status()

        with open(OUTPUT_AUDIO_PATH, "wb") as f:
            f.write(synth_res.content)
        print(f"🎉 [2. 音声合成] ({time.time() - t0:.2f}秒)")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"音声合成エラー: {e}")

    # 3. Colab GPU による動画生成
    t0 = time.time()
    try:
        print("🚀 Colab で口パク動画を合成中...")
        client = get_gradio_client()  # 安全にクライアントを取得
        result_video_path = client.predict(
            face_img_path=handle_file(FACE_IMG_PATH),
            audio_path=handle_file(OUTPUT_AUDIO_PATH),
            api_name="/process_pipeline"
        )
        shutil.copy(result_video_path, FINAL_VIDEO_PATH)
        print(f"🎬 [3. 動画合成+転送] ({time.time() - t0:.2f}秒)")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        # 接続が切れていた場合に次回再接続できるようリセット
        global gradio_client
        gradio_client = None
        raise HTTPException(status_code=500, detail=f"動画生成エラー: {e}")

    print("-" * 50)
    print(f"🏁 【全体の合計待ち時間】: {time.time() - total_start:.2f}秒")
    print("=" * 50 + "\n")

    encoded_reply = urllib.parse.quote(reply_text)
    return FileResponse(
        FINAL_VIDEO_PATH,
        media_type="video/mp4",
        filename="final_output.mp4",
        headers={"X-Reply-Text": encoded_reply, "Access-Control-Expose-Headers": "X-Reply-Text"}
    )


# ====================================================
# アバター自動生成・切り替えエンドポイント
# ====================================================
@app.post("/upload_avatar")
async def upload_avatar(file: UploadFile = File(...)):
    print("\n" + "=" * 50)
    print(f"📷 新しいアバター画像を受信: {file.filename}")
    print("=" * 50)

    # 1. ローカルの face.jpg を上書き保存
    temp_face_path = os.path.join(BASE_DIR, "face.jpg")
    content = await file.read()
    with open(temp_face_path, "wb") as f:
        f.write(content)
    print(f"✅ ローカルに保存しました: {temp_face_path}")

    # 2. Colab GPU で setup_avatar を呼び出し
    t0 = time.time()
    try:
        print("🚀 Colab で新しいアバターのループ動画＆キャッシュを生成中...")
        client = get_gradio_client()
        result_idle_path = client.predict(
            face_img_path=handle_file(temp_face_path),
            api_name="/setup_avatar"
        )

        # 3. 生成された待機動画を static/avatar_idle.mp4 に上書き配置
        dest_idle_path = os.path.join(STATIC_DIR, "avatar_idle.mp4")
        shutil.copy(result_idle_path, dest_idle_path)
        print(f"🎉 [アバター更新完了] ({time.time() - t0:.2f}秒): {dest_idle_path}")

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
    try:
        requests.post(SYNC_ENDPOINT, data=new_url, timeout=3)
    except Exception:
        pass
    return {"status": "success", "url": COLAB_GRADIO_URL}
