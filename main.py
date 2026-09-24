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

# 3. Colab 実行時に発行された gradio.live / trycloudflare の URL
COLAB_GRADIO_URL = "https://7e42be673fb0f0c312.gradio.live"
gradio_client = None

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
2. 会話のテンポを最優先するため、必ず1文のみ（15文字〜35文字程度）で短く簡潔に答えてください。
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

SYNC_ENDPOINTS = [
    "https://kvdb.io/A5pQ8K3wKqR2fP4s9Yx7Nz/colab_url",
    "https://api.cl1p.net/kaeru510-memorial"
]
CACHE_URL_FILE = os.path.join(BASE_DIR, "colab_url.txt")

def fetch_latest_colab_url():
    """クラウド同期エンドポイントまたはローカルキャッシュから最新のColab URLを取得"""
    global COLAB_GRADIO_URL, gradio_client
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
        
        # ★ 話速を 1.22倍 に設定（自然な早口で動画総フレーム数を大幅削減）
        query_data["speedScale"] = 1.22

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
        colab_url = fetch_latest_colab_url()
        video_success = False

        # 【超高速優先ルート】ダイレクトAPI (/api/generate) へ音声のみ一撃POST
        direct_url = f"{colab_url}/api/generate"
        try:
            with open(OUTPUT_AUDIO_PATH, "rb") as f:
                res = requests.post(
                    direct_url,
                    files={"audio": ("output.wav", f, "audio/wav")},
                    timeout=20
                )
            if res.ok and len(res.content) > 1000:
                with open(FINAL_VIDEO_PATH, "wb") as out_f:
                    out_f.write(res.content)
                video_success = True
                print(f"🎬 [3. 高速ダイレクト動画合成+転送] ({time.time() - t0:.2f}秒)")
        except Exception as dir_err:
            print(f"ℹ️ ダイレクトAPI待機/フォールバック: {dir_err}")

        # 【フォールバック】従来の Gradio Client 経由
        if not video_success:
            print("🔄 Gradio Client へフォールバックして動画生成中...")
            client = get_gradio_client()
            result_video_path = client.predict(
                face_img_path=handle_file(FACE_IMG_PATH),
                audio_path=handle_file(OUTPUT_AUDIO_PATH),
                api_name="/process_pipeline"
            )
            shutil.copy(result_video_path, FINAL_VIDEO_PATH)
            print(f"🎬 [3. Gradioフォールバック動画合成] ({time.time() - t0:.2f}秒)")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
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
        colab_url = fetch_latest_colab_url()
        setup_success = False
        dest_idle_path = os.path.join(STATIC_DIR, "avatar_idle.mp4")

        # 【超高速優先ルート】ダイレクトAPI (/api/setup_avatar) へ画像を一撃POST
        direct_url = f"{colab_url}/api/setup_avatar"
        try:
            with open(temp_face_path, "rb") as f:
                res = requests.post(
                    direct_url,
                    files={"image": ("face.jpg", f, "image/jpeg")},
                    timeout=90
                )
            if res.ok and len(res.content) > 1000:
                with open(dest_idle_path, "wb") as out_f:
                    out_f.write(res.content)
                setup_success = True
                print(f"🎉 [アバター更新完了 (ダイレクト)] ({time.time() - t0:.2f}秒): {dest_idle_path}")
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
