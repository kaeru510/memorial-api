import os
import time
import requests
import traceback
import shutil
import secrets
import urllib.parse
from fastapi import FastAPI, HTTPException, Depends, status
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
COLAB_GRADIO_URL = "https://509c9cab460d813a20.gradio.live"

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

# ====================================================
# 2. Colab Gradio クライアント（安全な遅延接続方式）
# ====================================================
gradio_client = None

def get_gradio_client():
    """リクエスト時に接続を試みる（Colabが未起動でもサーバー起動が落ちないようにする）"""
    global gradio_client
    if gradio_client is None:
        print(f"🌐 Colab サーバーへ接続中: {COLAB_GRADIO_URL}")
        try:
            gradio_client = Client(COLAB_GRADIO_URL)
            print("✅ Colab 接続完了！")
        except Exception as e:
            print(f"❌ Colab 接続失敗: {e}")
            raise HTTPException(
                status_code=503,
                detail="Colabサーバーに接続できませんでした。Colabセルが実行中か、URLが最新か確認してください。"
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