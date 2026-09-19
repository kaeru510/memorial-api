import os
import sys
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

AIVIS_URL = "http://127.0.0.1:10101"
SPEAKER_ID = 888753760
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)


phrases = {
    "nod_1.wav": "うん！",
    "nod_2.wav": "えーっとね、",
    "nod_3.wav": "そうだね、"
}

for filename, text in phrases.items():
    print(f"生成中: {text} ...")
    q = requests.post(f"{AIVIS_URL}/audio_query", params={"text": text, "speaker": SPEAKER_ID}).json()
    s = requests.post(f"{AIVIS_URL}/synthesis", params={"speaker": SPEAKER_ID}, json=q)
    out_path = os.path.join(STATIC_DIR, filename)
    with open(out_path, "wb") as f:
        f.write(s.content)
    print(f"✅ 作成完了: {out_path} ({text})")

print("🎉 すべての相槌音声が完成しました！")