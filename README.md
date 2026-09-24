# LipSync AI Chat (Memorial API)

対話型AIアバター（デジタルヒューマン）システム。  
ユーザーが音声またはテキストで話しかけると、**「Gemini（思考） ➔ AivisSpeech（音声合成） ➔ Google Colab GPU（Wav2Lip / LivePortrait 口パク動画合成）」** の超高速パイプラインが連動し、Webブラウザ上でアバターが滑らかに対話します。

さらに、**任意の顔写真1枚をアップロードするだけで、その人物の呼吸・まばたき待機動画と顔キャッシュを全自動生成**し、即座に対話相手を切り替えることができます。

---

## 1. システムアーキテクチャ

```mermaid
flowchart TD
    User([ユーザー]) -->|マイク音声 / テキスト入力| WebUI["フロントエンド (static/index.html)"]
    WebUI -->|POST /chat| FastAPI["バックエンド (main.py :8000)"]
    
    subgraph Pipeline [対話・動画生成パイプライン]
        FastAPI -->|1. テキスト生成| Gemini["Google Gemini API (gemini-3.5-flash-lite)"]
        Gemini -->|返答文 30〜60字| FastAPI
        FastAPI -->|2. 音声合成 (話速1.15倍)| Aivis["AivisSpeech (ローカル:10101)"]
        Aivis -->|output.wav| FastAPI
        FastAPI -->|3. 口パク動画生成| Colab["Google Colab GPU (Gradio API)"]
        Colab -->|final_output.mp4| FastAPI
    end
    
    FastAPI -->|MP4動画 + 返答字幕ヘッダー| WebUI
    WebUI -->|動画再生 & 字幕表示| User

    subgraph DynamicAvatar [アバター自動セットアップ]
        WebUI -->|📷 写真アップロード| FastAPI
        FastAPI -->|POST /upload_avatar| Colab
        Colab -->|LivePortrait + ピンポンループ + キャッシュ生成| Colab
        Colab -->|avatar_idle.mp4| WebUI
    end
```

---

## 2. フォルダ構成と主要ファイルの役割

```text
C:\dev\memorial-api\
├── main.py                     # FastAPI メインサーバー（ポート 8000）
├── colab_server.py             # Colab で動かす高速常駐サーバー（GitHub管理）
├── static/
│   ├── index.html              # Web UI（対話画面、音声認識、アバター変更UI）
│   └── avatar_idle.mp4         # 現在のアバター待機ループ動画
├── face.jpg                    # 現在のアバター顔写真（原画）
├── AivisSpeech-Engine-Windows-x64-1.2.0/ # ローカル音声合成エンジン（ポート 10101）
├── LivePortrait/               # LivePortrait 推論モジュール（Colab用）
├── liveportrait_service.py     # LivePortrait API ラッパー
├── run_client.py               # 単体疎通テスト用スクリプト
├── make_nod.py                 # 相槌音声事前生成スクリプト
└── README.md                   # このドキュメント
```

---

## 3. Python 実行環境

* **Anaconda 環境名**: `grave`
* **Python パス**: `C:\Users\yamada\anaconda3\envs\grave\python.exe`
* **主な導入パッケージ**: `fastapi`, `uvicorn`, `gradio_client`, `google-genai`, `opencv-python`, `torch`, `requests` 等

---

## 4. 起動手順（わずか2クリックで完了！）

### ステップ 1: Google Colab で「▶」を押す（1セル完結）
Google Colab ノートブックに以下のセルを作成し、**「▶（実行ボタン）」を1回押すだけ** です：

```python
# 🚀 1クリック全自動起動（Driveマウント & サーバー起動）
import os
from google.colab import drive

if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

!git clone https://github.com/kaeru510/memorial-api.git /content/memorial-api 2>/dev/null || (cd /content/memorial-api && git pull)
!python /content/memorial-api/colab_server.py
```
> [!IMPORTANT]
> **Colab のハードウェア設定は「T4 GPU」** を指定してください（メニューの「ランタイム」→「ランタイムのタイプを変更」→「T4 GPU」）。
> 起動すると、最新の接続URLがクラウド経由でローカルPCへ**全自動同期**されます（URLのコピペは一切不要です）。

---

### ステップ 2: デスクトップの起動アイコンをダブルクリック
デスクトップに作成された **`LipSync起動.bat`** をダブルクリックします。
- 音声合成エンジン（AivisSpeech）と FastAPI がバックグラウンド（最小化）で自動起動します。
- 準備が整うと、**自動でブラウザ（http://localhost:8000）が開いて即座に対話可能** になります！


---

## 5. 重要な便利機能と仕様

1. **URL完全自動同期システム**:
   * Colab 起動時に発行される `https://xxxx.gradio.live` のURLは、自動的にクラウド同期ストレージ（`cl1p.net`）へ送信されます。
   * ローカルの FastAPI（`main.py`）はリクエスト時にその最新URLを自動取得して接続するため、**URLのコピペ作業は一切不要** です。
   * Web画面（`http://localhost:8000`）の下部にも現在接続中のURLが表示され、鉛筆アイコンから手動変更も可能です。

2. **静止画1枚からのアバター自動セットアップ**:
   * Web画面右上の「📷 写真を変更」から任意の人物の顔写真をアップロードすると、Colab側で LivePortrait が動作し、まばたき・呼吸ループ動画と全フレーム顔座標キャッシュ（`base_avatar_cache.npz`）を自動生成します。
   * サーバー再起動なしで、即座に新しい人物のアバターとして対話できます。

3. **超高速 Wav2Lip パイプライン**:
   * 顔検出結果を事前にキャッシュ（`.npz`）化し、GPUテンソルに事前展開。
   * FFmpeg 標準入力（rawvideo パイプ）への直接流し込みによりディスクI/Oを極小化し、高速推論とアスペクト比維持の軽量エンコード（長辺480px）を実現しています。

