# プロジェクト共通ルール & コンテキスト

## プロジェクト概要
このプロジェクトは **対話型AIアバターシステム（LipSync AI Chat）** です。
詳細は `README.md` を参照してください。

- **バックエンド**: FastAPI (`main.py`)
- **フロントエンド**: `static/index.html` (Web Speech API + 動画シームレス切替)
- **AI対話**: Gemini API (`gemini-3.5-flash-lite`)
- **音声合成 (TTS)**: AivisSpeech (`http://127.0.0.1:10101`, Speaker: `888753760`)
- **動画生成**: Google Colab GPU (Wav2Lip + LivePortrait 高速パイプライン、Gradio API経由)
- **Python仮想環境**: Anaconda `grave` (`C:\Users\yamada\anaconda3\envs\grave\python.exe`)
- **GitHubリポジトリ**: `https://github.com/kaeru510/memorial-api.git`

---

## ターミナルコマンド実行に関する重要ルール
1. **自動検証コマンドの禁止**:
   - Pythonファイルなどのコードを編集・作成した後に、自動で `py_compile`、テスト、リンター等の構文チェック・検証コマンドをターミナルで実行しないでください。
   - バックグラウンドプロセスのタイムアウトやエディタのクラッシュを防止するためです。

2. **コマンドの実行方針**:
   - コマンドの実行は、ユーザーから明示的に依頼された場合（「〜を実行して」「インストールして」など）のみ行ってください。
   - ファイルの修正が完了したら、追加のコマンドは実行せずそのまま変更内容を報告してください。
   - 動作確認が必要な場合は、実行コマンドをテキストでユーザーに提示し、ユーザーに手動実行を促してください。
