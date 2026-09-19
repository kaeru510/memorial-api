from gradio_client import Client, handle_file

# Colabで発行されたURLを後でここに入力します
COLAB_URL = "ここにColabのURLを貼り付け"

print("🚀 ColabのGPUへリクエスト送信中...")
client = Client(COLAB_URL)

# LivePortraitで推論を実行
result = client.predict(
    source_image=handle_file("LivePortrait/assets/examples/source/s9.jpg"),
    driving_video=handle_file("LivePortrait/assets/examples/driving/d0.mp4"),
    api_name="/gpu"
)

print(f"✨ 生成完了！出力ファイル: {result}")