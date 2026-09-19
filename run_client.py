import os
import shutil
from gradio_client import Client, handle_file

# --- 基本設定 ---
SERVER_URL = "https://16c86684b24f91b54b.gradio.live"

# テスト用の入力ファイルパス（固定）
SOURCE_IMAGE = r"C:\Users\yamada\OneDrive - Shizuoka University\myfolder\memorial-api\face.jpg"
DRIVING_VIDEO = r"C:\Users\yamada\OneDrive - Shizuoka University\myfolder\memorial-api\motion.mp4"

print("🔗 Colabの推論サーバーに接続中...")
client = Client(SERVER_URL)

print("🚀 推論リクエストを送信中（Colab GPUで処理）...")

# 位置引数として順番に渡す（fn_index=18）
result = client.predict(
    handle_file(SOURCE_IMAGE),   # 1. source_image
    None,                        # 2. source_video
    handle_file(DRIVING_VIDEO),  # 3. driving_video
    None,                        # 4. driving_image
    None,                        # 5. driving_pickle
    False,                       # 6. flag_normalize_lip
    True,                        # 7. flag_relative_input
    True,                        # 8. flag_remap_input
    True,                        # 9. flag_stitching_input
    False,                       # 10. flag_crop_driving_video
    "all",                       # 11. animation_region
    "expression-friendly",       # 12. driving_option
    1.0,                         # 13. driving_multiplier
    True,                        # 14. flag_do_crop
    2.3,                         # 15. scale
    0.0,                         # 16. vx_ratio
    -0.125,                      # 17. vy_ratio
    2.2,                         # 18. scale_crop_driving_video
    0.0,                         # 19. vx_ratio_crop_driving_video
    -0.1,                        # 20. vy_ratio_crop_driving_video
    3e-7,                        # 21. driving_smooth_observation_variance
    "Image",                     # 22. tab_selection
    "Video",                     # 23. v_tab_selection
    fn_index=18
)

print(f"🎉 処理完了！生成結果: {result}")

# 生成された動画をカレントディレクトリに保存
output_video_path = result[0]
dest_path = r"C:\Users\yamada\OneDrive - Shizuoka University\myfolder\memorial-api\output_liveportrait.mp4"
shutil.copy(output_video_path, dest_path)
print(f"📁 動画を保存しました: {dest_path}")