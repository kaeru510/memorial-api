import os
import shutil
from gradio_client import Client, handle_file


class LivePortraitService:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = Client(server_url)

    def generate(
        self,
        source_image_path: str,
        driving_video_path: str,
        output_path: str,
        driving_option: str = "expression-friendly",
        driving_multiplier: float = 1.0,
        flag_do_crop: bool = True
    ) -> str:
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
        if not os.path.exists(driving_video_path):
            raise FileNotFoundError(f"Driving video not found: {driving_video_path}")

        # Gradio API 呼び出し (fn_index=18)
        result = self.client.predict(
            handle_file(source_image_path),   # 1. source_image
            None,                             # 2. source_video
            handle_file(driving_video_path),  # 3. driving_video
            None,                             # 4. driving_image
            None,                             # 5. driving_pickle
            False,                            # 6. flag_normalize_lip
            True,                             # 7. flag_relative_input
            True,                             # 8. flag_remap_input
            True,                             # 9. flag_stitching_input
            False,                            # 10. flag_crop_driving_video
            "all",                            # 11. animation_region
            driving_option,                   # 12. driving_option
            driving_multiplier,               # 13. driving_multiplier
            flag_do_crop,                     # 14. flag_do_crop
            2.3,                              # 15. scale
            0.0,                              # 16. vx_ratio
            -0.125,                           # 17. vy_ratio
            2.2,                              # 18. scale_crop_driving_video
            0.0,                              # 19. vx_ratio_crop_driving_video
            -0.1,                             # 20. vy_ratio_crop_driving_video
            3e-7,                             # 21. driving_smooth_observation_variance
            "Image",                          # 22. tab_selection
            "Video",                          # 23. v_tab_selection
            fn_index=18
        )

        output_video_path = result[0]
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copy(output_video_path, output_path)
        return os.path.abspath(output_path)