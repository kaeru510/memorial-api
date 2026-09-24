"""
procedural_motion.py - 完全動画レス自律モーション生成モジュール
外部の参照動画（mp4）を一切使わず、数学的アルゴリズムによって
自然な呼吸、頭部の微小ゆらぎ、自律まばたき、および感情別リアクション（うなずき、首傾げ、笑顔）
を含む LivePortrait 用のモーションテンプレート (.pkl) を生成します。
"""

import os
import pickle
import numpy as np


def get_rotation_matrix_np(pitch_deg, yaw_deg, roll_deg):
    """
    LivePortrait (camera.py) と完全互換の 3D Euler 回転行列 (1, 3, 3) を生成
    LivePortrait の定義:
        rot = rot_z @ rot_y @ rot_x
        return rot.T
    """
    x = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    z = np.radians(roll_deg)

    rot_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(x), -np.sin(x)],
        [0.0, np.sin(x), np.cos(x)]
    ], dtype=np.float32)

    rot_y = np.array([
        [np.cos(y), 0.0, np.sin(y)],
        [0.0, 1.0, 0.0],
        [-np.sin(y), 0.0, np.cos(y)]
    ], dtype=np.float32)

    rot_z = np.array([
        [np.cos(z), -np.sin(z), 0.0],
        [np.sin(z), np.cos(z), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    rot = rot_z @ rot_y @ rot_x
    return np.ascontiguousarray(rot.T[np.newaxis, ...], dtype=np.float32)


def generate_procedural_motion(
    motion_type="idle",
    duration_sec=8.0,
    fps=25,
    emotion=None
):
    """
    数式によって自律的なモーションテンプレート辞書を生成

    Parameters:
        motion_type: "idle", "nod", "happy", "curious"
        duration_sec: アニメーションの秒数（idle の場合は 8.0s 推奨: 整数周期で完全シームレスループ）
        fps: フレームレート (25)
        emotion: オプションの感情名 ("happy", "nod", "curious", "normal")
    Returns:
        driving_template_dct (dict): LivePortrait が直接読み込めるテンプレート形式
    """
    n_frames = int(duration_sec * fps)
    t = np.linspace(0, duration_sec, n_frames, endpoint=False)

    # 1. 基本の生体微動（呼吸 & ポスチャースウェイ）
    # 8.0秒周期で始点と終点が完全に一致する周波数（0.25Hz = 4秒周期, 0.125Hz = 8秒周期）
    f_breath = 0.25  # 呼吸: 4秒に1回
    f_sway = 0.125   # ゆらぎ: 8秒に1回

    # 呼吸による上下動 (translation Y)
    dy = 0.0035 * np.sin(2.0 * np.pi * f_breath * t)
    dx = 0.0010 * np.sin(2.0 * np.pi * f_sway * t)
    dz = np.zeros(n_frames, dtype=np.float32)

    # 頭部角度 (Pitch: 上下うなずき, Yaw: 左右振り, Roll: 首傾げ)
    pitch = 0.6 * np.cos(2.0 * np.pi * f_breath * t)
    yaw = 0.9 * np.sin(2.0 * np.pi * f_sway * t)
    roll = 0.5 * np.sin(2.0 * np.pi * f_breath * t + 0.5)

    # 2. 感情・アクションのオーバーレイ
    effective_motion = emotion if emotion else motion_type

    if effective_motion == "nod":
        # 0.2秒〜1.0秒にかけて丁寧な相槌（コクッとうなずく）
        nod_window = (t >= 0.2) & (t <= 1.0)
        nod_progress = (t[nod_window] - 0.2) / 0.8  # 0 to 1
        # ガウス風ベルカーブでピッチを約 -4.5度下げる
        nod_curve = -4.5 * np.sin(np.pi * nod_progress) ** 1.5
        pitch[nod_window] += nod_curve
        dy[nod_window] -= 0.005 * np.sin(np.pi * nod_progress)

    elif effective_motion == "happy":
        # 喜び: わずかに顔を上げて首を優しく傾げる
        happy_curve = np.sin(np.pi * np.linspace(0, 1, n_frames))
        roll += 2.5 * happy_curve
        pitch += 1.2 * happy_curve

    elif effective_motion == "curious":
        # 疑問: 語尾（または全体）で興味深そうに首を傾げる
        curious_curve = np.sin(np.pi * np.linspace(0, 1, n_frames))
        roll -= 3.2 * curious_curve
        pitch += 0.8 * curious_curve

    # 3. 自律まばたきカーブ (Eye close ratio)
    # 開眼時: 0.38, 閉眼時: 0.03
    eye_ratio = np.full(n_frames, 0.38, dtype=np.float32)

    # 8秒間に2回の自然な瞬きを配置 (t=2.4s, t=5.8s)
    blink_times = [2.4, 5.8]
    for b_time in blink_times:
        b_idx = int(b_time * fps)
        # まばたき持続フレーム: 4フレーム（約160ms）
        if b_idx + 4 < n_frames:
            eye_ratio[b_idx] = 0.22
            eye_ratio[b_idx + 1] = 0.04  # 完全閉眼
            eye_ratio[b_idx + 2] = 0.15
            eye_ratio[b_idx + 3] = 0.32

    # 4. モーションフレーム辞書リストの構築
    motion_list = []
    c_eyes_list = []
    c_lip_list = []

    for i in range(n_frames):
        R_mat = get_rotation_matrix_np(pitch[i], yaw[i], roll[i])
        t_vec = np.array([[dx[i], dy[i], dz[i]]], dtype=np.float32)
        scale_vec = np.array([[1.0]], dtype=np.float32)
        exp_vec = np.zeros((1, 21, 3), dtype=np.float32)

        # happy感情の場合、口角の表情コードをわずかに上向きに
        if effective_motion == "happy":
            # LivePortrait expression dim 19/20 are lip corners
            exp_vec[0, 19, 1] -= 0.015
            exp_vec[0, 20, 1] -= 0.015

        item_dct = {
            'scale': scale_vec,
            'R': R_mat,
            'exp': exp_vec,
            't': t_vec,
            'kp': np.zeros((1, 21, 3), dtype=np.float32),
            'x_s': np.zeros((1, 21, 3), dtype=np.float32)
        }
        motion_list.append(item_dct)

        e_ratio = float(eye_ratio[i])
        c_eyes_list.append(np.array([[e_ratio, e_ratio]], dtype=np.float32))
        c_lip_list.append(np.array([[0.0]], dtype=np.float32))

    template_dct = {
        'n_frames': n_frames,
        'output_fps': fps,
        'motion': motion_list,
        'c_eyes_lst': c_eyes_list,
        'c_lip_lst': c_lip_list
    }

    return template_dct


def save_procedural_motion_template(
    output_pkl_path,
    motion_type="idle",
    duration_sec=8.0,
    fps=25,
    emotion=None
):
    """
    指定パスに LivePortrait 互換 .pkl モーションテンプレートを出力保存
    """
    template_dct = generate_procedural_motion(
        motion_type=motion_type,
        duration_sec=duration_sec,
        fps=fps,
        emotion=emotion
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl_path)), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(template_dct, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ 自律モーションテンプレート生成完了: {output_pkl_path} ({template_dct['n_frames']} フレーム, {duration_sec}秒, タイプ: {motion_type})")
    return output_pkl_path


if __name__ == "__main__":
    test_pkl = "test_procedural_idle.pkl"
    save_procedural_motion_template(test_pkl, "idle", 8.0, 25)
    print(f"ファイルサイズ: {os.path.getsize(test_pkl)} bytes")
