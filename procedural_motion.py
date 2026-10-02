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


def expression_offsets(smile=0.0, eyebrow=0.0, mouth=0.0):
    """表情パラメータ → LivePortrait の表情キーポイント差分 exp (1, 21, 3)
    係数は ComfyUI-AdvancedLivePortrait の表情エディタ（calc_fe）に準拠。
      smile:   -0.3〜1.3 程度（正で口角が上がる、負で下がる）
      eyebrow: -10〜15 程度（正で眉が上がる、負で眉を寄せる）
      mouth:   0〜 （口の開き）
    """
    e = np.zeros((1, 21, 3), dtype=np.float32)
    e[0, 20, 1] += smile * -0.01
    e[0, 14, 1] += smile * -0.02
    e[0, 17, 1] += smile * 0.0065
    e[0, 17, 2] += smile * 0.003
    e[0, 13, 1] += smile * -0.00275
    e[0, 16, 1] += smile * -0.00275
    e[0, 3, 1] += smile * -0.0035
    e[0, 7, 1] += smile * -0.0035

    e[0, 19, 1] += mouth * 0.001
    e[0, 19, 2] += mouth * 0.0001
    e[0, 17, 1] += mouth * -0.0001

    if eyebrow > 0:
        e[0, 1, 1] += eyebrow * 0.001
        e[0, 2, 1] += eyebrow * -0.001
    else:
        e[0, 1, 0] += eyebrow * -0.001
        e[0, 2, 0] += eyebrow * 0.001
        e[0, 1, 1] += eyebrow * 0.0003
        e[0, 2, 1] += eyebrow * -0.0003
    return e


def generate_procedural_motion(
    motion_type="idle",
    duration_sec=8.0,
    fps=25,
    emotion=None,
    expression=None,
    eye_open=1.0
):
    """
    数式によって自律的なモーションテンプレート辞書を生成

    Parameters:
        motion_type: "idle", "nod", "happy", "curious"
        duration_sec: アニメーションの秒数（idle は 8 の倍数を推奨: 呼吸4秒・ゆらぎ8秒の整数周期で完全シームレスループ）
        fps: フレームレート (25)
        emotion: オプションの感情名 ("happy", "nod", "curious", "normal")。頭の動きも変わる
        expression: 表情だけを変える dict（expression_offsets の引数）。頭の動きは idle と同一のまま
        eye_open: 目の開き具合の倍率（1.0 で通常。笑顔で細める等）
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

    # ループ全体で1周・2周する遅いゆらぎを重ね、8秒ごとの同じ動きの繰り返し感を減らす
    # （周期がループ長の整数分の1なので、始点と終点は一致したまま）
    f_slow = 1.0 / duration_sec
    yaw += 0.6 * np.sin(2.0 * np.pi * f_slow * t)
    pitch += 0.3 * np.sin(2.0 * np.pi * 2 * f_slow * t + 1.0)

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
    eye_ratio = np.full(n_frames, 0.38 * eye_open, dtype=np.float32)

    # まばたきは 2.5〜5.5秒の不規則な間隔で配置（人は毎分15〜20回程度、等間隔だと機械的に見える）
    # 形: 2コマで閉じ → 1コマ閉眼 → 3コマで開く（約240ms。閉じる方が開くより速い）
    blink_shape = [0.26, 0.10, 0.04, 0.12, 0.24, 0.33]
    rng = np.random.default_rng(7)  # 毎回同じループになるよう固定シード
    b_time = 1.2
    while True:
        b_idx = int(b_time * fps)
        if b_idx + len(blink_shape) >= n_frames:  # ループの継ぎ目をまたがない
            break
        eye_ratio[b_idx:b_idx + len(blink_shape)] = np.minimum(blink_shape, 0.38 * eye_open)
        b_time += rng.uniform(2.5, 5.5)

    expr_vec = expression_offsets(**expression) if expression else None

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
        if expr_vec is not None:
            exp_vec = exp_vec + expr_vec

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
    emotion=None,
    wrap_pad_frames=0,
    expression=None,
    eye_open=1.0
):
    """
    指定パスに LivePortrait 互換 .pkl モーションテンプレートを出力保存

    wrap_pad_frames: ループ用。前後に周期的な続きのコマを足す（LivePortrait の平滑化が端で途切れて
        ループの継ぎ目が飛ぶのを防ぐ）。生成後の動画から前後この数のコマを切り落として使う。
    """
    template_dct = generate_procedural_motion(
        motion_type=motion_type,
        duration_sec=duration_sec,
        fps=fps,
        emotion=emotion,
        expression=expression,
        eye_open=eye_open
    )
    p = wrap_pad_frames
    if p > 0:
        for key in ("motion", "c_eyes_lst", "c_lip_lst"):
            lst = template_dct[key]
            template_dct[key] = lst[-p:] + lst + lst[:p]
        template_dct["n_frames"] += 2 * p

    if expression and p > 0:
        # 相対モーションでは各コマの表情が「先頭コマとの差」で適用されるため、全コマ同じ表情だと打ち消される。
        # 先頭コマ（切り落とす余白）だけ無表情にして基準にし、以降のコマに表情差分が乗るようにする
        first = dict(template_dct["motion"][0])
        first["exp"] = first["exp"] - expression_offsets(**expression)
        template_dct["motion"][0] = first
    os.makedirs(os.path.dirname(os.path.abspath(output_pkl_path)), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(template_dct, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ 自律モーションテンプレート生成完了: {output_pkl_path} ({template_dct['n_frames']} フレーム, {duration_sec}秒, タイプ: {motion_type})")
    return output_pkl_path


if __name__ == "__main__":
    test_pkl = "test_procedural_idle.pkl"
    save_procedural_motion_template(test_pkl, "idle", 8.0, 25)
    print(f"ファイルサイズ: {os.path.getsize(test_pkl)} bytes")
