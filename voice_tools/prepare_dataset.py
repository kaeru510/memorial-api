"""
prepare_dataset.py - 読み上げ録音から Style-Bert-VITS2 用の学習データを作る

入力: voice_data/recordings/{normal,happy,calm,sad}.(wav|m4a|mp3)  … 読み上げ台本.md の各章を1ファイルずつ
出力: voice_data/work/dataset/raw/{style}/{style}_{番号}.wav と esd.list

1. 左チャンネルだけを使う（ステレオ録音で左右の位相がずれていると、混ぜたときに声が打ち消し合うため）
2. 無音で区切り、faster-whisper で文字起こし
3. 台本の文と照らし合わせ、文の途中で分かれた区間はつなぎ、読み直しはより一致する方を採用
4. 書き起こしには台本の文（正しい表記）を使う

使い方: python voice_tools/prepare_dataset.py  （faster-whisper が必要）
"""
import difflib
import json
import os
import re
import subprocess
import sys
import unicodedata

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_DIR = os.path.join(ROOT, "voice_data", "recordings")
SCRIPT_MD = os.path.join(ROOT, "voice_data", "読み上げ台本.md")
OUT_DIR = os.path.join(ROOT, "voice_data", "work", "dataset")
STYLES = ["normal", "happy", "calm", "sad"]
SPEAKER = "custom"
SR_OUT = 44100
SR_VAD = 16000
PAD = 0.12          # 区間の前後に残す余白（秒）
MIN_SIM = 0.45      # 台本との一致度がこれ未満の対応は採用しない


def load_script():
    """台本 md から {style: [文, ...]} を取り出す"""
    text = open(SCRIPT_MD, encoding="utf-8").read()
    out, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"^## \d+\. .*（(\w+)）", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = re.match(r"^\d+\. (.+)$", line.strip())
        if cur and m:
            out[cur].append(m.group(1).strip())
    return out


def load_left(path, sr):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-af", "pan=mono|c0=c0",
                          "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def detect_segments(x, sr, frame=0.02, min_sil=0.45, min_speech=0.3):
    n = int(sr * frame)
    e = np.array([np.sqrt(np.mean(x[i:i + n] ** 2) + 1e-12) for i in range(0, len(x) - n, n)])
    db = 20 * np.log10(e)
    thr = max(np.percentile(db, 20) + 12, db.max() - 40)
    speech = db > thr
    segs, start, sil = [], None, 0
    for i, v in enumerate(speech):
        if v:
            start = i if start is None else start
            sil = 0
        elif start is not None:
            sil += 1
            if sil * frame >= min_sil:
                end = i - sil + 1
                if (end - start) * frame >= min_speech:
                    segs.append((start * frame, end * frame))
                start, sil = None, 0
    if start is not None and (len(speech) - start) * frame >= min_speech:
        segs.append((start * frame, len(speech) * frame))
    return segs


def norm(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s、。，．,.!！?？・「」『』（）()…ー〜~]", "", s)


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def align(trans, sentences, max_join=3):
    """区間の書き起こし列と台本の文列を、順序を保って対応づける（DP）。
    区間は最大 max_join 個までつないで1文に対応させられる。区間・文の読み飛ばしも許す。
    戻り値: [(文番号, 開始区間, 終了区間(含む), 一致度), ...]"""
    S, K = len(trans), len(sentences)
    NEG = -1e9
    best = np.full((S + 1, K + 1), NEG)
    back = {}
    best[0, 0] = 0.0
    for i in range(S + 1):
        for k in range(K + 1):
            cur = best[i, k]
            if cur == NEG:
                continue
            if i < S and cur > best[i + 1, k]:          # 区間を捨てる（雑音・言い間違い・読み直し前のもの）
                best[i + 1, k] = cur
                back[(i + 1, k)] = (i, k, None)
            if k < K and cur > best[i, k + 1]:          # 文を読み飛ばす（録音にない）
                best[i, k + 1] = cur
                back[(i, k + 1)] = (i, k, None)
            if k < K:
                for j in range(i, min(S, i + max_join)):
                    s = sim("".join(trans[i:j + 1]), sentences[k])
                    if s < MIN_SIM:
                        continue
                    score = cur + s
                    if score > best[j + 1, k + 1]:
                        best[j + 1, k + 1] = score
                        back[(j + 1, k + 1)] = (i, k, (k, i, j, s))
    pairs, state = [], (S, K)
    while state != (0, 0):
        i, k, m = back[state]
        if m:
            pairs.append(m)
        state = (i, k)
    return pairs[::-1]


def main():
    from faster_whisper import WhisperModel
    script = load_script()
    model = WhisperModel("small", device="cpu", compute_type="int8")
    os.makedirs(OUT_DIR, exist_ok=True)
    esd, report = [], {}

    for style in STYLES:
        src = next((os.path.join(REC_DIR, f) for f in os.listdir(REC_DIR)
                    if os.path.splitext(f)[0].rstrip(".") == style), None)
        if not src:
            print(f"⚠️ {style} の録音が見つかりません")
            continue
        x16 = load_left(src, SR_VAD)
        x44 = load_left(src, SR_OUT)
        segs = detect_segments(x16, SR_VAD)
        trans = []
        for s, e in segs:
            a, b = int(max(0, s - PAD) * SR_VAD), int((e + PAD) * SR_VAD)
            out, _ = model.transcribe(x16[a:b], language="ja", beam_size=5,
                                      initial_prompt="、".join(script[style][:3]))
            trans.append("".join(o.text for o in out).strip())
        pairs = align(trans, script[style])

        os.makedirs(os.path.join(OUT_DIR, "raw", style), exist_ok=True)
        peak_target = 10 ** (-3 / 20)  # ピークを -3dBFS にそろえる
        for k, i, j, s in pairs:
            a = int(max(0, segs[i][0] - PAD) * SR_OUT)
            b = int((segs[j][1] + PAD) * SR_OUT)
            clip = x44[a:b]
            clip = clip * (peak_target / max(np.abs(clip).max(), 1e-6))
            name = f"{style}/{style}_{k + 1:02d}.wav"
            pcm = (np.clip(clip, -1, 1) * 32767).astype(np.int16).tobytes()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR_OUT), "-ac", "1",
                            "-i", "-", os.path.join(OUT_DIR, "raw", name)], input=pcm, check=True)
            esd.append(f"{name}|{SPEAKER}|JP|{script[style][k]}")

        matched = {k for k, *_ in pairs}
        report[style] = {
            "segments": len(segs),
            "sentences": len(script[style]),
            "matched": len(pairs),
            "missing": [script[style][k] for k in range(len(script[style])) if k not in matched],
            "low_similarity": [(script[style][k], "".join(trans[i:j + 1]), round(s, 2))
                               for k, i, j, s in pairs if s < 0.75],
            "joined": [(script[style][k], j - i + 1) for k, i, j, s in pairs if j > i],
            "seconds": round(sum((segs[j][1] - segs[i][0]) for _, i, j, _ in pairs), 1),
        }

    with open(os.path.join(OUT_DIR, "esd.list"), "w", encoding="utf-8") as f:
        f.write("\n".join(esd) + "\n")
    with open(os.path.join(OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for style, r in report.items():
        print(f"{style:7s} 区間 {r['segments']:2d} → 採用 {r['matched']:2d}/{r['sentences']}文 ({r['seconds']}秒)"
              f"  つないだ文 {len(r['joined'])}  要確認 {len(r['low_similarity'])}  欠け {len(r['missing'])}")
    print(f"合計 {len(esd)} 文 → {OUT_DIR}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
