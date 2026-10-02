"""
voice_lab.py - 声の学習（Style-Bert-VITS2 の追加学習）を Colab で行うモジュール

colab_server.py から register(api_app, ...) で読み込まれ、次の API を追加する。
  POST /api/voice/dataset      学習データ（zip: raw/{style}/*.wav と esd.list）を受け取り Drive に置く
  POST /api/voice/train_start  前処理 → 学習 → ONNX 変換 → AIVMX 作成 → 音声エンジンへ登録 を裏で実行
  GET  /api/voice/status       進み具合
  GET  /api/voice/models       登録済みの学習済み声（スタイル名 → スタイルID）
  GET  /api/voice/aivmx        学習済み声の .aivmx（ローカルの音声エンジンにも入れる用）

Style-Bert-VITS2 は numpy<2 などを要求し、会話用サーバーの環境と衝突するため、
専用の Python 3.11 環境（uv）に入れて別プロセスで動かす。途中経過はすべて Google Drive に置き、
Colab が切れても train_start をもう一度呼べば続きから再開する。
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

SBV2_DIR = "/content/Style-Bert-VITS2"
SBV2_VENV = "/content/sbv2_venv"
SBV2_PY = f"{SBV2_VENV}/bin/python"
VOICE_ROOT = "/content/drive/MyDrive/lipsync_avatar/voice"
DATA_ROOT = f"{VOICE_ROOT}/Data"
ASSETS_ROOT = f"{VOICE_ROOT}/model_assets"
AIVMX_DIR = f"{VOICE_ROOT}/aivmx"
VOICES_JSON = f"{VOICE_ROOT}/voices.json"   # {model_name: {"uuid":..., "styles": {スタイル名: ID}}}
LOG_PATH = "/content/voice_train.log"
STATUS_PATH = "/content/voice_status.json"
REPO_DIR = "/content/memorial-api"

job = {"state": "idle", "detail": "", "model": None, "started": None, "elapsed": None, "step": None, "total_steps": None}


def _save_status():
    try:
        json.dump(job, open(STATUS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass


def _log(msg):
    job["detail"] = msg
    _save_status()
    print(f"🎙️ [声の学習] {msg}", flush=True)


def _run(cmd, cwd=None, env=None, on_line=None):
    """コマンドを実行し、出力をログファイルに追記。失敗時は末尾のログ付きで例外"""
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            log.write(line)
            tail = (tail + [line])[-30:]
            if on_line:
                on_line(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} 失敗:\n{''.join(tail[-15:])}")


def setup_sbv2():
    """Style-Bert-VITS2 を専用環境に用意（2回目以降は数秒）"""
    if os.path.exists(f"{SBV2_DIR}/.ready"):
        return
    os.environ["PATH"] = "/root/.local/bin:/root/.cargo/bin:" + os.environ["PATH"]
    _log("Style-Bert-VITS2 を取得中...")
    if not os.path.exists(SBV2_DIR):
        _run(["git", "clone", "-q", "--depth", "1", "https://github.com/litagin02/Style-Bert-VITS2.git", SBV2_DIR])
    _log("専用の Python 3.11 環境を作成中（数分）...")
    _run(["pip", "install", "-q", "uv"])
    if not os.path.exists(SBV2_PY):
        _run(["uv", "venv", "-q", "-p", "3.11", SBV2_VENV])
    _run(["uv", "pip", "install", "-q", "--python", SBV2_PY, "-r", f"{SBV2_DIR}/requirements-colab.txt", "aivmlib",
          "setuptools<81"])
    _log("事前学習モデルと BERT をダウンロード中...")
    _run([SBV2_PY, "initialize.py", "--skip_default_models"], cwd=SBV2_DIR)
    import yaml
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(ASSETS_ROOT, exist_ok=True)
    with open(f"{SBV2_DIR}/configs/paths.yml", "w", encoding="utf-8") as f:
        yaml.dump({"dataset_root": DATA_ROOT, "assets_root": ASSETS_ROOT}, f)
    open(f"{SBV2_DIR}/.ready", "w").close()


PREPROCESS_RUNNER = r'''
import os, sys, socket, subprocess, time
os.chdir("{sbv2}"); sys.path.insert(0, "{sbv2}")
# 読み解析（pyopenjtalk）の補助サーバーを先に起動して待つ。SBV2 は起動を10秒しか待たず、
# 初回は辞書の読み込み等でそれ以上かかることがあるため。出力はログに残す
port = 7861
log = open("/content/pyopenjtalk_worker.log", "w")
proc = subprocess.Popen([sys.executable, "-m", "style_bert_vits2.nlp.japanese.pyopenjtalk_worker", "--port", str(port)],
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
for _ in range(240):
    try:
        socket.create_connection((socket.gethostname(), port), timeout=1).close()
        break
    except OSError:
        if proc.poll() is not None:
            raise RuntimeError("読み解析サーバーが終了しました: " + open("/content/pyopenjtalk_worker.log").read()[-1500:])
        time.sleep(0.5)
else:
    raise RuntimeError("読み解析サーバーが120秒以内に起動しませんでした: " + open("/content/pyopenjtalk_worker.log").read()[-1500:])
from gradio_tabs.train import preprocess_all
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
pyopenjtalk_worker.initialize_worker()
preprocess_all(model_name="{model}", batch_size={batch}, epochs={epochs}, save_every_steps={save_every},
               num_processes=2, normalize=False, trim=False, freeze_EN_bert=False, freeze_JP_bert=False,
               freeze_ZH_bert=False, freeze_style=False, freeze_decoder=False, use_jp_extra=True,
               val_per_lang=0, log_interval=50, yomi_error="skip")
'''


def train(model, epochs=100, batch=4, save_every=500, install_fn=None):
    t0 = time.time()
    setup_sbv2()
    data_dir = f"{DATA_ROOT}/{model}"
    if not os.path.exists(f"{data_dir}/esd.list"):
        raise RuntimeError(f"学習データがありません: {data_dir}/esd.list")
    n_utt = sum(1 for line in open(f"{data_dir}/esd.list", encoding="utf-8") if line.strip())
    job["total_steps"] = (n_utt // batch + (n_utt % batch > 0)) * epochs

    # pyopenjtalk が pkg_resources を使うが、新しい setuptools では取り除かれているため古い版を入れる
    if subprocess.run([SBV2_PY, "-c", "import pkg_resources"], capture_output=True).returncode != 0:
        _log("不足している部品（setuptools<81）を追加中...")
        os.environ["PATH"] = "/root/.local/bin:/root/.cargo/bin:" + os.environ["PATH"]
        _run(["uv", "pip", "install", "-q", "--python", SBV2_PY, "setuptools<81"])

    resumed = bool(glob.glob(f"{data_dir}/models/G_*.pth"))
    if not resumed:
        _log(f"前処理中（{n_utt} 文、テキスト解析・BERT 特徴・スタイル）...")
        runner = "/content/sbv2_preprocess.py"
        open(runner, "w", encoding="utf-8").write(PREPROCESS_RUNNER.format(
            sbv2=SBV2_DIR, model=model, batch=batch, epochs=epochs, save_every=save_every))
        _run([SBV2_PY, runner], cwd=SBV2_DIR)
    else:
        _log("前回の続きから学習を再開します")

    import yaml
    with open(f"{SBV2_DIR}/default_config.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["model_name"] = model
    with open(f"{SBV2_DIR}/config.yml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    def on_line(line):
        m = re.search(r"(\d+)/(\d+) \[", line) or re.search(r"[Ss]tep[:= ]+(\d+)", line)
        if m:
            job["step"] = int(m.group(1))
            job["detail"] = f"学習中 {job['step']}/{job['total_steps']} ステップ"

    _log(f"学習中（約 {job['total_steps']} ステップ）...")
    _run([SBV2_PY, "train_ms_jp_extra.py", "--config", f"{data_dir}/config.json",
          "--model", data_dir, "--assets_root", ASSETS_ROOT], cwd=SBV2_DIR, on_line=on_line)

    # 最新の学習結果 → ONNX → AIVMX
    cands = sorted(glob.glob(f"{ASSETS_ROOT}/{model}/{model}_e*_s*.safetensors"),
                   key=lambda p: int(re.search(r"_s(\d+)\.safetensors$", p).group(1)))
    if not cands:
        raise RuntimeError("学習結果（.safetensors）が見つかりません")
    st = cands[-1]
    _log(f"ONNX に変換中（{os.path.basename(st)}）...")
    _run([SBV2_PY, "convert_onnx.py", "--model", st], cwd=SBV2_DIR)
    onnx = st.replace(".safetensors", ".onnx")
    os.makedirs(AIVMX_DIR, exist_ok=True)
    aivmx = f"{AIVMX_DIR}/{model}.aivmx"
    _log("AivisSpeech 用の形式（.aivmx）に変換中...")
    _run([f"{SBV2_VENV}/bin/aivmlib", "create-aivmx", "-o", aivmx, "-m", onnx,
          "-h", f"{ASSETS_ROOT}/{model}/config.json", "-s", f"{ASSETS_ROOT}/{model}/style_vectors.npy",
          "-a", "Style-Bert-VITS2 (JP-Extra)"])
    if install_fn:
        _log("音声エンジンに登録中...")
        install_fn(model, aivmx)
    job["elapsed"] = round(time.time() - t0)
    _log(f"完了（{job['elapsed']} 秒）")


def load_voices():
    try:
        return json.load(open(VOICES_JSON, encoding="utf-8"))
    except Exception:
        return {}


def install_aivmx(aivis_url, model, aivmx_path):
    """AIVMX を AivisSpeech Engine に登録し、スタイル名 → スタイルID を voices.json に記録"""
    import requests
    before = set(requests.get(f"{aivis_url}/aivm_models", timeout=30).json().keys())
    with open(aivmx_path, "rb") as f:
        r = requests.post(f"{aivis_url}/aivm_models/install", files={"file": (os.path.basename(aivmx_path), f)}, timeout=300)
    r.raise_for_status()
    models = requests.get(f"{aivis_url}/aivm_models", timeout=30).json()
    new = [u for u in models if u not in before] or [u for u, v in models.items()
                                                    if v.get("manifest", {}).get("name") == model]
    if not new:
        prev = load_voices().get(model, {}).get("uuid")  # 同じ UUID で再登録された場合
        new = [prev] if prev in models else []
    if not new:
        raise RuntimeError("登録したモデルが見つかりません")
    uuid = new[0]
    speaker_uuids = [sp["uuid"] for sp in models[uuid]["manifest"]["speakers"]]
    styles = {}
    for sp in requests.get(f"{aivis_url}/speakers", timeout=30).json():
        if sp.get("speaker_uuid") in speaker_uuids:
            styles.update({st["name"]: st["id"] for st in sp["styles"]})
    voices = load_voices()
    voices[model] = {"uuid": uuid, "styles": styles, "aivmx": aivmx_path}
    os.makedirs(VOICE_ROOT, exist_ok=True)
    json.dump(voices, open(VOICES_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"🎙️ 学習済みの声 [{model}] を登録: {styles}", flush=True)


def run_job(model, epochs, aivis_url):
    """学習プロセスの本体（python voice_lab.py train ... で別プロセスとして実行される）"""
    job.update(state="running", model=model, started=time.time(), elapsed=None, step=None, total_steps=None)
    _save_status()
    try:
        train(model, epochs=epochs, install_fn=lambda m, p: install_aivmx(aivis_url, m, p))
        job["state"] = "done"
    except Exception as e:
        job.update(state="error", detail=f"{type(e).__name__}: {e}")
        print(f"❌ [声の学習] {e}", flush=True)
    job["elapsed"] = round(time.time() - job["started"])
    _save_status()


def register(api_app, aivis_url, is_aivis_ready):
    """colab_server の FastAPI に声の学習用 API を追加。aivis_url はColab内の AivisSpeech Engine"""
    import requests
    from fastapi import File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse

    def reinstall_saved_voices():
        """Colab 起動時: Drive に保存済みの学習済み声を音声エンジンに入れ直す（エンジンのモデル置き場は毎回消えるため）"""
        for _ in range(300):
            if is_aivis_ready():
                break
            time.sleep(2)
        else:
            return
        for model, info in load_voices().items():
            if os.path.exists(info.get("aivmx", "")):
                try:
                    install_aivmx(aivis_url, model, info["aivmx"])
                except Exception as e:
                    print(f"⚠️ 学習済みの声 [{model}] の再登録に失敗: {e}", flush=True)

    threading.Thread(target=reinstall_saved_voices, daemon=True).start()

    @api_app.post("/api/voice/dataset")
    def api_voice_dataset(file: UploadFile = File(...), model_name: str = Form(...)):
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", model_name):
            raise HTTPException(status_code=400, detail="model_name は英数字・_・- のみ")
        data_dir = f"{DATA_ROOT}/{model_name}"
        tmp = f"/content/voice_{int(time.time())}.zip"
        with open(tmp, "wb") as f:
            f.write(file.file.read())
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)  # データを入れ替えるときは前処理・学習もやり直し
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(data_dir)
        os.remove(tmp)
        n = len(glob.glob(f"{data_dir}/raw/**/*.wav", recursive=True))
        return {"model_name": model_name, "wav_files": n}

    proc_holder = {"proc": None}

    def read_status():
        try:
            return json.load(open(STATUS_PATH, encoding="utf-8"))
        except Exception:
            return dict(job)

    @api_app.post("/api/voice/train_start")
    def api_voice_train_start(model_name: str = Form(...), epochs: int = Form(100)):
        p = proc_holder["proc"]
        if p is not None and p.poll() is None:
            raise HTTPException(status_code=409, detail="学習はすでに実行中です")
        # 学習処理は毎回 GitHub の最新コードで別プロセスとして動かす（不具合修正に Colab の再起動が要らない）
        subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"], check=False)
        json.dump({**job, "state": "running", "detail": "開始中", "model": model_name, "started": time.time()},
                  open(STATUS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
        out = open("/content/voice_job_stdout.log", "a")
        proc_holder["proc"] = subprocess.Popen(
            [sys.executable, f"{REPO_DIR}/voice_lab.py", "train", model_name, str(epochs), aivis_url],
            stdout=out, stderr=subprocess.STDOUT, cwd=REPO_DIR)
        return {"state": "running"}

    @api_app.get("/api/voice/status")
    def api_voice_status():
        st = read_status()
        p = proc_holder["proc"]
        if st.get("state") == "running" and p is not None and p.poll() is not None:
            st["state"], st["detail"] = "error", f"学習プロセスが異常終了しました (code {p.returncode})"
        if st.get("state") == "running" and st.get("started"):
            st["elapsed"] = round(time.time() - st["started"])
        try:
            st["log_tail"] = open(LOG_PATH, encoding="utf-8", errors="ignore").read()[-1500:]
        except Exception:
            st["log_tail"] = ""
        return st

    @api_app.get("/api/voice/models")
    def api_voice_models():
        return load_voices()

    @api_app.get("/api/voice/aivmx")
    def api_voice_aivmx(model_name: str):
        info = load_voices().get(model_name)
        if not info or not os.path.exists(info.get("aivmx", "")):
            raise HTTPException(status_code=404, detail="学習済みの声が見つかりません")
        return FileResponse(info["aivmx"], media_type="application/octet-stream", filename=f"{model_name}.aivmx")


if __name__ == "__main__":
    # python voice_lab.py train <model> <epochs> <aivis_url>
    if len(sys.argv) >= 5 and sys.argv[1] == "train":
        run_job(sys.argv[2], int(sys.argv[3]), sys.argv[4])
