"""
brain.py - アバターの「脳」：人格と、家族ごとの会話の記憶

  人格   : 質問への回答（brain/persona.json）から Gemini への人物設定を作る。
           回答を差し替えれば別の人物（例: おじいちゃん）になる。
  家族   : 話しかける人ごとに 名前・続柄・人格側からの呼び方 を登録（brain/people.json）。
  記憶   : 会話を家族ごとに保存し（brain/logs/{id}.jsonl）、数往復ごとに
           「その人について分かったこと」と「最近の会話の要約」を自動で更新して次の会話に使う。

brain/ は実在の人物の個人情報なので Git では管理しない（.gitignore 済み）。
"""
import json
import os
import re
import threading
import time
import uuid as _uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BRAIN_DIR = os.path.join(BASE_DIR, "brain")
PERSONA_PATH = os.path.join(BRAIN_DIR, "persona.json")
PEOPLE_PATH = os.path.join(BRAIN_DIR, "people.json")
LOG_DIR = os.path.join(BRAIN_DIR, "logs")
GUEST_ID = "guest"
MEMORY_UPDATE_EVERY = 4      # 何往復ごとに記憶を更新するか
MAX_FACTS = 40
RECENT_TURNS_IN_HISTORY = 12  # 起動し直したときに会話の流れとして読み込む直近の発言数

_lock = threading.Lock()
version = {"n": 0}  # 人格・記憶が変わったら増やす（会話セッションを作り直す合図）

# 回答しやすさを優先した質問（空欄でもよい）
PERSONA_QUESTIONS = [
    {"id": "name", "label": "名前（呼ばれ方も）", "placeholder": "例: 山田 太郎（みんなからは たろうさん と呼ばれる）"},
    {"id": "profile", "label": "年齢・生まれ・住んでいる所", "placeholder": "例: 1940年生まれ、静岡県出身。ずっと浜松で暮らしている"},
    {"id": "first_person", "label": "一人称と話し方（方言・敬語かどうか）", "placeholder": "例: 一人称は わし。遠州弁で、家族にはくだけた話し方"},
    {"id": "catchphrases", "label": "口癖・よく言う言葉", "placeholder": "例: まあ、ええら / なんとかなるで", "multiline": True},
    {"id": "personality", "label": "性格", "placeholder": "例: 穏やかで世話好き、でも頑固なところもある", "multiline": True},
    {"id": "likes", "label": "好きなこと・趣味・好きな食べ物", "placeholder": "例: 釣り、相撲を見ること、うなぎ", "multiline": True},
    {"id": "values", "label": "大切にしていること・よく言っていた教え", "placeholder": "例: 人には親切にしろ、と よく言っていた", "multiline": True},
    {"id": "history", "label": "人生の主な出来事・仕事・思い出", "placeholder": "例: 40年 大工をしていた。若いころ東京で修行した", "multiline": True},
    {"id": "family", "label": "家族のこと（誰がいるか、関係）", "placeholder": "例: 妻の花子、娘が二人、孫が三人", "multiline": True},
    {"id": "examples", "label": "実際に言いそうなセリフ（3つほど）", "placeholder": "例: おう、よう来たな / ちゃんと飯食っとるか？", "multiline": True},
]


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _bump():
    version["n"] += 1


# ----------------------------------------------------------------------------
# 人格
# ----------------------------------------------------------------------------
def get_persona():
    return _read(PERSONA_PATH, {"answers": {}, "prompt": "", "updated": None})


def build_persona_prompt(client, model, answers):
    """回答から人物設定の文章を Gemini に作らせる（回答にない事実は作らせない）"""
    qa = "\n".join(f"- {q['label']}: {answers.get(q['id'], '').strip() or '（未回答）'}" for q in PERSONA_QUESTIONS)
    instruction = (
        "以下はある実在の人物についての質問と回答です。この人物になりきって会話するための人物設定を作ってください。\n"
        "条件:\n"
        "・回答に書かれていない事実（経歴・家族・出来事）は作らない。未回答の項目は書かない\n"
        "・一人称、話し方（方言・語尾）、口癖、性格、価値観、好きなもの、人生の背景、家族、話し方の例を含める\n"
        "・箇条書きで、400〜800字程度。前置きや説明文は不要で、設定だけを書く\n\n"
        f"{qa}"
    )
    from google.genai import types
    res = client.models.generate_content(model=model, contents=instruction,
                                         config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=1200))
    return (res.text or "").strip()


def save_persona(client, model, answers, prompt_override=None):
    prompt = prompt_override.strip() if prompt_override else build_persona_prompt(client, model, answers)
    data = {"answers": answers, "prompt": prompt, "updated": time.strftime("%Y-%m-%d %H:%M")}
    with _lock:
        _write(PERSONA_PATH, data)
        _bump()
    return data


# ----------------------------------------------------------------------------
# 家族（話しかける人）
# ----------------------------------------------------------------------------
def list_people():
    return _read(PEOPLE_PATH, {})


def upsert_person(person):
    people = list_people()
    pid = person.get("id") or _uuid.uuid4().hex[:8]
    cur = people.get(pid, {"facts": [], "summary": "", "turns_since_update": 0})
    for key in ("name", "relation", "call_name", "notes"):
        if key in person:
            cur[key] = (person.get(key) or "").strip()
    people[pid] = cur
    with _lock:
        _write(PEOPLE_PATH, people)
        _bump()
    return pid, cur


def delete_person(pid):
    people = list_people()
    people.pop(pid, None)
    with _lock:
        _write(PEOPLE_PATH, people)
        _bump()


def reset_memory(pid):
    people = list_people()
    if pid in people:
        people[pid].update(facts=[], summary="", turns_since_update=0)
        with _lock:
            _write(PEOPLE_PATH, people)
    path = os.path.join(LOG_DIR, f"{pid}.jsonl")
    if os.path.exists(path):
        os.replace(path, path + f".reset{int(time.time())}")
    with _lock:
        _bump()


# ----------------------------------------------------------------------------
# 会話のシステム指示
# ----------------------------------------------------------------------------
def build_system_instruction(base_rules, pid):
    persona = get_persona()
    person = list_people().get(pid)
    parts = []
    if persona.get("prompt"):
        parts.append("あなたは次の人物です。この人物として、その人らしい口調・一人称・口癖で話してください。\n"
                     "人物設定にないことを聞かれたら、無理に作らず、その人らしくぼかして答えてください。\n"
                     "口癖や決まり文句は毎回使わず、ときどき（数回に1回程度）自然に混ぜる程度にしてください。\n"
                     f"【人物設定】\n{persona['prompt']}")
    else:
        parts.append("あなたは親しみやすい話し相手です。友達のように明るく親しみやすい口調で話してください。")
    if person:
        who = f"{person.get('name', '')}（あなたから見て {person.get('relation') or '関係不明'}）"
        call = person.get("call_name") or person.get("name", "")
        parts.append(f"【今話している相手】{who}。相手のことは {call} と呼んでください。"
                     + (f"\n補足: {person['notes']}" if person.get("notes") else ""))
        if person.get("facts"):
            parts.append("【この相手について覚えていること】\n" + "\n".join(f"・{x}" for x in person["facts"]))
        if person.get("summary"):
            parts.append(f"【この相手と最近話したこと】\n{person['summary']}")
        parts.append("覚えていることは、会話の流れに合うときだけ自然に話題にしてください。毎回持ち出す必要はありません。")
    else:
        parts.append("【今話している相手】名前の分からない来客です。")
    parts.append(base_rules)
    return "\n\n".join(parts)


def recent_history(pid, n=RECENT_TURNS_IN_HISTORY):
    """直近の発言を [(role, text), ...]（role は user / model）で返す"""
    path = os.path.join(LOG_DIR, f"{pid}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[-n:]
    if rows and rows[0]["role"] == "model":  # 履歴は相手の発言から始める
        rows = rows[1:]
    return [(r["role"], r["text"]) for r in rows]


# ----------------------------------------------------------------------------
# 記憶の更新
# ----------------------------------------------------------------------------
def log_turn(pid, user_text, model_text):
    os.makedirs(LOG_DIR, exist_ok=True)
    now = time.strftime("%Y-%m-%d %H:%M")
    with _lock, open(os.path.join(LOG_DIR, f"{pid}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": now, "role": "user", "text": user_text}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"t": now, "role": "model", "text": model_text}, ensure_ascii=False) + "\n")


def maybe_update_memory(client, model, pid):
    """数往復ごとに、その相手について分かったことと要約を更新（会話を止めないよう別スレッドで呼ぶ）"""
    people = list_people()
    person = people.get(pid)
    if not person:
        return
    person["turns_since_update"] = person.get("turns_since_update", 0) + 1
    if person["turns_since_update"] < MEMORY_UPDATE_EVERY:
        with _lock:
            _write(PEOPLE_PATH, people)
        return
    convo = "\n".join(f"{'相手' if r == 'user' else 'あなた'}: {t}" for r, t in recent_history(pid, MEMORY_UPDATE_EVERY * 2 + 4))
    instruction = (
        f"あなたは会話の記録係です。相手は {person.get('name', '')}（{person.get('relation', '')}）です。\n"
        "これまでに覚えていること・前回までの要約と、直近の会話から、記憶を更新してください。\n"
        "facts: 相手について分かった事実（好み・予定・出来事・家族・体調など）。短い文で、重複はまとめ、古くなった情報は新しい情報に置き換える。"
        f"最大{MAX_FACTS}個。会話から分からないことは書かない。\n"
        "summary: 最近の会話の流れの要約（200字以内）。次に話すときに思い出せるように。\n"
        'JSON で {"facts": [...], "summary": "..."} の形だけを出力してください。\n\n'
        f"【これまで覚えていること】\n{json.dumps(person.get('facts', []), ensure_ascii=False)}\n"
        f"【前回までの要約】\n{person.get('summary', '')}\n"
        f"【直近の会話】\n{convo}"
    )
    try:
        from google.genai import types
        res = client.models.generate_content(
            model=model, contents=instruction,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1500,
                                               response_mime_type="application/json"))
        text = re.sub(r"^```(json)?|```$", "", (res.text or "").strip()).strip()
        data = json.loads(text)
        people = list_people()  # 更新中に変わっている可能性があるので読み直す
        if pid in people:
            people[pid]["facts"] = [str(x).strip() for x in data.get("facts", []) if str(x).strip()][:MAX_FACTS]
            people[pid]["summary"] = str(data.get("summary", "")).strip()
            people[pid]["turns_since_update"] = 0
            people[pid]["memory_updated"] = time.strftime("%Y-%m-%d %H:%M")
            with _lock:
                _write(PEOPLE_PATH, people)
                _bump()
            print(f"🧠 記憶を更新しました [{people[pid].get('name')}]: {len(people[pid]['facts'])}件")
    except Exception as e:
        print(f"⚠️ 記憶の更新に失敗: {e}")
