# -*- coding: utf-8 -*-
"""
现实版宝可梦 · 单文件版（Python 标准库，零第三方依赖）

用法：
    python 宝可梦.py            # 默认 http://127.0.0.1:8000
    python 宝可梦.py 9000       # 指定端口

功能：拍照/选图 → AI 识别是否动物 → 自动抠图 → 生成图鉴卡 → 地图标记
      + 社区求助 + 宠物医院认证/集资/救治/领养 全流程

数据：程序同目录下 data.json（数据库）、uploads/（图片）。
上传到 GitHub 时这两个都在 .gitignore 里，运行后会自己生成。
"""

import base64
import json
import mimetypes
import os
import posixpath
import re
import socket
import sys
import threading
import time
import uuid
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

APP_TITLE = "现实版宝可梦 · 城市精灵图鉴"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.json")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ADMIN_FILE = os.path.join(BASE_DIR, "admin.html")
MAX_BODY = 32 * 1024 * 1024          # 单次请求最大 32MB

TABLES = ["pokedex", "hospitals", "fundraisers", "treatments",
          "reviews", "stray_reports", "donations"]

DB_LOCK = threading.Lock()
SEED_LOCK = threading.Lock()

# ============================================================ 数据库


def _blank():
    return {"rows": {t: [] for t in TABLES}, "seq": {t: 0 for t in TABLES}}


def load_db():
    if not os.path.exists(DATA_FILE):
        return _blank()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return _blank()
    rows = d.get("rows") or {}
    seq = d.get("seq") or {}
    for t in TABLES:
        rows.setdefault(t, [])
        seq.setdefault(t, len(rows[t]))
    return {"rows": rows, "seq": seq}


def save_db(db):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False)
    os.replace(tmp, DATA_FILE)


def _as_int(v):
    try:
        return int(v)
    except Exception:
        return None


def _match(row, col, val):
    cur = row.get(col)
    iv, ic = _as_int(val), _as_int(cur)
    if iv is not None and ic is not None:
        return iv == ic
    return str(cur) == str(val)


def run_query(q):
    table = q.get("table")
    if table not in TABLES:
        raise ValueError("未知的数据表：" + str(table))
    op = q.get("op") or "select"
    filters = q.get("filters") or []
    mode = q.get("mode")
    payload = q.get("payload")
    order_by = q.get("orderBy")
    limit_n = q.get("limitN")
    with_count = bool(q.get("count"))

    with DB_LOCK:
        db = load_db()
        rows = db["rows"].setdefault(table, [])

        if op == "select":
            out = [dict(r) for r in rows]
            for col, val in filters:
                out = [r for r in out if _match(r, col, val)]
            if isinstance(order_by, dict) and order_by.get("col"):
                col = order_by["col"]
                asc = order_by.get("asc", True)
                out.sort(key=lambda r: (r.get(col) is None,
                                        _key(r.get(col))), reverse=not asc)
            total = len(out)
            if isinstance(limit_n, int) and limit_n >= 0:
                out = out[:limit_n]
            if mode == "single":
                if not out:
                    return {"data": None, "count": total,
                            "error": "没有找到记录"}
                return {"data": out[0], "count": total, "error": None}
            if mode == "maybe":
                return {"data": out[0] if out else None,
                        "count": total, "error": None}
            return {"data": out, "count": total, "error": None}

        if op == "insert":
            items = payload if isinstance(payload, list) else [payload]
            created = []
            now = _now()
            for it in items:
                it = dict(it or {})
                db["seq"][table] = int(db["seq"].get(table, 0)) + 1
                row = {"id": db["seq"][table]}
                row.update(it)
                row.setdefault("created_at", now)
                if "owner_id" not in row:
                    row["owner_id"] = "trial-trainer"
                rows.append(row)
                created.append(dict(row))
            save_db(db)
            data = created if isinstance(payload, list) else created[0]
            return {"data": data, "error": None}

        if op == "update":
            patch = payload or {}
            hit = []
            for r in rows:
                if all(_match(r, c, v) for c, v in filters):
                    r.update(patch)
                    hit.append(dict(r))
            save_db(db)
            return {"data": hit, "error": None}

        if op == "delete":
            keep, removed = [], []
            for r in rows:
                if all(_match(r, c, v) for c, v in filters):
                    removed.append(dict(r))
                else:
                    keep.append(r)
            db["rows"][table] = keep
            save_db(db)
            return {"data": removed, "error": None}

        raise ValueError("未知的操作：" + str(op))


def _key(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    return str(v or "")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================ AI 识别（可选）
# 设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 / 通义千问）即可开启自动识别是否为动物 + 主体定位。
# 不设置也能用：跳过识别，直接按整图抠图。

VISION_MODEL = os.environ.get("VISION_MODEL", "qwen-vl-max-latest")
DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

PROMPT = (
    "判断这张照片的主体是不是真实的小动物（猫、狗、鸟、鱼、兔、鼠、蜥蜴等活体动物）。\n"
    "人、昆虫标本、玩具、玩偶、雕塑、插画、Logo、食物、风景、物品都不算动物。\n"
    "严格只输出 JSON，不要任何解释，不要写任何多余字段：\n"
    '{"is_animal":true或false,"bbox":[x,y,w,h]}\n'
    "bbox 是动物主体在原图中的位置，用 0-1 归一化，x/y 为左上角，w/h 为宽高。"
)


def ai_enabled():
    return bool(os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def models_payload():
    if not ai_enabled():
        return {"models": [], "ai": False,
                "hint": "未配置 AI。设置环境变量 DASHSCOPE_API_KEY 后重启即可自动识别。"}
    return {"models": [{"id": VISION_MODEL, "name": VISION_MODEL,
                        "supportsImages": True, "disabled": False}], "ai": True}


def ai_recognize(data_url):
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"ok": False, "reason": "no_ai"}
    body = {
        "model": VISION_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "你是严格的动物识别引擎，只判断照片主体是不是真实活体动物并给出主体位置，只输出JSON。"},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
    }
    req = urllib.request.Request(
        DASHSCOPE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except Exception as e:                     # 网络/额度问题 → 降级为手动
        return {"ok": False, "reason": "error", "error": "AI 服务调用失败：%s" % e}
    text = ""
    try:
        text = resp["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    except Exception:
        return {"ok": False, "reason": "bad_json"}
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(),
                  flags=re.M).strip()
    try:
        j = json.loads(text)
    except Exception:
        return {"ok": False, "reason": "bad_json"}
    if j.get("is_animal") is False:
        return {"ok": False, "reason": "not_animal"}
    return {"ok": True, "bbox": j.get("bbox"), "species": j.get("species")}


# ============================================================ 设置 & 高德


SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def save_settings(d):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("保存设置失败：", e)


def amap_search(name, address):
    """服务器代理高德 POI 搜索，取商家评分（绕过浏览器跨域）"""
    key = load_settings().get("amap_key") or os.environ.get("AMAP_KEY")
    if not key:
        return {"error": "还没配置高德 Key"}
    if not name:
        return {"error": "医院名称不能为空"}

    def get(url):
        req = urllib.request.Request(url, headers={"User-Agent": "PokemonPals/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        q = urllib.request.quote(name + " " + (address or ""))
        sr = get("https://restapi.amap.com/v3/place/text?keywords=%s&types=07200"
                 "&key=%s&offset=1&page=1" % (q, key))
        pois = sr.get("pois") or []
        if not pois:
            return {"error": "未在高德找到该医院，请确认名称与地址"}
        poi = pois[0]
        rating = (poi.get("biz_ext") or {}).get("rating")
        if not rating:
            dr = get("https://restapi.amap.com/v3/place/detail?id=%s&key=%s"
                     % (poi.get("id"), key))
            dpois = dr.get("pois") or []
            if dpois:
                rating = (dpois[0].get("biz_ext") or {}).get("rating")
        if not rating:
            return {"error": "该医院暂无高德评分"}
        return {"rating": str(rating), "poi": poi.get("name"), "address": poi.get("address")}
    except Exception as e:
        return {"error": "高德请求失败：%s" % e}


# ============================================================ 示例数据

SEED_POKEDEX = [
    # (名字, 稀有度, 元素标签, 备注, emoji, 纬度, 经度)
    ("橘子", "常见", "毛茸茸", "在小区花坛边遇到的橘猫，看见我就翻肚皮。", "🐱", 31.2304, 121.4737),
    ("旺财", "常见", "元气", "楼下便利店门口的小狗，尾巴摇得像螺旋桨。", "🐶", 31.2280, 121.4690),
    ("团子", "少见", "圆滚", "公园长椅下的白兔，耳朵上有一撮灰毛。", "🐰", 31.2350, 121.4800),
]


def seed_demo(db=None):
    """首次运行填充示例数据，让页面打开就有内容可看（只做一次）"""
    with SEED_LOCK:
        own = db is None
        if own:
            with DB_LOCK:
                d = load_db()
        else:
            d = db
        rows = d["rows"].setdefault("pokedex", [])
        if rows:
            return False
        now = int(time.time() * 1000)
        for i, (name, rar, el, note, emo, lat, lng) in enumerate(SEED_POKEDEX):
            d["seq"]["pokedex"] = int(d["seq"].get("pokedex", 0)) + 1
            rows.append({
                "id": d["seq"]["pokedex"],
                "owner_id": "trial-trainer",
                "name": name, "rarity": rar, "element": el,
                "description": note,
                "photo_path": "emoji:" + emo,
                "cutout_path": None, "card_path": None,
                "lat": lat, "lng": lng,
                "place": "示例地点",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(time.time() - i * 3600)),
            })
        if own:
            with DB_LOCK:
                save_db(d)
        return True


# 内嵌示例数据前端脚本：封面用 emoji 直接画，不依赖任何图片文件
SEED_JS = r"""
const EMOJI_COVER = emo => 'data:image/svg+xml,'+encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360">'+
  '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'+
  '<stop offset="0" stop-color="#d9f0e2"/><stop offset="1" stop-color="#f4e6c8"/></linearGradient></defs>'+
  '<rect width="480" height="360" fill="url(#g)"/>'+
  '<circle cx="240" cy="176" r="128" fill="#ffffff" opacity="0.55"/>'+
  '<text x="240" y="250" font-size="160" text-anchor="middle">'+emo+'</text></svg>');

/* 示例数据的封面：photo_path 存 emoji:🐱 时，前端直接画一张渐变卡面 */
function coverSrc(path){
  if(!path) return PAW_PLACEHOLDER;
  if(path.indexOf('emoji:') === 0) return EMOJI_COVER(path.slice(6));
  return path;
}
"""


CSS = r"""

:root{
  --bg:#f4f6f2; --card:#ffffff; --ink:#22301f; --sub:#7a8577; --line:#e5e9e0;
  --green:#3a9e6e; --green-d:#2f7f58; --orange:#ff8a3d; --red:#e5484d; --gold:#eab308;
  --blue:#3b82f6; --purple:#8b5cf6; --teal:#14b8a6; --gray:#94a3b8;
  --wechat:#07c160; --qq:#12b7f5;
  --r:16px; --shadow:0 6px 24px rgba(34,48,31,.08);
}
*{box-sizing:border-box; margin:0; padding:0; -webkit-tap-highlight-color:transparent}
html,body{height:100%}
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif; background:#e8ece4; color:var(--ink); display:flex; justify-content:center;}
#phone{width:100%; max-width:520px; min-height:100vh; background:var(--bg); position:relative; display:flex; flex-direction:column; box-shadow:0 0 40px rgba(0,0,0,.12);}
header{position:sticky; top:0; z-index:50; background:linear-gradient(135deg,#3a9e6e,#2f7f58); color:#fff; padding:14px 16px; display:flex; align-items:center; gap:10px;}
header .logo{font-size:22px}
header h1{font-size:17px; font-weight:700; flex:1}
header .hd-btn{background:rgba(255,255,255,.18); border:none; color:#fff; padding:7px 12px; border-radius:999px; font-size:13px; cursor:pointer}
main{flex:1; padding:14px 14px 84px; }
.view{display:none} .view.active{display:block}
.tabbar{position:fixed; bottom:0; z-index:60; width:100%; max-width:520px; background:#fff; border-top:1px solid var(--line); display:flex; padding:6px 0 calc(6px + env(safe-area-inset-bottom));}
.tabbar button{flex:1; background:none; border:none; font-size:11px; color:var(--sub); cursor:pointer; padding:4px 0}
.tabbar button .ic{font-size:20px; display:block; line-height:24px}
.tabbar button.on{color:var(--green-d); font-weight:700}
.card{background:var(--card); border-radius:var(--r); box-shadow:var(--shadow); padding:14px; margin-bottom:12px}
.h-row{display:flex; align-items:center; gap:10px}
.h-title{font-size:16px; font-weight:700}
.sub{color:var(--sub); font-size:12px}
.btn{display:inline-flex; align-items:center; justify-content:center; gap:6px; border:none; border-radius:12px; padding:11px 16px; font-size:14px; font-weight:600; cursor:pointer; background:var(--green); color:#fff; width:100%;}
.btn.sec{background:#eef3ec; color:var(--ink)}
.btn.warn{background:var(--orange)} .btn.danger{background:var(--red)}
.btn.sm{padding:7px 12px; font-size:12px; width:auto; border-radius:10px}
.btn:disabled{opacity:.5; cursor:not-allowed}
input,textarea,select{width:100%; border:1.5px solid var(--line); border-radius:12px; padding:11px 12px; font-size:14px; background:#fbfcfa; color:var(--ink); font-family:inherit; outline:none;}
input:focus,textarea:focus,select:focus{border-color:var(--green)}
textarea{resize:vertical; min-height:70px}
label.fl{display:block; font-size:12px; color:var(--sub); margin:10px 0 5px; font-weight:600}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:10px}
.grid3{display:grid; grid-template-columns:repeat(3,1fr); gap:8px}
.badge{display:inline-block; font-size:11px; font-weight:700; padding:3px 9px; border-radius:999px; color:#fff}
.b-open{background:var(--red)} .b-captured{background:var(--blue)} .b-fundraising{background:var(--orange)}
.b-treating{background:var(--teal)} .b-adoptable{background:var(--purple)} .b-adopted{background:var(--green)} .b-released{background:var(--gray)}
.rarity{font-size:11px; font-weight:700; padding:3px 9px; border-radius:999px; color:#fff}
.r-common{background:var(--gray)} .r-rare{background:var(--blue)} .r-epic{background:var(--purple)} .r-legend{background:linear-gradient(90deg,#f59e0b,#ef4444)}
.el-tag{font-size:11px; padding:3px 9px; border-radius:6px; background:#eef3ec; color:var(--green-d); font-weight:700}
.dex-grid{display:grid; grid-template-columns:1fr 1fr; gap:12px}
.dex-card{background:#fff; border-radius:var(--r); box-shadow:var(--shadow); overflow:visible; cursor:pointer}
.dex-card .ph{position:relative; height:120px; display:flex; align-items:flex-end; justify-content:center; font-size:44px}
.dex-card .ph::before{content:''; position:absolute; inset:0; border-radius:var(--r) var(--r) 0 0; background:linear-gradient(135deg,#dff0e4,#f5efe0)}
.dex-card .ph img.cover{position:relative; z-index:1; width:100%; height:120px; object-fit:cover; border-radius:var(--r) var(--r) 0 0}
.dex-card .ph img.cover.cf{object-position:top center}
.dex-card .ph img.pop{position:relative; z-index:2; max-height:128px; max-width:80%; object-fit:contain; margin-bottom:8px; filter:drop-shadow(0 8px 12px rgba(90,50,10,.28))}
.dex-card .info{position:relative; z-index:1; background:#fff; padding:10px 12px}
/* 合成卡面图 */
.cardface-img{display:block; width:100%; border-radius:18px; box-shadow:0 10px 26px rgba(60,40,10,.18); cursor:zoom-in}
.dex-card .nm{font-weight:800; font-size:14px}
.dex-card .sp{font-size:11px; color:var(--sub); margin:2px 0 6px}
.stat-row{display:flex; align-items:center; gap:6px; font-size:10px; margin-top:3px}
.stat-row .lb{width:26px; color:var(--sub); font-weight:700}
.stat-row .bar{flex:1; height:6px; border-radius:3px; background:#eef1ea; overflow:hidden}
.stat-row .bar i{display:block; height:100%; border-radius:3px}
.fab{position:fixed; right:18px; bottom:96px; z-index:55; width:58px; height:58px; border-radius:50%; border:none; background:var(--orange); color:#fff; font-size:26px; box-shadow:0 8px 20px rgba(255,138,61,.45); cursor:pointer}
#mapTab{height:calc(100vh - 190px); min-height:380px; border-radius:var(--r); overflow:hidden; box-shadow:var(--shadow)}
.modal-mask{position:fixed; inset:0; z-index:100; background:rgba(15,25,15,.5); display:flex; align-items:flex-end; justify-content:center;}
.modal{background:#fff; width:100%; max-width:520px; max-height:92vh; overflow-y:auto; border-radius:20px 20px 0 0; padding:18px 16px calc(20px + env(safe-area-inset-bottom));}
.modal .grip{width:40px; height:4px; border-radius:2px; background:#dfe5db; margin:0 auto 14px}
.modal h3{font-size:17px; margin-bottom:4px}
.modal .x{float:right; background:none; border:none; font-size:20px; color:var(--sub); cursor:pointer}
.progress{height:10px; background:#eef1ea; border-radius:5px; overflow:hidden; margin:8px 0}
.progress i{display:block; height:100%; background:linear-gradient(90deg,var(--orange),#ffb35c); border-radius:5px}
.timeline{border-left:2px solid var(--line); margin:8px 0 0 6px; padding-left:14px}
.timeline .ti{margin-bottom:14px; position:relative}
.timeline .ti::before{content:''; position:absolute; left:-20px; top:4px; width:9px; height:9px; border-radius:50%; background:var(--green)}
.timeline .ti .tt{font-size:13px; font-weight:700}
.timeline .ti .td{font-size:12px; color:var(--sub); margin-top:2px; white-space:pre-wrap; word-break:break-all}
.timeline img,.timeline video{max-width:100%; border-radius:10px; margin-top:6px}
.stars{color:var(--gold); font-size:14px; letter-spacing:1px}
.toast{position:fixed; top:18%; left:50%; transform:translateX(-50%); z-index:200; background:rgba(20,30,20,.88); color:#fff; padding:11px 18px; border-radius:12px; font-size:14px; max-width:80%; text-align:center; animation:fadein .2s}
@keyframes fadein{from{opacity:0; transform:translate(-50%,-8px)}to{opacity:1; transform:translate(-50%,0)}}
.empty{text-align:center; color:var(--sub); font-size:13px; padding:36px 10px}
.empty .big{font-size:40px; display:block; margin-bottom:8px}
.seg{display:flex; background:#eef1ea; border-radius:12px; padding:4px; margin-bottom:12px}
.seg button{flex:1; border:none; background:none; padding:8px 4px; border-radius:9px; font-size:13px; color:var(--sub); cursor:pointer; font-weight:600}
.seg button.on{background:#fff; color:var(--ink); box-shadow:0 2px 6px rgba(0,0,0,.08)}
.count-box{background:#fffbe8; border:1.5px dashed #f2d478; border-radius:12px; padding:10px 12px; font-size:13px; margin:10px 0; color:#8a6d1a}
.scan-preview{width:100%; max-height:220px; object-fit:cover; border-radius:12px; background:#eef1ea}
.spin{display:inline-block; width:16px; height:16px; border:2px solid rgba(255,255,255,.4); border-top-color:#fff; border-radius:50%; animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.loading{text-align:center; color:var(--sub); padding:30px; font-size:13px}
.pill-row{display:flex; gap:8px; overflow-x:auto; padding:2px 0 8px}
.pill{flex-shrink:0; border:1.5px solid var(--line); background:#fff; border-radius:999px; padding:6px 14px; font-size:12px; font-weight:600; color:var(--sub); cursor:pointer}
.pill.on{background:var(--green); border-color:var(--green); color:#fff}
.map-mini{height:200px; border-radius:12px; overflow:hidden; margin:10px 0}
.notice{background:#e8f3ec; border-radius:12px; padding:12px; font-size:13px; color:#2f5b43; margin-bottom:12px; line-height:1.6}
.notice-warn{background:#fff8e8; border:1.5px solid #f0d79a; border-radius:12px; padding:12px; font-size:12.5px; line-height:1.65; color:#7a5c14; margin-top:10px}
.avatar{width:34px; height:34px; border-radius:50%; background:#dff0e4; display:flex; align-items:center; justify-content:center; font-size:16px}
.li{display:flex; gap:10px; padding:10px 0; border-bottom:1px solid var(--line); align-items:flex-start}
.li:last-child{border-bottom:none}
.li .lbody{flex:1; min-width:0}
.li .ltop{display:flex; align-items:center; gap:6px; flex-wrap:wrap}
a.link{color:var(--blue); text-decoration:none; font-size:12px}
/* ---- 登录页 ---- */
.auth-hero{font-size:44px; text-align:center; margin-bottom:4px}
.lm{display:flex; align-items:center; justify-content:center; gap:8px; width:100%; padding:13px; border-radius:14px; font-size:15px; font-weight:700; border:none; cursor:pointer; margin-bottom:10px; font-family:inherit}
.lm .ic{font-size:18px}
.lm.wechat{background:var(--wechat); color:#fff; box-shadow:0 6px 16px rgba(7,193,96,.28)}
.lm.qq{background:var(--qq); color:#fff; box-shadow:0 6px 16px rgba(18,183,245,.28)}
.lm.phone{background:#eef3ec; color:var(--ink); border:1.5px solid var(--line)}
.divider{display:flex; align-items:center; gap:10px; color:var(--sub); font-size:11px; margin:14px 0 10px}
.divider::before,.divider::after{content:''; flex:1; height:1px; background:var(--line)}
.tiny{font-size:11px; color:var(--sub); text-align:center; margin-top:10px; line-height:1.6}
/* ---- 拍照 / 抠图 / 虚拟动物 ---- */
.pick-row{display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px}
.pick{position:relative; display:flex; flex-direction:column; align-items:center; gap:6px; padding:16px 10px; border:1.5px dashed #cfe0d3; border-radius:16px; background:#f7fbf8; cursor:pointer; font-size:13px; font-weight:600; color:var(--ink)}
.pick:active{background:#eef7f0}
.pick input{position:absolute; width:1px; height:1px; opacity:0; pointer-events:none}
.pick .pi{font-size:24px}
.cutbox{position:relative; margin-top:12px; border-radius:16px; padding:10px; min-height:150px; display:flex; align-items:center; justify-content:center; border:1.5px solid var(--line);
  background:linear-gradient(45deg,#eef3ee 25%,transparent 25%) 0 0/16px 16px,
    linear-gradient(-45deg,#eef3ee 25%,transparent 25%) 0 8px/16px 16px,
    linear-gradient(45deg,transparent 75%,#eef3ee 75%) 8px -8px/16px 16px,
    linear-gradient(-45deg,transparent 75%,#eef3ee 75%) -8px 0/16px 16px, #fff}
.cutbox img{max-width:100%; max-height:230px; display:block}
.cutbox.checker{background-image:linear-gradient(45deg,#e9ece6 25%,transparent 25%),linear-gradient(-45deg,#e9ece6 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#e9ece6 75%),linear-gradient(-45deg,transparent 75%,#e9ece6 75%);background-size:14px 14px;background-position:0 0,0 7px,7px -7px,-7px 0;background-color:#fff}
/* 卡面式抠图预览：原图铺底 + 主体浮层 */
.cutbox.petcard{padding:0; border:none; overflow:hidden; background:#f5b382}
.cutbox.petcard .mini-bg{position:absolute; inset:0; width:100%; height:100%; object-fit:cover}
.cutbox.petcard #scanCutImg{position:relative; z-index:1; max-height:210px; filter:drop-shadow(0 10px 16px rgba(60,35,5,.3))}
/* ---- 修图工作台 ---- */
.ed-mask{position:fixed; inset:0; z-index:150; background:#f4f6f2; display:flex; flex-direction:column}
.ed-head{display:flex; align-items:center; gap:10px; padding:10px 14px; border-bottom:1px solid var(--line); background:#fff}
.ed-head b{font-size:15px}
.ed-head .x2{margin-left:auto; background:none; border:none; font-size:20px; color:var(--sub); cursor:pointer; line-height:1}
.ed-stage{position:relative; flex:1; min-height:0; overflow:auto; display:flex; padding:10px;
  background:linear-gradient(45deg,#e9ece6 25%,transparent 25%) 0 0/14px 14px,
    linear-gradient(-45deg,#e9ece6 25%,transparent 25%) 0 7px/14px 14px,
    linear-gradient(45deg,transparent 75%,#e9ece6 75%) 7px -7px/14px 14px,
    linear-gradient(-45deg,transparent 75%,#e9ece6 75%) -7px 0/14px 14px, #fff}
.ed-stage canvas{display:block; margin:auto; touch-action:none; max-width:none; max-height:none; cursor:crosshair}
.ed-bar{background:#fff; border-top:1px solid var(--line); padding:10px 14px calc(12px + env(safe-area-inset-bottom))}
.ed-tools{display:flex; gap:6px; overflow-x:auto; padding-bottom:2px}
.ed-t{flex:none; border:1.5px solid var(--line); background:#fff; color:var(--ink); border-radius:999px; padding:7px 12px; font-size:12.5px; font-weight:700; font-family:inherit; cursor:pointer}
.ed-t.on{background:var(--green); border-color:var(--green); color:#fff}
.ed-chip{flex:none; font-size:12px; color:var(--sub); align-self:center; padding:0 4px; font-weight:700}
.ed-cursor{position:fixed; display:none; border:1.5px solid #fff; box-shadow:0 0 0 1.5px rgba(0,0,0,.55), inset 0 0 0 1.5px rgba(0,0,0,.3); border-radius:50%; pointer-events:none; z-index:300}
.ed-loading{position:absolute; inset:0; display:flex; align-items:center; justify-content:center; background:rgba(255,255,255,.8); font-size:13px; color:#4a5a48; font-weight:700; z-index:5}
/* 卡面：原图做背景 + 抠图主体“破框”浮在上面（宝可梦卡牌风） */
.pet-stage{position:relative; height:224px; border-radius:18px; margin:14px 0 12px; display:flex; align-items:flex-end; justify-content:center}
.pet-stage.pop{margin-top:48px}
.stage-bg{position:absolute; inset:0; border-radius:18px; overflow:hidden; background:radial-gradient(circle at 50% 28%, #ffecc7 0%, #ffd7a8 46%, #f5b382 100%); box-shadow:var(--shadow)}
.pet-stage .stage-photo{position:absolute; inset:0; width:100%; height:100%; object-fit:cover}
.pet-stage .stage-dim{position:absolute; inset:0; background:linear-gradient(180deg, rgba(255,248,230,.06), rgba(122,74,31,.20))}
.pet-stage img.cut-pop{position:relative; z-index:2; max-height:246px; max-width:88%; object-fit:contain; margin-bottom:10px; transform:translateY(-30px); filter:drop-shadow(0 14px 22px rgba(90,50,10,.32)); animation:petFloatPop 3.4s ease-in-out infinite; cursor:pointer}
.pet-stage img.cut-photo{position:relative; z-index:2; max-height:180px; max-width:82%; margin-bottom:16px; border-radius:14px; box-shadow:0 10px 18px rgba(90,50,10,.22); animation:petFloat 3.4s ease-in-out infinite; cursor:pointer}
.pet-stage .paw{position:relative; z-index:2; font-size:74px; margin-bottom:20px; cursor:pointer; animation:petFloat 3.4s ease-in-out infinite}
@keyframes petFloat{0%,100%{transform:translateY(0) rotate(-1deg)}50%{transform:translateY(-9px) rotate(1deg)}}
@keyframes petFloatPop{0%,100%{transform:translateY(-30px) rotate(-1deg)}50%{transform:translateY(-40px) rotate(1deg)}}
.pet-stage.bounce img{animation:petBounce .62s ease}
.pet-stage.bounce img.cut-pop{animation:petBouncePop .62s ease}
@keyframes petBounce{0%{transform:scale(1)}35%{transform:scale(1.1) translateY(-14px)}70%{transform:scale(.97)}100%{transform:scale(1)}}
@keyframes petBouncePop{0%{transform:translateY(-30px) scale(1)}35%{transform:translateY(-46px) scale(1.08)}70%{transform:translateY(-28px) scale(.97)}100%{transform:translateY(-30px) scale(1)}}
.pet-bubble{position:absolute; top:12px; left:50%; transform:translate(-50%,-8px) scale(.92); opacity:0; background:#fff; color:#3f4a3f; border-radius:14px; padding:7px 12px; font-size:13px; font-weight:700; box-shadow:0 6px 18px rgba(0,0,0,.12); transition:.22s; white-space:nowrap; max-width:90%; z-index:3}
.pet-bubble.on{opacity:1; transform:translate(-50%,0) scale(1)}
.heart{position:absolute; font-size:16px; animation:heartUp 1.1s ease-out forwards; pointer-events:none; z-index:4}
@keyframes heartUp{0%{opacity:1; transform:translateY(0) scale(.6)}100%{opacity:0; transform:translateY(-62px) scale(1.35)}}
.pet-acts{display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:10px 0}
.pet-acts button{padding:10px 4px; border-radius:14px; border:1.5px solid var(--line); background:#fff; font-size:13px; font-weight:700; font-family:inherit; cursor:pointer; color:var(--ink)}
.pet-acts button:active{transform:scale(.97); background:#f4f9f5}
.pet-stat{display:flex; align-items:center; gap:8px; font-size:12px; margin-top:6px}
.pet-stat .lb{width:46px; color:var(--sub); flex:none}
.pet-stat .bar{flex:1; height:8px; border-radius:5px; background:#eef2ee; overflow:hidden}
.pet-stat .bar i{display:block; height:100%; border-radius:5px; transition:width .3s}
"""

BODY = r"""

<div id="phone">
  <header>
    <span class="logo">🐾</span>
    <h1>现实版宝可梦</h1>
    <button class="hd-btn" id="hdAuthBtn" onclick="authClick()">试用中</button>
  </header>

  <main>
    <!-- 图鉴 -->
    <section class="view" id="view-pokedex">
      <div class="card" id="dexStats"></div>
      <div class="pill-row" id="dexFilter"></div>
      <div class="dex-grid" id="dexGrid"></div>
      <button class="fab" onclick="openScan()" title="扫描小动物">📷</button>
    </section>

    <!-- 地图 -->
    <section class="view" id="view-map">
      <div class="card" style="padding:10px 14px">
        <div class="h-row"><span class="h-title">遭遇地图</span><span class="sub" id="mapCount"></span></div>
        <div class="sub" style="margin-top:4px">展示你的图鉴精灵被遇到的地点 · 点击标记查看</div>
      </div>
      <div id="mapTab"></div>
      <div id="mapList" style="margin-top:12px"></div>
    </section>

    <!-- 社区 -->
    <section class="view" id="view-community">
      <div class="seg" id="cSeg"></div>
      <button class="btn" style="margin-bottom:12px" onclick="openNewReport()">🆘 发现受伤/生病的小动物，发布求助</button>
      <div id="reportList"></div>
    </section>

    <!-- 医院 -->
    <section class="view" id="view-hospital"><div id="hospitalBody" class="loading">加载中…</div></section>

    <!-- 我的 -->
    <section class="view" id="view-me"><div id="meBody"></div></section>
  </main>

  <nav class="tabbar">
    <button data-tab="pokedex" class="on" onclick="switchTab('pokedex')"><span class="ic">📕</span>图鉴</button>
    <button data-tab="map" onclick="switchTab('map')"><span class="ic">🗺️</span>地图</button>
    <button data-tab="community" onclick="switchTab('community')"><span class="ic">社区</span>社区</button>
    <button data-tab="hospital" onclick="switchTab('hospital')"><span class="ic">🏥</span>医院</button>
    <button data-tab="me" onclick="switchTab('me')"><span class="ic">👤</span>我的</button>
  </nav>

  <div id="modalRoot"></div>
</div>
"""

JS = r"""
const PAW_SVG = 'data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="30" height="38"><path d="M15 0C6.7 0 0 6.4 0 14.3c0 6.6 4.6 9.6 8.8 11.4 2.4 1 3.4 2.6 3.9 5.1L15 38l2.3-7.2c.5-2.5 1.5-4.1 3.9-5.1 4.2-1.8 8.8-4.8 8.8-11.4C30 6.4 23.3 0 15 0z" fill="%23ff8a3d"/><circle cx="15" cy="14" r="5.5" fill="%23fff"/></svg>')
const PAW_PLACEHOLDER = 'data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300"><rect width="300" height="300" rx="24" fill="#f1f3ee"/><text x="150" y="180" font-size="96" text-anchor="middle">🐾</text></svg>');

/* ================= 初始化 =================
   数据层：本文件由 Python 服务端提供，读写统一走 /api（数据存在服务器 data.json），
   图片/视频上传到服务器 uploads/ 目录。换设备、多人访问共享同一份数据。 */
const GUEST = { id:'trial-trainer', email:'trial@local' };
/* DEMO 默认为 false：数据一律走服务器。保留该变量名，代码里少量 DEMO 分支不会报错 */
const DEMO = false;

const API = {
  async call(op, payload){
    const r = await fetch('/api/' + op, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload || {})
    });
    let j = null;
    try{ j = await r.json(); }catch(e){ throw new Error('服务端返回异常（HTTP '+r.status+'）'); }
    if(!r.ok) throw new Error((j && j.error) || ('请求失败 HTTP '+r.status));
    return j;
  }
};

/* 与 Supabase 风格保持一致的链式查询，落到 /api/query */
function serverDB(table){
  const q = { table, op:'select', filters:[], orderBy:null, limitN:null, mode:null, payload:null, count:false };
  const run = async ()=>{
    try{
      const j = await API.call('query', q);
      return { data:j.data, count:j.count, error:null };
    }catch(e){
      console.warn('数据操作失败', e);
      return { data:null, error:{ message:e.message || String(e) } };
    }
  };
  const b = {
    select(cols='*', opt){ if(opt && opt.count) q.count = true; return b; },
    insert(row){ q.op='insert'; q.payload=row; return b; },
    update(patch){ q.op='update'; q.payload=patch; return b; },
    delete(){ q.op='delete'; return b; },
    eq(col,val){ q.filters.push([col,val]); return b; },
    order(col, opt){ q.orderBy = { col, asc: !(opt && opt.ascending===false) }; return b; },
    limit(n){ q.limitN = n; return b; },
    single(){ q.mode='single'; return b; },
    maybeSingle(){ q.mode='maybe'; return b; },
    then(res, rej){ return run().then(res, rej); }
  };
  return b;
}
const localDB = serverDB;
function dbf(t){ return serverDB(t); }

/* ---- 媒体：上传到服务器 / 由服务器返回 URL ---- */
const memMedia = new Map();
const mediaCache = new Map();
/* 媒体缓存（预留） */
function mediaPut(key, blob){ memMedia.set(key, blob); }
async function mediaGet(key){ return memMedia.get(key) || null; }

/** 上传文件 → 返回可直接放进 <img src> 的 URL（/uploads/xxx.jpg） */
async function uploadFile(file, folder){
  const fd = new FormData();
  fd.append('file', file, (file && file.name) || 'upload.jpg');
  fd.append('folder', folder || 'misc');
  const r = await fetch('/api/upload', { method:'POST', body: fd });
  let j = null;
  try{ j = await r.json(); }catch(e){ throw new Error('上传失败（HTTP '+r.status+'）'); }
  if(!r.ok || !j.path) throw new Error((j && j.error) || '上传失败');
  return j.path;                       /* 形如 /uploads/pokedex/1699xxxx.jpg */
}

/** 把记录里的媒体字段解析成可显示的 URL（兼容旧数据里存的内联 data: 与 local:// 键） */
async function mediaURL(path){
  if(!path) return null;
  if(/^(data:|https?:|\/)/i.test(path)) return path;
  if(path.startsWith('local://')){
    const blob = memMedia.get(path.slice(8));
    if(blob) return URL.createObjectURL(blob);
    throw new Error('本地媒体已丢失');
  }
  return path;
}
function resolveMediaPath(path){ return path || null; }

/* ---- 试用示例数据（首次打开时填充，方便直接看到完整流程） ---- */
function demoCover(emoji, c1, c2){
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient></defs><rect width="480" height="360" fill="url(#g)"/><circle cx="240" cy="176" r="132" fill="#ffffff" opacity="0.2"/><text x="240" y="252" font-size="168" text-anchor="middle">${emoji}</text></svg>`;
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
}
/* 透明背景版的示例“抠图” */
function demoCut(emoji){
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="480" height="360"><text x="240" y="268" font-size="220" text-anchor="middle">${emoji}</text></svg>`;
  return 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
}


/* 老数据补字段（示例精灵的抠图 / 宠物状态 / 审核演示样本） */

/* 审核台演示样本：保证第一次打开 admin.html 时有待审数据可走流程（只补一次） */
function demoAuditSeed(){
  if(DEMO_DB.auditSeeded) return;
  DEMO_DB.auditSeeded = true;
  const R = DEMO_DB.rows || (DEMO_DB.rows = {});
  const now = Date.now(), H = 3600000;
  const hasPendingH = (R.hospitals||[]).some(h=>h.status==='pending');
  if(!hasPendingH){
    demoTable('hospitals').push({
      id: ++DEMO_DB.seq.hospitals, owner_id:'demo-hosp-2', name:'暖爪动物医院',
      license_no:'91310115MA1K8P7X2B',
      license_path: demoCover('📄','#dbeafe','#93c5fd'),
      address:'上海市浦东新区张杨路 500 号', phone:'021-5888 1234',
      intro:'社区宠物诊疗 · 流浪动物救助协作，可提供术后观察床位',
      status:'pending', created_at:new Date(now - 26*H).toISOString()
    });
  }
  const hasPendingF = (R.fundraisers||[]).some(f=>f.status==='pending_review');
  if(!hasPendingF){
    demoTable('fundraisers').push({
      id: ++DEMO_DB.seq.fundraisers, rescue_id:1, owner_id:'demo-hosp-2',
      hospital_name:'暖爪动物医院', title:'三花猫后腿骨折内固定手术',
      target:3200, raised:0, status:'pending_review',
      created_at:new Date(now - 5*H).toISOString()
    });
  }
  demoSave();
}

const S = { user:null, hospital:null, admin:false, tab:'pokedex',
  dex:[], reports:[], cFilter:'all', models:null, visionModel:null };

const ST_MAP = { open:['待救助','b-open'], captured:['已捕捉','b-captured'], fundraising:['集资中','b-fundraising'],
  treating:['救治中','b-treating'], adoptable:['待领养','b-adoptable'], adopted:['已领养','b-adopted'], released:['已放生','b-released'] };

function esc(s){ return String(s??'').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function toast(msg, ms=2400){ const t=document.createElement('div'); t.className='toast'; t.textContent=msg; document.body.appendChild(t); setTimeout(()=>t.remove(), ms); }
/* 按钮忙碌态：禁用 + 改文案，结束后恢复 */
function busy(el, isBusy, label){
  if(!el) return;
  if(isBusy){
    if(el.dataset.txt===undefined) el.dataset.txt = el.innerHTML;
    el.disabled = true;
    el.style.opacity = '.65';
    el.style.pointerEvents = 'none';
    if(label) el.textContent = label;
  }else{
    el.disabled = false;
    el.style.opacity = '';
    el.style.pointerEvents = '';
    if(el.dataset.txt!==undefined){ el.innerHTML = el.dataset.txt; delete el.dataset.txt; }
  }
}
function money(n){ return '¥' + Number(n||0).toFixed(2).replace(/\.00$/,''); }
function fmtTime(ts){ if(!ts) return ''; const d=new Date(ts); return `${d.getMonth()+1}月${d.getDate()}日 ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`; }
function stBadge(st){ const [t,c]=ST_MAP[st]||[st,'b-open']; return `<span class="badge ${c}">${t}</span>`; }
function needLogin(){ return false; }
function myName(){ return '训练家'; }
function starsOf(v){ v=Math.round(Number(v)||0); return '★'.repeat(v)+'☆'.repeat(5-v); }

async function attachMedia(rootEl){
  const nodes = [...rootEl.querySelectorAll('[data-media]')];
  await Promise.all(nodes.map(async n=>{
    try{ const url = coverSrc(await mediaURL(n.dataset.media)); n.src=url; n.dataset.media=''; }
    catch(e){ n.outerHTML='<span class="sub">[媒体加载失败]</span>'; }
  }));
}

/* ================= 图片压缩 ================= */
function compressImage(file, maxSide=900, quality=.82){
  return new Promise(resolve=>{
    let done = false;
    const finish = v => { if(!done){ done = true; clearTimeout(timer); resolve(v); } };
    const fallback = () => finish(file);          /* 解码失败时退回原图，不阻塞流程 */
    const timer = setTimeout(fallback, 8000);     /* 兜底：任何情况下都不卡死 */
    try{
      const img = new Image();
      const fr = new FileReader();
      fr.onerror = fallback;
      fr.onload = ()=>{
        img.onload = ()=>{
          try{
            let {width:w, height:h} = img;
            if(!w || !h) return fallback();
            if(Math.max(w,h)>maxSide){ const k=maxSide/Math.max(w,h); w=Math.round(w*k); h=Math.round(h*k); }
            const cv=document.createElement('canvas'); cv.width=w; cv.height=h;
            cv.getContext('2d').drawImage(img,0,0,w,h);
            cv.toBlob(b=>{
              if(!b) return fallback();
              finish(new File([b], 'photo.jpg', {type:'image/jpeg'}));
            }, 'image/jpeg', quality);
          }catch(e){ fallback(); }
        };
        img.onerror = fallback;
        img.src = fr.result;
      };
      fr.readAsDataURL(file);
    }catch(e){ fallback(); }
  });
}
function fileToDataURL(file){ return new Promise((res,rej)=>{ const fr=new FileReader(); fr.onload=()=>res(fr.result); fr.onerror=rej; fr.readAsDataURL(file); }); }

/* ================= 认证 ================= */
function authClick(){ openTrialInfo(); }
function openTrialInfo(){
  openModal(`
    <h3>🐾 现实版宝可梦</h3>
    <div class="notice">打开就能用，图鉴、地图、社区、宠物医院全部开放。</div>
    <div class="sub" style="line-height:1.9">
      · 你拍的每一只小动物、发布的求助、医院的认证与集资，都会保存到运行本程序的服务器<br>
      · 数据文件为 data.json，图片保存在 uploads 目录，备份时一起拷走即可<br>
      · 管理审核台位于 /admin（admin.html），可审核医院资质与集资申请
    </div>
    <div class="grid2" style="margin-top:14px">
      <button class="btn sec" onclick="closeModal()">开始收集</button>
      <button class="btn" onclick="location.href='/admin'">打开审核台</button>
    </div>`);
}

/* ================= 登录后 ================= */
async function afterLogin(){
  S.user = GUEST; S.demo = true; S.admin = true;
  const btn = document.getElementById('hdAuthBtn');
  if(btn) btn.textContent = '训练家';
  try{
    const { data } = await dbf('hospitals').select('*').eq('owner_id', S.user.id).maybeSingle();
    S.hospital = data || null;
  }catch(e){ S.hospital = null; }
  await ensureModels();
}
function refreshAll(){
  S.user = S.user || GUEST;
  afterLogin().then(()=>{ renderMe(); renderHospitalTab(); }).catch(e=>console.warn(e));
  /* 各视图独立渲染，任何一个出错都不影响其他页面 */
  [renderPokedex, renderCommunity, renderMe, renderHospitalTab].forEach(fn=>{
    try{ fn(); }catch(e){ console.warn('渲染失败', fn.name, e); }
  });
}

/* ================= 模型 ================= */
async function ensureModels(){
  if(S.models) return;
  try{
    const r = await fetch('/api/models');
    const j = await r.json();
    const list = Array.isArray(j.models)? j.models : [];
    S.models = list.filter(m=>m.disabled!==true);
    S.visionModel = S.models.find(m=>m.supportsImages===true) || null;
  }catch(e){ S.models=[]; }
}

/* ================= 导航 ================= */
function switchTab(tab){
  S.tab=tab;
  document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
  document.getElementById('view-'+tab).classList.add('active');
  document.querySelectorAll('.tabbar button').forEach(b=>b.classList.toggle('on', b.dataset.tab===tab));
  if(tab==='pokedex') loadPokedex();
  if(tab==='map'){ loadPokedex(true).then(()=>initMapTab()); }
  if(tab==='community') loadReports();
  if(tab==='hospital') renderHospitalTab();
  if(tab==='me') renderMe();
}

/* ================= 图鉴 ================= */
async function loadPokedex(silent){
  if(!S.user){ document.getElementById('dexGrid').innerHTML = `<div class="empty" style="grid-column:1/-1"><span class="big">🐾</span>图鉴加载中…<br><button class="btn sm" style="margin-top:12px" onclick="openTrialInfo()">查看试用说明</button></div>`; document.getElementById('dexStats').innerHTML=''; document.getElementById('dexFilter').innerHTML=''; return; }
  if(!silent) document.getElementById('dexGrid').innerHTML='<div class="loading" style="grid-column:1/-1">加载图鉴…</div>';
  const { data, error } = await dbf('pokedex').select('*').order('created_at',{ascending:false});
  if(error){ toast('图鉴加载失败：'+error.message); return; }
  S.dex = data||[];
  await stripAutoAttrs(S.dex);
  renderPokedex();
}

/* 老数据清理：把以前自动生成的属性/描述/数值删掉，并让旧排版卡面重画 */
const AUTO_FIELDS = ['species','element','rarity','look','description','stats'];
async function stripAutoAttrs(list){
  const todo = (list||[]).filter(d => AUTO_FIELDS.some(k => d[k] !== undefined && d[k] !== null));
  if(!todo.length) return;
  const patch = { species:null, element:null, rarity:null, look:null, description:null, stats:null, card_path:null };
  todo.forEach(d => Object.assign(d, patch));
  await Promise.all(todo.map(d => {
    try{ return Promise.resolve(dbf('pokedex').update(patch).eq('id', d.id)).catch(()=>{}); }
    catch(e){ return null; }
  }));
}
function renderPokedex(){
  const grid=document.getElementById('dexGrid'), stats=document.getElementById('dexStats'), fl=document.getElementById('dexFilter');
  if(!S.user) return loadPokedex();
  stats.innerHTML = `<div class="h-row"><span class="h-title">我的图鉴</span><span class="sub">${S.dex.length} 只伙伴</span></div>`;
  fl.innerHTML = '';
  const list = S.dex;
  if(!list.length){ grid.innerHTML = `<div class="empty" style="grid-column:1/-1"><span class="big">📷</span>还没有精灵，点击右下角相机去扫描身边的小动物吧！</div>`; return; }
  grid.innerHTML = list.map(d=>{
    const cover = d.cutout_path ? coverSrc(d.cutout_path) : coverSrc(d.photo_path);
    return `<div class="dex-card" onclick="showCard(${d.id})">
      <div class="ph">${d.card_path
        ? `<img data-media="${esc(d.card_path)}" class="cover cf" alt="${esc(d.name)}">`
        : cover?`<img data-media="${esc(cover)}" class="${d.cutout_path?'pop':'cover'}" alt="${esc(d.name)}">`:('🐾')}</div>
      <div class="info">
        <div class="h-row" style="justify-content:space-between"><span class="nm">${esc(d.name)}</span></div>
        <div class="sp">${fmtTime(d.created_at)} 收录</div>
      </div></div>`;
  }).join('');
  attachMedia(grid);
}
async function showCard(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  openModal(`
    <h3>${esc(d.name)}</h3>
    <div class="sub">收录于 ${fmtTime(d.created_at)}</div>
    <div id="cardFaceBox" style="margin:12px 0 4px">${d.card_path?`<img class="cardface-img" data-media="${esc(d.card_path)}" alt="卡面" onclick="saveCardFace(${d.id})">`:'<div class="loading" style="border-radius:18px">🎨 正在绘制卡面…</div>'}</div>
    <div class="tiny" style="margin:2px 0 6px">${d.card_path?'👆 点卡面可保存图片':'首次打开会自动把照片画成这张卡面'}</div>
    <div id="petBox">${petStageHTML(d)}</div>
    <div class="tiny" style="margin-top:6px">点它会和你互动 · 喂食 / 摸摸 / 玩耍 会提升亲密度</div>
    <div class="sub" style="margin-top:10px">📍 ${d.lat
      ? `遭遇地点：${(+d.lat).toFixed(5)}, ${(+d.lng).toFixed(5)} <a class="link" href="javascript:closeModal();switchTab('map')">在地图查看</a> · <a class="link" href="javascript:openGeoPicker(${d.id})">调整</a>`
      : `<a class="link" href="javascript:openGeoPicker(${d.id})">补标记遭遇地点</a>（还没在地图上留下爪印）`}</div>
    <div class="grid3" style="margin-top:14px">
      <button class="btn sec sm" onclick="saveCardFace(${d.id})">🖼️ 保存卡面</button>
      <button class="btn sec sm" onclick="openRecut(${d.id})">✂️ 重新抠图</button>
      <button class="btn danger sm" onclick="delEntry(${d.id})">放归</button>
    </div>`);
  await attachMedia(document.getElementById('modalRoot'));
  /* 没有卡面就现画一张 */
  if(!d.card_path){
    const path = await ensureCardFace(d);
    const box = document.getElementById('cardFaceBox');
    if(box && path){
      box.innerHTML = `<img class="cardface-img" src="${await mediaURL(path)}" alt="卡面" onclick="saveCardFace(${d.id})">`;
    }else if(box){ box.innerHTML = '<div class="notice">卡面生成失败（缺少照片）</div>'; }
  }
}
async function saveCardFace(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  try{
    let path = d.card_path;
    if(!path) path = await ensureCardFace(d, true);
    if(!path) return toast('还没有照片，画不了卡面');
    const a = document.createElement('a');
    a.href = await mediaURL(path); a.download = `${d.name}-卡面.png`;
    document.body.appendChild(a); a.click(); a.remove();
    toast('卡面已保存 🎴');
  }catch(e){ toast('保存失败：'+(e.message||e)); }
}

async function delEntry(id){
  if(!confirm('确定把这只精灵从图鉴放归吗？')) return;
  const { data, error } = await dbf('pokedex').delete().eq('id', id).select();
  if(error) return toast('删除失败：'+error.message);
  if(!data.length) return toast('无法删除（可能不属于你）');
  toast('已放归 🌿'); closeModal(); loadPokedex();
}

/* ---- 扫描 / 识别 / 抠图 / 虚拟动物 ---- */
let scanCtx = { file:null, dataURL:null, ai:null, cutoutBlob:null, cutoutURL:null };
const delay = ms => new Promise(r=>setTimeout(r, ms));

function openScan(){
  if(needLogin()) return;
  scanCtx = { file:null, dataURL:null, ai:null, cutoutBlob:null, cutoutURL:null };
  openModal(`
    <h3>📷 扫描小动物</h3>
    <div class="sub">拍照或从相册选一张照片 · 只会识别真实小动物</div>
    <div id="scanStep1">
      <div class="pick-row">
        <label class="pick"><span class="pi">📷</span><span class="pt">拍照</span>
          <input type="file" accept="image/*" capture="environment" onchange="handleScanFile(this)"></label>
        <label class="pick"><span class="pi">🖼️</span><span class="pt">从相册选择</span>
          <input type="file" accept="image/*" onchange="handleScanFile(this)"></label>
      </div>
      <div id="scanPrev"></div>
      <div id="scanGoWrap" style="display:none;margin-top:12px">
        <button class="btn" id="scanGo" onclick="runScan()">开始识别</button>
      </div>
    </div>
    <div id="scanStep2" style="display:none"></div>
    <div class="tiny">识别与抠图都在本机浏览器完成，照片不会外传。</div>`);
}

async function handleScanFile(input){
  const f = input.files?.[0]; if(!f) return;
  const prev = document.getElementById('scanPrev');
  const wrap = document.getElementById('scanGoWrap');
  if(prev) prev.innerHTML = '<div class="sub" style="margin-top:10px">正在读取照片…</div>';
  try{
    const cf = await compressImage(f);
    scanCtx.file = cf;
    scanCtx.dataURL = await fileToDataURL(cf);
    const img = document.createElement('img');
    img.className = 'scan-preview';
    img.onerror = ()=>{
      scanCtx.dataURL = null;
      prev.innerHTML = '<div class="notice">这个图片格式浏览器无法预览（如 iPhone 的 HEIC），请换成 JPG / PNG 再试。</div>';
    };
    img.src = scanCtx.dataURL;
    if(prev){ prev.innerHTML=''; prev.appendChild(img); }
    if(wrap) wrap.style.display = 'block';
  }catch(e){
    if(prev) prev.innerHTML = '<div class="notice">读取照片失败，请换一张再试。</div>';
    toast('读取照片失败：'+(e.message||e));
  }
}

/* 识别：只判定是否动物 + 定位主体（不再生成任何属性/数值/描述） */
async function recognizeAnimal(dataURL){
  if(!dataURL) return { ok:false, reason:'no_image' };
  await Promise.race([ensureModels(), delay(2500)]);
  if(!S.visionModel) return { ok:false, reason:'no_ai' };
  let j = null;
  try{
    const r = await fetch('/api/vision', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image:dataURL })
    });
    j = await r.json();
    if(!r.ok) return { ok:false, reason:'error', message:(j && j.error) || ('HTTP '+r.status) };
  }catch(e){ return { ok:false, reason:'error', message:e.message }; }
  if(!j || j.ok !== true){
    if(j && j.reason === 'not_animal') return { ok:false, reason:'not_animal', data:j };
    return { ok:false, reason:'bad_json' };
  }
  return { ok:true, data:{ bbox: j.bbox || null } };
}

async function runScan(){
  const btn = document.getElementById('scanGo');
  if(!scanCtx.dataURL) return toast('请先拍照或选择照片');
  busy(btn, true, 'AI 识别中…');
  let r = null;
  try{ r = await recognizeAnimal(scanCtx.dataURL); }
  catch(e){ console.warn('识别异常', e); r = { ok:false, reason:'error', message:e.message }; }
  busy(btn, false);
  if(r.reason === 'not_animal') return renderNotAnimal();
  scanCtx.ai = r.ok ? r.data : null;
  await prepareCutout();
  const tip = r.ok ? '🤖 已确认是小动物并自动抠图，取个名字就能收服'
    : (r.reason==='no_ai' ? '🧪 当前拿不到识别模型，已按照片完成抠图，直接命名收服'
    : '🧪 这次没能确认，也可以直接命名收服');
  openScanForm(tip);
}

function renderNotAnimal(){
  const step1 = document.getElementById('scanStep1');
  const step2 = document.getElementById('scanStep2');
  if(step1) step1.style.display = 'none';
  if(step2){
    step2.style.display = 'block';
    step2.innerHTML = `
      <div class="notice-warn" style="margin-top:0">
        <b>没有识别到小动物</b><br>
        这张照片看起来是物品、风景或图案。图鉴只收录真实的小动物（猫、狗、鸟、鱼、兔子等）。
      </div>
      <div id="scanPrev2" style="margin-top:10px"></div>
      <button class="btn" style="margin-top:14px" onclick="openScan()">重新拍照 / 换一张</button>`;
    if(scanCtx.dataURL){
      const img = document.createElement('img');
      img.className = 'scan-preview';
      img.src = scanCtx.dataURL;
      document.getElementById('scanPrev2').appendChild(img);
    }
  }
}

/* ---------- 智能抠图 v3（自适应主体框 + 颜色模型 + 软边 alpha + 边缘去色） ---------- */
function loadImage(src){
  return new Promise((res,rej)=>{
    const img = new Image();
    img.onload = ()=>res(img);
    img.onerror = ()=>rej(new Error('图片解码失败'));
    img.src = src;
  });
}
function canvasToBlob(cv){
  return new Promise(res=>cv.toBlob(b=>res(b), 'image/png'));
}
const CUT_MAX_SIDE = 900;
const CUT_BINS = 32768;                     // 5bit/通道 颜色量化

function cutBin(px,i){ return ((px[i]>>3)<<10) | ((px[i+1]>>3)<<5) | (px[i+2]>>3); }

/* 归一化 bbox → 像素矩形（外扩，留出边缘）；不像 bbox 就返回 null */
function bboxRect(bbox,w,h){
  if(!Array.isArray(bbox)||bbox.length<4) return null;
  let [x,y,bw,bh] = bbox.map(Number);
  if(![x,y,bw,bh].every(v=>isFinite(v))) return null;
  if(Math.max(bw,bh) > 1.5){ x/=w; y/=h; bw/=w; bh/=h; }     // 模型返回了像素值
  if(x<0||y<0||bw<=0||bh<=0||x>=1||y>=1) return null;
  bw = Math.min(bw,1-x); bh = Math.min(bh,1-y);
  if(bw<0.05||bh<0.05) return null;
  const ex = 0.12*bw, ey = 0.12*bh;
  const x0 = Math.max(0,x-ex)*w, y0 = Math.max(0,y-ey)*h;
  const x1 = Math.min(1,x+bw+ex)*w, y1 = Math.min(1,y+bh+ey)*h;
  if(x1-x0 < w*0.1 || y1-y0 < h*0.1) return null;
  return [Math.round(x0), Math.round(y0), Math.round(x1)-1, Math.round(y1)-1];
}

/* 直方图 → 3x3x3 平滑后的概率表（让模型对相近颜色有泛化） */
function smoothProb(counts,total){
  const out = new Float32Array(CUT_BINS);
  for(let r=0;r<32;r++) for(let g=0;g<32;g++) for(let b=0;b<32;b++){
    let s = 0;
    for(let dr=-1;dr<=1;dr++){
      const rr=r+dr; if(rr<0||rr>31) continue;
      for(let dg=-1;dg<=1;dg++){
        const gg=g+dg; if(gg<0||gg>31) continue;
        for(let db=-1;db<=1;db++){
          const bb=b+db; if(bb<0||bb>31) continue;
          s += counts[(rr<<10)|(gg<<5)|bb] * (dr?0.5:1) * (dg?0.5:1) * (db?0.5:1);
        }
      }
    }
    out[(r<<10)|(g<<5)|b] = s;
  }
  const norm = 1/Math.max(1,total*8);
  for(let i=0;i<CUT_BINS;i++) out[i] *= norm;
  return out;
}

/* 可分离 3 抽头模糊（用于边缘抗锯齿） */
function blurAlpha(a,w,h,passes){
  let src=a, dst=new Uint8Array(w*h);
  for(let p=0;p<passes;p++){
    for(let y=0;y<h;y++){
      const row=y*w;
      for(let x=0;x<w;x++){
        const l=src[row+(x>0?x-1:0)], c=src[row+x], r=src[row+(x<w-1?x+1:x)];
        dst[row+x] = (l + c*2 + r) >> 2;
      }
    }
    const t=src; src=dst; dst=t;
  }
  return src;
}

/* 两遍倒角距离变换：mask 为 1 的地方距离为 0，其余为到最近 1 的距离 */
function distTransform(mask, w, h){
  const INF = 1e7, d = new Float32Array(w*h);
  for(let i=0;i<w*h;i++) d[i] = mask[i] ? 0 : INF;
  const A = 1, B = 1.4142136;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    const i=y*w+x; let v=d[i]; if(v===0) continue;
    if(x>0) v=Math.min(v,d[i-1]+A);
    if(y>0) v=Math.min(v,d[i-w]+A);
    if(x>0&&y>0) v=Math.min(v,d[i-w-1]+B);
    if(x<w-1&&y>0) v=Math.min(v,d[i-w+1]+B);
    d[i]=v;
  }
  for(let y=h-1;y>=0;y--) for(let x=w-1;x>=0;x--){
    const i=y*w+x; let v=d[i]; if(v===0) continue;
    if(x<w-1) v=Math.min(v,d[i+1]+A);
    if(y<h-1) v=Math.min(v,d[i+w]+A);
    if(x<w-1&&y<h-1) v=Math.min(v,d[i+w+1]+B);
    if(x>0&&y<h-1) v=Math.min(v,d[i+w-1]+B);
    d[i]=v;
  }
  return d;
}

/* 没有 AI 主体框时：用「和画面边缘颜色的差异」找显著区域的外接框 */
function autoROI(px, w, h){
  const B = Math.max(2, Math.round(Math.min(w,h)*0.02));
  let br=0,bg=0,bb=0,cnt=0;
  for(let y=0;y<h;y+=2) for(let x=0;x<w;x+=2){
    if(!(x<B||y<B||x>=w-B||y>=h-B)) continue;
    const p=(y*w+x)*4; if(px[p+3]<16) continue;
    br+=px[p]; bg+=px[p+1]; bb+=px[p+2]; cnt++;
  }
  if(cnt < 40) return null;
  br/=cnt; bg/=cnt; bb/=cnt;
  const sw = Math.ceil(w/2), sh = Math.ceil(h/2);
  const sal = new Uint8Array(sw*sh);
  let tot = 0;
  for(let y=0;y<sh;y++) for(let x=0;x<sw;x++){
    const p = ((y*2)*w + x*2)*4;
    if(px[p+3]<16) continue;
    const dr=px[p]-br, dg=px[p+1]-bg, db=px[p+2]-bb;
    if(dr*dr+dg*dg+db*db > 2400){ sal[y*sw+x]=1; tot++; }
  }
  if(tot < sw*sh*0.015) return null;
  const seen = new Uint8Array(sw*sh), st = new Int32Array(sw*sh);
  let best = null;
  for(let s=0;s<sw*sh;s++){
    if(!sal[s]||seen[s]) continue;
    let top=0, area=0, x0=sw, y0=sh, x1=0, y1=0;
    seen[s]=1; st[top++]=s;
    while(top>0){
      const j=st[--top]; area++;
      const x=j%sw, y=(j/sw)|0;
      if(x<x0)x0=x; if(x>x1)x1=x; if(y<y0)y0=y; if(y>y1)y1=y;
      if(x>0   && sal[j-1] && !seen[j-1]){ seen[j-1]=1; st[top++]=j-1; }
      if(x<sw-1&& sal[j+1] && !seen[j+1]){ seen[j+1]=1; st[top++]=j+1; }
      if(y>0   && sal[j-sw]&& !seen[j-sw]){ seen[j-sw]=1; st[top++]=j-sw; }
      if(y<sh-1&& sal[j+sw]&& !seen[j+sw]){ seen[j+sw]=1; st[top++]=j+sw; }
    }
    if(!best || area>best.area) best={area,x0,y0,x1,y1};
  }
  if(!best) return null;
  const pad = 0.06;
  const bx=best.x0*2, by=best.y0*2, bw=(best.x1-best.x0+1)*2, bh=(best.y1-best.y0+1)*2;
  const X  = Math.max(0, Math.round(bx - bw*pad));
  const Y  = Math.max(0, Math.round(by - bh*pad));
  const X1 = Math.min(w-1, Math.round(bx + bw + bw*pad));
  const Y1 = Math.min(h-1, Math.round(by + bh + bh*pad));
  if(X1-X < w*0.08 || Y1-Y < h*0.08) return null;
  return [X,Y,X1,Y1];
}

/**
 * 抠图主函数
 * @param img      图片元素
 * @param strength weak | auto | strong
 * @param bbox     AI 返回的主体框（归一化 或 像素）
 */
async function cutoutFromImage(img, strength='auto', bbox=null){
  const sw = img.width || img.naturalWidth || 1;
  const sh = img.height || img.naturalHeight || 1;
  const scale = Math.min(1, CUT_MAX_SIDE/Math.max(sw,sh));
  const w = Math.max(8, Math.round(sw*scale));
  const h = Math.max(8, Math.round(sh*scale));
  const cv = document.createElement('canvas'); cv.width=w; cv.height=h;
  const ctx = cv.getContext('2d', {willReadFrequently:true});
  ctx.drawImage(img, 0, 0, w, h);
  const px = ctx.getImageData(0,0,w,h).data;

  /* 有 AI / 手动框就按框来；没有就用显著性自动找主体，避免硬切正中 40% */
  let rect = bboxRect(bbox, w, h);
  if(!rect) rect = autoROI(px, w, h);
  const mx = Math.round(w*0.04), my = Math.round(h*0.04);
  const rx0 = rect? rect[0] : mx, ry0 = rect? rect[1] : my;
  const rx1 = rect? rect[2] : w-1-mx, ry1 = rect? rect[3] : h-1-my;
  const roi = [rx0, ry0, rx1, ry1];
  const rw = Math.max(2, rx1-rx0+1), rh = Math.max(2, ry1-ry0+1);

  /* ---- 1-4. 迭代式前景/背景建模（GrabCut 思路，纯颜色分类 + 连通块）----
     背景样本 = ROI 外圈 + ROI 边缘与核心之间的环带（权重 0.5，那里常含阴影渐变）
     前景样本 = 第 1 轮用核心，之后用上一轮结果 */
  const tol = strength==='weak' ? -0.45 : strength==='strong' ? 0.35 : 0;
  const coreBand = new Uint8Array(w*h);        // 深度核心（前景种子）
  const ringBand = new Uint8Array(w*h);        // 核心与 ROI 边缘之间的环带
  {
    const cx0=Math.round(rx0+rw*0.3), cx1=Math.round(rx0+rw*0.7);
    const cy0=Math.round(ry0+rh*0.3), cy1=Math.round(ry0+rh*0.7);
    for(let y=ry0;y<=ry1;y++) for(let x=rx0;x<=rx1;x++){
      const i=y*w+x;
      if(x>=cx0&&x<=cx1&&y>=cy0&&y<=cy1) coreBand[i]=1;
      else ringBand[i]=1;
    }
  }
  let keep = new Uint8Array(w*h), keepArea = 0, coverage = 0, scoreArr = null;
  const fgC = new Float32Array(CUT_BINS), bgC = new Float32Array(CUT_BINS);
  const cstack = new Int32Array(w*h);

  for(let iter=0; iter<3; iter++){
    fgC.fill(0); bgC.fill(0);
    let fgN=0, bgN=0;
    for(let y=0;y<h;y++){
      for(let x=0;x<w;x++){
        const i=y*w+x, p=i*4;
        if(px[p+3] < 16) continue;
        const bin = cutBin(px,p);
        if(iter===0 ? coreBand[i] : keep[i]){ fgC[bin]++; fgN++; }
        else if(x<rx0 || x>rx1 || y<ry0 || y>ry1){ bgC[bin]++; bgN++; }
        else if(ringBand[i]){ bgC[bin]+=0.5; bgN+=0.5; }
      }
    }
    /* 背景样本太少（主体几乎占满画面）→ 退用最外一圈 */
    if(bgN < w*h*0.012){
      bgC.fill(0); bgN = 0;
      const band = Math.max(2, Math.round(Math.min(w,h)*0.07));
      for(let y=0;y<h;y++) for(let x=0;x<w;x++){
        if(x>=band && x<w-band && y>=band && y<h-band) continue;
        const p=(y*w+x)*4;
        if(px[p+3] < 16) continue;
        bgC[cutBin(px,p)]++; bgN++;
      }
    }
    if(fgN < 60 || bgN < 60) break;

    const fgP = smoothProb(fgC, fgN), bgP = smoothProb(bgC, bgN);
    const score = new Float32Array(w*h);
    scoreArr = score;
    for(let i=0,p=0;i<w*h;i++,p+=4){
      if(px[p+3] < 16){ score[i] = -9; continue; }
      const b = ((px[p]>>3)<<10)|((px[p+1]>>3)<<5)|(px[p+2]>>3);
      score[i] = Math.log((fgP[b]+2e-4)/(bgP[b]+2e-4));
    }

    /* 颜色分类 + 连通块过滤（不依赖泛洪连通，过渡色带不再挡路） */
    const label = new Int32Array(w*h).fill(-1);
    const comps = [];
    for(let start=0;start<w*h;start++){
      if(score[start] < tol || label[start]>=0) continue;
      const id = comps.length;
      let top = 0, area = 0, minX=w, minY=h, maxX=0, maxY=0, inCore=false;
      cstack[top++] = start; label[start] = id;
      while(top > 0){
        const j = cstack[--top]; area++;
        const x = j%w, y = (j/w)|0;
        if(x<minX) minX=x; if(x>maxX) maxX=x;
        if(y<minY) minY=y; if(y>maxY) maxY=y;
        if(coreBand[j]) inCore=true;
        if(x>0   && score[j-1]>=tol && label[j-1]<0){ label[j-1]=id; cstack[top++]=j-1; }
        if(x<w-1 && score[j+1]>=tol && label[j+1]<0){ label[j+1]=id; cstack[top++]=j+1; }
        if(y>0   && score[j-w]>=tol && label[j-w]<0){ label[j-w]=id; cstack[top++]=j-w; }
        if(y<h-1 && score[j+w]>=tol && label[j+w]<0){ label[j+w]=id; cstack[top++]=j+w; }
      }
      comps.push({id, area, minX, minY, maxX, maxY, inCore});
    }
    comps.sort((a,b)=>b.area-a.area);
    keep = new Uint8Array(w*h); keepArea = 0;
    if(comps.length){
      const main = comps[0];
      /* 主体膨胀 6px：只有贴着主体的碎块（胡须/衔着的树枝）保留，远处碎屑丢弃 */
      const near = new Uint8Array(w*h);
      const R = 6;
      let mSrc = new Uint8Array(w*h);
      for(let i=0;i<w*h;i++) if(label[i]===main.id) mSrc[i]=1;
      let cur = mSrc;
      for(let step=0; step<R; step++){
        const nxt = new Uint8Array(w*h);
        for(let y=0;y<h;y++) for(let x=0;x<w;x++){
          const i=y*w+x;
          if(cur[i]){ nxt[i]=1; continue; }
          if((x>0&&cur[i-1])||(x<w-1&&cur[i+1])||(y>0&&cur[i-w])||(y<h-1&&cur[i+w])) nxt[i]=1;
        }
        cur = nxt;
      }
      for(let i=0;i<w*h;i++) if(cur[i]) near[i]=1;
      const okIds = new Set([main.id]);
      for(const c of comps){
        if(c.id===main.id) continue;
        if(c.area >= main.area*0.2){ okIds.add(c.id); continue; }
        if(c.area < 40) continue;
        let touch=false;
        for(let y=c.minY;y<=c.maxY&&!touch;y++) for(let x=c.minX;x<=c.maxX;x++){
          if(near[y*w+x] && label[y*w+x]===c.id){ touch=true; break; }
        }
        if(touch) okIds.add(c.id);
      }
      for(let i=0;i<w*h;i++){ if(label[i]>=0 && okIds.has(label[i])){ keep[i]=1; keepArea++; } }
    }
    coverage = keepArea/(w*h);

    /* 孔洞填充：被主体完全包住的小区域填回（大头是颜色误判），大空隙保留透明 */
    if(iter===2 || coverage>0){
      const seen = new Uint8Array(w*h);
      const hstack = new Int32Array(w*h);
      const seeds = [];
      for(let x=rx0;x<=rx1;x++){ seeds.push(x, (ry0)*w+x, (ry1)*w+x); }
      for(let y=ry0;y<=ry1;y++){ seeds.push(y*w+rx0, y*w+rx1); }
      let top=0;
      const pushH=i=>{ if(!keep[i] && !seen[i]){ seen[i]=1; hstack[top++]=i; } };
      for(const i of seeds) pushH(i);
      while(top>0){
        const i=hstack[--top], x=i%w, y=(i/w)|0;
        if(x>0) pushH(i-1);
        if(x<w-1) pushH(i+1);
        if(y>0) pushH(i-w);
        if(y<h-1) pushH(i+w);
      }
      /* 未被访问且非主体的 → 被包住的孔洞 */
      let holePixels=[];
      for(let i=0;i<w*h;i++) if(!keep[i] && !seen[i]) holePixels.push(i);
      if(holePixels.length && holePixels.length < keepArea*0.35){
        for(const i of holePixels){ keep[i]=1; }
        keepArea += holePixels.length;
        coverage = keepArea/(w*h);
      }
    }
  }

  /* ---- 5. 判断是否抠成功；失败时退回“圆角柔和框”，不硬留背景 ---- */
  const ok = coverage > 0.008 && coverage < 0.93;
  if(!ok){
    return softFallback(img, roi, w, h);
  }

  /* ---- 6. 软边 alpha：距离场做出 ~1.5px 过渡带，再按颜色置信度微调（毛边更自然） ---- */
  const inside = new Uint8Array(w*h), outside = new Uint8Array(w*h);
  for(let i=0;i<w*h;i++){ if(keep[i]) inside[i]=1; else outside[i]=1; }
  const dIn  = distTransform(outside, w, h);   // 前景像素 → 最近背景的距离
  const dOut = distTransform(inside,  w, h);   // 背景像素 → 最近前景的距离
  const alpha = new Uint8Array(w*h);
  const sc = scoreArr || new Float32Array(w*h);
  for(let i=0;i<w*h;i++){
    const sd = keep[i] ? (dIn[i]-0.35) : -(dOut[i]-0.35);
    let a = sd >= 0.9 ? 1 : (sd <= -0.7 ? 0 : (sd+0.7)/1.6);
    if(a > 0.03 && a < 0.99){
      a *= 0.55 + 0.45*Math.max(0, Math.min(1, (sc[i]+0.45)/1.25));
    }
    alpha[i] = Math.round(a*255);
  }
  const soft = blurAlpha(alpha, w, h, 1);

  /* ---- 6b. 边缘去污：反解半透明像素的前景色，贴到别的底色不再泛白边 ---- */
  const fpx = new Uint8ClampedArray(px);
  for(let y=2;y<h-2;y++) for(let x=2;x<w-2;x++){
    const i=y*w+x, a=soft[i];
    if(a < 60 || a > 246) continue;   /* 太透明的像素反解会放大噪点，直接跳过 */
    let br=0,bg=0,bb=0,n=0;
    for(let dy=-2;dy<=2;dy++) for(let dx=-2;dx<=2;dx++){
      if(!dx && !dy) continue;
      const j=(y+dy)*w+(x+dx);
      if(soft[j] >= 18) continue;
      const p=j*4; br+=px[p]; bg+=px[p+1]; bb+=px[p+2]; n++;
    }
    if(!n) continue;
    br/=n; bg/=n; bb/=n;
    const af=a/255, p=i*4;
    const fix=(c,b)=>{ const v=(c-(1-af)*b)/af; return v<0?0:(v>255?255:v); };
    fpx[p]=fix(px[p],br); fpx[p+1]=fix(px[p+1],bg); fpx[p+2]=fix(px[p+2],bb);
  }

  /* ---- 7. 裁掉空白，输出透明 PNG ---- */
  let minX=w, minY=h, maxX=-1, maxY=-1;
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    if(soft[y*w+x] > 6){
      if(x<minX) minX=x; if(x>maxX) maxX=x;
      if(y<minY) minY=y; if(y>maxY) maxY=y;
    }
  }
  if(maxX < 0){ return softFallback(img, roi, w, h); }
  const pad = Math.round(Math.max(maxX-minX, maxY-minY)*0.04);
  const cx0 = Math.max(0, minX-pad), cy0 = Math.max(0, minY-pad);
  const cx1 = Math.min(w-1, maxX+pad), cy1 = Math.min(h-1, maxY+pad);
  const outW = cx1-cx0+1, outH = cy1-cy0+1;

  const out = document.createElement('canvas');
  out.width = outW; out.height = outH;
  const octx = out.getContext('2d');
  const oid = octx.createImageData(outW, outH);
  const od = oid.data;
  for(let y=0;y<outH;y++){
    for(let x=0;x<outW;x++){
      const si = (y+cy0)*w + (x+cx0);
      const di = (y*outW + x)*4;
      od[di]   = fpx[si*4];
      od[di+1] = fpx[si*4+1];
      od[di+2] = fpx[si*4+2];
      od[di+3] = soft[si];
    }
  }
  octx.putImageData(oid, 0, 0);
  const blob = await canvasToBlob(out);
  if(!blob) throw new Error('导出失败');
  return { blob, url:URL.createObjectURL(blob), muted:false, coverage };
}

/* 抠不出干净轮廓时，用主体框做圆角+柔边（比硬切一块背景好看） */
async function softFallback(img, rect, w, h){
  const x0 = rect?rect[0]:0, y0 = rect?rect[1]:0;
  const x1 = rect?rect[2]:w-1, y1 = rect?rect[3]:h-1;
  const bw = x1-x0+1, bh = y1-y0+1;
  const cv = document.createElement('canvas'); cv.width=bw; cv.height=bh;
  const ctx = cv.getContext('2d');
  ctx.drawImage(img, x0*(img.width||img.naturalWidth)/w, y0*(img.height||img.naturalHeight)/h,
    bw*(img.width||img.naturalWidth)/w, bh*(img.height||img.naturalHeight)/h, 0, 0, bw, bh);
  const out = document.createElement('canvas'); out.width=bw; out.height=bh;
  const octx = out.getContext('2d');
  const r = Math.round(Math.min(bw,bh)*0.18);
  octx.save();
  octx.beginPath();
  if(octx.roundRect) octx.roundRect(0,0,bw,bh,r); else octx.rect(0,0,bw,bh);
  octx.clip();
  octx.drawImage(cv,0,0);
  octx.restore();
  /* 四周柔化 8px，避免硬边 */
  const feather = Math.max(4, Math.round(Math.min(bw,bh)*0.05));
  const grd = octx.createLinearGradient(0,0,0,feather);
  octx.globalCompositeOperation = 'destination-out';
  const fade = (x,y,wd,ht,horizontal)=>{
    const g = horizontal ? octx.createLinearGradient(x,y,x+wd,y) : octx.createLinearGradient(x,y,x,y+ht);
    g.addColorStop(0,'rgba(0,0,0,1)'); g.addColorStop(1,'rgba(0,0,0,0)');
    octx.fillStyle = g; octx.fillRect(x,y,wd,ht);
  };
  fade(0,0,bw,feather,false);
  fade(0,bh-feather,bw,feather,false);
  fade(0,0,feather,bh,true);
  fade(bw-feather,0,feather,bh,true);
  octx.globalCompositeOperation = 'source-over';
  const blob = await canvasToBlob(out);
  return { blob, url:URL.createObjectURL(blob), muted:true, coverage:1 };
}


async function prepareCutout(strength='auto'){
  if(!scanCtx.dataURL) return null;
  try{
    const img = await loadImage(scanCtx.dataURL);
    const c = await cutoutFromImage(img, strength, scanCtx.ai && scanCtx.ai.bbox);
    scanCtx.cutoutBlob = c.blob;
    scanCtx.cutoutURL = c.url;
    scanCtx.cutoutMuted = !!c.muted;
    scanCtx.cutStrength = strength;
    return c;
  }catch(e){
    console.warn('抠图失败', e);
    scanCtx.cutoutBlob = null;
    scanCtx.cutoutURL = null;
    return null;
  }
}

/* ---------- 生成卡片 ---------- */
async function rescanCut(strength){
  const img = document.getElementById('scanCutImg');
  if(img) img.style.opacity='.4';
  const c = await prepareCutout(strength);
  const el = document.getElementById('scanCutImg');
  if(el){ el.style.opacity='1'; if(c) el.src = c.url; }
  toast(c && !c.muted ? '抠图完成 ✅' : (c?'背景较复杂，已用柔边兜底，可点「精修边缘」手修':'抠图失败'));
}
/* 扫描流程里的精修：保存后直接应用到当前这次收服 */
async function editScanCut(){
  if(!scanCtx.dataURL) return toast('请先选择照片');
  if(!scanCtx.cutoutBlob){
    const c = await prepareCutout();
    if(!c) return toast('自动抠图失败，请换一张照片');
  }
  await openRetouch({
    title:'✂️ 精修 · 即将收服',
    blob: scanCtx.cutoutBlob,
    photoURL: scanCtx.dataURL,
    bbox: scanCtx.ai && scanCtx.ai.bbox,
    onSave: async (b)=>{
      scanCtx.cutoutBlob = b;
      scanCtx.cutoutURL = URL.createObjectURL(b);
      scanCtx.cutoutMuted = false;
      const img = document.getElementById('scanCutImg');
      if(img) img.src = scanCtx.cutoutURL;
      toast('已应用到本次收服 ✅');
    }
  });
}
function openScanForm(tip){
  const step1 = document.getElementById('scanStep1');
  const step2 = document.getElementById('scanStep2');
  if(step1) step1.style.display = 'none';
  step2.style.display = 'block';
  step2.innerHTML = `
    <div class="notice">${tip || '🤖 已自动抠图，取个名字就能收服'}</div>
    <div class="cutbox petcard">${scanCtx.dataURL?`<img id="scanCutBg" class="mini-bg" src="${scanCtx.dataURL}" alt="">`:''}<img id="scanCutImg" src="${scanCtx.cutoutURL || scanCtx.dataURL || ''}" alt="抠图预览"></div>
    <div class="tiny" style="text-align:left;margin-top:6px">↑ 卡面预览：主体会像图鉴卡一样浮在原图上${scanCtx.cutoutMuted?' · 这张背景较复杂，点下面「精修边缘」用笔刷或框选主体再处理一次':''}</div>
    ${scanCtx.dataURL?`<div class="grid3" style="margin-top:8px">
      <button class="btn sec sm" onclick="rescanCut('weak')">保守抠</button>
      <button class="btn sec sm" onclick="rescanCut('auto')">标准抠</button>
      <button class="btn sec sm" onclick="rescanCut('strong')">强力抠</button>
    </div>
    <button class="btn sec sm" style="width:100%;margin-top:8px" onclick="editScanCut()">✂️ 精修边缘（放大笔刷 / 框选主体）</button>`:''}
    <label class="fl">精灵名称（你来命名！）</label>
    <input id="scName" placeholder="给它起个名字，比如：大橘队长" maxlength="12" onkeydown="if(event.key==='Enter')saveEntry()">
    <div class="tiny" style="text-align:left;margin-top:6px">只记录名字和这张照片，其他内容都不再自动生成</div>
    <label class="fl" style="display:flex;align-items:center;gap:8px;font-size:14px;color:var(--ink)">
      <input type="checkbox" id="scGeo" checked style="width:auto"> 📍 同时在地图标记遭遇地点（使用当前位置）
    </label>
    <button class="btn" style="margin-top:14px" id="scSave" onclick="saveEntry()">Pokéball 收服！</button>`;
  document.getElementById('scName').focus();
}

async function saveEntry(){
  const name = document.getElementById('scName').value.trim();
  if(!name) return toast('给它起个名字吧！');
  const geo = document.getElementById('scGeo').checked;
  const row = {
    name,
    ai_bbox: scanCtx.ai ? scanCtx.ai.bbox || null : null,
    pet: { affinity:8, hunger:70, mood:70, lastAt:Date.now() }
  };
  busy(document.getElementById('scSave'), true, '收服中…');
  try{
    if(scanCtx.file) row.photo_path = await uploadFile(scanCtx.file, 'pokedex');
    else row.photo_path = PAW_PLACEHOLDER;
    if(scanCtx.cutoutBlob){
      row.cutout_path = await uploadFile(new File([scanCtx.cutoutBlob], 'cutout.png', {type:'image/png'}), 'cutout');
    }
    if(geo){
      const pos = await new Promise(res=>navigator.geolocation.getCurrentPosition(
        p=>res({lat:p.coords.latitude, lng:p.coords.longitude}), ()=>res(null), {timeout:6000}));
      if(pos){ row.lat = pos.lat; row.lng = pos.lng; }
    }
    const { data, error } = await dbf('pokedex').insert(row).select();
    if(error) throw error;
    toast(`🎉 收服成功！${name} 已加入图鉴`);
    closeModal(); loadPokedex();
    if(data && data[0]){
      ensureCardFace(data[0]).catch(()=>{});   /* 后台画卡面 */
      if(row.lat) setTimeout(()=>showCard(data[0].id), 300);
      else{
        /* 定位被拒绝/失败 → 手动在地图上点一下，别让遭遇地点缺席 */
        toast('未获取到定位，请在地图上标记遇到它的位置 📍');
        setTimeout(()=>openGeoPicker(data[0].id), 400);
      }
    }
  }catch(e){
    toast('收服失败：'+(e.message||e));
    busy(document.getElementById('scSave'), false);
  }
}

/* ---------- 虚拟动物互动 ---------- */
const PET_ACTS = {
  feed: { label:'🍖 喂食', lines:['好吃！','再来一口～','嚼嚼嚼…','这个味道我喜欢'] },
  pet:  { label:'🤚 摸摸', lines:['呼噜呼噜…','好舒服～','再摸摸头','眼睛都眯起来了'] },
  play: { label:'🎾 玩耍', lines:['接住！','再来一次！','好玩好玩','看我跳得多高'] }
};
const PET_IDLE = ['？','盯着你看…','打了个哈欠','歪了歪头','尾巴扫了一下'];

function petOf(d){
  const now = Date.now();
  d.pet = Object.assign({ affinity:8, hunger:70, mood:70, lastAt:now }, d.pet || {});
  const hrs = (now - (d.pet.lastAt || now)) / 3600000;
  if(hrs > 0.5){
    d.pet.hunger = Math.max(0, Math.round(d.pet.hunger - hrs*3));
    d.pet.mood = Math.max(0, Math.round(d.pet.mood - hrs*2));
    d.pet.lastAt = now;
  }
  return d.pet;
}
function petLevel(p){ return Math.max(1, 1 + Math.floor((p.affinity||0)/20)); }
function petBar(label, val, color){
  return `<div class="pet-stat"><span class="lb">${label}</span><div class="bar"><i id="pb_${label}" style="width:${Math.max(0,Math.min(100,val))}%;background:${color}"></i></div></div>`;
}
function petStageHTML(d){
  const p = petOf(d);
  const pop = !!d.cutout_path;
  const bg = d.photo_path ? `<img class="stage-photo" data-media="${esc(d.photo_path)}" alt="">` : '';
  const main = pop
    ? `<img class="cut-pop" data-media="${esc(d.cutout_path)}" alt="${esc(d.name)}">`
    : d.photo_path
      ? `<img class="cut-photo" data-media="${esc(d.photo_path)}" alt="${esc(d.name)}">`
      : `<div class="paw">🐾</div>`;
  return `
    <div class="pet-stage${pop?' pop':''}" id="petStage" onclick="petTap(${d.id})">
      <div class="stage-bg">${bg}<div class="stage-dim"></div></div>
      <div class="pet-bubble" id="petBubble">…</div>
      ${main}
      <div id="petLv" style="position:absolute;left:10px;top:8px;font-size:11px;font-weight:700;color:#7a4a1f;background:#ffffffaa;border-radius:8px;padding:3px 7px;z-index:3">Lv.${petLevel(p)} · 亲密度 ${p.affinity}</div>
    </div>
    <div class="pet-acts">
      ${Object.entries(PET_ACTS).map(([k,v])=>`<button onclick="event.stopPropagation();petAction(${d.id},'${k}')">${v.label}</button>`).join('')}
    </div>
    <div id="petStats">
      ${petBar('亲密度', p.affinity, '#f472b6')}
      ${petBar('饱食度', p.hunger, '#f59e0b')}
      ${petBar('心情', p.mood, '#22c55e')}
    </div>`;
}
function showPetBubble(text){
  const b = document.getElementById('petBubble');
  if(!b) return;
  b.textContent = text;
  b.classList.add('on');
  clearTimeout(b._t);
  b._t = setTimeout(()=>b.classList.remove('on'), 1800);
}
function heartBurst(){
  const stage = document.getElementById('petStage');
  if(!stage) return;
  for(let i=0;i<3;i++){
    const h = document.createElement('span');
    h.className = 'heart';
    h.textContent = ['❤️','✨','💛'][i%3];
    h.style.left = (35 + Math.random()*30) + '%';
    h.style.top = '55%';
    h.style.animationDelay = (i*0.08)+'s';
    stage.appendChild(h);
    setTimeout(()=>h.remove(), 1400);
  }
}
function petTap(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  const p = petOf(d);
  p.mood = Math.min(100, p.mood + 1);
  p.affinity += 1;
  p.lastAt = Date.now();
  showPetBubble(PET_IDLE[Math.floor(Math.random()*PET_IDLE.length)]);
  heartBurst();
  updatePetBars(d);
  dbf('pokedex').update({ pet:p }).eq('id', id).select().catch(()=>{});
}
async function petAction(id, kind){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  const p = petOf(d);
  const act = PET_ACTS[kind]; if(!act) return;
  if(kind==='feed'){ p.hunger = Math.min(100, p.hunger+18); p.mood = Math.min(100, p.mood+6); p.affinity += 2; }
  if(kind==='pet'){ p.mood = Math.min(100, p.mood+10); p.affinity += 3; }
  if(kind==='play'){ p.mood = Math.min(100, p.mood+14); p.hunger = Math.max(0, p.hunger-8); p.affinity += 4; }
  p.lastAt = Date.now();
  showPetBubble(act.lines[Math.floor(Math.random()*act.lines.length)]);
  heartBurst();
  const stage = document.getElementById('petStage');
  if(stage){ stage.classList.add('bounce'); setTimeout(()=>stage.classList.remove('bounce'), 650); }
  updatePetBars(d);
  await dbf('pokedex').update({ pet:p }).eq('id', id).select().catch(()=>{});
}
function updatePetBars(d){
  const p = petOf(d);
  [['亲密度',p.affinity],['饱食度',p.hunger],['心情',p.mood]].forEach(([k,v])=>{
    const el = document.getElementById('pb_'+k);
    if(el) el.style.width = Math.max(0, Math.min(100, v)) + '%';
  });
  const lv = document.getElementById('petLv');
  if(lv) lv.textContent = `Lv.${petLevel(p)} · 亲密度 ${p.affinity}`;
}

/* ---------- 抠图保存 / 重新抠图 ---------- */
async function saveCutout(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  if(!d.cutout_path) return toast('还没有抠图，先生成一次');
  try{
    const url = await mediaURL(d.cutout_path);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${d.name}-抠图.png`;
    document.body.appendChild(a); a.click(); a.remove();
    toast('已保存透明 PNG 🖼️');
  }catch(e){ toast('保存失败：'+(e.message||e)); }
}
function openRecut(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  if(!d.photo_path && !d.cutout_path) return toast('这只精灵没有图片');
  (async()=>{
    let blob = null, photoURL = null;
    try{
      if(d.photo_path) photoURL = coverSrc(await mediaURL(d.photo_path));
      if(d.cutout_path) blob = await (await fetch(coverSrc(await mediaURL(d.cutout_path)))).blob();
      if(!blob){
        if(!photoURL) throw new Error('缺少原图');
        const img = await loadImage(photoURL);
        blob = (await cutoutFromImage(img, 'auto', d.ai_bbox)).blob;
      }
    }catch(e){ return toast('图片读取失败：'+(e.message||e)); }
    await openRetouch({
      title:'✂️ 修图 · '+d.name, blob, photoURL, bbox:d.ai_bbox,
      onSave: async (b)=>{
        const path = await uploadFile(new File([b], 'cutout.png', {type:'image/png'}), 'cutout');
        await dbf('pokedex').update({ cutout_path:path }).eq('id', d.id).select();
        d.cutout_path = path;
        toast('已保存抠图 ✅ 正在重画卡面…');
        await ensureCardFace(d, true);   /* 抠图变了 → 卡面重画 */
        await loadPokedex(true);
        showCard(d.id);
      }
    });
  })();
}
/* ================= 卡面合成（水彩插画背景 + 主体破框 + 名字牌） ================= */
const CARD_W = 750, CARD_H = 1050;

function rr(ctx,x,y,w,h,r){
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r);
  ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath();
}

/* 照片 → 水彩插画风：柔焦 + 色块量化 + 纸纹 + 暖光暗角 */
function paintWatercolor(ctx, img, x, y, w, h){
  const t = document.createElement('canvas');
  t.width = Math.max(2, Math.round(w*0.5)); t.height = Math.max(2, Math.round(h*0.5));
  const tc = t.getContext('2d');
  tc.filter = 'blur(2px) saturate(1.4) brightness(1.1) contrast(0.9)';
  tc.drawImage(img, 0, 0, t.width, t.height);
  const id = tc.getImageData(0,0,t.width,t.height), px = id.data, step = 22;
  for(let i=0;i<px.length;i+=4){
    px[i]   = Math.round(px[i]/step)*step;
    px[i+1] = Math.round(px[i+1]/step)*step;
    px[i+2] = Math.round(px[i+2]/step)*step;
  }
  tc.putImageData(id,0,0);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(t, x, y, w, h);
  /* 纸纹 */
  const n = document.createElement('canvas'); n.width = 140; n.height = 140;
  const nc = n.getContext('2d'), nd = nc.createImageData(140,140);
  for(let i=0;i<nd.data.length;i+=4){
    const v = 195 + Math.random()*60;
    nd.data[i]=v; nd.data[i+1]=v-6; nd.data[i+2]=v-18; nd.data[i+3]=13;
  }
  nc.putImageData(nd,0,0);
  ctx.fillStyle = ctx.createPattern(n,'repeat'); ctx.fillRect(x,y,w,h);
  /* 暖光 + 暗角 */
  const g = ctx.createRadialGradient(x+w/2, y+h*0.32, 24, x+w/2, y+h*0.45, w*0.78);
  g.addColorStop(0,'rgba(255,242,204,0.22)'); g.addColorStop(1,'rgba(115,78,30,0.20)');
  ctx.fillStyle = g; ctx.fillRect(x,y,w,h);
  ctx.restore();
}

/* 主体抠图 + 白色柔边（先白晕描边再画本体） */
function drawCutoutPop(ctx, img, cx, bottomY, maxH, maxW){
  const s = Math.min(maxH/img.height, maxW/img.width);
  const w = img.width*s, h = img.height*s;
  const x = cx - w/2, y = bottomY - h;
  ctx.save();
  ctx.shadowColor = 'rgba(255,255,255,0.95)'; ctx.shadowBlur = 7;
  for(let i=0;i<2;i++) ctx.drawImage(img, x, y, w, h);
  ctx.shadowColor = 'rgba(90,50,10,0.35)'; ctx.shadowBlur = 18; ctx.shadowOffsetY = 10;
  ctx.drawImage(img, x, y, w, h);
  ctx.restore();
  return {x, y, w, h};
}

/* 斜纹底（记录区） */
function paintHatch(ctx, x, y, w, h){
  ctx.save();
  rr(ctx,x,y,w,h,16); ctx.clip();
  ctx.fillStyle = '#f6efdd'; ctx.fillRect(x,y,w,h);
  ctx.strokeStyle = 'rgba(190,160,110,0.28)'; ctx.lineWidth = 7;
  for(let i=-h;i<w+h;i+=26){
    ctx.beginPath(); ctx.moveTo(x+i,y+h+8); ctx.lineTo(x+i+h+16,y-8); ctx.stroke();
  }
  ctx.restore();
  ctx.save(); rr(ctx,x,y,w,h,16);
  ctx.strokeStyle = '#3a3428'; ctx.lineWidth = 2.5; ctx.stroke(); ctx.restore();
}

async function makeCardFace(d){
  if(!d.photo_path) return null;
  const photo = await loadImage(coverSrc(await mediaURL(d.photo_path)));
  let cut = null;
  try{ if(d.cutout_path) cut = await loadImage(coverSrc(await mediaURL(d.cutout_path))); }catch(e){}
  const cv = document.createElement('canvas'); cv.width = CARD_W; cv.height = CARD_H;
  const ctx = cv.getContext('2d');
  const ink = '#3a3428';

  /* 卡底 */
  rr(ctx,0,0,CARD_W,CARD_H,30); ctx.fillStyle='#fbf6ea'; ctx.fill();
  rr(ctx,14,14,CARD_W-28,CARD_H-28,22);
  ctx.strokeStyle = ink; ctx.lineWidth = 1.6; ctx.setLineDash([2,7]); ctx.stroke(); ctx.setLineDash([]);

  /* 头图：水彩背景 + 主体破框 */
  const HX=44, HY=44, HW=CARD_W-88, HH=560;
  ctx.save(); rr(ctx,HX,HY,HW,HH,18); ctx.clip();
  paintWatercolor(ctx, photo, HX, HY, HW, HH);
  ctx.restore();
  rr(ctx,HX,HY,HW,HH,18); ctx.strokeStyle = ink; ctx.lineWidth = 2.5; ctx.stroke();
  if(cut){
    drawCutoutPop(ctx, cut, HX+HW/2, HY+HH-8, HH*0.98, HW*0.86);
  }else{
    ctx.save(); rr(ctx,HX+10,HY+10,HW-20,HH-20,12); ctx.clip();
    ctx.drawImage(photo, HX, HY, HW, HH); ctx.restore();
  }

  /* 名字牌（手写感，压在头图下缘） */
  const name = (d.name||'神秘生物').slice(0,12);
  ctx.font = "bold 44px 'KaiTi','Kaiti SC','STKaiti',serif";
  const nw = ctx.measureText(name).width;
  const pbW = nw + 92, pbH = 64, pbX = CARD_W/2 - pbW/2, pbY = HY + HH - 34;
  rr(ctx, pbX, pbY, pbW, pbH, 12);
  ctx.fillStyle = '#fffdf4'; ctx.fill();
  ctx.strokeStyle = ink; ctx.lineWidth = 2.5; ctx.stroke();
  ctx.fillStyle = ink; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(name, CARD_W/2 - 10, pbY + pbH/2 + 2);
  ctx.font = '26px sans-serif';
  ctx.fillText('🐾', pbX + pbW - 30, pbY + pbH/2 + 2);

  /* 遭遇记录（只保留事实信息，不再有任何自动生成的属性） */
  let y = pbY + pbH + 26;
  ctx.fillStyle = '#e07a2f'; rr(ctx, 52, y+4, 7, 30, 3); ctx.fill();
  ctx.fillStyle = ink; ctx.textAlign='left'; ctx.textBaseline='alphabetic';
  ctx.font = "bold 30px 'KaiTi','Kaiti SC','STKaiti',serif";
  ctx.fillText('遭遇记录', 70, y + 30);
  y += 52;

  paintHatch(ctx, 44, y, CARD_W-88, 250);
  const inX = 76, inR = CARD_W - 76;
  const dt = new Date(d.created_at || Date.now());
  const pad2 = n => String(n).padStart(2,'0');
  const dateStr = `${dt.getFullYear()}.${pad2(dt.getMonth()+1)}.${pad2(dt.getDate())}`;
  const timeStr = `${pad2(dt.getHours())}:${pad2(dt.getMinutes())}`;
  const geoStr = d.lat ? `${(+d.lat).toFixed(4)}, ${(+d.lng).toFixed(4)}` : '尚未标记';

  const recRow = (rowY, emoji, label, value)=>{
    ctx.textAlign='left'; ctx.textBaseline='alphabetic';
    ctx.font = '24px sans-serif'; ctx.fillStyle = '#6b6250';
    ctx.fillText(emoji, inX, rowY);
    ctx.font = 'bold 20px "PingFang SC","Microsoft YaHei",sans-serif';
    ctx.fillText(label, inX + 46, rowY);
    ctx.font = 'bold 27px "PingFang SC","Microsoft YaHei",sans-serif';
    ctx.fillStyle = ink; ctx.textAlign='right';
    ctx.fillText(value, inR, rowY);
    ctx.textAlign='left';
  };
  recRow(y + 92, '📅', '收服时间', `${dateStr}  ${timeStr}`);
  /* 虚线分隔 */
  ctx.beginPath(); ctx.moveTo(inX, y + 128); ctx.lineTo(inR, y + 128);
  ctx.strokeStyle = 'rgba(58,52,40,.28)'; ctx.lineWidth = 1.6;
  ctx.setLineDash([4,7]); ctx.stroke(); ctx.setLineDash([]);
  recRow(y + 188, '📍', '遭遇地点', geoStr);

  /* 底部小字 */
  ctx.font = '16px "PingFang SC","Microsoft YaHei",sans-serif'; ctx.fillStyle = '#8a8070';
  ctx.textAlign='center';
  ctx.fillText('🐾 现实版宝可梦 · 我的图鉴', CARD_W/2, CARD_H - 30);
  ctx.textAlign='left';
  return await canvasToBlob(cv);
}

/* 生成并保存卡面；force=true 时强制重画 */
async function ensureCardFace(d, force){
  try{
    if(!force && d.card_path) return d.card_path;
    const blob = await makeCardFace(d);
    if(!blob) return null;
    const path = await uploadFile(new File([blob], 'card.png', {type:'image/png'}), 'card');
    d.card_path = path;
    dbf('pokedex').update({ card_path: path }).eq('id', d.id).select().catch(()=>{});
    return path;
  }catch(e){ console.warn('卡面生成失败', e); return null; }
}

/* ================= 修图工作台（擦除 / 恢复 / 撤销 / 缩放 / 框选重抠） ================= */
let ED = null;
let _edScratch = null;
const ED_PASSES = [[1,1],[0.72,0.55],[0.46,0.3]];

function edEl(id){ return document.getElementById(id); }
function edMsg(txt){
  const s = edEl('edStage'); if(!s) return;
  let m = edEl('edLoading');
  if(!m){ m = document.createElement('div'); m.className='ed-loading'; m.id='edLoading'; s.appendChild(m); }
  m.textContent = txt || '';
  m.style.display = txt ? 'flex' : 'none';
}
function edScratch(w,h){
  if(!_edScratch) _edScratch = document.createElement('canvas');
  if(_edScratch.width!==w || _edScratch.height!==h){ _edScratch.width=w; _edScratch.height=h; }
  return _edScratch;
}

async function openRetouch(opts){
  const old = edEl('edMask'); if(old) old.remove();
  const mask = document.createElement('div');
  mask.className = 'ed-mask'; mask.id = 'edMask';
  mask.innerHTML = `
    <div class="ed-head">
      <b>${esc(opts.title || '✂️ 修图')}</b>
      <button class="x2" onclick="edClose()">✕</button>
    </div>
    <div class="ed-stage" id="edStage"><canvas id="edCanvas"></canvas></div>
    <div class="ed-bar">
      <div class="ed-tools">
        <button class="ed-t on" id="edTErase" onclick="edTool('erase')">🧽 擦除</button>
        <button class="ed-t" id="edTRestore" onclick="edTool('restore')">🖌 恢复</button>
        <button class="ed-t" id="edTMove" onclick="edTool('move')">✋ 移动</button>
        <button class="ed-t" onclick="edUndo()">↶ 撤销</button>
        <button class="ed-t" onclick="edReset()">🗑 重来</button>
        <button class="ed-t" onclick="edTighten()">✂️ 自动修边</button>
        ${opts.photoURL ? '<button class="ed-t" onclick="edBBox()">🎯 框选主体</button>' : ''}
      </div>
      <div class="ed-tools" style="margin-top:6px;align-items:center">
        <button class="ed-t" onclick="edZoom(-1)">🔍 缩小</button>
        <span class="ed-chip" id="edZoomTxt">100%</span>
        <button class="ed-t" onclick="edZoom(1)">🔍 放大</button>
        <span class="ed-chip" id="edStrokeTxt">0 笔</span>
        <button class="ed-t" onclick="edRecut('weak')">轻抠</button>
        <button class="ed-t" onclick="edRecut('auto')">标准抠</button>
        <button class="ed-t" onclick="edRecut('strong')">强抠</button>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin-top:10px">
        <span class="sub" style="flex:none">笔刷 <b id="edSizeTxt">34</b>px</span>
        <input type="range" min="4" max="90" step="2" value="34" style="flex:1;padding:0;border:none;background:none" oninput="edSize(this.value)">
      </div>
      <div class="grid2" style="margin-top:10px">
        <button class="btn" id="edSaveBtn" onclick="edSave()">💾 保存</button>
        <button class="btn sec" onclick="edDownload()">⬇ 下载 PNG</button>
      </div>
      <div class="tiny" style="margin-top:8px">按住拖动修边 · 放大后用「移动」平移 · 松手后仍可撤销</div>
    </div>`;
  document.body.appendChild(mask);
  const cur = document.createElement('div');
  cur.className = 'ed-cursor'; cur.id = 'edCursor';
  document.body.appendChild(cur);

  ED = { baseBmp:null, origBmp:null, photoBmp:null, strokes:[], tool:'erase', size:34,
         zoom:1, cv:null, ctx:null, drawing:false, cur:null, pan:null,
         mode:'brush', bbox:opts.bbox||null, rectDrag:null, onSave:opts.onSave, photoURL:opts.photoURL };
  const cv = edEl('edCanvas');
  ED.cv = cv; ED.ctx = cv.getContext('2d');
  cv.addEventListener('pointerdown', edDown);
  cv.addEventListener('pointermove', edMove);
  cv.addEventListener('pointerup', edUp);
  cv.addEventListener('pointercancel', edUp);
  cv.addEventListener('pointerleave', ()=>{ const c = edEl('edCursor'); if(c) c.style.display='none'; });

  edMsg('正在载入…');
  try{
    ED.baseBmp = await createImageBitmap(opts.blob);
    ED.origBmp = ED.baseBmp;
  }catch(e){ edMsg('图片载入失败'); return; }
  if(opts.photoURL){
    try{ ED.photoBmp = await createImageBitmap(await (await fetch(opts.photoURL)).blob()); }
    catch(e){ ED.photoBmp = null; }
  }
  edMsg('');
  edSetCanvas(ED.baseBmp.width, ED.baseBmp.height);
  edRender();
  edTool('erase'); edSize(34);
}

function edClose(){
  const m = edEl('edMask'); if(m) m.remove();
  const c = edEl('edCursor'); if(c) c.remove();
  ED = null;
}
function edSetCanvas(w,h){
  if(!ED) return;
  ED.cv.width = w; ED.cv.height = h;
  ED.natW = w; ED.natH = h;
  const st = edEl('edStage');
  const avail = Math.max(160, (st ? st.clientWidth : 420) - 20);
  ED.baseW = Math.min(w, avail);        /* 小图按 1:1 显示，不糊 */
  edApplyZoom();
}
function edApplyZoom(){
  if(!ED) return;
  ED.cv.style.width = Math.round((ED.baseW||ED.cv.width) * ED.zoom)+'px';
  ED.cv.style.height = 'auto';
  const t = edEl('edZoomTxt'); if(t) t.textContent = Math.round(ED.zoom*100)+'%';
}
function edRender(){
  if(!ED || !ED.baseBmp) return;
  const {cv,ctx} = ED;
  ctx.setTransform(1,0,0,1,0,0);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(ED.baseBmp,0,0);
  for(const s of ED.strokes) edPaintStroke(s);
  edSyncCount();
}
function edSyncCount(){
  const t = edEl('edStrokeTxt');
  if(t && ED) t.textContent = ED.strokes.length + ' 笔';
}
/* 沿点集画一笔（单点画圆） */
function edPath(ctx, pts, lw){
  ctx.lineWidth = Math.max(0.5, lw);
  ctx.beginPath();
  if(pts.length === 1){
    ctx.arc(pts[0].x, pts[0].y, Math.max(0.5, lw/2), 0, Math.PI*2);
    ctx.fill();
    return;
  }
  ctx.moveTo(pts[0].x, pts[0].y);
  for(let i=1;i<pts.length-1;i++){
    const a=pts[i], b=pts[i+1];
    ctx.quadraticCurveTo(a.x, a.y, (a.x+b.x)/2, (a.y+b.y)/2);
  }
  const last = pts[pts.length-1];
  ctx.lineTo(last.x, last.y);
  ctx.stroke();
}
/* 完整重绘一笔：擦除=destination-out；恢复=用原始底图裁剪回笔迹 */
function edPaintStroke(s){
  if(!s || !s.pts.length) return;
  const {cv,ctx} = ED;
  if(s.mode === 'erase'){
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000'; ctx.fillStyle = '#000';
    for(const [k,a] of ED_PASSES){
      ctx.globalAlpha = a;
      edPath(ctx, s.pts, Math.max(1, s.w*k));
    }
    ctx.restore();
  }else{
    const sc = edScratch(cv.width, cv.height), sx = sc.getContext('2d');
    sx.setTransform(1,0,0,1,0,0);
    sx.globalCompositeOperation = 'source-over'; sx.globalAlpha = 1;
    sx.clearRect(0,0,sc.width,sc.height);
    sx.drawImage(ED.baseBmp,0,0);
    sx.globalCompositeOperation = 'destination-in';
    sx.lineCap='round'; sx.lineJoin='round'; sx.strokeStyle='#000'; sx.fillStyle='#000';
    edPath(sx, s.pts, Math.max(1, s.w));
    sx.globalCompositeOperation = 'source-over';
    ctx.save(); ctx.globalCompositeOperation='source-over'; ctx.globalAlpha=1;
    ctx.drawImage(sc,0,0); ctx.restore();
  }
}
/* 拖动过程中的增量绘制（松手后会整体重绘，边缘更干净） */
function edDrawSeg(s, from, to){
  const {ctx} = ED;
  const lw = Math.max(1, s.w);
  if(s.mode === 'erase'){
    ctx.save(); ctx.globalCompositeOperation='destination-out';
    ctx.lineCap='round'; ctx.lineJoin='round'; ctx.strokeStyle='#000';
    for(const [k,a] of ED_PASSES){
      ctx.globalAlpha = a; ctx.lineWidth = Math.max(1, lw*k);
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
    }
    ctx.restore();
  }else{
    edPaintStroke(s);
  }
}

function edPos(e){
  const cv = ED.cv, r = cv.getBoundingClientRect();
  const k = r.width ? cv.width/r.width : 1;
  return { x:(e.clientX-r.left)*k, y:(e.clientY-r.top)*(r.height ? cv.height/r.height : 1), k };
}
function edDown(e){
  if(!ED) return;
  if(ED.mode === 'bbox'){ edBBoxStart(e); return; }
  if(ED.tool === 'move'){
    const st = edEl('edStage');
    ED.pan = { x:e.clientX, y:e.clientY, l:st.scrollLeft, t:st.scrollTop };
    return;
  }
  if(!ED.baseBmp) return;
  const p = edPos(e);
  ED.cur = { mode:ED.tool, w:Math.max(1, ED.size*p.k), pts:[{x:p.x, y:p.y}] };
  ED.drawing = true;
  try{ ED.cv.setPointerCapture(e.pointerId); }catch(_){}
  edPaintStroke(ED.cur);
  e.preventDefault();
}
function edMove(e){
  if(!ED) return;
  const c = edEl('edCursor');
  if(ED.pan){
    const st = edEl('edStage');
    st.scrollLeft = ED.pan.l - (e.clientX - ED.pan.x);
    st.scrollTop  = ED.pan.t - (e.clientY - ED.pan.y);
    e.preventDefault(); return;
  }
  if(ED.mode === 'bbox'){ if(ED.rectDrag) edBBoxMove(e); return; }
  if(c && ED.baseBmp && ED.tool !== 'move'){
    const r = ED.cv.getBoundingClientRect();
    if(e.clientX>=r.left && e.clientX<=r.right && e.clientY>=r.top && e.clientY<=r.bottom){
      c.style.display='block';
      c.style.width = ED.size+'px'; c.style.height = ED.size+'px';
      c.style.left = (e.clientX - ED.size/2)+'px'; c.style.top = (e.clientY - ED.size/2)+'px';
    }else{ c.style.display='none'; }
  }
  if(!ED.drawing || !ED.cur) return;
  const p = edPos(e), pts = ED.cur.pts, last = pts[pts.length-1];
  if(Math.hypot(p.x-last.x, p.y-last.y) < Math.max(1, ED.cur.w*0.14)) return;
  pts.push({x:p.x, y:p.y});
  edDrawSeg(ED.cur, last, {x:p.x, y:p.y});
  e.preventDefault();
}
function edUp(e){
  if(!ED) return;
  if(ED.mode === 'bbox'){ edBBoxEnd(e); return; }
  if(ED.pan){ ED.pan = null; return; }
  if(!ED.drawing) return;
  ED.drawing = false;
  if(ED.cur && ED.cur.pts.length){
    ED.strokes.push(ED.cur);
    ED.cur = null;
    edRender();
  }
  try{ ED.cv.releasePointerCapture(e.pointerId); }catch(_){}
}

function edTool(t){
  if(!ED) return;
  ED.tool = t;
  [['edTErase','erase'],['edTRestore','restore'],['edTMove','move']].forEach(([id,k])=>{
    const b = edEl(id); if(b) b.classList.toggle('on', k===t);
  });
  ED.cv.style.cursor = t==='move' ? 'grab' : 'crosshair';
  const c = edEl('edCursor'); if(c && t==='move') c.style.display='none';
}
function edSize(v){ if(!ED) return; ED.size = +v; const t = edEl('edSizeTxt'); if(t) t.textContent = v; }
function edZoom(dir){
  if(!ED) return;
  ED.zoom = Math.max(1, Math.min(4, ED.zoom + (dir>0?1:-1)));
  edApplyZoom();
}
function edUndo(){
  if(!ED) return;
  if(ED.mode === 'bbox') return toast('先框选主体完成重抠，再撤销笔划');
  if(!ED.strokes.length) return toast('没有可撤销的笔划');
  ED.strokes.pop(); edRender();
}
function edReset(){
  if(!ED) return;
  if(ED.mode === 'bbox') return toast('先框选主体完成重抠');
  ED.strokes = []; ED.baseBmp = ED.origBmp; edRender();
  toast('已回到自动抠图结果');
}
/* 一键收紧 1px 边缘：去掉背景残留的描边 */
async function edTighten(){
  if(!ED || !ED.baseBmp) return;
  const w = ED.baseBmp.width, h = ED.baseBmp.height;
  const t = document.createElement('canvas'); t.width=w; t.height=h;
  const tc = t.getContext('2d', {willReadFrequently:true});
  tc.drawImage(ED.baseBmp,0,0);
  const im = tc.getImageData(0,0,w,h), a = im.data;
  const src = new Uint8Array(w*h);
  for(let i=0;i<w*h;i++) src[i] = a[i*4+3];
  const er = new Uint8Array(w*h);
  for(let y=0;y<h;y++) for(let x=0;x<w;x++){
    let m = 255;
    for(let dy=-1;dy<=1;dy++){
      const yy=y+dy; if(yy<0||yy>=h) continue;
      for(let dx=-1;dx<=1;dx++){
        const xx=x+dx; if(xx<0||xx>=w) continue;
        const v = src[yy*w+xx]; if(v<m) m=v;
      }
    }
    er[y*w+x] = m;
  }
  const sm = blurAlpha(er, w, h, 1);
  for(let i=0;i<w*h;i++) a[i*4+3] = sm[i];
  tc.putImageData(im,0,0);
  try{ ED.baseBmp = await createImageBitmap(t); }catch(e){ return toast('处理失败'); }
  ED.strokes = []; edRender();
  toast('边缘已收紧 1px ✅');
}

/* 用原图重新抠一版（可带框） */
async function edRecut(strength, bbox){
  if(!ED) return;
  if(!ED.photoBmp) return toast('缺少原图，只能手动修边');
  edMsg('正在重新抠图…');
  try{
    const c = await cutoutFromImage(ED.photoBmp, strength, bbox || ED.bbox);
    ED.baseBmp = await createImageBitmap(c.blob);
    ED.origBmp = ED.baseBmp;
    ED.strokes = []; ED.zoom = 1; ED.mode = 'brush';
    edSetCanvas(ED.baseBmp.width, ED.baseBmp.height);
    const z = edEl('edZoomTxt'); if(z) z.textContent='100%';
    edRender();
    toast(c.muted ? '背景较复杂，可用笔刷修一下' : '重新抠图完成 ✅');
  }catch(e){ toast('抠图失败：'+(e.message||e)); }
  finally{ edMsg(''); }
}

/* 框选主体：框外一律当背景，比 AI 猜的框更准 */
function edBBox(){
  if(!ED || !ED.photoBmp) return toast('缺少原图，无法框选');
  ED.mode = 'bbox'; ED.strokes = []; ED.zoom = 1; ED.rectDrag = null;
  edSetCanvas(ED.photoBmp.width, ED.photoBmp.height);
  const z = edEl('edZoomTxt'); if(z) z.textContent='100%';
  const c = edEl('edCursor'); if(c) c.style.display='none';
  edBBoxPaint();
  toast('在原图上拖出一个框，圈住主体');
}
function edBBoxPaint(){
  if(!ED || !ED.photoBmp) return;
  const {ctx,cv} = ED, d = ED.rectDrag;
  ctx.setTransform(1,0,0,1,0,0);
  ctx.globalCompositeOperation='source-over'; ctx.globalAlpha=1;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(ED.photoBmp,0,0);
  if(!d) return;
  const x=Math.min(d.x0,d.x1), y=Math.min(d.y0,d.y1);
  const w=Math.abs(d.x1-d.x0), h=Math.abs(d.y1-d.y0);
  ctx.save();
  ctx.fillStyle='rgba(0,0,0,.38)';
  ctx.beginPath(); ctx.rect(0,0,cv.width,cv.height); ctx.rect(x,y,w,h);
  ctx.fill('evenodd');
  ctx.strokeStyle='#3a9e6e'; ctx.lineWidth=Math.max(2, cv.width/320);
  ctx.strokeRect(x,y,w,h);
  ctx.restore();
}
function edBBoxStart(e){
  const p = edPos(e);
  ED.rectDrag = {x0:p.x, y0:p.y, x1:p.x, y1:p.y};
  try{ ED.cv.setPointerCapture(e.pointerId); }catch(_){}
  edBBoxPaint();
  e.preventDefault();
}
function edBBoxMove(e){
  const p = edPos(e);
  ED.rectDrag.x1 = p.x; ED.rectDrag.y1 = p.y;
  edBBoxPaint();
  e.preventDefault();
}
function edBBoxEnd(e){
  if(!ED || !ED.rectDrag) return;
  const cv = ED.cv, d = ED.rectDrag; ED.rectDrag = null;
  const w = Math.abs(d.x1-d.x0), h = Math.abs(d.y1-d.y0);
  if(w < cv.width*0.05 || h < cv.height*0.05){
    toast('框太小了，重新画一个'); edBBoxPaint(); return;
  }
  const bbox = [Math.min(d.x0,d.x1)/cv.width, Math.min(d.y0,d.y1)/cv.height, w/cv.width, h/cv.height];
  ED.bbox = bbox;
  edRecut('auto', bbox);
}

async function edSave(){
  if(!ED) return;
  if(ED.mode === 'bbox') return toast('请先框选主体，等重抠完成后再保存');
  const btn = edEl('edSaveBtn'); busy(btn, true, '保存中…');
  try{
    const blob = await new Promise(r=>ED.cv.toBlob(r,'image/png'));
    if(!blob) throw new Error('导出失败');
    await ED.onSave(blob);
    edClose();
  }catch(e){ toast('保存失败：'+(e.message||e)); busy(btn, false); }
}
async function edDownload(){
  if(!ED) return;
  const blob = await new Promise(r=>ED.cv.toBlob(r,'image/png'));
  if(!blob) return toast('导出失败');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '抠图.png';
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(()=>URL.revokeObjectURL(a.href), 4000);
  toast('已保存透明 PNG 🖼️');
}

/* ================= 地图 ================= */
let mapObj=null, markerLayer=null;
function pawStyles(){
  return { paw:new TMap.MarkerStyle({ width:30, height:38, anchor:{x:15,y:38}, src:PAW_SVG }) };
}
function initMapTab(){
  const box=document.getElementById('mapTab');
  if(typeof TMap==='undefined'){
    box.innerHTML='<div class="loading">地图加载中…</div>';
    let n=0;
    const t=setInterval(()=>{
      if(typeof TMap!=='undefined'){ clearInterval(t); initMapTab(); }
      else if(++n>12){ clearInterval(t); box.innerHTML='<div class="empty"><span class="big">🗺️</span>地图组件加载失败，请检查网络后刷新<br><span class="sub" style="margin-top:8px;display:block">遭遇地点仍保存在图鉴里，可点开卡片查看坐标</span></div>'; renderMapList(); }
    }, 400);
    return;
  }
  if(!mapObj){
    mapObj = new TMap.Map('mapTab', { zoom:12, center:new TMap.LatLng(31.2304,121.4737) });
    markerLayer = new TMap.MultiMarker({ map:mapObj, styles:pawStyles() });
    markerLayer.on('click', async (evt)=>{
      const id=evt.geometry?.id; const d=S.dex.find(x=>String(x.id)===String(id));
      if(d) showCard(d.id);
    });
  }
  refreshMapMarkers();
}
function refreshMapMarkers(){
  if(!markerLayer||!mapObj) { renderMapList(); return; }
  const pts = S.dex.filter(d=>d.lat&&d.lng).map(d=>({
    id:String(d.id), styleId:'paw', position:new TMap.LatLng(+d.lat,+d.lng), properties:{name:d.name}
  }));
  markerLayer.setGeometries(pts);
  document.getElementById('mapCount').textContent = `${pts.length} 个遭遇地点 · ${S.dex.length-pts.length} 个未标记`;
  renderMapList();
  if(pts.length){
    const last=pts[0];
    if(!mapObj._fitDone){ mapObj.setCenter(last.position); mapObj.setZoom(14); mapObj._fitDone=true; }
  }
}
/* 地图下方列表：即使地图瓦片加载失败，位置信息也始终可见 */
function renderMapList(){
  const el=document.getElementById('mapList'); if(!el) return;
  const marked=S.dex.filter(d=>d.lat&&d.lng), unmarked=S.dex.filter(d=>!(d.lat&&d.lng));
  el.innerHTML =
    (marked.length?`<div class="card" style="padding:10px 14px;margin-bottom:10px"><b style="font-size:13px">🐾 最近遭遇</b>
      ${marked.map(d=>`<div class="h-row" style="padding:7px 0;border-bottom:1px solid var(--line);cursor:pointer" onclick="showCard(${d.id})">
        <span style="font-size:15px">${esc(d.name)}</span>
        <span class="sub">${(+d.lat).toFixed(4)}, ${(+d.lng).toFixed(4)}</span>
        <span class="sub" style="margin-left:auto">${fmtTime(d.created_at)}</span></div>`).join('')}</div>`:'')
  + (unmarked.length?`<div class="card" style="padding:10px 14px"><b style="font-size:13px">📍 未标记的伙伴</b>
      ${unmarked.map(d=>`<div class="h-row" style="padding:7px 0;border-bottom:1px solid var(--line)">
        <span style="font-size:15px">${esc(d.name)}</span>
        <button class="btn sm" style="margin-left:auto" onclick="openGeoPicker(${d.id})">补标记</button></div>`).join('')}</div>`:'')
  + (!S.dex.length?'<div class="empty"><span class="big">🐾</span>还没有精灵，先去图鉴扫描一只吧</div>':'');
}

/* ---------- 手动选点 / 补标记位置 ---------- */
let geoPick=null;   // { map, layer, picked }
function openGeoPicker(id){
  const d = S.dex.find(x=>x.id===id); if(!d) return;
  geoPick = { picked: d.lat? {lat:+d.lat, lng:+d.lng} : null, map:null, layer:null };
  openModal(`
    <h3>📍 ${d.lat?'调整':'标记'}「${esc(d.name)}」的遭遇地点</h3>
    <div class="sub">在地图上点击选点${d.lat?'，爪印会移动到新位置':'，或直接使用手机当前定位'}</div>
    <div id="geoMap" style="height:290px;border-radius:14px;overflow:hidden;margin:10px 0;border:1.5px solid var(--line)"></div>
    <div class="sub" id="geoTxt">${geoPick.picked?`当前：${geoPick.picked.lat.toFixed(5)}, ${geoPick.picked.lng.toFixed(5)}`:'未选择位置'}</div>
    <div class="grid2" style="margin-top:10px">
      <button class="btn sec sm" onclick="geoUseCurrent()">📱 用当前位置</button>
      <button class="btn sm" onclick="geoSave(${id})">💾 保存位置</button>
    </div>`);
  setTimeout(()=>{
    const box=document.getElementById('geoMap'); if(!box) return;
    if(typeof TMap==='undefined'){ box.innerHTML='<div class="empty" style="padding:60px 10px">地图组件加载失败<br>请改用「用当前位置」</div>'; return; }
    const center = geoPick.picked? new TMap.LatLng(geoPick.picked.lat,geoPick.picked.lng)
      : (mapObj? mapObj.getCenter() : new TMap.LatLng(31.2304,121.4737));
    geoPick.map = new TMap.Map('geoMap',{ zoom:14, center });
    geoPick.layer = new TMap.MultiMarker({ map:geoPick.map, styles:pawStyles() });
    if(geoPick.picked) geoPick.layer.setGeometries([{id:'pick',styleId:'paw',position:new TMap.LatLng(geoPick.picked.lat,geoPick.picked.lng)}]);
    geoPick.map.on('click', e=>{
      if(!e.latLng) return;
      geoPick.picked = { lat:e.latLng.lat, lng:e.latLng.lng };
      geoPick.layer.setGeometries([{id:'pick',styleId:'paw',position:new TMap.LatLng(geoPick.picked.lat,geoPick.picked.lng)}]);
      const t=document.getElementById('geoTxt');
      if(t) t.textContent=`已选：${geoPick.picked.lat.toFixed(5)}, ${geoPick.picked.lng.toFixed(5)}`;
    });
  }, 120);
}
function geoMovePin(){
  if(!geoPick||!geoPick.picked||!geoPick.layer) return;
  geoPick.layer.setGeometries([{id:'pick',styleId:'paw',position:new TMap.LatLng(geoPick.picked.lat,geoPick.picked.lng)}]);
  if(geoPick.map){ geoPick.map.setCenter(new TMap.LatLng(geoPick.picked.lat,geoPick.picked.lng)); }
  const t=document.getElementById('geoTxt');
  if(t) t.textContent=`已选：${geoPick.picked.lat.toFixed(5)}, ${geoPick.picked.lng.toFixed(5)}`;
}
function geoUseCurrent(){
  if(!geoPick) return;
  const t=document.getElementById('geoTxt');
  if(t) t.textContent='定位中…';
  navigator.geolocation.getCurrentPosition(p=>{
    geoPick.picked={lat:p.coords.latitude, lng:p.coords.longitude};
    geoMovePin();
  }, err=>{
    if(t) t.textContent='定位失败（'+(err.message||err.code)+'）· 请在地图上点击选点';
  }, {timeout:8000, enableHighAccuracy:true});
}
async function geoSave(id){
  if(!geoPick||!geoPick.picked) return toast('请先在地图上点击选点，或用当前定位');
  const {lat,lng} = geoPick.picked;
  try{
    await dbf('pokedex').update({ lat, lng }).eq('id', id).select();
    const d=S.dex.find(x=>x.id===id);
    if(d){ d.lat=lat; d.lng=lng; }
    toast('已标记遭遇地点 🐾');
    closeModal();
    if(markerLayer) refreshMapMarkers();
    renderMapList();
  }catch(e){ toast('保存失败：'+(e.message||e)); }
}

/* ================= 社区（求助/救助） ================= */
function openModal(html){
  document.getElementById('modalRoot').innerHTML =
    `<div class="modal-mask" onclick="if(event.target===this)closeModal()"><div class="modal">
      <button class="x" onclick="closeModal()">✕</button><div class="grip"></div>${html}</div></div>`;
}
function closeModal(){ document.getElementById('modalRoot').innerHTML=''; if(window._detailMap){ window._detailMap.destroy(); window._detailMap=null; } }

/* 兼容刷新入口：社区列表复用 loadReports */
function renderCommunity(){ if(document.getElementById('reportList')) loadReports(); }

async function loadReports(){
  const seg=document.getElementById('cSeg');
  const filters=[['all','全部'],['open','待救助'],['fundraising','集资中'],['treating','救治中'],['adoptable','待领养'],['adopted','已领养'],['released','已放生']];
  seg.innerHTML=filters.map(([k,l])=>`<button class="${S.cFilter===k?'on':''}" onclick="S.cFilter='${k}';loadReports()">${l}</button>`).join('');
  const list=document.getElementById('reportList');
  list.innerHTML='<div class="loading">加载救助动态…</div>';
  const { data, error } = await dbf('stray_reports').select('*').order('created_at',{ascending:false}).limit(50);
  if(error){ list.innerHTML=`<div class="empty">加载失败：${esc(error.message)}</div>`; return; }
  S.reports=data||[];
  const shown = S.cFilter==='all'? S.reports : S.reports.filter(r=>r.status===S.cFilter);
  if(!shown.length){ list.innerHTML='<div class="empty"><span class="big">🌿</span>暂无相关动态</div>'; return; }
  list.innerHTML = shown.map(r=>`
    <div class="card" onclick="openReportDetail(${r.id})" style="cursor:pointer">
      <div class="h-row">
        ${r.photo_path?`<img data-media="${esc(r.photo_path)}" style="width:64px;height:64px;border-radius:12px;object-fit:cover">`:'<div class="avatar">🐾</div>'}
        <div class="lbody">
          <div class="ltop">${stBadge(r.status)} <b style="font-size:14px">${esc(r.reporter_name||'爱心市民')}</b><span class="sub">${fmtTime(r.created_at)}</span></div>
          <div style="font-size:13px;margin-top:4px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${esc(r.description)}</div>
          <div class="sub" style="margin-top:4px">${r.captured_by_name?'🏥 '+esc(r.captured_by_name):''} ${r.lat?'📍 已标记位置':''}</div>
        </div>
      </div>
    </div>`).join('');
  attachMedia(list);
}
let newReportCtx={photo:null, lat:null, lng:null};
function openNewReport(){
  if(needLogin()) return;
  newReportCtx={photo:null,lat:null,lng:null};
  openModal(`
    <h3>🆘 发布求助</h3>
    <div class="sub">描述小动物的情况，附上照片和位置，等待官方认证医院前往救助</div>
    <label class="fl">现场照片</label>
    <input type="file" accept="image/*" capture="environment" onchange="nrPhoto(this)">
    <div id="nrPrev" style="margin-top:10px"></div>
    <label class="fl">情况描述</label>
    <textarea id="nrDesc" placeholder="例如：小区东门绿化带有一只受伤的流浪猫，后腿疑似骨折，精神萎靡…"></textarea>
    <label class="fl">位置</label>
    <button class="btn sec sm" onclick="nrLocate()">📍 自动获取当前位置</button>
    <span class="sub" id="nrLocTxt" style="margin-left:8px">未获取</span>
    <button class="btn" style="margin-top:16px" id="nrSubmit" onclick="submitReport()">发布求助</button>`);
}
async function nrPhoto(input){
  const f=input.files?.[0]; if(!f) return;
  newReportCtx.photo=await compressImage(f);
  document.getElementById('nrPrev').innerHTML=`<img class="scan-preview" src="${await fileToDataURL(newReportCtx.photo)}">`;
}
async function nrLocate(){
  const el=document.getElementById('nrLocTxt'); el.textContent='定位中…';
  navigator.geolocation.getCurrentPosition(p=>{
    newReportCtx.lat=p.coords.latitude; newReportCtx.lng=p.coords.longitude;
    el.textContent=`已获取 (${p.coords.latitude.toFixed(4)}, ${p.coords.longitude.toFixed(4)})`;
  }, ()=>{ el.textContent='定位失败，请检查权限'; }, {timeout:8000});
}
async function submitReport(){
  const desc=document.getElementById('nrDesc').value.trim();
  if(!desc) return toast('请描述情况');
  if(!newReportCtx.photo) return toast('请上传现场照片');
  if(!newReportCtx.lat) return toast('请先获取位置（方便医院找到小动物）');
  busy(document.getElementById('nrSubmit'),true,'发布中…');
  try{
    const photo=await uploadFile(newReportCtx.photo,'stray');
    const name=myName();
    const { error } = await dbf('stray_reports').insert({
      reporter_name:name, description:desc, photo_path:photo,
      lat:newReportCtx.lat, lng:newReportCtx.lng, status:'open' });
    if(error) throw error;
    toast('已发布，等待医院响应 🙏'); closeModal(); loadReports();
  }catch(e){ toast('发布失败：'+(e.message||e)); busy(document.getElementById('nrSubmit'),false); }
}

async function openReportDetail(id){
  const r=S.reports.find(x=>x.id===id); if(!r) return;
  window._currentReportId=id;
  const mine = S.user && r.owner_id===S.user.id;
  const iAmCapturer = S.user && r.captured_by===S.user.id;
  let fr=null, treatments=[], donors=[];
  if(r.status!=='open'){
    const frR = await dbf('fundraisers').select('*').eq('rescue_id', r.id).order('created_at',{ascending:false}).limit(1);
    fr = frR.data?.[0]||null;
  }
  if(fr){ const dR=await dbf('donations').select('*').eq('fundraiser_id',fr.id).order('created_at',{ascending:false}).limit(20); donors=dR.data||[]; }
  if(['captured','fundraising','treating','adoptable','adopted','released'].includes(r.status)){
    const tR=await dbf('treatments').select('*').eq('rescue_id', r.id).order('created_at',{ascending:true}); treatments=tR.data||[];
  }
  const pct = fr? Math.min(100, (Number(fr.raised)/Number(fr.target))*100):0;
  let adoptBox='';
  if(r.status==='adoptable' && r.adopt_deadline){
    const days=Math.ceil((new Date(r.adopt_deadline)-Date.now())/86400000);
    adoptBox = r.outcome_kind==='adoption' ? `<div class="count-box">🏡 <b>开放领养中</b> · 领养等待期 <b>${days>0?days:0}</b> 天 / 共 7 天<br>
      <span class="sub">${days>0?'领养申请将在等待期结束后确认':'等待期已结束'}</span></div>`
      : `<div class="notice">🌿 这只小动物已被治愈并<strong>放归自然</strong></div>`;
  }
  openModal(`
    <h3>${stBadge(r.status)} 救助档案 #${r.id}</h3>
    <div class="sub">发布：${esc(r.reporter_name)} · ${fmtTime(r.created_at)}${r.captured_at?` · 捕捉：${esc(r.captured_by_name||'')} ${fmtTime(r.captured_at)}`:''}</div>
    ${r.photo_path?`<img data-media="${esc(r.photo_path)}" style="width:100%;border-radius:14px;margin:12px 0;max-height:260px;object-fit:cover">`:''}
    <p style="font-size:14px;line-height:1.7">${esc(r.description)}</p>
    ${r.capture_photo?`<div><div class="fl" style="font-size:12px;color:var(--sub);margin:12px 0 5px;font-weight:600">✅ 捕捉证明照片</div><img data-media="${esc(r.capture_photo)}" style="width:100%;border-radius:14px;max-height:220px;object-fit:cover"></div>`:''}
    <div id="dtMap" class="map-mini"></div>
    ${adoptBox}
    ${fr?`<div class="card" style="box-shadow:none;border:1.5px solid var(--line);margin:12px 0">
      <div class="h-row"><b>💰 ${esc(fr.title||'救治集资')}</b><span class="sub" style="margin-left:auto">${fr.status==='pending_review'?'待后台审核':fr.status==='active'?'集资中':fr.status==='closed'?'已达标关闭':'未通过'}</span></div>
      ${fr.status!=='pending_review'&&fr.status!=='rejected'?`
      <div class="progress"><i style="width:${pct}%"></i></div>
      <div class="h-row"><b style="color:var(--orange)">${money(fr.raised)}</b><span class="sub">/ 目标 ${money(fr.target)}</span>
      <span class="sub" style="margin-left:auto">${pct.toFixed(0)}%</span></div>
      ${fr.status==='active'?`<div class="grid2" style="margin-top:10px"><input id="donAmt" type="number" min="1" step="1" placeholder="爱心金额¥"><button class="btn warn sm" onclick="donate(${fr.id})">献爱心</button></div>`:''}
      ${donors.length?`<div class="sub" style="margin-top:8px">❤️ ${donors.map(d=>esc(d.donor_name)+' '+money(d.amount)).join(' · ')}</div>`:''}`:''}
    </div>`:''}
    ${treatments.length?`<div style="margin:14px 0"><b style="font-size:14px">🩺 救治过程</b><div class="timeline">${treatments.map(t=>`
      <div class="ti"><div class="tt">${esc(t.hospital_name||'医院')}</div><div class="sub">${fmtTime(t.created_at)}</div>
      <div class="td">${esc(t.content)}</div>${(t.media||[]).map(m=>`<div data-mbox="${esc(m)}"></div>`).join('')}</div>`).join('')}</div></div>`:''}
    ${mine&&r.status==='open'?`<div class="sub" style="margin:10px 0">你的求助已发布，等待医院捕捉救助</div>`:''}
    ${iAmCapturer&&r.status==='captured'&&!fr?`<button class="btn warn" style="margin-top:12px" onclick="openCreateFundraiser(${r.id})">💰 设置救助金额，申请开放集资</button>`:''}
  `);
  attachMedia(document.getElementById('modalRoot'));
  document.querySelectorAll('[data-mbox]').forEach(async box=>{
    const path=box.dataset.mbox;
    try{
      const url=await mediaURL(path);
      const isVid=/\.(mp4|mov|webm|m4v)$/i.test(path)||path.includes('video');
      box.outerHTML = isVid? `<video src="${url}" controls style="width:100%;border-radius:10px;margin-top:6px"></video>`
        : `<img src="${url}" style="width:100%;border-radius:10px;margin-top:6px">`;
    }catch(e){ box.outerHTML='<span class="sub">[媒体加载失败]</span>'; }
  });
  if(typeof TMap!=='undefined' && r.lat){
    try{
      const m=new TMap.Map('dtMap',{zoom:15, center:new TMap.LatLng(+r.lat,+r.lng)});
      window._detailMap=m;
      new TMap.MultiMarker({ map:m, styles:{ p:new TMap.MarkerStyle({width:26,height:34,anchor:{x:13,y:34},
        src:'data:image/svg+xml,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="26" height="34"><path d="M13 0C5.8 0 0 5.5 0 12.4c0 5.7 4 8.3 7.6 9.9 2.1.9 2.9 2.2 3.4 4.4L13 34l2-7.3c.5-2.2 1.3-3.5 3.4-4.4C22 20.7 26 18.1 26 12.4 26 5.5 20.2 0 13 0z" fill="%23e5484d"/><circle cx="13" cy="12" r="4.8" fill="%23fff"/></svg>')})},
        geometries:[{id:'1',styleId:'p',position:new TMap.LatLng(+r.lat,+r.lng)}]});
    }catch(e){}
  }
}
async function donate(fid){
  if(needLogin()) return;
  const amt=+document.getElementById('donAmt').value;
  if(!amt||amt<=0) return toast('请输入有效的爱心金额');
  const name=myName();
  const { error } = await dbf('donations').insert({ fundraiser_id:fid, donor_name:name, amount:amt });
  if(error) return toast('捐赠失败：'+error.message);
  toast('感谢你的爱心！❤️'); loadReports();
  if(window._currentReportId) setTimeout(()=>openReportDetail(window._currentReportId), 400);
}

/* ================= 医院端 ================= */
async function renderHospitalTab(){
  const body=document.getElementById('hospitalBody');
  if(!S.user){ body.innerHTML=`<div class="empty"><span class="big">🏥</span>正在加载医院名录…<br><button class="btn sm" style="margin-top:12px" onclick="openTrialInfo()">查看试用说明</button></div>`; return; }
  if(!S.hospital){
    const { data } = await dbf('hospitals').select('*').eq('status','approved').order('created_at',{ascending:false});
    body.innerHTML=`
      <div class="notice">你是普通训练家。如你是宠物医院，可注册官方账号参与救助。</div>
      <button class="btn" style="margin-bottom:14px" onclick="openHospitalRegister()">🏥 注册宠物医院官方账号</button>
      <div class="h-title" style="margin-bottom:10px">认证医院名录</div>
      <div id="hospDir"></div>`;
    renderHospDir(data||[]);
    return;
  }
  if(S.hospital.status==='pending'){
    body.innerHTML=`<div class="card"><div class="h-title">⏳ 资质审核中</div>
      <p class="sub" style="margin-top:8px">医院「${esc(S.hospital.name)}」的营业执照已提交，等待平台后台审核。审核通过后即可参与救助。</p>
      <button class="btn sec sm" style="margin-top:12px" onclick="refreshHospital()">刷新状态</button></div>`;
    return;
  }
  if(S.hospital.status==='rejected'){
    body.innerHTML=`<div class="card"><div class="h-title">❌ 认证未通过</div>
      ${S.hospital.review_note?`<div class="notice" style="margin-top:10px;white-space:pre-wrap;line-height:1.7">${esc(S.hospital.review_note)}</div>`
        :'<p class="sub" style="margin-top:8px">未通过原因未填写，请联系平台确认。</p>'}
      <p class="sub" style="margin-top:8px">按上述条款补正材料后可重新提交认证。</p>
      <button class="btn sm" style="margin-top:10px" onclick="refreshHospital()">刷新</button></div>`;
    return;
  }
  const [openR, mineR] = await Promise.all([
    dbf('stray_reports').select('*').eq('status','open').order('created_at',{ascending:false}).limit(20),
    dbf('stray_reports').select('*').eq('captured_by', S.user.id).order('created_at',{ascending:false}).limit(50)
  ]);
  const open=openR.data||[], mine=mineR.data||[];
  const rescued=mine.filter(r=>['captured','fundraising','treating','adoptable','adopted','released'].includes(r.status)).length;
  const myFrs = (await dbf('fundraisers').select('*').eq('owner_id',S.user.id).order('created_at',{ascending:false}).limit(50)).data||[];
  const pendingFr = myFrs.filter(f=>f.status==='pending_review').length;
  const revR = await dbf('reviews').select('*').eq('hospital_id',S.hospital.id).order('created_at',{ascending:false}).limit(30);
  const revs=revR.data||[];
  const avg = revs.length? revs.reduce((s,r)=>s+Number(r.rating),0)/revs.length : null;
  body.innerHTML=`
    <div class="card">
      <div class="h-row"><div class="avatar">🏥</div><div><div class="h-title">${esc(S.hospital.name)}</div>
      <span class="badge b-adopted">✅ 官方认证</span></div></div>
      <div class="grid3" style="margin-top:12px;text-align:center">
        <div><div style="font-size:22px;font-weight:800">${rescued}</div><div class="sub">累计救助</div></div>
        <div><div style="font-size:22px;font-weight:800">${avg?avg.toFixed(1):'-'}</div><div class="stars">${avg?starsOf(avg):'暂无'}</div></div>
        <div><div style="font-size:22px;font-weight:800">${S.hospital.amap_rating||'-'}</div><div class="sub">高德评分</div></div>
      </div>
      <button class="btn sec sm" style="margin-top:12px" onclick="syncAmap()">🔄 从高德地图同步评分与评价</button>
      ${revs.length?`<div style="margin-top:10px">${revs.slice(0,5).map(r=>`<div class="li"><div class="lbody"><div class="ltop"><b style="font-size:13px">${esc(r.reviewer_name)}</b><span class="stars">${starsOf(r.rating)}</span><span class="sub">${fmtTime(r.created_at)}</span></div><div style="font-size:12px">${esc(r.content||'')}</div></div></div>`).join('')}</div>`:''}
    </div>
    ${pendingFr?`<div class="count-box">⏳ 你有 ${pendingFr} 个集资申请等待后台审核</div>`:''}
    <div class="h-title" style="margin:6px 0 10px">🎯 待救助求助（去捕捉）</div>
    ${open.length? open.map(r=>`<div class="card"><div class="h-row">
      ${r.photo_path?`<img data-media="${esc(r.photo_path)}" style="width:56px;height:56px;border-radius:10px;object-fit:cover">`:''}
      <div class="lbody"><div style="font-size:13px">${esc(r.description).slice(0,60)}</div>
      <div class="sub" style="margin-top:2px">${esc(r.reporter_name)} · ${fmtTime(r.created_at)} · 📍${(+r.lat).toFixed(4)},${(+r.lng).toFixed(4)}</div></div>
      <button class="btn sm" onclick="doCapture(${r.id})">捕捉</button></div></div>`).join('')
      : '<div class="empty">暂无待救助求助</div>'}
    <div class="h-title" style="margin:16px 0 10px">🩺 我的救助管理</div>
    ${mine.length? mine.map(r=>{
      const fr=myFrs.find(f=>f.rescue_id===r.id);
      let act='';
      if(r.status==='captured'&&(!fr||fr.status==='rejected')) act=`<button class="btn warn sm" onclick="openCreateFundraiser(${r.id})">${fr?'重新提交救助金额':'设置救助金额'}</button>`;
      if(fr&&fr.status==='active'&&r.status==='captured') act=`<span class="sub">集资中，达标后开始救治</span> <button class="btn sm" onclick="startTreating(${r.id})">开始救治</button>`;
      if(r.status==='treating') act=`<button class="btn sm" onclick="openTreatment(${r.id})">上传救治进展</button> <button class="btn warn sm" onclick="finishTreatment(${r.id})">救治完成</button>`;
      if(r.status==='adoptable'&&r.outcome_kind==='adoption'&&r.adopt_status==='open') act=`<span class="sub">等待期 ${Math.max(0,Math.ceil((new Date(r.adopt_deadline)-Date.now())/86400000))}/7 天</span> <button class="btn sm" onclick="markAdopted(${r.id})">确认领养</button>`;
      return `<div class="card"><div class="h-row">
        ${r.photo_path?`<img data-media="${esc(r.photo_path)}" style="width:52px;height:52px;border-radius:10px;object-fit:cover">`:''}
        <div class="lbody"><div class="ltop">${stBadge(r.status)} <span class="sub">#${r.id} · ${fmtTime(r.created_at)}</span></div>
        <div style="font-size:13px;margin-top:3px">${esc(r.description).slice(0,50)}</div>
        ${fr?`<div class="sub" style="margin-top:3px">💰 集资 ${money(fr.raised)}/${money(fr.target)}（${fr.status==='active'?'进行中':fr.status==='closed'?'已达标':fr.status==='rejected'?'已驳回':'审核中'}）</div>`:''}
        ${fr&&fr.status==='rejected'&&fr.review_note?`<div class="notice" style="margin-top:6px;white-space:pre-wrap;line-height:1.65;font-size:12px">${esc(fr.review_note)}</div>`:''}
        ${r.outcome_kind==='adoption'&&r.adopt_deadline?`<div class="sub">🏡 领养等待期剩余 ${Math.max(0,Math.ceil((new Date(r.adopt_deadline)-Date.now())/86400000))} 天（7天期限）</div>`:''}
        </div></div>${act?`<div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">${act}</div>`:''}</div>`;
    }).join('') : '<div class="empty">在上方捕捉求助后，这里会出现管理面板</div>'}`;
  attachMedia(body);
}
async function refreshHospital(){
  const { data } = await dbf('hospitals').select('*').eq('owner_id', S.user.id).maybeSingle();
  S.hospital=data||null; renderHospitalTab();
}
function renderHospDir(list){
  const el=document.getElementById('hospDir');
  if(!list.length){ el.innerHTML='<div class="empty">还没有认证医院</div>'; return; }
  el.innerHTML=list.map(h=>`<div class="card"><div class="h-row"><div class="avatar">🏥</div>
    <div class="lbody"><div class="ltop"><b>${esc(h.name)}</b><span class="badge b-adopted">官方认证</span></div>
    <div class="sub" style="margin-top:3px">${esc(h.address||'')}${h.amap_rating?` · ⭐ 高德 ${h.amap_rating}`:''}</div></div>
    <button class="btn sec sm" onclick="openHospitalProfile(${h.id})">主页</button></div></div>`).join('');
}
function openHospitalRegister(){
  openModal(`
    <h3>🏥 医院官方账号认证</h3>
    <div class="sub">提交营业执照，平台后台审核通过后即可参与救助</div>
    <label class="fl">医院名称</label><input id="hName" placeholder="XX宠物医院">
    <label class="fl">统一社会信用代码 / 执照号</label><input id="hLicNo" placeholder="执照编号">
    <label class="fl">营业执照照片</label><input type="file" accept="image/*" id="hLic" onchange="hLicPrev(this)">
    <div id="hLicPrev" style="margin-top:8px"></div>
    <label class="fl">地址</label><input id="hAddr" placeholder="省市区 + 街道门牌">
    <label class="fl">联系电话</label><input id="hPhone" placeholder="电话">
    <label class="fl">医院简介</label><textarea id="hIntro" placeholder="科室、医生团队、擅长领域…"></textarea>
    <button class="btn" style="margin-top:16px" id="hReg" onclick="submitHospital()">提交认证申请</button>`);
}
function hLicPrev(input){
  const f=input.files?.[0]; if(!f) return;
  window._licFile=f; fileToDataURL(f).then(u=>document.getElementById('hLicPrev').innerHTML=`<img class="scan-preview" src="${u}">`);
}
async function submitHospital(){
  const name=document.getElementById('hName').value.trim();
  if(!name||!document.getElementById('hLicNo').value.trim()) return toast('请填写名称和执照编号');
  if(!window._licFile) return toast('请上传营业执照照片');
  busy(document.getElementById('hReg'),true,'提交中…');
  try{
    const lic=await uploadFile(window._licFile,'licenses');
    const { error } = await dbf('hospitals').insert({ name, license_no:document.getElementById('hLicNo').value.trim(),
      license_path:lic, address:document.getElementById('hAddr').value.trim(),
      phone:document.getElementById('hPhone').value.trim(), intro:document.getElementById('hIntro').value.trim(), status:'pending' });
    if(error) throw error;
    toast('已提交，等待后台审核 ✅'); closeModal(); refreshHospital();
  }catch(e){ toast('提交失败：'+(e.message||e)); busy(document.getElementById('hReg'),false); }
}
async function doCapture(rid){
  openModal(`
    <h3>✅ 上传捕捉证明</h3>
    <div class="sub">捕捉成功后，请上传现场捕捉照片作为凭证</div>
    <input type="file" accept="image/*" capture="environment" id="capFile">
    <div id="capPrev" style="margin-top:8px"></div>
    <button class="btn" style="margin-top:14px" id="capGo" onclick="submitCapture(${rid})">确认已捕捉</button>`);
  document.getElementById('capFile').onchange=async e=>{
    const f=e.target.files?.[0]; if(!f) return;
    window._capFile=await compressImage(f);
    document.getElementById('capPrev').innerHTML=`<img class="scan-preview" src="${await fileToDataURL(window._capFile)}">`;
  };
}
async function submitCapture(rid){
  if(!window._capFile) return toast('请上传捕捉照片');
  busy(document.getElementById('capGo'),true,'提交中…');
  try{
    const photo=await uploadFile(window._capFile,'captures');
    const { data, error } = await dbf('stray_reports').update({
      status:'captured', captured_by:S.user.id, captured_by_name:S.hospital.name,
      capture_photo:photo, captured_at:new Date().toISOString()
    }).eq('id',rid).select();
    if(error) throw error;
    if(!data.length) return toast('操作失败：可能已被其他医院捕捉');
    window._capFile=null; toast('捕捉成功！接下来设置救助金额 💰'); closeModal(); renderHospitalTab(); loadReports();
  }catch(e){ toast('捕捉失败：'+(e.message||e)); busy(document.getElementById('capGo'),false); }
}
function openCreateFundraiser(rid){
  openModal(`
    <h3>💰 设置救助金额</h3>
    <div class="sub">提交后由平台后台审核，审核通过即开放集资通道；金额达标自动关闭</div>
    <label class="fl">集资标题</label><input id="frTitle" value="小动物救治募捐" maxlength="20">
    <label class="fl">目标金额（元）</label><input id="frTarget" type="number" min="1" step="1" placeholder="例如 800">
    <button class="btn warn" style="margin-top:14px" id="frGo" onclick="submitFundraiser(${rid})">提交审核</button>`);
}
async function submitFundraiser(rid){
  const target=+document.getElementById('frTarget').value;
  if(!target||target<=0) return toast('请输入有效目标金额');
  busy(document.getElementById('frGo'),true,'提交中…');
  const { error } = await dbf('fundraisers').insert({ rescue_id:rid, owner_id:S.user.id,
    hospital_name:S.hospital.name, title:document.getElementById('frTitle').value.trim()||'小动物救治募捐',
    target, raised:0, status:'pending_review' });
  if(error) return toast('提交失败：'+(error.message||error));
  toast('已提交后台审核 ⏳'); closeModal(); renderHospitalTab();
}
async function startTreating(rid){
  const { data, error } = await dbf('stray_reports').update({ status:'treating' }).eq('id',rid).select();
  if(error||!data.length) return toast('操作失败');
  toast('已开始救治 🩺'); renderHospitalTab();
}
function openTreatment(rid){
  openModal(`
    <h3>🩺 上传救治进展</h3>
    <div class="sub">图文或视频记录，将公开展示在救助档案中</div>
    <label class="fl">进展说明</label><textarea id="trTxt" placeholder="例如：已完成骨折复位手术，恢复良好，已能自主进食…"></textarea>
    <label class="fl">图片 / 视频（可多选）</label><input type="file" id="trFiles" multiple accept="image/*,video/*">
    <button class="btn" style="margin-top:14px" id="trGo" onclick="submitTreatment(${rid})">发布进展</button>`);
}
async function submitTreatment(rid){
  const txt=document.getElementById('trTxt').value.trim();
  if(!txt) return toast('请填写进展说明');
  busy(document.getElementById('trGo'),true,'上传中…');
  try{
    const files=[...document.getElementById('trFiles').files];
    const media=[];
    for(const f of files){ media.push(await uploadFile(f, 'treatments')); }
    const { error } = await dbf('treatments').insert({ rescue_id:rid, owner_id:S.user.id,
      hospital_name:S.hospital.name, content:txt, media });
    if(error) throw error;
    toast('救治进展已发布 ✅'); closeModal(); renderHospitalTab();
  }catch(e){ toast('发布失败：'+(e.message||e)); busy(document.getElementById('trGo'),false); }
}
function finishTreatment(rid){
  openModal(`
    <h3>救治完成 · 选择去向</h3>
    <div class="sub">领养将开放 7 天等待期，期满确认领养人；放生则回归自然</div>
    <div class="grid2" style="margin-top:14px">
      <button class="btn" onclick="setOutcome(${rid},'adoption')">🏡 开放领养（7天等待期）</button>
      <button class="btn sec" onclick="setOutcome(${rid},'release')">🌿 治愈放生</button>
    </div>`);
}
async function setOutcome(rid, kind){
  const patch={ status:'adoptable', outcome_kind:kind,
    outcome_note: kind==='adoption'?'开放领养，等待期7天':'治愈放归自然', outcome_at:new Date().toISOString() };
  if(kind==='adoption'){ patch.adopt_deadline=new Date(Date.now()+7*86400000).toISOString(); patch.adopt_status='open'; }
  const { data, error } = await dbf('stray_reports').update(patch).eq('id',rid).select();
  if(error||!data.length) return toast('操作失败');
  toast(kind==='adoption'?'已开放领养，等待期 7 天 ⏳':'放生成功，回归自然 🌿'); closeModal(); renderHospitalTab(); loadReports();
}
async function markAdopted(rid){
  const adopter=prompt('领养人姓名/联系方式：'); if(!adopter) return;
  const { data, error } = await dbf('stray_reports').update({ status:'adopted', adopt_status:'adopted', adopter }).eq('id',rid).select();
  if(error||!data.length) return toast('操作失败');
  toast('领养成功 🎉'); renderHospitalTab(); loadReports();
}
async function syncAmap(){
  let key = null;
  try{ key = (await fetch('/api/amap_key').then(r=>r.json())).key || null; }catch(e){}
  if(!key){
    key = prompt('请输入高德开放平台 Web服务 Key（保存到本服务器，所有人共用）。\n申请入口：lbs.amap.com → 控制台 → 应用管理 → 创建应用 → 添加 Key（服务平台选「Web服务」）');
    if(!key) return;
    await fetch('/api/amap_key', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ key }) });
  }
  toast('正在从高德同步评分…');
  try{
    /* 由服务器代请求，绕开浏览器跨域限制 */
    const r = await fetch('/api/amap_search', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ name:S.hospital.name, address:S.hospital.address||'' })
    });
    const j = await r.json();
    if(!j.rating) return toast(j.error || '该医院暂无高德评分');
    const { error } = await dbf('hospitals').update({ amap_rating:String(j.rating) }).eq('id',S.hospital.id).select();
    if(error) return toast('同步失败：'+error.message);
    toast(`已同步高德评分 ${j.rating} ⭐`);
    S.hospital.amap_rating = String(j.rating);
    refreshHospital();
  }catch(e){ toast('同步失败：'+(e.message||'网络错误')); }
}
async function openHospitalProfile(hid){
  const { data:h } = await dbf('hospitals').select('*').eq('id',hid).maybeSingle();
  if(!h) return toast('医院不存在');
  const revs=(await dbf('reviews').select('*').eq('hospital_id',hid).order('created_at',{ascending:false}).limit(30)).data||[];
  const avg=revs.length?revs.reduce((s,r)=>s+Number(r.rating),0)/revs.length:null;
  const rescued=(await dbf('stray_reports').select('*',{count:'exact',head:true}).eq('captured_by',h.owner_id)).count||0;
  openModal(`
    <h3>🏥 ${esc(h.name)}</h3>
    <div class="ltop" style="margin:4px 0 10px"><span class="badge b-adopted">官方认证</span>${h.amap_rating?`<span class="sub">⭐ 高德评分 ${h.amap_rating}</span>`:''}</div>
    <div class="grid3" style="text-align:center;margin-bottom:10px">
      <div><div style="font-size:22px;font-weight:800">${rescued}</div><div class="sub">救助动物数</div></div>
      <div><div style="font-size:22px;font-weight:800">${avg?avg.toFixed(1):'-'}</div><div class="stars">${avg?starsOf(avg):'暂无评价'}</div></div>
      <div><div style="font-size:22px;font-weight:800">${revs.length}</div><div class="sub">评价数</div></div>
    </div>
    ${h.address?`<div class="sub">📍 ${esc(h.address)}</div>`:''}
    ${h.phone?`<div class="sub">📞 ${esc(h.phone)}</div>`:''}
    ${h.intro?`<p style="font-size:13px;margin-top:8px;line-height:1.7">${esc(h.intro)}</p>`:''}
    <div style="margin-top:14px"><b style="font-size:14px">用户评价</b>
      ${revs.length?revs.map(r=>`<div class="li"><div class="lbody"><div class="ltop"><b style="font-size:13px">${esc(r.reviewer_name)}</b><span class="stars">${starsOf(r.rating)}</span><span class="sub">${fmtTime(r.created_at)}</span></div><div style="font-size:12px">${esc(r.content||'')}</div></div></div>`).join(''):'<div class="sub" style="padding:10px 0">暂无评价</div>'}
    </div>
    ${S.user&&S.user.id!==h.owner_id?`<div class="grid2" style="margin-top:12px">
      <select id="rvStars"><option value="5">★★★★★</option><option value="4">★★★★</option><option value="3">★★★</option><option value="2">★★</option><option value="1">★</option></select>
      <button class="btn sm" onclick="submitReview(${hid})">发表评价</button></div>
      <textarea id="rvText" placeholder="说说这家医院的服务体验…" style="margin-top:8px"></textarea>`:''}`);
}
async function submitReview(hid){
  if(needLogin()) return;
  const rating=+document.getElementById('rvStars').value;
  const content=document.getElementById('rvText').value.trim();
  const name=myName();
  const { error } = await dbf('reviews').insert({ hospital_id:hid, reviewer_name:name, rating, content });
  if(error) return toast('评价失败：'+error.message);
  toast('评价成功 ⭐'); closeModal(); openHospitalProfile(hid);
}

/* ================= 我的 / 管理后台 ================= */
async function renderMe(){
  const body=document.getElementById('meBody');
  if(!S.user){ body.innerHTML=`<div class="card"><div class="h-title">👤 正在进入试用模式</div><p class="sub" style="margin:8px 0 12px">无需登录，打开即用</p><button class="btn" onclick="openTrialInfo()">查看试用说明</button></div>`; return; }
  const dons=(await dbf('donations').select('*').eq('owner_id',S.user.id).order('created_at',{ascending:false}).limit(20)).data||[];
  const uname = '训练家';
  const usub = '本机服务器 · 数据保存在 data.json';
  body.innerHTML=`
    <div class="card"><div class="h-row"><div class="avatar" style="width:48px;height:48px;font-size:24px">🐦</div>
      <div><div class="h-title">${esc(uname)}</div><div class="sub">${esc(usub)}</div></div>
      ${S.hospital&&S.hospital.status==='approved'?'<span class="badge b-adopted" style="margin-left:auto">认证医院</span>':''}</div></div>
    ${S.hospital&&S.hospital.status==='approved'?`<button class="btn sec" style="margin-bottom:12px" onclick="switchTab('hospital')">🏥 进入医院工作台</button>`:''}
    <div class="card"><div class="h-title" style="margin-bottom:8px">❤️ 我的爱心捐赠</div>
      ${dons.length?dons.map(d=>`<div class="li"><div class="lbody"><div class="ltop"><b style="color:var(--orange)">${money(d.amount)}</b><span class="sub">${fmtTime(d.created_at)}</span></div></div></div>`).join(''):'<div class="sub">暂无捐赠记录</div>'}</div>`;
}
/* 审核功能已迁移到独立页面 admin.html（用户端不保留任何审核入口与审核能力） */

/* ================= 启动 ================= */
(async function(){
  S.user = GUEST; S.demo = true;
  try{ await afterLogin(); }catch(e){ console.warn(e); }
  try{ refreshAll(); }catch(e){ console.warn('初始化失败', e); }
  switchTab('pokedex');
})();
"""


def _read(rel):
    """外部同名文件优先（方便二次开发），没有就用内嵌版本"""
    p = os.path.join(BASE_DIR, rel)
    if not os.path.exists(p):
        return ""
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _html_or_inline(rel, inline):
    v = _read(rel)
    return v if v.strip() else inline


# 管理审核台页面（内嵌 admin.html）
ADMIN_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>审核台 · 现实版宝可梦</title>
<style>
*{box-sizing:border-box; -webkit-tap-highlight-color:transparent}
:root{
  --bg:#f4f5f7; --card:#fff; --line:#e3e5e8; --line2:#eef0f2;
  --ink:#1b1d21; --sub:#6b7280; --hint:#9aa1ab;
  --blue:#1f6feb; --green:#1a7f4b; --red:#c93b3b; --amber:#b45309;
  --r:10px;
}
html,body{height:100%}
body{margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
button{font-family:inherit; cursor:pointer}
input,textarea,select{font-family:inherit}

/* ---------- 顶栏 ---------- */
.top{position:sticky; top:0; z-index:30; background:#fff; border-bottom:1px solid var(--line);
  padding:12px 20px; display:flex; align-items:center; gap:16px; flex-wrap:wrap}
.top h1{font-size:15px; margin:0; font-weight:600; letter-spacing:.2px}
.top .tagline{font-size:12px; color:var(--hint); border-left:1px solid var(--line); padding-left:16px}
.stats{margin-left:auto; display:flex; gap:8px; flex-wrap:wrap}
.stat{display:flex; align-items:baseline; gap:5px; background:var(--bg); border:1px solid var(--line);
  border-radius:8px; padding:5px 11px; font-size:12px; color:var(--sub)}
.stat b{font-size:15px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums}
.stat.pend b{color:var(--amber)}
.stat.ok b{color:var(--green)}
.stat.bad b{color:var(--red)}
.ghost{border:1px solid var(--line); background:#fff; color:var(--sub); border-radius:8px; padding:6px 12px; font-size:12.5px}
.ghost:hover{border-color:#c9ccd1; color:var(--ink)}

/* ---------- 筛选 ---------- */
.filters{position:sticky; top:53px; z-index:29; background:rgba(244,245,247,.96); backdrop-filter:blur(6px);
  padding:10px 20px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; border-bottom:1px solid var(--line)}
.fgroup{display:flex; align-items:center; gap:6px}
.flabel{font-size:12px; color:var(--hint); margin-right:2px}
.chip{border:1px solid var(--line); background:#fff; color:var(--sub); border-radius:99px;
  padding:5px 13px; font-size:12.5px; line-height:1.4; transition:.12s}
.chip:hover{border-color:#c9ccd1}
.chip.on{background:var(--ink); border-color:var(--ink); color:#fff}
.chip .n{margin-left:5px; font-variant-numeric:tabular-nums; opacity:.7}
.searchbox{margin-left:auto; display:flex; align-items:center; gap:6px; background:#fff; border:1px solid var(--line);
  border-radius:8px; padding:5px 10px}
.searchbox input{border:none; outline:none; font-size:13px; width:170px; background:transparent; color:var(--ink)}

/* ---------- 主区 ---------- */
.layout{display:grid; grid-template-columns:380px 1fr; gap:0; height:calc(100vh - 105px)}
.list{overflow-y:auto; border-right:1px solid var(--line); background:#fff}
.list-head{position:sticky; top:0; background:#fff; padding:10px 16px; font-size:12px; color:var(--hint);
  border-bottom:1px solid var(--line2); z-index:2}

/* 列表项 */
.item{padding:13px 16px 13px 14px; border-bottom:1px solid var(--line2); cursor:pointer; position:relative; transition:background .12s}
.item:hover{background:#fafbfc}
.item.on{background:#f0f5ff}
.item.on::before{content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background:var(--blue)}
.item-top{display:flex; align-items:center; gap:7px; margin-bottom:5px}
.ty{font-size:11px; padding:2px 7px; border-radius:5px; font-weight:500; letter-spacing:.3px; flex:none}
.ty.hospital{background:#eaf0fb; color:#1d4ed8}
.ty.fundraiser{background:#fdf2e3; color:#b45309}
.ty.treatment{background:#e9f6ef; color:#15803d}
.st{font-size:11px; padding:2px 7px; border-radius:5px; margin-left:auto; flex:none}
.st.pending{background:#fef3e2; color:#b45309}
.st.passed{background:#e9f6ef; color:#15803d}
.st.rejected{background:#fbeaea; color:#b02525}
.item-title{font-size:13.5px; font-weight:500; line-height:1.45; margin-bottom:4px;
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden}
.item-meta{font-size:11.5px; color:var(--hint); display:flex; gap:8px; flex-wrap:wrap; align-items:center}
.item-thumbs{display:flex; gap:4px; margin-top:7px}
.item-thumbs img{width:44px; height:44px; object-fit:cover; border-radius:5px; background:#f0f1f3; border:1px solid var(--line2)}
.empty{padding:60px 24px; text-align:center; color:var(--hint); font-size:13px; line-height:1.9}

/* ---------- 详情 ---------- */
.detail{overflow-y:auto; padding:0}
.detail-empty{display:flex; align-items:center; justify-content:center; height:100%; color:var(--hint); font-size:13px; text-align:center; line-height:2}
.dhead{padding:20px 24px 16px; border-bottom:1px solid var(--line); background:#fff}
.dhead .row1{display:flex; align-items:center; gap:8px; margin-bottom:8px}
.dhead h2{font-size:17px; margin:0 0 6px; font-weight:600; line-height:1.4}
.dhead .meta{font-size:12.5px; color:var(--sub); display:flex; gap:14px; flex-wrap:wrap}
.dbody{padding:20px 24px 120px}
.sect{margin-bottom:22px}
.sect h3{font-size:12.5px; font-weight:600; color:var(--hint); letter-spacing:.6px; margin:0 0 10px;
  text-transform:uppercase; display:flex; align-items:center; gap:8px}
.sect h3::after{content:''; flex:1; height:1px; background:var(--line2)}
.kv{background:var(--card); border:1px solid var(--line); border-radius:var(--r); overflow:hidden}
.kv-row{display:flex; padding:10px 14px; border-bottom:1px solid var(--line2); font-size:13px; gap:14px}
.kv-row:last-child{border-bottom:none}
.kv-k{width:96px; flex:none; color:var(--sub); font-size:12.5px}
.kv-v{flex:1; word-break:break-all}
.kv-v.mono{font-family:ui-monospace,Menlo,Consolas,monospace; font-size:12.5px; letter-spacing:.2px}

.imgs{display:grid; grid-template-columns:repeat(auto-fill,minmax(132px,1fr)); gap:10px}
.imgbox{position:relative; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#f7f8f9; cursor:zoom-in}
.imgbox img{display:block; width:100%; height:132px; object-fit:cover; transition:transform .18s}
.imgbox:hover img{transform:scale(1.04)}
.imgbox .cap{position:absolute; left:0; right:0; bottom:0; background:linear-gradient(transparent,rgba(0,0,0,.62));
  color:#fff; font-size:11px; padding:14px 8px 6px}

.risk{border:1px solid #f3d6a8; background:#fffaf1; border-radius:var(--r); padding:12px 14px; margin-bottom:18px}
.risk h4{margin:0 0 8px; font-size:12.5px; color:#8a5a08; font-weight:600}
.risk ul{margin:0; padding-left:18px; font-size:12.5px; color:#7a5410; line-height:1.85}
.risk code{background:#f6e6c8; padding:1px 5px; border-radius:4px; font-size:11.5px}
.risk.clear{border-color:#cfe6d6; background:#f5fbf7}
.risk.clear h4{color:#1a7f4b}
.risk.clear p{margin:0; font-size:12.5px; color:#2c7a52}

.log{border-left:2px solid var(--line); margin-left:4px; padding-left:16px}
.log-item{position:relative; padding-bottom:14px; font-size:12.5px}
.log-item::before{content:''; position:absolute; left:-21px; top:4px; width:8px; height:8px; border-radius:50%;
  background:#fff; border:2px solid var(--line)}
.log-item.ok::before{border-color:var(--green)}
.log-item.bad::before{border-color:var(--red)}
.log-item .lt{font-weight:500; margin-bottom:2px}
.log-item .lm{color:var(--hint); font-size:11.5px}
.log-item .lr{color:var(--sub); margin-top:4px; line-height:1.7; background:#fafbfc; border:1px solid var(--line2);
  border-radius:6px; padding:7px 10px; white-space:pre-wrap}

/* 底部操作条 */
.actbar{position:fixed; bottom:0; left:380px; right:0; background:#fff; border-top:1px solid var(--line);
  padding:13px 24px; display:flex; gap:10px; align-items:center; z-index:25; box-shadow:0 -2px 12px rgba(20,30,40,.05)}
.actbar .hint{font-size:12px; color:var(--hint); margin-right:auto}
.btn{border:1px solid transparent; border-radius:8px; padding:9px 22px; font-size:13.5px; font-weight:500; transition:.12s}
.btn.pass{background:var(--green); color:#fff}
.btn.pass:hover{background:#166b3f}
.btn.reject{background:#fff; color:var(--red); border-color:#e6bcbc}
.btn.reject:hover{background:#fdf4f4; border-color:#d99c9c}
.btn:disabled{opacity:.45; cursor:not-allowed}
.btn.plain{background:#fff; color:var(--sub); border-color:var(--line)}
.btn.plain:hover{color:var(--ink)}

/* ---------- 驳回弹窗 ---------- */
.mask{position:fixed; inset:0; background:rgba(20,26,34,.42); z-index:60; display:flex; align-items:center; justify-content:center; padding:24px}
.modal{background:#fff; width:100%; max-width:720px; max-height:86vh; border-radius:14px; display:flex; flex-direction:column; overflow:hidden}
.mhead{padding:18px 22px 14px; border-bottom:1px solid var(--line)}
.mhead h3{margin:0 0 4px; font-size:15.5px; font-weight:600}
.mhead p{margin:0; font-size:12.5px; color:var(--sub)}
.mbody{overflow-y:auto; padding:16px 22px}
.group{margin-bottom:16px}
.group h4{font-size:12.5px; color:var(--hint); font-weight:600; margin:0 0 8px; letter-spacing:.4px}
.clause{display:flex; gap:10px; padding:8px 10px; border:1px solid var(--line); border-radius:8px; margin-bottom:6px;
  cursor:pointer; align-items:flex-start; transition:.12s; background:#fff}
.clause:hover{border-color:#c9ccd1; background:#fcfcfd}
.clause.sel{border-color:#e3b4b4; background:#fdf7f7}
.clause input{margin-top:3px; accent-color:var(--red); flex:none}
.clause .cc{font-size:11px; font-family:ui-monospace,monospace; color:var(--hint); flex:none; width:22px; margin-top:1px}
.clause .ct{flex:1}
.clause .cname{font-size:13px; font-weight:500; display:flex; align-items:center; gap:7px; flex-wrap:wrap}
.clause .cdesc{font-size:11.5px; color:var(--sub); margin-top:2px; line-height:1.6}
.lv{font-size:10.5px; padding:1px 6px; border-radius:4px; font-weight:500; letter-spacing:.2px}
.lv.block{background:#fbeaea; color:#b02525}
.lv.fix{background:#fef3e2; color:#b45309}
.lv.warn{background:#eef1f4; color:#5b6472}
.note{margin-top:6px}
.note label{font-size:12.5px; color:var(--sub); display:block; margin-bottom:6px}
.note textarea{width:100%; min-height:72px; border:1px solid var(--line); border-radius:8px; padding:10px 12px;
  font-size:13px; resize:vertical; outline:none; line-height:1.65}
.note textarea:focus{border-color:#c9ccd1}
.mfoot{padding:14px 22px; border-top:1px solid var(--line); display:flex; gap:10px; align-items:center; background:#fcfcfd}
.mfoot .tip{font-size:12px; color:var(--hint); margin-right:auto}

/* lightbox */
.lb{position:fixed; inset:0; background:rgba(12,14,18,.92); z-index:80; display:flex; align-items:center; justify-content:center}
.lb img{max-width:88vw; max-height:84vh; object-fit:contain; border-radius:6px}
.lb .x{position:absolute; top:18px; right:22px; background:rgba(255,255,255,.14); color:#fff; border:none;
  width:38px; height:38px; border-radius:50%; font-size:19px}
.lb .nav{position:absolute; top:50%; transform:translateY(-50%); background:rgba(255,255,255,.14); color:#fff;
  border:none; width:44px; height:64px; border-radius:8px; font-size:20px}
.lb .nav.prev{left:20px} .lb .nav.next{right:20px}
.lb .count{position:absolute; bottom:22px; left:50%; transform:translateX(-50%); color:#cfd4db; font-size:12.5px}

.toast{position:fixed; left:50%; bottom:84px; transform:translateX(-50%) translateY(8px); background:#1b1d21; color:#fff;
  padding:10px 18px; border-radius:9px; font-size:13px; opacity:0; pointer-events:none; transition:.2s; z-index:90}
.toast.on{opacity:1; transform:translateX(-50%) translateY(0)}

@media (max-width:900px){
  .layout{grid-template-columns:1fr; height:auto}
  .list{max-height:52vh; border-right:none; border-bottom:1px solid var(--line)}
  .actbar{left:0}
  .detail{min-height:60vh}
  .searchbox{display:none}
}
</style>
</head>
<body>

<div class="top">
  <h1>审核台</h1>
  <span class="tagline">医院资质 · 救治集资 · 救治动态</span>
  <div class="stats" id="stats"></div>
  <button class="ghost" onclick="reload()">刷新数据</button>
</div>

<div class="filters">
  <div class="fgroup">
    <span class="flabel">状态</span>
    <button class="chip on" data-state="pending" onclick="setState('pending')">待审核<span class="n" id="cPending"></span></button>
    <button class="chip" data-state="passed" onclick="setState('passed')">已通过<span class="n" id="cPassed"></span></button>
    <button class="chip" data-state="rejected" onclick="setState('rejected')">已驳回<span class="n" id="cRejected"></span></button>
    <button class="chip" data-state="all" onclick="setState('all')">全部<span class="n" id="cAll"></span></button>
  </div>
  <div class="fgroup">
    <span class="flabel">类型</span>
    <button class="chip on" data-type="all" onclick="setType('all')">全部</button>
    <button class="chip" data-type="hospital" onclick="setType('hospital')">医院认证</button>
    <button class="chip" data-type="fundraiser" onclick="setType('fundraiser')">救治集资</button>
    <button class="chip" data-type="treatment" onclick="setType('treatment')">救治动态</button>
  </div>
  <div class="searchbox">
    <input id="q" placeholder="搜索标题 / 提交人" oninput="renderList()">
  </div>
</div>

<div class="layout">
  <div class="list">
    <div class="list-head" id="listHead">加载中…</div>
    <div id="listBody"></div>
  </div>
  <div class="detail" id="detail">
    <div class="detail-empty">左侧选择一条记录开始审核<br><span style="font-size:12px">待审核记录会排在最前面</span></div>
  </div>
</div>

<div class="actbar" id="actbar" style="display:none"></div>
<div id="modalRoot"></div>
<div class="toast" id="toast"></div>

<script>
/* ============================================================
   一、本地数据（与主程序共用 localStorage + IndexedDB）
   ============================================================ */
const DB_KEY = 'pk_trial_db_v1';
const IDB_NAME = 'pk_trial_media';

function loadDB(){
  try{
    const raw = localStorage.getItem(DB_KEY);
    const db = raw ? JSON.parse(raw) : null;
    if(!db || !db.rows) return { rows:{} };
    return db;
  }catch(e){ return { rows:{} }; }
}
function saveDB(db){
  localStorage.setItem(DB_KEY, JSON.stringify(db));
}
function idbGet(key){
  return new Promise((res,rej)=>{
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = ()=>{ const d=r.result; if(!d.objectStoreNames.contains('files')) d.createObjectStore('files'); };
    r.onsuccess = ()=>{
      const d = r.result;
      if(!d.objectStoreNames.contains('files')) return res(null);
      const tx = d.transaction('files','readonly');
      const g = tx.objectStore('files').get(key);
      g.onsuccess = ()=>res(g.result || null);
      g.onerror = ()=>rej(g.error);
    };
    r.onerror = ()=>rej(r.error);
  });
}
const mediaCache = new Map();
async function mediaURL(path){
  if(!path) return null;
  if(/^data:/i.test(path)) return path;
  if(/^https?:/i.test(path)) return path;
  if(path.startsWith('local://')){
    if(mediaCache.has(path)) return mediaCache.get(path);
    const blob = await idbGet(path.slice(8));
    if(!blob) return null;
    const url = URL.createObjectURL(blob);
    mediaCache.set(path, url);
    return url;
  }
  return path;
}
function mediaKind(path){
  if(!path) return '';
  if(/\.(mp4|mov|webm|m4v)$/i.test(path)) return 'video';
  return 'image';
}

/* ============================================================
   二、参考规范条款（依据抖音社区自律公约 / 医疗健康内容规范
        / 公益募捐相关规则整理，非官方原文，按内部审核口径使用）
   ============================================================ */
const CLAUSE_GROUPS = [
  {
    key:'qual', title:'A · 主体与资质',
    items:[
      { code:'A1', lv:'block', name:'证照缺失或无法辨认',
        desc:'营业执照/动物诊疗许可证缺失，或照片模糊、裁切、反光、遮盖关键信息，无法核验证照编号与有效期。' },
      { code:'A2', lv:'block', name:'主体信息不一致',
        desc:'证照登记名称与提交的医院名称、收款账户主体不一致，或存在借用、挂靠资质嫌疑。' },
      { code:'A3', lv:'block', name:'经营范围不含动物诊疗',
        desc:'执照经营范围未包含动物诊疗/宠物医疗服务，不具备开展救治的主体资格。' },
      { code:'A4', lv:'block', name:'证件疑似伪造或已过期',
        desc:'证件存在拼接、涂改、PS 痕迹，或有效期已届满、年检信息缺失。' },
      { code:'A5', lv:'fix', name:'联系方式无法核验',
        desc:'联系电话空号、非本地号段，或与执照登记信息不符，需补充可核验的联系方式。' }
    ]
  },
  {
    key:'med', title:'B · 医疗内容与图片',
    items:[
      { code:'B1', lv:'block', name:'血腥伤口未处理',
        desc:'开放性伤口、手术切口、大量血迹等特写未经遮挡或打码，易引起强烈不适。' },
      { code:'B2', lv:'block', name:'展示虐待或伤害过程',
        desc:'出现殴打、踩踏、烫伤、捕兽夹等伤害过程画面，或对受伤过程进行展示性描述。' },
      { code:'B3', lv:'block', name:'引人不适画面',
        desc:'尸体、濒死状态、寄生虫、腐坏组织等特写，超出必要医疗记录范围。' },
      { code:'B4', lv:'fix', name:'非医疗必要的强制保定',
        desc:'动物被绳索、胶带、器械过度束缚，且非手术或检查所必需，需补充说明必要性。' },
      { code:'B5', lv:'block', name:'疑似摆拍或人为致伤',
        desc:'伤情与描述不符、存在人为制造伤情的迹象，或同一动物反复出现"受伤—募捐"循环。' },
      { code:'B6', lv:'warn', name:'夸大疗效或绝对化表述',
        desc:'使用"包治""百分百治愈""最权威"等绝对化、承诺性用语，建议修改后再发布。' }
    ]
  },
  {
    key:'fund', title:'C · 募捐与资金',
    items:[
      { code:'C1', lv:'block', name:'金额与病情明显不符',
        desc:'募捐目标金额显著高于病历、诊断证明或医院报价单所列费用，且无合理解释。' },
      { code:'C2', lv:'fix', name:'缺少费用凭证',
        desc:'未提供诊断证明、费用清单、治疗方案报价等任一凭证，无法判断金额合理性。' },
      { code:'C3', lv:'block', name:'收款账户异常',
        desc:'使用个人账户收款，或收款账户与认证医院主体不一致，存在资金安全风险。' },
      { code:'C4', lv:'block', name:'同一病例重复募捐',
        desc:'同一救助档案已有进行中或已完成的募捐，仍再次发起，或拆分金额多次发起。' },
      { code:'C5', lv:'block', name:'承诺回报诱导捐赠',
        desc:'以返现、抽奖、赠品、曝光引流等回报方式诱导捐赠，不属于公益募捐范畴。' },
      { code:'C6', lv:'block', name:'已达标仍继续募捐',
        desc:'募捐金额已达到或超过目标，通道未关闭仍继续收款。' }
    ]
  },
  {
    key:'safe', title:'D · 内容与信息安全',
    items:[
      { code:'D1', lv:'block', name:'站外引流',
        desc:'图片或文字中出现微信号、二维码、私人电话、外部链接、群号等站外联系方式。' },
      { code:'D2', lv:'block', name:'泄露他人隐私',
        desc:'出现可识别的人脸、门牌号、车牌、身份证件、病历姓名等个人信息，未做遮挡。' },
      { code:'D3', lv:'fix', name:'涉嫌侵犯著作权',
        desc:'使用他人拍摄的照片、影视截图、品牌 Logo 等，未获得授权且非必要使用。' },
      { code:'D4', lv:'block', name:'涉及保护动物或野生动物交易',
        desc:'涉及国家重点保护野生动物的捕捉、买卖、饲养，或野生动物制品交易。' },
      { code:'D5', lv:'block', name:'违法违规或违禁内容',
        desc:'涉及赌博、色情、违禁药品、管制器具、虚假医疗广告等违法违规内容。' }
    ]
  },
  {
    key:'info', title:'E · 信息完整性',
    items:[
      { code:'E1', lv:'fix', name:'关键信息缺失',
        desc:'缺少动物当前状态、伤情描述、所在位置、发现时间等关键信息，无法判断救助真实性。' },
      { code:'E2', lv:'block', name:'图文不符',
        desc:'照片与文字描述明显不是同一只动物、同一处伤情，或图片与所述病例无关。' },
      { code:'E3', lv:'fix', name:'时间线矛盾',
        desc:'治疗记录时间早于救助时间、抓捕时间晚于手术时间等逻辑冲突，需核实。' }
    ]
  }
];
const ALL_CLAUSES = CLAUSE_GROUPS.flatMap(g=>g.items.map(i=>({...i, group:g.title})));
const clauseByCode = c => ALL_CLAUSES.find(x=>x.code===c);
const LV_TEXT = { block:'阻断', fix:'需补正', warn:'提醒' };

/* ============================================================
   三、把三张表归一成"审核单"
   ============================================================ */
const TYPE_META = {
  hospital:{ label:'医院认证', cls:'hospital' },
  fundraiser:{ label:'救治集资', cls:'fundraiser' },
  treatment:{ label:'救治动态', cls:'treatment' }
};
const STATE_META = {
  pending:{ label:'待审核', cls:'pending' },
  passed:{ label:'已通过', cls:'passed' },
  rejected:{ label:'已驳回', cls:'rejected' }
};

let DB = null;
let ITEMS = [];
let curState = 'pending';
let curType = 'all';
let curId = null;   /* `${type}:${id}` */

function fmtTime(iso){
  if(!iso) return '—';
  const d = new Date(iso);
  if(isNaN(d)) return String(iso);
  const p = n=>String(n).padStart(2,'0');
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function ago(iso){
  const t = new Date(iso).getTime();
  if(!t) return '';
  const m = Math.floor((Date.now()-t)/60000);
  if(m < 1) return '刚刚';
  if(m < 60) return m+' 分钟前';
  if(m < 1440) return Math.floor(m/60)+' 小时前';
  return Math.floor(m/1440)+' 天前';
}
function money(n){ return '¥' + Number(n||0).toLocaleString('zh-CN'); }

function stateOf(type, row){
  if(type==='hospital') return row.status==='approved' ? 'passed' : row.status==='rejected' ? 'rejected' : 'pending';
  if(type==='fundraiser') return (row.status==='active'||row.status==='closed') ? 'passed' : row.status==='rejected' ? 'rejected' : 'pending';
  const s = row.audit_status || 'pending';   /* 救治动态：新字段，旧数据视为待审 */
  return s==='approved' ? 'passed' : s==='rejected' ? 'rejected' : 'pending';
}

function buildItems(){
  const rows = DB.rows || {};
  const reps = rows.stray_reports || [];
  const trs = rows.treatments || [];
  const logs = rows.audit_logs || [];
  const out = [];

  (rows.hospitals||[]).forEach(h=>{
    out.push({
      key:'hospital:'+h.id, type:'hospital', id:h.id, row:h,
      title: h.name || '（未命名医院）',
      by: h.owner_id || '—',
      at: h.created_at,
      state: stateOf('hospital', h),
      extra: h.license_no ? '执照 ' + h.license_no : '',
      images: h.license_path ? [{ path:h.license_path, label:'营业执照' }] : [],
      logs: logs.filter(l=>l.target_type==='hospital' && String(l.target_id)===String(h.id))
    });
  });

  (rows.fundraisers||[]).forEach(f=>{
    const rep = reps.find(r=>String(r.id)===String(f.rescue_id)) || null;
    const trt = trs.filter(t=>String(t.rescue_id)===String(f.rescue_id));
    const images = [];
    if(rep && rep.photo_path) images.push({ path:rep.photo_path, label:'求助现场照' });
    if(rep && rep.capture_photo) images.push({ path:rep.capture_photo, label:'捕捉凭证' });
    trt.forEach((t,i)=>(t.media||[]).forEach((m,j)=>images.push({ path:m, label:`治疗记录 ${i+1}-${j+1}` })));
    out.push({
      key:'fundraiser:'+f.id, type:'fundraiser', id:f.id, row:f,
      title: f.title || '（未命名集资）',
      by: f.hospital_name || f.owner_id || '—',
      at: f.created_at,
      state: stateOf('fundraiser', f),
      extra: `${money(f.target)} · 已筹 ${money(f.raised)}`,
      images, rep, treatments: trt,
      logs: logs.filter(l=>l.target_type==='fundraiser' && String(l.target_id)===String(f.id))
    });
  });

  (rows.treatments||[]).forEach(t=>{
    const rep = reps.find(r=>String(r.id)===String(t.rescue_id)) || null;
    out.push({
      key:'treatment:'+t.id, type:'treatment', id:t.id, row:t,
      title: (t.content||'（无内容）').slice(0,42),
      by: t.hospital_name || t.owner_id || '—',
      at: t.created_at,
      state: stateOf('treatment', t),
      extra: ((t.media||[]).length ? (t.media.length)+' 个附件' : '无附件'),
      images: (t.media||[]).map((m,i)=>({ path:m, label:'动态附件 '+(i+1) })),
      rep,
      logs: logs.filter(l=>l.target_type==='treatment' && String(l.target_id)===String(t.id))
    });
  });

  out.sort((a,b)=>{
    const pa = a.state==='pending'?0:1, pb = b.state==='pending'?0:1;
    if(pa!==pb) return pa-pb;
    return new Date(b.at||0) - new Date(a.at||0);
  });
  return out;
}

/* ============================================================
   四、渲染
   ============================================================ */
function render(){
  DB = loadDB();
  ITEMS = buildItems();
  renderStats();
  renderList();
  if(curId){
    const it = ITEMS.find(x=>x.key===curId);
    if(it) renderDetail(it); else { curId=null; renderDetail(null); }
  } else {
    renderDetail(null);
  }
}
function renderStats(){
  const pend = ITEMS.filter(i=>i.state==='pending').length;
  const pass = ITEMS.filter(i=>i.state==='passed').length;
  const rej  = ITEMS.filter(i=>i.state==='rejected').length;
  const byType = t => ITEMS.filter(i=>i.type===t && i.state==='pending').length;
  document.getElementById('stats').innerHTML = `
    <div class="stat pend"><b>${pend}</b>待审核</div>
    <div class="stat"><b>${byType('hospital')}</b>医院</div>
    <div class="stat"><b>${byType('fundraiser')}</b>集资</div>
    <div class="stat"><b>${byType('treatment')}</b>动态</div>
    <div class="stat ok"><b>${pass}</b>已通过</div>
    <div class="stat bad"><b>${rej}</b>已驳回</div>`;
  const set = (id,n)=>{ const el=document.getElementById(id); if(el) el.textContent = n; };
  set('cPending', pend); set('cPassed', pass); set('cRejected', rej); set('cAll', ITEMS.length);
}
function setState(s){
  curState = s;
  document.querySelectorAll('[data-state]').forEach(b=>b.classList.toggle('on', b.dataset.state===s));
  renderList();
}
function setType(t){
  curType = t;
  document.querySelectorAll('[data-type]').forEach(b=>b.classList.toggle('on', b.dataset.type===t));
  renderList();
}
function filtered(){
  const q = (document.getElementById('q').value||'').trim().toLowerCase();
  return ITEMS.filter(i=>{
    if(curState!=='all' && i.state!==curState) return false;
    if(curType!=='all' && i.type!==curType) return false;
    if(q && !(i.title+' '+i.by).toLowerCase().includes(q)) return false;
    return true;
  });
}
function renderList(){
  const list = filtered();
  document.getElementById('listHead').textContent =
    `${list.length} 条记录 · ${STATE_META[curState].label}${curType!=='all'?' · '+TYPE_META[curType].label:''}`;
  const box = document.getElementById('listBody');
  if(!list.length){
    const noData = !ITEMS.length;
    box.innerHTML = `<div class="empty">${noData
      ? '还没有任何提交记录<br><span style="font-size:12px">先打开主程序（index.html），用医院账号提交一次认证或集资</span>'
      : '当前筛选条件下没有记录'}</div>`;
    return;
  }
  box.innerHTML = list.map(i=>{
    const t = TYPE_META[i.type], s = STATE_META[i.state];
    const thumbs = i.images.slice(0,3).map(im=>`<img data-src="${esc(im.path)}" alt="">`).join('');
    return `<div class="item ${i.key===curId?'on':''}" onclick="openItem('${i.key}')">
      <div class="item-top">
        <span class="ty ${t.cls}">${t.label}</span>
        <span class="st ${s.cls}">${s.label}</span>
      </div>
      <div class="item-title">${esc(i.title)}</div>
      <div class="item-meta">
        <span>${esc(i.by)}</span>
        <span>${fmtTime(i.at)}</span>
        ${i.extra?`<span>${esc(i.extra)}</span>`:''}
      </div>
      ${thumbs?`<div class="item-thumbs">${thumbs}</div>`:''}
    </div>`;
  }).join('');
  hydrateThumbs(box);
}
async function hydrateThumbs(root){
  for(const img of root.querySelectorAll('img[data-src]')){
    const p = img.dataset.src;
    try{
      const url = await mediaURL(p);
      if(url) img.src = url;
    }catch(e){}
    img.removeAttribute('data-src');
  }
}
function esc(s){
  return String(s==null?'':s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ---------- 详情 ---------- */
function openItem(key){
  curId = key;
  const it = ITEMS.find(x=>x.key===key);
  renderList();
  renderDetail(it);
}
function kvRow(k, v, mono){
  if(v===undefined || v===null || v==='') v = '—';
  return `<div class="kv-row"><div class="kv-k">${esc(k)}</div><div class="kv-v ${mono?'mono':''}">${esc(v)}</div></div>`;
}
function renderDetail(it){
  const d = document.getElementById('detail');
  const bar = document.getElementById('actbar');
  if(!it){
    d.innerHTML = '<div class="detail-empty">左侧选择一条记录开始审核<br><span style="font-size:12px">待审核记录会排在最前面</span></div>';
    bar.style.display = 'none';
    return;
  }
  const t = TYPE_META[it.type], s = STATE_META[it.state];
  const row = it.row;
  let body = '';

  /* 自动风险提示 */
  body += riskPanel(it);

  if(it.type==='hospital'){
    body += `<div class="sect"><h3>提交信息</h3><div class="kv">
      ${kvRow('医院名称', row.name)}
      ${kvRow('执照编号', row.license_no, true)}
      ${kvRow('地址', row.address)}
      ${kvRow('联系电话', row.phone, true)}
      ${kvRow('简介', row.intro)}
      ${kvRow('提交人 ID', row.owner_id, true)}
      ${kvRow('提交时间', fmtTime(row.created_at))}
      ${kvRow('当前状态', s.label)}
    </div></div>`;
  }
  if(it.type==='fundraiser'){
    body += `<div class="sect"><h3>募捐信息</h3><div class="kv">
      ${kvRow('集资标题', row.title)}
      ${kvRow('发起医院', row.hospital_name)}
      ${kvRow('目标金额', money(row.target))}
      ${kvRow('已筹金额', money(row.raised))}
      ${kvRow('关联救助编号', '#'+row.rescue_id, true)}
      ${kvRow('提交时间', fmtTime(row.created_at))}
      ${kvRow('当前状态', s.label)}
    </div></div>`;
    if(it.rep){
      body += `<div class="sect"><h3>关联求助 / 病例资料</h3><div class="kv">
        ${kvRow('求助人', it.rep.reporter_name || it.rep.owner_id)}
        ${kvRow('情况描述', it.rep.description)}
        ${kvRow('位置', it.rep.lat ? (+it.rep.lat).toFixed(5)+', '+(+it.rep.lng).toFixed(5) : '—', true)}
        ${kvRow('求助时间', fmtTime(it.rep.created_at))}
        ${kvRow('救助状态', it.rep.status)}
        ${kvRow('捕捉医院', it.rep.captured_by_name)}
        ${kvRow('捕捉时间', fmtTime(it.rep.captured_at))}
      </div></div>`;
    }
    if(it.treatments && it.treatments.length){
      body += `<div class="sect"><h3>治疗记录（${it.treatments.length}）</h3><div class="kv">
        ${it.treatments.map((tr,i)=>`<div class="kv-row"><div class="kv-k">${fmtTime(tr.created_at)}</div>
          <div class="kv-v">${esc(tr.content)}${(tr.media||[]).length?`<div style="color:var(--hint);font-size:11.5px;margin-top:3px">含 ${tr.media.length} 个附件</div>`:''}</div></div>`).join('')}
      </div></div>`;
    }else{
      body += `<div class="sect"><h3>治疗记录</h3><div class="kv"><div class="kv-row"><div class="kv-v" style="color:var(--hint)">暂无治疗记录 —— 金额依据需靠费用凭证核验</div></div></div></div>`;
    }
  }
  if(it.type==='treatment'){
    body += `<div class="sect"><h3>动态内容</h3><div class="kv">
      ${kvRow('发布医院', row.hospital_name)}
      ${kvRow('正文', row.content)}
      ${kvRow('关联救助编号', '#'+row.rescue_id, true)}
      ${kvRow('发布时间', fmtTime(row.created_at))}
      ${kvRow('当前状态', s.label)}
    </div></div>`;
    if(it.rep){
      body += `<div class="sect"><h3>关联求助</h3><div class="kv">
        ${kvRow('求助人', it.rep.reporter_name || it.rep.owner_id)}
        ${kvRow('情况描述', it.rep.description)}
      </div></div>`;
    }
  }

  /* 图片 */
  body += `<div class="sect"><h3>原始上传图片（${it.images.length}）</h3>${
    it.images.length
      ? `<div class="imgs" id="imgGrid">${it.images.map((im,idx)=>`
          <div class="imgbox" onclick="openLB('${it.key}',${idx})">
            <img data-full="${esc(im.path)}" alt="${esc(im.label)}">
            <div class="cap">${esc(im.label)}</div>
          </div>`).join('')}</div>`
      : `<div class="kv"><div class="kv-row"><div class="kv-v" style="color:var(--hint)">该记录没有任何图片凭证</div></div></div>`
  }</div>`;

  /* 审核历史 */
  body += `<div class="sect"><h3>审核记录（${it.logs.length}）</h3>${
    it.logs.length
      ? `<div class="log">${it.logs.slice().sort((a,b)=>new Date(b.created_at)-new Date(a.created_at)).map(l=>`
          <div class="log-item ${l.result==='passed'?'ok':'bad'}">
            <div class="lt">${l.result==='passed'?'通过':'驳回'}</div>
            <div class="lm">${fmtTime(l.created_at)} · 审核人 ${esc(l.reviewer||'—')}</div>
            ${l.reason?`<div class="lr">${esc(l.reason)}</div>`:''}
          </div>`).join('')}</div>`
      : `<div class="kv"><div class="kv-row"><div class="kv-v" style="color:var(--hint)">还没有审核记录</div></div></div>`
  }</div>`;

  d.innerHTML = `
    <div class="dhead">
      <div class="row1"><span class="ty ${t.cls}">${t.label}</span><span class="st ${s.cls}">${s.label}</span></div>
      <h2>${esc(it.title)}</h2>
      <div class="meta">
        <span>提交人：${esc(it.by)}</span>
        <span>${fmtTime(it.at)}（${ago(it.at)}）</span>
        <span>编号 ${esc(it.key)}</span>
      </div>
    </div>
    <div class="dbody">${body}</div>`;

  for(const img of d.querySelectorAll('img[data-full]')){
    const p = img.dataset.full;
    mediaURL(p).then(u=>{ if(u) img.src = u; }).catch(()=>{});
  }
  /* 操作条 */
  bar.style.display = 'flex';
  bar.innerHTML = `<span class="hint">${it.state==='pending'
      ? '审核结论会立即写入本地数据，并生成一条不可编辑的审核记录'
      : '这条记录已审核过，可以重新审核覆盖结论（历史记录会保留）'}</span>
    <button class="btn reject" onclick="openReject()">驳回</button>
    <button class="btn pass" onclick="doPass()">通过</button>`;
}

/* 自动风险提示：把能用规则判断的问题先摆出来 */
function riskPanel(it){
  const risks = [];
  const row = it.row;
  if(!it.images.length) risks.push('该记录<b>没有任何图片凭证</b>，无法核验真实性（对应 E1）');
  if(it.type==='hospital'){
    const lic = String(row.license_no||'');
    if(!row.license_path) risks.push('未上传营业执照照片（对应 A1）');
    if(lic && (lic.length < 15 || lic.length > 20)) risks.push(`执照编号长度为 ${lic.length} 位，与 18 位统一社会信用代码格式不符（对应 A4）`);
    if(!row.phone) risks.push('未填写联系电话（对应 A5）');
    else if(!/^[0-9-+\s]{7,}$/.test(String(row.phone))) risks.push('联系电话格式异常（对应 A5）');
  }
  if(it.type==='fundraiser'){
    const cents = Number(row.target||0);
    if(cents >= 5000) risks.push(`目标金额 ${money(cents)} 超过 5000 元，需费用清单/报价单佐证（对应 C1、C2）`);
    if(!(it.treatments||[]).length) risks.push('没有任何治疗记录，金额缺少病情依据（对应 C2）');
    const same = (DB.rows.fundraisers||[]).filter(f=>String(f.rescue_id)===String(row.rescue_id) && String(f.id)!==String(row.id));
    const dup = same.filter(f=>f.status==='active' || f.status==='closed');
    if(dup.length) risks.push(`同一救助已有 ${dup.length} 条进行中/已完成的募捐，疑似重复募捐（对应 C4）`);
    if(Number(row.raised||0) >= Number(row.target||0) && row.status==='active') risks.push('已筹金额已达目标仍处于集资中（对应 C6）');
    if(Number(row.target||0) <= 0) risks.push('目标金额无效（对应 C1）');
  }
  if(it.type==='treatment'){
    const txt = String(row.content||'');
    if(txt.length < 15) risks.push('正文过短，关键信息不足（对应 E1）');
    if(/(微信|vx|VX|加我|二维码|扫码|qq|QQ)/.test(txt)) risks.push('正文疑似含站外联系方式（对应 D1）');
    if(/(包治|百分百|100%|最权威|第一)/.test(txt)) risks.push('正文含绝对化/承诺性用语（对应 B6）');
    if(row.rescue_id && it.rep && row.created_at && it.rep.created_at &&
       new Date(row.created_at) < new Date(it.rep.created_at)) risks.push('发布时间早于关联求助时间，时间线矛盾（对应 E3）');
  }
  const desc = String(it.rep && it.rep.description || row.description || '');
  if(desc && desc.length < 12) risks.push('描述字数过少，关键信息可能缺失（对应 E1）');
  if(/(微信|vx|加我|扫码|二维码)/i.test(desc)) risks.push('描述中疑似含站外引流信息（对应 D1）');

  if(!risks.length) return `<div class="risk clear"><h4>自动预检</h4><p>未发现规则可判定的风险项，仍需人工核对图片与凭证。</p></div>`;
  return `<div class="risk"><h4>自动预检发现 ${risks.length} 项需关注</h4><ul>${risks.map(r=>`<li>${r}</li>`).join('')}</ul></div>`;
}

/* ============================================================
   五、审核动作
   ============================================================ */
function nowISO(){ return new Date().toISOString(); }
function curItem(){ return ITEMS.find(x=>x.key===curId) || null; }

function doPass(){
  const it = curItem(); if(!it) return;
  const msg = it.state==='pending'
    ? `确认通过「${it.title}」？\n\n通过后该记录会对用户端可见${it.type==='fundraiser'?'，集资通道立即开放':''}。`
    : `这条记录已经审核过（${STATE_META[it.state].label}），确认重新判定为通过？`;
  if(!confirm(msg)) return;
  applyReview(it, 'passed', '', []);
}

function openReject(){
  const it = curItem(); if(!it) return;
  const html = `
    <div class="mask" onclick="if(event.target===this)closeModal()">
      <div class="modal">
        <div class="mhead">
          <h3>驳回 · ${esc(it.title)}</h3>
          <p>勾选违反的规范条款（至少一条），驳回原因会回传给提交方</p>
        </div>
        <div class="mbody">
          ${CLAUSE_GROUPS.map(g=>`
            <div class="group">
              <h4>${g.title}</h4>
              ${g.items.map(c=>`
                <label class="clause" id="cl_${c.code}">
                  <input type="checkbox" value="${c.code}" onchange="toggleClause(this)">
                  <span class="cc">${c.code}</span>
                  <span class="ct">
                    <span class="cname">${esc(c.name)}<span class="lv ${c.lv}">${LV_TEXT[c.lv]}</span></span>
                    <span class="cdesc">${esc(c.desc)}</span>
                  </span>
                </label>`).join('')}
            </div>`).join('')}
          <div class="note">
            <label>补充说明（选填，会拼在条款后一起回传）</label>
            <textarea id="rjNote" placeholder="例如：请重新拍摄营业执照，需能看清统一社会信用代码与有效期"></textarea>
          </div>
        </div>
        <div class="mfoot">
          <span class="tip" id="rjTip">尚未选择条款</span>
          <button class="btn plain" onclick="closeModal()">取消</button>
          <button class="btn reject" id="rjGo" onclick="submitReject()" disabled>确认驳回</button>
        </div>
      </div>
    </div>`;
  document.getElementById('modalRoot').innerHTML = html;
}
function toggleClause(cb){
  cb.closest('.clause').classList.toggle('sel', cb.checked);
  const picked = [...document.querySelectorAll('.mbody input[type=checkbox]:checked')].map(x=>x.value);
  const tip = document.getElementById('rjTip');
  const go = document.getElementById('rjGo');
  const blocking = picked.some(c=>(clauseByCode(c)||{}).lv==='block');
  tip.textContent = picked.length
    ? `已选 ${picked.length} 条${blocking?' · 含阻断级':''}`
    : '尚未选择条款';
  go.disabled = picked.length === 0;
  go.textContent = blocking ? '确认驳回（阻断级）' : '确认驳回';
}
function closeModal(){ document.getElementById('modalRoot').innerHTML = ''; }

function submitReject(){
  const it = curItem(); if(!it) return;
  const picked = [...document.querySelectorAll('.mbody input[type=checkbox]:checked')].map(x=>x.value);
  if(!picked.length) return toast('请至少勾选一条违规条款');
  const note = document.getElementById('rjNote').value.trim();
  const lines = picked.map(c=>{
    const cl = clauseByCode(c);
    return `【${cl.code} ${cl.name}】${cl.desc}`;
  });
  const reason = lines.join('\n') + (note ? `\n补充说明：${note}` : '');
  closeModal();
  applyReview(it, 'rejected', reason, picked);
}

function applyReview(it, result, reason, clauses){
  DB = loadDB();
  const rows = DB.rows;
  rows.audit_logs = rows.audit_logs || [];
  const log = {
    id: (Math.max(0, ...rows.audit_logs.map(l=>Number(l.id)||0)) + 1),
    target_type: it.type,
    target_id: it.id,
    target_title: it.title,
    target_owner: it.by,
    result,
    reason: reason || '',
    clauses: clauses || [],
    reviewer: '本地审核员',
    created_at: nowISO()
  };
  rows.audit_logs.push(log);

  /* 回写业务状态 */
  if(it.type==='hospital'){
    const h = (rows.hospitals||[]).find(x=>String(x.id)===String(it.id));
    if(h){ h.status = result==='passed' ? 'approved' : 'rejected';
      h.review_note = reason || ''; h.reviewed_at = log.created_at; }
  }
  if(it.type==='fundraiser'){
    const f = (rows.fundraisers||[]).find(x=>String(x.id)===String(it.id));
    if(f){
      if(result==='passed') f.status = (Number(f.raised||0) >= Number(f.target||0)) ? 'closed' : 'active';
      else f.status = 'rejected';
      f.review_note = reason || ''; f.reviewed_at = log.created_at;
    }
  }
  if(it.type==='treatment'){
    const t = (rows.treatments||[]).find(x=>String(x.id)===String(it.id));
    if(t){ t.audit_status = result==='passed' ? 'approved' : 'rejected';
      t.review_note = reason || ''; t.reviewed_at = log.created_at; }
  }

  saveDB(DB);
  toast(result==='passed' ? '已通过，结论已写入本地数据' : '已驳回，原因已记录');
  render();
}

/* ============================================================
   六、图片放大
   ============================================================ */
let lbList = [], lbIdx = 0;
async function openLB(key, idx){
  const it = ITEMS.find(x=>x.key===key); if(!it) return;
  lbList = it.images; lbIdx = idx;
  await showLB();
}
async function showLB(){
  const im = lbList[lbIdx]; if(!im) return;
  const url = await mediaURL(im.path);
  document.getElementById('modalRoot').innerHTML = `
    <div class="lb" onclick="if(event.target===this)closeModal()">
      <button class="x" onclick="closeModal()">×</button>
      ${lbList.length>1?`<button class="nav prev" onclick="lbStep(-1)">‹</button>
      <button class="nav next" onclick="lbStep(1)">›</button>`:''}
      ${mediaKind(im.path)==='video'
        ? `<video src="${url||''}" controls style="max-width:88vw;max-height:84vh"></video>`
        : `<img src="${url||''}" alt="${esc(im.label)}">`}
      <div class="count">${esc(im.label)} · ${lbIdx+1}/${lbList.length}</div>
    </div>`;
}
function lbStep(d){
  lbIdx = (lbIdx + d + lbList.length) % lbList.length;
  showLB();
}

/* ============================================================
   七、启动
   ============================================================ */
function reload(){
  mediaCache.clear();
  render();
  toast('已重新读取本地数据');
}
let toastTimer = null;
function toast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('on');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>t.classList.remove('on'), 2200);
}
document.addEventListener('keydown', e=>{
  if(e.key==='Escape') closeModal();
});
/* 主程序在另一个标签页写入时，自动同步 */
window.addEventListener('storage', e=>{
  if(e.key === DB_KEY) render();
});
render();
</script>
</body>
</html>
"""

CSS = _html_or_inline("_t_css.txt", CSS)
BODY = _html_or_inline("_t_body.txt", BODY)
JS = _html_or_inline("_t_js9.txt", JS)

EXTRA_CSS = """
/* ---- 放大使用场景：桌面端容器更宽 ---- */
#phone{max-width:760px}
@media (min-width:900px){
  #phone{max-width:880px; box-shadow:0 0 0 1px rgba(0,0,0,.04), 0 24px 80px rgba(34,48,31,.16)}
}
header .hd-btn:active{transform:scale(.96)}
"""


CSS = CSS.rstrip() + "\n" + EXTRA_CSS


def build_index():
    return "".join([
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head>\n",
        "<meta charset=\"utf-8\">\n",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n",
        "<title>", APP_TITLE, "</title>\n",
        "<script>\n  window._TMapSecurityConfig = { serviceHost: '' };\n</script>\n",
        "<script src=\"https://map.qq.com/api/gljs?v=1.exp\"></script>\n",
        "<style>\n", CSS, "\n</style>\n</head>\n<body>\n",
        BODY, "\n<script>\n", SEED_JS, "\n", JS, "\n</script>\n</body>\n</html>\n",
    ])


# ============================================================ HTTP 服务


class Handler(BaseHTTPRequestHandler):
    server_version = "PokemonPals/1.0"
    protocol_version = "HTTP/1.1"

    # ---------- 工具 ----------
    def log_message(self, fmt, *args):
        if os.environ.get("QUIET") != "1":
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _err(self, msg, code=400):
        self._json({"error": str(msg)}, code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        if n > MAX_BODY:
            raise ValueError("请求体过大（上限 %d MB）" % (MAX_BODY // 1024 // 1024))
        return self.rfile.read(n)

    # ---------- 路由 ----------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if not JS or not CSS:
                return self._err("缺少 _t_css.txt / _t_body.txt / _t_js5.txt，"
                                 "请把它们与 宝可梦.py 放在同一目录", 500)
            return self._send(200, build_index(), "text/html; charset=utf-8")
        if path in ("/admin", "/admin.html"):
            if not ADMIN_HTML:
                return self._send(404, "<h3>未找到 admin.html</h3>", "text/html; charset=utf-8")
            return self._send(200, ADMIN_HTML, "text/html; charset=utf-8")
        if path == "/api/stats":
            with DB_LOCK:
                db = load_db()
            return self._json({t: len(db["rows"].get(t) or []) for t in TABLES})
        if path == "/api/models":
            return self._json(models_payload())
        if path == "/api/amap_key":
            return self._json({"key": load_settings().get("amap_key")})
        if path.startswith("/uploads/"):
            return self._serve_file(path[len("/uploads/"):])
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        return self._err("Not Found: " + path, 404)

    def _serve_file(self, rel):
        rel = posixpath.normpath("/" + rel).lstrip("/")
        full = os.path.abspath(os.path.join(UPLOAD_DIR, rel))
        if not full.startswith(os.path.abspath(UPLOAD_DIR) + os.sep) or not os.path.isfile(full):
            return self._err("Not Found", 404)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/query":
                q = json.loads(self._body().decode("utf-8") or "{}")
                if (q.get("op") or "select") == "insert" and q.get("table") == "pokedex":
                    pass
                return self._json(run_query(q))
            if path == "/api/upload":
                return self._json(self._handle_upload())
            if path == "/api/vision":
                j = json.loads(self._body().decode("utf-8") or "{}")
                return self._json(ai_recognize(j.get("image") or ""))
            if path == "/api/reset":
                with DB_LOCK:
                    save_db(_blank())
                    d = load_db()
                    seed_demo(d)
                    save_db(d)
                return self._json({"ok": True})
            if path == "/api/amap_key":
                j = json.loads(self._body().decode("utf-8") or "{}")
                k = (j.get("key") or "").strip()
                s = load_settings()
                s["amap_key"] = k
                save_settings(s)
                return self._json({"ok": True, "key": k})
            if path == "/api/amap_search":
                j = json.loads(self._body().decode("utf-8") or "{}")
                return self._json(amap_search(j.get("name") or "", j.get("address") or ""))
            return self._err("Not Found: " + path, 404)
        except Exception as e:
            return self._err(e, 500)

    def _handle_upload(self):
        ctype = self.headers.get("Content-Type") or ""
        m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype)
        if "multipart/form-data" not in ctype or not m:
            raise ValueError("请用 multipart/form-data 上传")
        boundary = (m.group(1) or m.group(2)).strip().encode()
        raw = self._body()
        parts = raw.split(b"--" + boundary)
        file_bytes, filename, folder = None, "file", "misc"
        for part in parts:
            if b"\r\n\r\n" not in part:
                continue
            head, content = part.split(b"\r\n\r\n", 1)
            content = content.rstrip(b"\r\n")
            if content.endswith(b"--"):
                content = content[:-2]
            head_txt = head.decode("utf-8", "replace")
            nm = re.search(r'name="([^"]*)"', head_txt)
            if not nm:
                continue
            name = nm.group(1)
            if name == "folder":
                folder = content.decode("utf-8", "replace").strip() or "misc"
            elif name == "file":
                file_bytes = content
                fn = re.search(r'filename="([^"]*)"', head_txt)
                if fn and fn.group(1):
                    filename = fn.group(1)
        if not file_bytes:
            raise ValueError("没有收到文件")
        folder = re.sub(r"[^a-zA-Z0-9_-]", "", folder) or "misc"
        ext = (os.path.splitext(filename)[1] or ".jpg").lower()
        ext = re.sub(r"[^a-z0-9.]", "", ext)[:8] or ".jpg"
        d = os.path.join(UPLOAD_DIR, folder)
        os.makedirs(d, exist_ok=True)
        fn = "%d-%s%s" % (int(time.time() * 1000), uuid.uuid4().hex[:6], ext)
        with open(os.path.join(d, fn), "wb") as f:
            f.write(file_bytes)
        return {"path": "/uploads/%s/%s" % (folder, fn), "size": len(file_bytes)}


# ============================================================ 启动


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with DB_LOCK:
        d = load_db()
        if seed_demo(d):
            save_db(d)

    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    ip = local_ip()
    print("=" * 52)
    print("  🐾 现实版宝可梦 已启动")
    print("=" * 52)
    print("  本机访问:  http://127.0.0.1:%d" % port)
    if ip:
        print("  手机访问:  http://%s:%d   （同一 WiFi）" % (ip, port))
    print("  审核后台:  http://127.0.0.1:%d/admin" % port)
    print("  AI 识别:   %s" % ("已开启（%s）" % VISION_MODEL if ai_enabled()
                              else "未配置（设置 DASHSCOPE_API_KEY 后重启）"))
    print("  数据文件:  %s" % DATA_FILE)
    print("  图片目录:  %s" % UPLOAD_DIR)
    print("=" * 52)
    print("  Ctrl + C 停止")
    # 自动打开浏览器（设置 NO_BROWSER=1 可关闭）
    if os.environ.get("NO_BROWSER") != "1":
        try:
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(
                "http://127.0.0.1:%d" % port)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
