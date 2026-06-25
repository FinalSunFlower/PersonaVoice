"""前端 API 全流程测试 (1111.mp3 + 人格联动)."""
import os
import requests, base64, json, time, sys
from pathlib import Path

# examples/api_demo.py → 项目根目录 = 上两级
PROJECT_ROOT = Path(__file__).resolve().parent.parent
API = os.environ.get("PERSONAVOICE_API", "http://localhost:8000")
AUDIO = str(PROJECT_ROOT / "1111.mp3")

print("="*70)
print("前端 API 全流程测试 (1111.mp3 + 人格联动)")
print("="*70)

# ── 测试 1: 健康检查 ──
print("\n[1/5] 健康检查")
r = requests.get(f"{API}/api/health", timeout=10)
data = r.json()
assert data["status"] == "ok", f"健康检查失败: {data}"
assert data["personavoice_loaded"], "PersonaVoice 未加载"
print(f"  ✓ status={data['status']}, version={data['version']}")
print(f"  ✓ 所有模块: {data['modules']}")

# ── 测试 2: 架构状态 ──
print("\n[2/5] 架构状态查询")
r = requests.get(f"{API}/api/architecture", timeout=10)
arch = r.json()
print(f"  ✓ 架构状态: {list(arch.keys())[:5]}...")

# ── 测试 3: 纯声音克隆 (无人格) ──
print("\n[3/5] 纯声音克隆 (1111.mp3, 无人格)")
t0 = time.time()
with open(AUDIO, "rb") as f:
    files = {"audio": ("1111.mp3", f, "audio/mpeg")}
    data = {"text": "你好,这是前端 API 全流程测试。"}
    r = requests.post(f"{API}/api/clone", files=files, data=data, timeout=120)
elapsed = time.time() - t0
assert r.status_code == 200, f"克隆失败: {r.status_code} {r.text[:200]}"
result = r.json()
assert "audio_base64" in result, f"返回缺少 audio_base64 字段: {list(result.keys())}"
audio_b64 = result["audio_base64"]
print(f"  ✓ 耗时: {elapsed:.2f}s")
print(f"  ✓ audio_base64 长度: {len(audio_b64)}")
print(f"  ✓ 返回字段: {list(result.keys())}")
if "speaker_embedding_norm" in result:
    print(f"  ✓ speaker_emb norm: {result['speaker_embedding_norm']:.4f}")
if "wave_duration" in result:
    print(f"  ✓ 音频时长: {result['wave_duration']:.2f}s")

# 保存音频
audio_data = base64.b64decode(audio_b64)
out_path = "outputs/frontend_test_3.wav"
Path("outputs").mkdir(exist_ok=True)
Path(out_path).write_bytes(audio_data)
print(f"  ✓ 音频保存: {out_path} ({len(audio_data)} bytes)")

# ── 测试 4: 声音克隆 + 人格联动 ──
print("\n[4/5] 声音克隆 + 人格联动 (1111.mp3 + 聊天记录)")
chat_history = json.dumps([
    {"role": "user", "content": "你好,我今天心情很好,想和你聊聊。"},
    {"role": "assistant", "content": "你好!很高兴听到你心情不错,我也很期待我们的对话。你今天有什么特别的事情想分享吗?"},
    {"role": "user", "content": "我刚完成了一个重要的项目,感觉非常有成就感。"},
    {"role": "assistant", "content": "太棒了!恭喜你完成项目。能感受到你的兴奋和自豪。这是什么类型的项目呢?"},
], ensure_ascii=False)

t0 = time.time()
with open(AUDIO, "rb") as f:
    files = {"audio": ("1111.mp3", f, "audio/mpeg")}
    data = {
        "text": "刚完成了一个重要项目,感觉非常有成就感,想和你分享一下这份喜悦!",
        "chat_history": chat_history,
    }
    r = requests.post(f"{API}/api/clone", files=files, data=data, timeout=180)
elapsed = time.time() - t0
assert r.status_code == 200, f"人格克隆失败: {r.status_code} {r.text[:200]}"
result = r.json()
assert "audio_base64" in result, f"返回缺少 audio_base64 字段: {list(result.keys())}"
audio_b64 = result["audio_base64"]
print(f"  ✓ 耗时: {elapsed:.2f}s")
print(f"  ✓ audio_base64 长度: {len(audio_b64)}")
if "persona_representation_norm" in result:
    print(f"  ✓ persona_repr norm: {result['persona_representation_norm']:.4f} (应 > 0 表示人格注入)")

# 保存音频
audio_data = base64.b64decode(audio_b64)
out_path = "outputs/frontend_test_4_persona.wav"
Path(out_path).write_bytes(audio_data)
print(f"  ✓ 音频保存: {out_path} ({len(audio_data)} bytes)")

# ── 测试 5: 人格分析接口 ──
print("\n[5/5] 人格分析接口")
# api_server 期望 chat_history 是 List[str]
chat_str_list = [
    "你好,我今天心情很好,想和你聊聊。",
    "我刚完成了一个重要的项目,感觉非常有成就感。",
    "太棒了!恭喜你完成项目。能感受到你的兴奋和自豪。",
]
r = requests.post(
    f"{API}/api/persona/analyze",
    json={"chat_history": chat_str_list},
    timeout=60,
)
assert r.status_code == 200, f"人格分析失败: {r.status_code} {r.text[:200]}"
persona = r.json()
print(f"  ✓ 人格分析返回: {list(persona.keys())}")
if "trait_radar" in persona:
    print(f"  ✓ Big Five traits: {persona['trait_radar']['labels']}")
    print(f"  ✓ Big Five values: {[f'{v:.3f}' for v in persona['trait_radar']['values']]}")
if "persona_representation" in persona:
    print(f"  ✓ persona_emb: shape={persona['persona_representation']['shape']}, "
          f"norm={persona['persona_representation']['norm']:.4f}")

print("\n" + "="*70)
print("✓ 前端全流程测试通过!")
print("="*70)
