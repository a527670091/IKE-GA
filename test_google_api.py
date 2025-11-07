import os
import requests
import json
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# 从环境变量获取API密钥（支持多种环境变量名）
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ 错误：未找到Gemini API密钥")
    print("   请通过以下方式之一设置API密钥：")
    print("   1. 在.env文件中设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
    print("   2. 设置环境变量 GEMINI_API_KEY 或 GOOGLE_API_KEY")
    exit(1)

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"

print("🔵 正在测试 Gemini API 连接...")
print("=" * 60)
print(f"📝 使用的API密钥: {API_KEY[:8]}...{API_KEY[-4:]}")
print("=" * 60)

try:
    # 准备请求数据
    headers = {
        "Content-Type": "application/json",
    }
    
    data = {
        "contents": [{
            "parts": [{
                "text": "用一句话解释什么是人工智能"
            }]
        }]
    }
    
    print("📡 正在发送测试请求...")
    print(f"   请求URL: {API_URL.split('?')[0]}...")
    
    # 发送请求
    response = requests.post(
        API_URL,
        headers=headers,
        json=data,
        timeout=30
    )
    
    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            print("✅ Gemini API 连接成功！")
            print("=" * 60)
            print("📝 模型回复：")
            print(content)
            print("=" * 60)
        else:
            print("❌ API返回了意外的响应格式")
            print(f"响应内容：{json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"❌ Gemini API 请求失败！")
        print(f"   状态码：{response.status_code}")
        print(f"   错误信息：{response.text}")
        print("=" * 60)
        
except requests.exceptions.Timeout:
    print("❌ Gemini API 连接超时！")
    print("   可能是网络连接问题，请检查网络设置")
    print("=" * 60)
except requests.exceptions.ConnectionError as e:
    print("❌ Gemini API 连接失败！")
    print(f"   连接错误：{str(e)}")
    print("   可能是网络无法访问Google服务")
    print("=" * 60)
except Exception as e:
    print("❌ Gemini API 测试失败！")
    print("=" * 60)
    print(f"错误类型：{type(e).__name__}")
    print(f"错误信息：{str(e)}")
    print("=" * 60)
    raise