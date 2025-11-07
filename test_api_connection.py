#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API连接测试脚本
用于测试Gemini和OpenAI API是否可以正常访问

使用方法：
1. 创建.env文件，添加以下内容：
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here

2. 运行脚本：
   python test_api_connection.py
"""

import os
import sys
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

def test_gemini_api():
    """测试Gemini API连接"""
    print("=" * 60)
    print("🔵 正在测试 Gemini API...")
    print("=" * 60)
    
    try:
        # 尝试导入Gemini库
        from google import genai
        
        # 获取API密钥
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("❌ 错误：未找到Gemini API密钥")
            print("   请在.env文件中设置 GEMINI_API_KEY 或 GOOGLE_API_KEY")
            return False
        
        # 创建客户端
        client = genai.Client(api_key=api_key)
        
        # 测试API调用
        print("📡 正在发送测试请求...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="用一句话解释什么是人工智能"
        )
        
        # 显示结果
        print("✅ Gemini API 连接成功！")
        print(f"📝 模型回复：{response.text}")
        return True
        
    except ImportError:
        print("❌ 错误：未安装 google-generativeai 库")
        print("   请运行：pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ Gemini API 连接失败：{str(e)}")
        print(f"   错误类型：{type(e).__name__}")
        return False


def test_openai_api():
    """测试OpenAI API连接"""
    print("\n" + "=" * 60)
    print("🟢 正在测试 OpenAI API...")
    print("=" * 60)
    
    try:
        # 尝试导入OpenAI库
        from openai import OpenAI
        
        # 获取API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("❌ 错误：未找到OpenAI API密钥")
            print("   请在.env文件中设置 OPENAI_API_KEY")
            return False
        
        # 创建客户端
        client = OpenAI(api_key=api_key)
        
        # 测试API调用
        print("📡 正在发送测试请求...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "用一句话解释什么是人工智能"}
            ],
            max_tokens=100
        )
        
        # 显示结果
        print("✅ OpenAI API 连接成功！")
        print(f"📝 模型回复：{response.choices[0].message.content}")
        return True
        
    except ImportError:
        print("❌ 错误：未安装 openai 库")
        print("   请运行：pip install openai")
        return False
    except Exception as e:
        print(f"❌ OpenAI API 连接失败：{str(e)}")
        print(f"   错误类型：{type(e).__name__}")
        return False


def main():
    """主函数"""
    print("\n" + "🚀 API连接测试工具".center(60, "="))
    print("本脚本将测试Gemini和OpenAI API是否可以正常访问\n")
    
    # 检查.env文件是否存在
    if not os.path.exists(".env"):
        print("⚠️  警告：未找到.env文件")
        print("   建议创建.env文件并添加API密钥")
        print("   可以参考.env.example文件\n")
    
    # 测试结果
    results = {
        "Gemini": False,
        "OpenAI": False
    }
    
    # 测试Gemini API
    results["Gemini"] = test_gemini_api()
    
    # 测试OpenAI API
    results["OpenAI"] = test_openai_api()
    
    # 显示总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    for api_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{api_name:10s}: {status}")
    
    print("\n" + "=" * 60)
    
    # 如果都成功，返回0；否则返回1
    if all(results.values()):
        print("🎉 所有API测试通过！")
        return 0
    else:
        print("⚠️  部分API测试失败，请检查API密钥和网络连接")
        return 1


if __name__ == "__main__":
    sys.exit(main())

