import os
import asyncio

from Config.LLMConfig import LLMConfig, LLMType
from Provider.LLMProviderRegister import create_llm_instance

api_key = os.getenv("SILICONFLOW_API_KEY")
if not api_key:
    raise ValueError("Please set SILICONFLOW_API_KEY in the environment or .env")

config = LLMConfig(
    model="Qwen/Qwen3-8B",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3,
    max_token=4096,
    provider=LLMType.OPENAI,
)

llm = create_llm_instance(config)

async def test_llm():
    rsp = await llm.acompletion_text(
        messages=[
            {"role": "user", "content": "测试一下硅基流动是否可用"}
        ],
        stream=False
    )
    print("LLM Output:", rsp)

asyncio.run(test_llm())