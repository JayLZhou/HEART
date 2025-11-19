from Config.LLMConfig import LLMConfig, LLMType
from Provider.LLMProviderRegister import create_llm_instance

config = LLMConfig(
    model="Qwen/QwQ-32B",
    api_key="sk-kpacfuoklmioauxqlqrpityhhbjarjqcpiknxleuvizduyxm",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3,
    max_token=4096,
    provider=LLMType.OPENAI,
)

llm = create_llm_instance(config)

import asyncio

async def test_llm():
    rsp = await llm.acompletion_text(
        messages=[
            {"role": "user", "content": "测试一下硅基流动是否可用"}
        ],
        stream=False
    )
    print("LLM Output:", rsp)

asyncio.run(test_llm())