import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import prompt


load_dotenv(Path(__file__).with_name(".env"), override=True)

proxy_bypass = os.getenv("NO_PROXY", "")
os.environ["NO_PROXY"] = ",".join(
    part
    for part in (
        proxy_bypass,
        "dashscope.aliyuncs.com",
        "google.serper.dev",
    )
    if part
)

import tool_search  # 注册 search 工具
import tool_visit  # 注册 visit 工具
from prompt import SYSTEM_PROMPT_MULTI, USER_PROMPT
from react_agent import MultiTurnReactAgent


model = os.getenv("DASHSCOPE_MAIN_MODEL", "qwen3-max")
agent = MultiTurnReactAgent(
    llm={
        "model": model,
        "generate_cfg": {
            "max_input_tokens": 320000,
            "max_retries": 10,
            "temperature": 0.6,
            "top_p": 0.95,
        },
        "model_type": "qwen_dashscope",
    },
    function_list=["search", "visit"],
    system_message=(
        SYSTEM_PROMPT_MULTI
        + "\nCurrent date: "
        + datetime.now().strftime("%Y-%m-%d")
    ),
)

question = prompt("请输入问题：")
result = agent._run(
    {
        "item": {"question": question, "answer": ""},
        "rollout_id": 1,
    },
    model,
    USER_PROMPT,
)

print("\n=== 最终答案 ===")
print(result["prediction"])
