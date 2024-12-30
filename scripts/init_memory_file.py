import json
import os
from datetime import datetime

def init_memory_file():
    memory_file_path = "config/user_memories.json"
    
    # 确保config目录存在
    os.makedirs("config", exist_ok=True)
    
    # 创建初始记忆文件，包含一些示例记忆
    initial_memories = {
        "default_user": [
            {
                "content": "用户说: 我最喜欢吃英式早餐\n助手回答: 好的，我记住了！",
                "type": "dialogue",
                "timestamp": datetime.now().isoformat()
            },
            {
                "content": "Jamie最喜欢的人是他的老婆和多米",
                "type": "fact",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    # 写入文件
    with open(memory_file_path, "w", encoding="utf-8") as f:
        json.dump(initial_memories, f, ensure_ascii=False, indent=2)
    
    print(f"Created initial memory file at {memory_file_path}")
    print("Initial memories:", json.dumps(initial_memories, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    init_memory_file() 