# 智能长期记忆对话系统

一个基于长期记忆的智能对话系统，通过管理和利用历史对话内容，提供个性化且连贯的对话体验。系统能够像人类一样形成并利用长期记忆，实现更自然的人机交互。

## 🌟 主要特点

### 🧠 智能记忆管理
- 使用先进的 BGE (BAAI General Embedding) 中文向量模型
- 多类型记忆支持（事实、偏好、对话）
- 记忆的自动存储与检索
- 智能记忆重构与优化
- 自动清理重复/无效记忆
- 记忆拆分功能

### 💬 对话能力
- 基于上下文的智能回复
- 个性化对话风格适配
- 多轮对话连贯性保持
- 记忆融入对话生成

### 🛠️ 可视化管理
- 直观的记忆管理界面
- 记忆分类与标签管理
- 全文搜索功能
- 记忆编辑与重构工具

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- 现代浏览器（支持ES6+）

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/long-memory-chat.git
cd long-memory-chat
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 初始化配置
```bash
python scripts/init_memory_file.py
```

4. 启动服务
```bash
python main.py
```

5. 访问系统
浏览器打开 `http://localhost:5000`

## 📁 项目结构

```
project/
├── config/                # 配置文件
│   ├── config.yaml       # 系统配置
│   └── user_memories.json # 记忆存储
├── models/               # 核心模型
│   ├── agent.py         # 对话代理
│   └── memory_manager.py # 记忆管理器
├── services/            # 服务层
│   └── context_manager.py # 上下文管理
├── templates/           # 前端页面
│   ├── index.html      # 对话界面
│   └── memory_manager.html # 记忆管理界面
├── scripts/            # 工具脚本
└── main.py            # 主程序入口
```

## 💡 记忆系统说明

### 记忆类型
- **事实记忆**：客观信息，如用户背景、习惯等
- **偏好记忆**：用户的喜好、观点等主观信息
- **对话记忆**：重要的对话片段和上下文

### 记忆管理功能
- **添加记忆**：手动/自动记录重要信息
- **编辑记忆**：更新和修正已有记忆
- **重构记忆**：优化记忆表达，提升可用性
- **拆分记忆**：将复杂记忆拆分为多个关联记忆
- **清理记忆**：自动去重和清理无效记忆

## ⚙️ 配置说明

在 `config/config.yaml` 中可配置：

```yaml
server:
  host: "0.0.0.0"
  port: 5000

memory:
  max_memories: 1000
  clean_threshold: 0.85
  
llm:
  model: "gpt-3.5-turbo"
  temperature: 0.7
```

## 🔜 开发计划

- [ ] 多用户支持
- [ ] 记忆优先级管理
- [ ] 记忆关联网络
- [ ] 记忆导入/导出
- [ ] 更多记忆类型
- [ ] API接口支持

## 🤝 参与贡献

1. Fork 本项目
2. 创建新分支 `git checkout -b feature/AmazingFeature`
3. 提交更改 `git commit -m 'Add some AmazingFeature'`
4. 推送分支 `git push origin feature/AmazingFeature`
5. 提交 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或通过以下方式联系：

- 项目Issue页面
- Email: your.email@example.com

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者！