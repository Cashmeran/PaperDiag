# PaperDiag

PaperDiag 是一个面向中文学术论文的 AIGC 检测工具。

开源、免费、本地运行。

## 快速开始

```bash
git clone https://github.com/<你的用户名>/paperdiag.git
cd paperdiag
pip install -e .
python -m paperdiag.cli webui
```

浏览器打开  http://localhost:5000 粘贴论文点检测即可。

## 依赖

Python 3.10+，jieba, numpy, scikit-learn, Flask, sentence-transformers。

首次运行需要下载 Qwen3-Embedding-0.6B（~1.2GB），之后不需要联网。

### 输出

- **AI 率**：中重度疑似句子字数 / 总字数
- **逐句热力图**：红（重度）/ 橙（中度）/ 黄（轻度）/ 绿（安全）
- **改写提示词**：一键复制，含逐句诊断病灶和修改指令，可粘贴到任意大模型进行改写

### 检测原理

把论文拆成句子，每句提取 19 个维度的特征：

- **统计特征**：词汇多样性、字符熵、词重复度、唯一字占比
- **困惑度**：bigram 概率、 自困惑度
- **Embedding**：段内同质化、人类参考池相似度
- **词表匹配**：高/中/低三级连接词、Slop 模式、char n-gram TF-IDF 指纹
- **结构模式**：模板句式、排比检测、学术套话、句长均匀度

19 维特征 → 随机森林分类器 → 每句 AI 概率 → 红（≥0.60）/ 橙（≥0.45）/ 黄（≥0.28）/ 绿四级标注。

### 训练数据

NLPCC 2025 Shared Task 1 的 CSL 子集

## 项目结构

```
paperdiag/
├── rules_engine.py      # 核心检测
├── webui.py             # Web 界面
├── preprocessor.py      # 文档解析
├── embedding_layer.py   # embedding 模型封装
├── data/                # 连接词黑名单、Slop 模式库、术语白名单
├── templates/           # 前端页面
└── tests/               # 测试
```


## License

MIT
