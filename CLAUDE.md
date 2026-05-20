# PaperDiag — 中文学术论文 AIGC 诊断工具

## 项目定位

逐句诊断论文"哪里读起来像AI"。开源、免费、本地运行。

## 架构

```
输入 → 预处理(docx/txt解析、专有名词保护、参考文献识别)
  → 句子级19维特征提取(统计/困惑度/Embedding/词表/结构)
  → 随机森林分类器(CV 91.6%) → P(AI|句子)
  → 三级判定(红≥0.60/橙≥0.45/黄≥0.28) → AI率 = (红+橙字数)/总字数
  → 逐句热力图 + 三色圆环图 + 复制改写提示词
```

## 当前状态

检测层完成。改写层已探索但4B小模型不可行。

## 关键文件

- `paperdiag/rules_engine.py` — 19维特征提取 + ML检测器（核心）
- `paperdiag/webui.py` — Flask Web应用
- `paperdiag/preprocessor.py` — docx/txt解析 + 专有名词保护
- `paperdiag/embedding_layer.py` — Qwen3-Embedding-0.6B封装
- `paperdiag/data/` — 连接词黑名单、Slop模式库、术语白名单等
- `paperdiag/templates/index.html` — Warm Paper WebUI
- `paperdiag/tests/` — 37个核心测试 + 10个Embedding测试

## 常用命令

```bash
pip install -e .
python -m paperdiag.cli webui    # 启动Web界面 → http://localhost:5000
python -m paperdiag.cli scan <论文>   # CLI诊断
python paperdiag/tests/test_core.py   # 运行测试
```
