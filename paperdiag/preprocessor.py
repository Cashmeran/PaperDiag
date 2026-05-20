"""预处理模块：文档解析、段落切分、术语保护"""

import re
import json
from pathlib import Path
from typing import Optional

# 受保护内容的正则模式
PROTECTED_PATTERNS = [
    (r'\[[\d,\s\-;]+\]', 'CITATION'),           # [1], [1,2,3]
    (r'\([^)]*\d{4}[^)]*\)', 'CITATION'),       # (张三 2024)
    (r'\d+\.\d+(?:\.\d+)*', 'NUMBER'),           # 小数/版本号
    (r'(?:https?://|www\.)[^\s]+', 'URL'),       # URL
    (r'[A-Za-z_][A-Za-z0-9_]*\([^)]*\)', 'FUNC'), # 英文函数名
    (r'[①②③④⑤⑥⑦⑧⑨⑩]', 'NUM_MARKER'),           # 序号
    (r'(?:图|表|公式|Fig|Table|Eq)\s*\d+', 'REF'), # 图表公式
    (r'`[^`]+`', 'CODE'),                         # 行内代码
    (r'```[\s\S]*?```', 'CODE_BLOCK'),            # 代码块
    # ---- 专有名词保护 ----
    (r'《[^》]+》', 'TITLE'),                      # 书名号《xxx》
    (r'"[^"]{2,30}"', 'QUOTE'),                   # 双引号引用
    (r'[一-鿿]{2,4}(?:教授|博士|院士|先生|女士|同志)', 'PERSON_TITLE'),  # 人名+头衔
    (r'(?:[京津沪渝]|[冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青川藏琼宁])(?:市|省|自治区|特别行政区)', 'PLACE'),  # 省市级地名
    (r'(?:北京|上海|广州|深圳|杭州|南京|武汉|成都|重庆|天津|西安|长沙|郑州|济南|青岛|大连|厦门|苏州)', 'PLACE'),
    (r'(?:北京大学|清华大学|复旦大学|上海交大|浙江大学|南京大学|武汉大学|中山大学|中科院|中国科学院|中国工程院|中国社会科学院)', 'ORG'),
    (r'(?:第[一二三四五六七八九十百千\d]+章|第[一二三四五六七八九十百千\d]+节)', 'HEADING'),  # 章节标题
    (r'[《〈]?\s*[一-鿿]{3,8}\s*[》〉]?\s*[:：]', 'BOOK_TITLE'),  # 可能被误判为AI的书籍标题
]

DATA_DIR = Path(__file__).parent / "data"


def load_term_whitelist(discipline: Optional[str] = None) -> set:
    """加载术语白名单。discipline可选: cs, med, econ, edu, law, soc, psych, env, mgmt, phil"""
    whitelist = set()
    if discipline:
        path = DATA_DIR / f"term_whitelist_{discipline}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                whitelist.update(data.get("terms", []))
    # 始终加载通用术语
    for term in ["参考文献", "致谢", "附录", "摘要", "关键词", "Abstract", "Keywords"]:
        whitelist.add(term)
    return whitelist


class ProtectedSpan:
    """受保护内容的标记"""
    def __init__(self, start: int, end: int, label: str, text: str):
        self.start = start
        self.end = end
        self.label = label
        self.text = text

    def __repr__(self):
        return f"ProtectedSpan({self.start}:{self.end}, {self.label})"


class Paragraph:
    """段落对象"""
    def __init__(self, index: int, text: str, style: str = "body"):
        self.index = index
        self.text = text.strip()
        self.style = style
        self.char_count = len(self.text)
        self.protected_spans: list[ProtectedSpan] = []

    def __repr__(self):
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Paragraph({self.index}, \"{preview}\")"


def _detect_protected(text: str) -> list[ProtectedSpan]:
    """检测文本中的受保护内容"""
    spans = []
    seen = set()
    for pattern, label in PROTECTED_PATTERNS:
        for m in re.finditer(pattern, text):
            start, end = m.start(), m.end()
            # 防止重叠
            if not any(s.start <= start < s.end or s.start < end <= s.end for s in spans):
                if (start, end) not in seen:
                    spans.append(ProtectedSpan(start, end, label, m.group()))
                    seen.add((start, end))
    return sorted(spans, key=lambda s: s.start)


def parse_docx(filepath: str) -> list[Paragraph]:
    """解析 docx 文件，返回段落列表"""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = []
        idx = 0
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                p = Paragraph(idx, text, para.style.name if para.style else "body")
                p.protected_spans = _detect_protected(text)
                paragraphs.append(p)
                idx += 1
        return paragraphs
    except ImportError:
        raise ImportError("需要安装 python-docx: pip install python-docx")
    except Exception as e:
        raise RuntimeError(f"解析 docx 失败: {e}")


def parse_txt(filepath: str) -> list[Paragraph]:
    """解析纯文本文件，返回段落列表"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # 以空行分隔段落
    raw_paras = re.split(r'\n\s*\n', content)
    paragraphs = []
    idx = 0
    for rp in raw_paras:
        text = rp.strip()
        if text and len(text) > 10:  # 忽略过短段落
            p = Paragraph(idx, text)
            p.protected_spans = _detect_protected(text)
            paragraphs.append(p)
            idx += 1
    return paragraphs


def parse_text(text: str) -> list[Paragraph]:
    """直接解析文本字符串"""
    raw_paras = re.split(r'\n\s*\n', text)
    paragraphs = []
    idx = 0
    for rp in raw_paras:
        t = rp.strip()
        if t and len(t) > 10:
            p = Paragraph(idx, t)
            p.protected_spans = _detect_protected(t)
            paragraphs.append(p)
            idx += 1
    return paragraphs


def auto_parse(filepath_or_text: str) -> list[Paragraph]:
    """自动判断输入类型并解析"""
    path = Path(filepath_or_text)
    if path.exists():
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return parse_docx(str(path))
        elif suffix in (".txt", ".md", ".markdown"):
            return parse_txt(str(path))
        else:
            return parse_txt(str(path))
    else:
        return parse_text(filepath_or_text)


def get_protected_text(para: Paragraph) -> str:
    """返回用占位符替换受保护内容后的文本"""
    if not para.protected_spans:
        return para.text
    result = []
    last_end = 0
    for span in para.protected_spans:
        result.append(para.text[last_end:span.start])
        result.append(f"__{span.label}_{span.start}__")
        last_end = span.end
    result.append(para.text[last_end:])
    return "".join(result)


def restore_protected(modified_text: str, original_para: Paragraph) -> str:
    """将受保护内容还原到改写后的文本中"""
    if not original_para.protected_spans:
        return modified_text
    result = modified_text
    for span in original_para.protected_spans:
        placeholder = f"__{span.label}_{span.start}__"
        if placeholder in result:
            result = result.replace(placeholder, span.text, 1)
    return result
