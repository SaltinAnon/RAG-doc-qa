"""文本处理工具：清洗、哈希、去重、截断。

中文文档的脏数据比英文多得多（PDF 提取尤其严重），
这个模块是数据管道的第一道防线。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------- 正则预编译（性能：避免每次调用重新编译） ----------------

# 零宽字符与 BOM：PDF 提取常见，肉眼看不见但会污染向量
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]")
# 控制字符（保留 \n \t）
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 3 个及以上换行 → 2 个
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 2 个及以上空格 → 1 个
_MULTI_SPACE = re.compile(r"[ \t\u3000]{2,}")
# 连字符断行："docu-\nment" → "document"（英文 PDF 常见）
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
# 单个**符号**成行（PDF 表格错位、竖线残留）—— 注意只匹配非文字字符，
# 不能匹配单个字母/汉字，否则会把正文里的短行误删（见下方注释）
_LONE_CHAR_LINE = re.compile(r"^[^\w\u4e00-\u9fff]\s*$", re.MULTILINE)
# 中文句子结束标点
CJK_SENT_END = "。！？；!?;"


def normalize_text(text: str, *, merge_hyphen: bool = True) -> str:
    """清洗文本，返回可直接用于切分和向量化的干净字符串。

    做这些事：
    1. Unicode NFKC 归一化（全角字母/数字 → 半角，兼容字符 → 标准字符）
    2. 去掉零宽字符、控制字符（PDF 提取的重灾区）
    3. 合并英文连字符断行
    4. 压缩多余空白与空行

    Args:
        text: 原始文本。
        merge_hyphen: 是否合并英文断词。纯中文文档可以关掉。

    Returns:
        清洗后的文本；输入为空时返回空字符串。
    """
    if not text:
        return ""

    # NFKC 会把 「（）」这类全角括号转成半角——对中文其实不好看，
    # 但能统一检索时的匹配形式，利大于弊；显示层不做这个转换。
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if merge_hyphen:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _LONE_CHAR_LINE.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # 逐行 strip，但保留空行结构
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def content_hash(text: str) -> str:
    """计算内容的稳定哈希（用于 doc_id 与去重）。

    与 Python 内置 `hash()` 的区别：内置哈希带进程随机盐，
    重启后同一个文档会得到不同 ID，导致重复入库。这里必须用哈希算法本身。

    Args:
        text: 任意文本。

    Returns:
        16 位十六进制字符串。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_hash(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """计算文件内容哈希（分块读，避免大文件吃满内存）。

    Args:
        path: 文件路径。
        chunk_size: 每次读取的字节数，默认 1MB。

    Returns:
        16 位十六进制字符串。
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def truncate(text: str, max_len: int = 200, suffix: str = "…") -> str:
    """按字符数截断（用于生成引用片段）。

    Args:
        text: 原文。
        max_len: 最大长度（含省略号）。
        suffix: 省略号。

    Returns:
        截断后的字符串。中文按字符算，一个汉字算 1。
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)].rstrip() + suffix


def chinese_ratio(text: str) -> float:
    """中文字符占比（用于判断该用哪个切分器 / 分词策略）。

    Args:
        text: 待检测文本。

    Returns:
        0.0 ~ 1.0 的比例；空文本返回 0.0。
    """
    if not text:
        return 0.0
    total = len([c for c in text if not c.isspace()])
    if total == 0:
        return 0.0
    cjk = len([c for c in text if "\u4e00" <= c <= "\u9fff"])
    return cjk / total


def tokenize_for_bm25(text: str) -> list[str]:
    """为 BM25 做分词。

    为什么不用 jieba：
    - 多一个依赖，且首次加载词典慢；
    - BM25 场景下 **中文按字切分（character bigram）** 效果已经接近分词，
      且不会因为词典缺失把专业术语切错。

    策略：
    - 连续 CJK 字符 → 拆成 unigram + bigram（"年假" → ["年","假","年假"]）
    - 连续 ASCII 字母数字 → 小写整词保留
    - 去掉标点与单字符噪声

    Args:
        text: 原始文本。

    Returns:
        token 列表。
    """
    tokens: list[str] = []

    # CJK 段
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(seg)  # unigram
        if len(seg) >= 2:
            tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))  # bigram

    # 英文/数字段
    for seg in re.findall(r"[A-Za-z0-9_]+", text):
        # 单个字母是噪声（"a" 到处都是），但**单个数字是有意义的**
        # （问「年假 5 天」时，"5" 必须保留，否则 BM25 完全匹配不上）
        if len(seg) > 1 or seg.isdigit():
            tokens.append(seg.lower())

    return tokens


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度（用于近似重复检测）。

    Args:
        a: 集合 A。
        b: 集合 B。

    Returns:
        0.0 ~ 1.0；任一边为空返回 0.0。
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def idf_weights(corpus_terms: list[set[str]], query_terms: set[str]) -> dict[str, float]:
    """为查询词计算 IDF 权重（**这是提升检索与拒答准确率的关键技巧**）。

    ## 解决什么问题

    朴素的「查询词覆盖率」会被**常见词污染**。举例：

        用户问：「公司的股票代码是多少？」
        文档写：「公司为全体正式员工缴纳五险一金，住房公积金比例为 12%。」

        朴素覆盖率：查询词 {公司, 股票, 代码} 里「公司」命中了 → 覆盖率 1/3 = 33%
        但「公司」在每一篇文档里都出现，几乎零信息量；
        真正有区分度的是「股票」「代码」——两个都没命中。

        结果：系统认为这个片段「有点相关」，于是**硬答**而不是拒答。

    ## 解决思路

    用 **IDF（逆文档频率）**给每个查询词加权：

        idf(t) = log((1 + N) / (1 + df(t))) + 1

    - `N`   = 候选片段总数
    - `df(t)`= 包含该词的片段数

    于是「公司」「员工」这类到处都是的词权重趋近 1.0，
    而「股票」「代码」这类只出现在个别片段的词权重可达 3~4 倍。
    覆盖率从「命中了几个词」变成「命中了多少**信息量**」。

    ## 为什么用「候选片段」而不是全库做统计

    全库统计需要额外把整个知识库拉出来算一遍，成本高且要维护缓存。
    而 RRF 融合后的候选（通常 20~30 条）已经是一个有代表性的样本，
    且**不需要任何额外 IO**。这是一个精度与成本的合理折中。

    ## 没有出现在任何候选里的查询词

    这类词其实**区分度最高**（说明知识库里根本没有相关内容），
    所以给它们最高权重 `log(N + 1) + 1`。
    这样「全都没命中」会让覆盖率迅速掉到很低，从而触发拒答。

    Args:
        corpus_terms: 候选片段的 token 集合列表。
        query_terms: 需要计算权重的查询词集合。

    Returns:
        `{查询词: 权重}`，包含 `query_terms` 里的每一个词。
        语料为空时所有词权重都是 1.0。
    """
    n = len(corpus_terms)
    if n == 0 or not query_terms:
        return {t: 1.0 for t in query_terms}

    df: Counter[str] = Counter()
    for ts in corpus_terms:
        for t in ts:
            df[t] += 1

    # 未出现在任何候选里的词 → 最高权重
    max_idf = math.log(n + 1) + 1.0

    return {
        t: (math.log((1 + n) / (1 + df[t])) + 1.0 if t in df else max_idf)
        for t in query_terms
    }


def safe_filename(name: str) -> str:
    """把用户上传的文件名清洗成安全文件名，防止路径穿越。

    Args:
        name: 原始文件名。

    Returns:
        只保留字母数字、下划线、短横线、点、以及中文字符的基名。
    """
    name = Path(name).name  # 去掉任何路径部分：../../etc/passwd → passwd
    name = re.sub(r"[^\w\u4e00-\u9fff.\-]", "_", name, flags=re.UNICODE)
    return name[:120] or "unnamed"
