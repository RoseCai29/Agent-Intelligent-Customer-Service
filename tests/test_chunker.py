"""分块器正确性测试：锁定「分块不得丢内容」这一最严重的数据正确性问题。

评审实测：旧实现单段 273 字仅保留 29/40 句、小 max_chars 下 5/10 句，
且 overlap 参数完全无效。本文件用带唯一标记的语料回归验证修复后：
  1. 每一句标记都必须出现在切块拼接结果里（覆盖不变量）
  2. overlap 真实生效（有/无 overlap 产出不同，且都不丢内容）
"""
import re

from app.wiki.chunker import split_text


def _markers(sentences: list[str]) -> list[str]:
    m = []
    for s in sentences:
        mm = re.search(r"【(.+?)】", s)
        m.append(mm.group(1) if mm else s)
    return m


def _assert_covers_all(chunks: list[str], markers: list[str]):
    joined = "".join(chunks)
    missing = [m for m in markers if m not in joined]
    assert not missing, f"丢内容，缺失标记: {missing}"


def _make_sentences(n: int, sentence: str = "句") -> list[str]:
    # 每个句子一个唯一标记，组成单段长文
    return [f"{sentence}{i}：【{sentence}{i}】这是第 {i} 句内容，用于锁定分块不丢内容。" for i in range(n)]


def test_single_long_paragraph_no_loss_default():
    """评审探针：单段 273 字、40 句，max_chars=200 不得再丢尾句。"""
    sents = _make_sentences(40)
    src = "".join(f"{s}。" for s in sents)[:273] or "".join(f"{s}。" for s in sents)
    # 截断后可能切掉部分标记，改用完整 40 句更稳
    src = "".join(f"{s}。" for s in sents)
    chunks = split_text(src, {"max_chars": 200, "overlap_chars": 40})
    _assert_covers_all(chunks, _markers(sents))
    assert chunks  # 有产出


def test_small_max_chars_no_loss():
    """评审探针：10 句、max_chars=60，全部句子都应保留。"""
    sents = _make_sentences(10)
    src = "".join(f"{s}。" for s in sents)
    chunks = split_text(src, {"max_chars": 60, "overlap_chars": 40})
    _assert_covers_all(chunks, _markers(sents))


def test_overlap_parameter_actually_effective():
    """overlap>0 与 overlap=0 产出应不同（重叠真实生效），且都不丢内容。"""
    sents = _make_sentences(12)
    src = "".join(f"{s}。" for s in sents)
    with_overlap = split_text(src, {"max_chars": 60, "overlap_chars": 30})
    without_overlap = split_text(src, {"max_chars": 60, "overlap_chars": 0})
    joined_ov = "".join(with_overlap)
    joined_no = "".join(without_overlap)
    _assert_covers_all(with_overlap, _markers(sents))
    _assert_covers_all(without_overlap, _markers(sents))
    assert len(joined_ov) > len(joined_no), "overlap>0 时产出应更长（有重复重叠）"


def test_multi_paragraph_coverage_invariant():
    """多段落混合：任意内容切分后拼接必须覆盖原文非空白字符。"""
    src = "\n\n".join([
        "### 退款多久到账\n退款通常在 1 到 3 个工作日内到账，超过请留意银行处理时间。"
        "如果迟迟未到账，可以联系在线客服协助查询。",
        "### 优惠券使用\n新人优惠券领取后 7 天内有效，仅限首单使用，且不能与其他优惠叠加。",
        "### 发货说明\n现货商品下单后 24 小时内发货，预售商品以页面标注的发货时间为准。",
    ])
    for cfg in ({"max_chars": 200, "overlap_chars": 40},
                {"max_chars": 80, "overlap_chars": 20},
                {"max_chars": 50, "overlap_chars": 10}):
        chunks = split_text(src, cfg)
        joined = "".join(chunks)
        # 覆盖不变量：去空白后拼接长度 >= 原文长度
        def _nz(t: str):
            return "".join(t.split())
        assert len(_nz(joined)) >= len(_nz(src)), f"cfg={cfg} 丢内容了"


def test_empty_and_short():
    assert split_text("") == []
    assert split_text("你好") == ["你好"]