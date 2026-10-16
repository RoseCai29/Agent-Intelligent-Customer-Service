"""Wiki 知识库导入 / 校验脚本。

用法：
  python -m app.ingest                 # 解析 data/wiki，打印结构化统计
  python -m app.ingest --dir <path>    # 额外解析指定目录
"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .wiki.loader import load_directory


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    cfg = load_config()
    dirs = list(cfg["wiki"]["data_dirs"])
    if args.dir:
        dirs.append(args.dir)

    chunk_cfg = cfg["wiki"].get("chunker", {})
    wiki = load_directory(dirs, chunk_cfg=chunk_cfg)

    print(f"解析文档目录: {dirs}")
    print(f"共 {len(wiki.documents)} 篇文档")
    chunks = wiki.chunks
    print(f"共 {len(chunks)} 个可检索块(chunk)")
    print(f"共 {len(wiki.faqs)} 条 FAQ")
    print("-" * 40)
    for doc in wiki.documents:
        print(f"  [{doc.category}] {doc.title}  (sections={len(doc.sections)}, "
              f"chunks={len(doc.chunks)}, faqs={len(doc.faqs)})")
        for sec in doc.sections:
            print(f"      - #{sec.level} {sec.title}  path='{sec.path}'  chunks={len(sec.chunks)}")
    if not chunks:
        print("警告：没有解析出任何块，请检查 markdown 格式。", file=sys.stderr)
        return 1
    return 0


def _parse(argv):
    p = argparse.ArgumentParser(description="导入并校验 wiki 知识库")
    p.add_argument("--dir", default=None, help="额外解析的 markdown 目录")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())