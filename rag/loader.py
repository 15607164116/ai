# -*- coding: utf-8 -*-
"""文档解析：Markdown / 纯文本 -> 纯文本。

注意：调用方 listdir_with_allowed_type() 返回的是字符串路径，
因此这里统一先转成 Path 再取后缀，避免 AttributeError: 'str' object has no attribute 'suffix'。
"""
from pathlib import Path
from typing import Union

# 允许解析的纯文本后缀
TEXT_SUFFIXES = (".md", ".txt", ".markdown")


def parse_file(path: Union[str, Path]) -> str:
    path = Path(path)                      # 兼容 str 与 Path
    suffix = path.suffix.lower()

    if suffix in TEXT_SUFFIXES:
        # utf-8-sig 可自动吃掉 Windows 记事本写入的 BOM
        return path.read_text(encoding="utf-8-sig", errors="ignore")

    raise ValueError(f"暂不支持的文件类型: {suffix}（当前仅支持 md/txt/markdown）")
