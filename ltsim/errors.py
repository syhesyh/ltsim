"""LTSim 错误类型。

所有诊断错误都用带编号的码（见设计文档附录 A），便于测试与用户文档引用。
"""
from __future__ import annotations


class LTSError(Exception):
    """LTSim 诊断错误。

    Attributes:
        code: 诊断码，如 "LTS-E-MACHINE"、"LTS-E-GRAPH-HOP"。
        msg:  人类可读的错误描述。
    """

    def __init__(self, code: str, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"{code}: {msg}")
