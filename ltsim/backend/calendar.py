"""B3 资源日历（backend.md §5.1）。只追加、不删除；只用标准库（bisect）。

时间一律是绝对拍数，区间左闭右开。
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right

from ..errors import LTSError


class ExclusiveCalendar:
    """独占资源：已占用的区间。提交的区间彼此不重叠（调度器只提交 conflict 为 None 的区间），
    所以按起点排序后，终点也是有序的。"""

    def __init__(self) -> None:
        self._starts: list[int] = []
        self._iv: list[tuple[int, int, str]] = []        # (start, end, owner)，按 start 排序

    def conflict(self, s: int, e: int) -> tuple[int, str] | None:
        """与 [s, e) 重叠的已占区间里，结束最晚的那个的 (end, owner)；没有重叠返回 None。空区间不与任何区间重叠。"""
        if s >= e:
            return None
        i = bisect_left(self._starts, e)                 # 起点 < e 的区间是 [0, i)
        if i == 0:
            return None
        start, end, owner = self._iv[i - 1]              # 区间互不重叠：起点最大者终点也最大
        return (end, owner) if end > s else None

    def commit(self, s: int, e: int, owner: str) -> None:
        if s >= e:
            return
        if self.conflict(s, e) is not None:
            raise AssertionError(f"commit overlaps an occupied interval: [{s},{e}) by {owner}")
        i = bisect_left(self._starts, s)
        self._starts.insert(i, s)
        self._iv.insert(i, (s, e, owner))

    def intervals(self) -> list[tuple[int, int, str]]:
        return list(self._iv)


class CounterCalendar:
    """计数资源：分段常数的占用量。_t 是断点，_lvl[i] 是 [_t[i], _t[i+1]) 上的占用量；
    第一个断点之前和最后一个断点之后占用量为 0。"""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._t: list[int] = []
        self._lvl: list[int] = []
        self._res: list[tuple[int, int, int, str]] = []  # 提交过的 (start, end, amount, owner)
        self._first: dict[str, int] = {}                 # owner → 第一次提交的序号（平局规则用）

    def level_at(self, t: int) -> int:
        i = bisect_right(self._t, t) - 1
        return self._lvl[i] if i >= 0 else 0

    def conflict(self, s: int, e: int, k: int) -> tuple[int, str] | None:
        """[s, e) 内占用量 > cap − k 的最后一拍 t：返回 (t + 1, 那一拍贡献最大的 owner)；放得下返回 None。
        贡献相同取最先提交的 owner。"""
        if s >= e:
            return None
        limit = self.cap - k
        if limit < 0:          # 单段用量就超过容量：B2 的链级检查已拦下，这里兜底
            raise LTSError("LTS-E-CHAIN-SELF", f"a single segment needs {k} > capacity {self.cap}")
        # 从右往左扫与 [s, e) 相交的分段
        i = bisect_right(self._t, e - 1) - 1             # 包含 e−1 的分段（或其左边最近的）
        while i >= 0:
            seg_s = self._t[i]
            seg_e = self._t[i + 1] if i + 1 < len(self._t) else None
            if seg_e is not None and seg_e <= s:
                break
            if self._lvl[i] > limit:
                last = (e - 1) if seg_e is None or seg_e > e - 1 else seg_e - 1
                return last + 1, self._blocker(last)
            if seg_s <= s:
                break
            i -= 1
        return None

    def _blocker(self, t: int) -> str:
        contrib: dict[str, int] = {}
        for s, e, k, owner in self._res:
            if s <= t < e:
                contrib[owner] = contrib.get(owner, 0) + k
        return max(contrib, key=lambda o: (contrib[o], -self._first[o]))

    def _breakpoint(self, x: int) -> int:
        """确保 x 是断点，返回它的下标。"""
        i = bisect_left(self._t, x)
        if i < len(self._t) and self._t[i] == x:
            return i
        lvl = self._lvl[i - 1] if i > 0 else 0
        self._t.insert(i, x)
        self._lvl.insert(i, lvl)
        return i

    def commit(self, s: int, e: int, k: int, owner: str) -> None:
        if s >= e:
            return
        if self.conflict(s, e, k) is not None:
            raise AssertionError(f"commit exceeds capacity {self.cap}: [{s},{e}) +{k} by {owner}")
        a = self._breakpoint(s)
        b = self._breakpoint(e)
        for i in range(a, b):
            self._lvl[i] += k
        self._res.append((s, e, k, owner))
        self._first.setdefault(owner, len(self._first))

    def steps(self) -> list[tuple[int, int]]:
        """(时刻, 从该时刻起的占用量)，供 trace 的计数器事件使用。"""
        return list(zip(self._t, self._lvl))
