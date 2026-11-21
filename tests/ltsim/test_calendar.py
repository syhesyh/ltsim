"""B3 资源日历（backend.md §5.1）：无重叠、部分重叠、刚好放满、超限一拍。"""
import pytest

from ltsim.backend.calendar import CounterCalendar, ExclusiveCalendar


class TestExclusive:
    def test_empty(self):
        assert ExclusiveCalendar().conflict(0, 10) is None

    def test_touching_is_not_overlap(self):
        c = ExclusiveCalendar()
        c.commit(10, 20, "A")
        assert c.conflict(0, 10) is None and c.conflict(20, 30) is None

    @pytest.mark.parametrize("s,e", [(5, 15), (19, 25), (12, 18), (0, 40)])
    def test_overlap_returns_end_and_owner(self, s, e):
        c = ExclusiveCalendar()
        c.commit(10, 20, "A")
        assert c.conflict(s, e) == (20, "A")

    def test_latest_end_among_overlaps(self):
        c = ExclusiveCalendar()
        c.commit(10, 20, "A")
        c.commit(30, 40, "B")
        assert c.conflict(15, 35) == (40, "B")
        assert c.conflict(15, 25) == (20, "A")
        assert c.conflict(20, 30) is None              # 恰好落在两段之间的空隙

    def test_commit_overlap_rejected(self):
        c = ExclusiveCalendar()
        c.commit(10, 20, "A")
        with pytest.raises(AssertionError):
            c.commit(15, 25, "B")


class TestCounter:
    def test_empty_fits_up_to_cap(self):
        c = CounterCalendar(4)
        assert c.conflict(0, 10, 4) is None

    def test_exactly_full_and_over_by_one(self):
        c = CounterCalendar(4)
        c.commit(10, 20, 3, "A")
        assert c.conflict(10, 20, 1) is None           # 3 + 1 = 4 刚好放满
        assert c.conflict(10, 20, 2) == (20, "A")      # 超限：最后一拍 19 → 返回 20
        assert c.conflict(0, 11, 2) == (11, "A")       # 只有第 10 拍超限
        assert c.conflict(0, 10, 4) is None and c.conflict(20, 30, 4) is None

    def test_last_overloaded_cycle(self):
        c = CounterCalendar(4)
        c.commit(10, 20, 3, "A")
        c.commit(15, 25, 1, "B")                       # [10,15)=3, [15,20)=4, [20,25)=1
        assert [c.level_at(t) for t in (9, 10, 15, 19, 20, 24, 25)] == [0, 3, 4, 4, 1, 1, 0]
        assert c.conflict(12, 30, 1) == (20, "A")      # >3 只在 [15,20)；第 19 拍 A 贡献 3、B 贡献 1

    def test_last_overloaded_cycle_threshold(self):
        c = CounterCalendar(4)
        c.commit(10, 20, 3, "A")
        c.commit(15, 25, 1, "B")
        # k = 3 → 阈值 cap − k = 1：占用 > 1 的拍是 [10,20)，最后一拍 19 → 返回 (20, A)
        assert c.conflict(12, 30, 3) == (20, "A")
        # k = 4 → 阈值 0：占用 > 0 的拍是 [10,25)，最后一拍 24，只有 B 在占 → (25, B)
        assert c.conflict(12, 30, 4) == (25, "B")

    def test_blocker_tie_goes_to_first_committed(self):
        c = CounterCalendar(4)
        c.commit(30, 40, 2, "C")
        c.commit(30, 40, 2, "D")
        assert c.conflict(30, 31, 1) == (31, "C")

    def test_steps(self):
        c = CounterCalendar(8)
        c.commit(0, 5, 2, "A")
        c.commit(3, 9, 4, "B")
        assert c.steps() == [(0, 2), (3, 6), (5, 4), (9, 0)]

    def test_commit_over_capacity_rejected(self):
        c = CounterCalendar(4)
        c.commit(0, 10, 3, "A")
        with pytest.raises(AssertionError):
            c.commit(5, 15, 2, "B")
