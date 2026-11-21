"""machine/model.py 的单元测试。"""
import os

import pytest

from ltsim.errors import LTSError
from ltsim.machine.model import load_machine

MACHINE_YAML = os.path.join(os.path.dirname(__file__), "..", "..", "ltsim", "machine", "lpu-tile-v0.yaml")


@pytest.fixture(scope="module")
def m():
    return load_machine(MACHINE_YAML)


class TestParam:
    def test_simple_path(self, m):
        assert m.param("mxm.lw_to_acc_gap") == 1
        assert m.param("mxm.block_switch_gap") == 5
        assert m.param("srf.streams_per_direction") == 32
        assert m.param("mem.read_vectors_per_cycle") == 44

    def test_per_dtype_mxm_params(self, m):
        # 只有延迟随精度变化：f16/bf16 是 i8 的 2 倍；acc_ii 不变
        assert m.param("mxm.acc_latency.i8") == 25
        assert m.param("mxm.acc_latency.f16") == 50
        assert m.param("mxm.acc_latency.bf16") == 50
        assert m.param("mxm.acc_ii") == 1
        # 权重装载：f16/bf16 每行 2 个向量，速率减半
        assert m.param("mxm.lw_rows_per_cycle.i8") == 16
        assert m.param("mxm.lw_rows_per_cycle.f16") == 8
        assert m.param("mxm.lw_rows_per_cycle.bf16") == 8

    def test_missing_raises(self, m):
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            m.param("mxm.no_such_field")

    def test_missing_section_raises(self, m):
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            m.param("nosuch.field")


class TestHop:
    def test_mem_w_to_mxm_w(self, m):
        assert m.hop("MEM_W", "MXM_W") == 8

    def test_mxm_w_to_vxm(self, m):
        assert m.hop("MXM_W", "VXM") == 21

    def test_vxm_to_mem_e(self, m):
        assert m.hop("VXM", "MEM_E") == 2

    def test_missing_hop_raises(self, m):
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            m.hop("MEM_W", "MEM_W")


class TestOpLatency:
    def test_gelu(self, m):
        assert m.op_latency("gelu.f32") == 18

    def test_add_i32(self, m):
        assert m.op_latency("add.i32") == 2

    def test_missing_raises(self, m):
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            m.op_latency("nosuch.op")


class TestResources:
    def test_mxm_planes_exclusive(self, m):
        res = m.resources()
        for p in ("mxm.w0", "mxm.w1", "mxm.e0", "mxm.e1"):
            assert res[f"{p}.weight"].kind == "exclusive"
            assert res[f"{p}.compute"].kind == "exclusive"
            assert res[f"{p}.weight"].cap == 1

    def test_vxm_counter(self, m):
        assert m.resources()["vxm.alu"].cap == 16

    def test_srf_counter(self, m):
        res = m.resources()
        assert res["srf.W2E"].cap == 32
        assert res["srf.E2W"].cap == 32

    def test_mem_counters(self, m):
        res = m.resources()
        for h in ("W", "E"):
            assert res[f"mem.{h}.rd"].cap == 44
            assert res[f"mem.{h}.wr"].cap == 44

    def test_sxm_exclusive(self, m):
        res = m.resources()
        assert res["sxm.w"].kind == "exclusive"
        assert res["sxm.e"].kind == "exclusive"


class TestLoadMachine:
    def test_bad_schema(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("schema: wrong\ngeometry: {}\n")
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            load_machine(str(p))

    def test_missing_section(self, tmp_path):
        p = tmp_path / "bad2.yaml"
        p.write_text("schema: lts-machine/0.1\ngeometry: {}\n")
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            load_machine(str(p))


class TestGeometry:
    def _write_variant(self, tmp_path, mutate):
        """读入真实 yaml，用 mutate(raw) 修改后写到临时文件。"""
        import yaml
        raw = yaml.safe_load(open(MACHINE_YAML))
        mutate(raw)
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump(raw))
        return str(p)

    def test_values(self, m):
        assert m.param("geometry.superlanes") == 20
        assert m.param("geometry.lanes_per_superlane") == 16
        assert m.param("geometry.vector_bytes") == 320
        assert m.param("srf.wave_skew_cycles") == 19

    def test_vector_bytes_mismatch(self, tmp_path):
        p = self._write_variant(tmp_path, lambda r: r["geometry"].update(vector_bytes=256))
        with pytest.raises(LTSError, match="LTS-E-MACHINE.*vector_bytes"):
            load_machine(p)

    def test_wave_skew_mismatch(self, tmp_path):
        p = self._write_variant(tmp_path, lambda r: r["srf"].update(wave_skew_cycles=20))
        with pytest.raises(LTSError, match="LTS-E-MACHINE.*wave_skew"):
            load_machine(p)

    def test_consistent_variant_ok(self, tmp_path):
        """16 superlane：vector_bytes = 16×16 = 256，wave_skew = 15，三者自洽即可加载。"""
        def mut(r):
            r["geometry"].update(superlanes=16, vector_bytes=256)
            r["srf"].update(wave_skew_cycles=15)
        m2 = load_machine(self._write_variant(tmp_path, mut))
        assert m2.param("geometry.vector_bytes") == 256

    def test_superlanes_change_alone_fails(self, tmp_path):
        """只改 superlanes、不改推导量：先撞上 vector_bytes 校验。"""
        p = self._write_variant(tmp_path, lambda r: r["geometry"].update(superlanes=16))
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            load_machine(p)

    def test_missing_lanes_per_superlane(self, tmp_path):
        p = self._write_variant(tmp_path, lambda r: r["geometry"].pop("lanes_per_superlane"))
        with pytest.raises(LTSError, match="LTS-E-MACHINE"):
            load_machine(p)
