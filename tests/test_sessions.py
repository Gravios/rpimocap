"""Tests for multi-session discovery + inventory."""
import numpy as np
import tifffile

from rpimocap.io import sessions as S


def _make_dataset(raw, specs):
    """specs: list of (timestamp, n_cam0, n_cam1, has_meta)."""
    import os
    os.makedirs(raw, exist_ok=True)
    for ts, n0, n1, meta in specs:
        if n0 is not None:
            tifffile.imwrite(os.path.join(raw, f"cam0_{ts}_raw.tif"),
                             np.zeros((n0, 16, 16), np.uint16))
        if n1 is not None:
            tifffile.imwrite(os.path.join(raw, f"cam1_{ts}_raw.tif"),
                             np.zeros((n1, 16, 16), np.uint16))
        if meta:
            with open(os.path.join(raw, f"{ts}_metadata.csv"), "w") as fh:
                fh.write("frame,ts\n")


class TestDiscovery:

    def test_finds_all_sessions(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [
            ("20260214_021058", 100, 100, True),
            ("20260214_021255", 120, 120, True),
            ("20260214_021410", 80, 80, True),
            ("20260214_021722", 1300, 950, True),
        ])
        sess = S.discover_sessions(raw)
        assert len(sess) == 4
        assert all(s.complete for s in sess)
        # sorted chronologically
        assert [s.timestamp for s in sess] == sorted(
            s.timestamp for s in sess)

    def test_metadata_linked(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [("20260214_021058", 10, 10, True),
                            ("20260214_021255", 10, 10, False)])
        sess = {s.timestamp: s for s in S.discover_sessions(raw)}
        assert sess["20260214_021058"].metadata_path is not None
        assert sess["20260214_021255"].metadata_path is None

    def test_incomplete_session(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [("20260214_021058", 10, None, True)])
        sess = S.discover_sessions(raw)
        assert len(sess) == 1
        assert not sess[0].complete
        assert sess[0].cam1_path is None

    def test_empty_dir(self, tmp_path):
        import os
        raw = str(tmp_path / "raw")
        os.makedirs(raw)
        assert S.discover_sessions(raw) == []


class TestInventory:

    def test_frame_counts_and_mismatch(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [
            ("20260214_021058", 100, 100, True),
            ("20260214_021722", 1300, 950, True),
        ])
        rows = {r["timestamp"]: r for r in S.inventory(raw)}
        assert rows["20260214_021058"]["frames_match"] is True
        assert rows["20260214_021058"]["stereo_overlap"] == 100
        bad = rows["20260214_021722"]
        assert bad["frames_match"] is False
        assert bad["cam0_frames"] == 1300
        assert bad["cam1_frames"] == 950
        assert bad["stereo_overlap"] == 950        # the usable range

    def test_session_frame_counts(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [("20260214_021058", 42, 37, True)])
        s = S.discover_sessions(raw)[0]
        n0, n1 = S.session_frame_counts(s)
        assert n0 == 42 and n1 == 37

    def test_inventory_has_metadata_flag(self, tmp_path):
        raw = str(tmp_path / "raw")
        _make_dataset(raw, [("20260214_021058", 10, 10, False)])
        rows = S.inventory(raw)
        assert rows[0]["has_metadata"] is False
