"""haversack.torchfree: a checkpoint's tensors without torch.load, and without its pickle.

The oracle is torch itself: every array this reader returns must be what torch.load returns,
bit for bit, dtype for dtype. The rest pins the two changes from the nnunet-inference-mlx
reader it came from (2026-10-22): a storage is read once however many names view it, and a
view must lie inside its storage before as_strided touches memory.
"""
import collections
import io
import pickle
import unittest
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from haversack import torchfree


def _checkpoint():
    """A checkpoint shaped like nnU-Net's: weights with an aliased tensor (ResEnc's
    decoder.encoder.*), an optimizer's momentum, a numpy training log, a torch.device."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv3d(1, 4, 3), torch.nn.InstanceNorm3d(4, affine=True))
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net(torch.randn(1, 1, 5, 5, 5)).sum().backward()
    opt.step()
    weights = collections.OrderedDict(net.state_dict())
    weights["decoder.encoder.0.weight"] = weights["0.weight"]          # one tensor, two names
    return {"network_weights": weights, "optimizer_state": opt.state_dict(),
            "trainer_name": "nnUNetTrainer",
            "init_args": {"configuration": "3d_fullres", "device": torch.device("cpu")},
            "logging": {"mean_fg_dice": [np.float32(0.5), np.float32(0.25)]},
            "inference_allowed_mirroring_axes": (0, 1, 2),
            "extras": {"transposed": torch.arange(12, dtype=torch.float16).reshape(3, 4).t(),
                       "scalar": torch.tensor(3, dtype=torch.int64),
                       "mask": torch.tensor([True, False]),
                       "empty": torch.zeros(0, 3)}}


def _saved(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


class _Counting(zipfile.ZipFile):
    """A ZipFile that records which members are read."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reads = []

    def read(self, name, pwd=None):
        self.reads.append(name)
        return super().read(name, pwd)


class ReadsWhatTorchReads(unittest.TestCase):
    def setUp(self):
        self.ck = _checkpoint()
        self.data = _saved(self.ck)

    def test_every_array_is_torchs_bit_for_bit(self):
        out = torchfree.load(io.BytesIO(self.data), realize=("network_weights", "extras"))
        for part in ("network_weights", "extras"):
            want = self.ck[part]
            self.assertEqual(list(out[part]), list(want))
            for k, t in want.items():
                with self.subTest(tensor=f"{part}.{k}"):
                    got = out[part][k]
                    self.assertEqual(got.dtype, t.numpy().dtype)
                    self.assertEqual(got.shape, tuple(t.shape))
                    self.assertTrue(np.array_equal(got, t.numpy()))
                    self.assertTrue(got.flags.writeable and got.flags.owndata or got.size == 0)

    def test_plain_values_come_back_and_tensors_not_asked_for_do_not(self):
        out = torchfree.load(io.BytesIO(self.data))
        self.assertEqual(out["trainer_name"], "nnUNetTrainer")
        self.assertEqual(out["init_args"]["configuration"], "3d_fullres")
        self.assertEqual(tuple(out["inference_allowed_mirroring_axes"]), (0, 1, 2))
        self.assertEqual([float(x) for x in out["logging"]["mean_fg_dice"]], [0.5, 0.25])
        self.assertNotIn("optimizer_state", out)          # it holds tensors, and was not asked for
        self.assertNotIn("extras", out)

    def test_only_the_storages_asked_for_are_read_and_each_once(self):
        zf = _Counting(io.BytesIO(self.data))
        out = torchfree.load(zf, realize=("network_weights",))
        storages = [n for n in zf.reads if "/data/" in n]
        self.assertEqual(len(storages), len(set(storages)), "a storage was read twice")
        want = {t.untyped_storage().data_ptr() for t in self.ck["network_weights"].values()}
        self.assertEqual(len(storages), len(want))        # the weights' storages, no momentum
        w = out["network_weights"]
        self.assertIs(w["decoder.encoder.0.weight"], w["0.weight"])   # one array, as torch shares


class RefusesWhatItShould(unittest.TestCase):
    def test_a_pickle_naming_any_other_global_is_refused_before_it_runs(self):
        called = []

        class Evil:
            def __reduce__(self):
                return (print, ("pwned",))       # builtins.print: not a checkpoint's global

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data.pkl", pickle.dumps({"x": Evil()}, protocol=2))
        with unittest.mock.patch("builtins.print", side_effect=lambda *a: called.append(a)):
            with self.assertRaisesRegex(torchfree.CheckpointError, r"builtins?__\.print|builtins\.print"):
                torchfree.load(io.BytesIO(buf.getvalue()))
        self.assertEqual(called, [])

    def test_bfloat16_is_refused_by_name(self):
        data = _saved({"network_weights": {"w": torch.zeros(2, dtype=torch.bfloat16)}})
        with self.assertRaisesRegex(torchfree.CheckpointError, "BFloat16Storage"):
            torchfree.load(io.BytesIO(data))

    def test_a_view_outside_its_storage_is_refused(self):
        st = torchfree._Storage(None, "archive/data/0", torchfree._StorageType("FloatStorage"),
                                numel=6, big_endian=False)
        torchfree._Tensor(st, 0, (2, 3), (3, 1))                     # exactly fits
        with self.assertRaises(torchfree.CheckpointError):
            torchfree._Tensor(st, 1, (2, 3), (3, 1))                 # one past the end
        with self.assertRaises(torchfree.CheckpointError):
            torchfree._Tensor(st, 0, (3, 3), (3, 1))

    def test_a_storage_of_the_wrong_length_is_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("archive/data/0", b"\0" * 8)
        zf = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
        st = torchfree._Storage(zf, "archive/data/0", torchfree._StorageType("FloatStorage"),
                                numel=3, big_endian=False)
        with self.assertRaises(torchfree.CheckpointError):
            st.flat()


class Digest(unittest.TestCase):
    def test_it_is_of_values_not_of_bytes_on_disk(self):
        a = {"x": np.arange(6, dtype="<f4").reshape(2, 3), "y": np.ones(2, dtype="<i8")}
        b = {"y": np.ones(2, dtype="<i8"), "x": np.asfortranarray(a["x"])}   # order, layout
        self.assertEqual(torchfree.state_digest(a), torchfree.state_digest(b))
        big = {"x": a["x"].astype(">f4"), "y": a["y"]}
        self.assertEqual(torchfree.state_digest(a), torchfree.state_digest(big))

    def test_a_value_a_name_a_dtype_or_a_shape_changes_it(self):
        a = {"x": np.arange(6, dtype="<f4").reshape(2, 3)}
        d = torchfree.state_digest(a)
        v = a["x"].copy()
        v.view(np.uint32)[0, 0] ^= 1                                 # one bit
        for other in ({"x": v}, {"z": a["x"]}, {"x": a["x"].astype("<f8")},
                      {"x": a["x"].reshape(3, 2)}):
            self.assertNotEqual(torchfree.state_digest(other), d)


class StoredMember(unittest.TestCase):
    def test_it_is_the_members_own_bytes(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("a.txt", b"first")
            zf.writestr("m/fold_0/checkpoint_final.pth", b"payload-bytes")
        raw = buf.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            off, size = torchfree.stored_member(zf, "m/fold_0/checkpoint_final.pth")
        self.assertEqual(raw[off:off + size], b"payload-bytes")

    def test_a_compressed_member_is_refused(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("x.pth", b"x" * 100)
        with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
            with self.assertRaisesRegex(torchfree.CheckpointError, "compressed"):
                torchfree.stored_member(zf, "x.pth")


class WhatTheMutantsFound(unittest.TestCase):
    """Each written to kill a mutant the first tests let through (2026-10-22)."""

    # T1: two DIFFERENT views of one storage are read once (the _realize memo cannot cover it)
    def test_two_different_views_of_one_storage_read_it_once(self):
        base = torch.arange(12, dtype=torch.float32)
        data = _saved({"w": {"a": base, "b": base[2:8].view(2, 3)}})
        zf = _Counting(io.BytesIO(data))
        out = torchfree.load(zf, realize=("w",))
        storages = [n for n in zf.reads if "/data/" in n]
        self.assertEqual(len(storages), 1, storages)
        self.assertTrue(np.array_equal(out["w"]["b"], base[2:8].view(2, 3).numpy()))

    # T10: a view at a nonzero storage offset
    def test_a_view_at_an_offset_is_its_own_elements(self):
        base = torch.arange(10, dtype=torch.int64)
        data = _saved({"w": {"tail": base[3:]}})
        out = torchfree.load(io.BytesIO(data), realize=("w",))
        self.assertTrue(np.array_equal(out["w"]["tail"], np.arange(3, 10)))

    # T9: a big-endian checkpoint reads as the same numbers, in native order
    def test_a_big_endian_checkpoint_reads_the_same_numbers(self):
        t = torch.arange(6, dtype=torch.float32) * 1.5
        src = zipfile.ZipFile(io.BytesIO(_saved({"w": t})))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as dst:
            for n in src.namelist():
                raw = src.read(n)
                if n.endswith("/byteorder"):
                    raw = b"big"
                elif "/data/" in n and not n.endswith(".pkl"):
                    raw = np.frombuffer(raw, "<f4").astype(">f4").tobytes()
                dst.writestr(n, raw)
        out = torchfree.load(io.BytesIO(buf.getvalue()), realize=("w",))
        self.assertTrue(np.array_equal(out["w"], t.numpy()))
        self.assertEqual(out["w"].dtype, np.dtype("<f4"))

    # T11: a negative offset is refused, not wrapped around
    def test_a_negative_offset_is_refused(self):
        st = torchfree._Storage(None, "archive/data/0", torchfree._StorageType("FloatStorage"),
                                numel=6, big_endian=False)
        with self.assertRaises(torchfree.CheckpointError):
            torchfree._Tensor(st, -1, (2,), (1,))
        with self.assertRaises(torchfree.CheckpointError):
            torchfree._Tensor(st, 0, (2,), (-1,))

    # T6: the same bytes under another dtype digest differently
    def test_the_dtype_is_in_the_digest_even_when_the_bytes_are_the_same(self):
        a = np.arange(6, dtype="<f4")
        self.assertNotEqual(torchfree.state_digest({"x": a}),
                            torchfree.state_digest({"x": a.view("<i4")}))

    # T12: a local header with an extra field (Info-ZIP's UT/ux, which `zip` writes)
    def test_a_member_whose_local_header_has_an_extra_field(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            zi = zipfile.ZipInfo("m/fold_0/checkpoint_final.pth")
            zi.extra = b"\x55\x54\x05\x00\x01\x00\x00\x00\x00"       # UT: mtime
            zf.writestr(zi, b"payload-bytes")
        raw = buf.getvalue()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            off, size = torchfree.stored_member(zf, "m/fold_0/checkpoint_final.pth")
        self.assertEqual(raw[off:off + size], b"payload-bytes")


import unittest.mock  # noqa: E402 - used above inside a test body
