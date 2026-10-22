"""Read a PyTorch checkpoint's tensors without torch, and without running its pickle.

A ``.pth`` written by ``torch.save`` (the zip format, torch >= 1.6) is a zip of one pickle,
``<prefix>/data.pkl``, and one raw member per tensor storage, ``<prefix>/data/<key>``. The
pickle says which storage each tensor is a view of; the bytes are plain little-endian arrays.
This reads the pickle with an unpickler that refuses every global but the handful a
checkpoint's tensors and plain values need, and reads a storage's bytes only when a tensor
that is asked for uses it - so ``network_weights`` comes out of an nnU-Net checkpoint without
its optimizer state ever being read, which is half the file, and over HTTP Range half the
download (2026-10-14, LNQ's 815 MB folds: 440 MB fetched for 407 MB of weights).

Why not ``torch.load(weights_only=True)``: nnU-Net checkpoints keep their training log as
numpy scalars, which torch's restricted loader refuses (torch 2.14), so nnU-Net's own predictor
loads with ``weights_only=False`` - a full unpickle of a third party's file. And why not the
original: this is the reader from nnunet-inference-mlx 0.4.0 (``_torchfree``, May 2026), with
two changes. A storage is read ONCE however many tensors view it - ResEnc's state dict names its
encoder twice (``encoder.*`` and ``decoder.encoder.*``), and the original fetched those bytes
again for every alias (1.55 GB for a 407 MB model over HTTP). And a view must lie inside its
storage, checked before ``as_strided`` touches memory, so a damaged or hostile file is an error
and not an out-of-bounds read.

numpy only, at call time from the pipeline layer; no torch anywhere in this module.
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import zipfile

import numpy as np

#: torch storage class -> numpy dtype. bfloat16 has no numpy dtype, and is refused by name
#: rather than handed back as raw uint16 bits a caller could mistake for numbers.
_STORAGE_DTYPE = {
    "DoubleStorage": np.dtype("<f8"), "FloatStorage": np.dtype("<f4"),
    "HalfStorage": np.dtype("<f2"), "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"), "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"), "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


class CheckpointError(ValueError):
    """The file is not a checkpoint this reader can read safely: a global outside the allowed
    set, a dtype numpy cannot hold, a view outside its storage."""


class _StorageType:
    def __init__(self, name: str):
        if name not in _STORAGE_DTYPE:
            raise CheckpointError(f"checkpoint uses torch.{name}, which this reader does not read")
        self.name, self.dtype = name, _STORAGE_DTYPE[name]


class _Storage:
    """One ``data/<key>`` member, read on first use and kept: every tensor viewing it shares
    the one array."""

    def __init__(self, zf, member: str, stype: _StorageType, numel: int, big_endian: bool):
        self.zf, self.member, self.dtype = zf, member, stype.dtype
        self.numel, self.big_endian = int(numel), big_endian
        self._flat = None

    def flat(self) -> np.ndarray:
        if self._flat is None:
            raw = self.zf.read(self.member)
            dtype = self.dtype.newbyteorder(">") if self.big_endian else self.dtype
            if len(raw) != self.numel * dtype.itemsize:
                raise CheckpointError(f"{self.member}: {len(raw)} bytes for {self.numel} "
                                      f"{self.dtype} elements")
            flat = np.frombuffer(raw, dtype=dtype)
            self._flat = flat.astype(self.dtype) if self.big_endian else flat
        return self._flat


class _Tensor:
    """A view of a storage, not yet read."""

    def __init__(self, storage: _Storage, offset, shape, stride):
        self.storage, self.offset = storage, int(offset)
        self.shape, self.stride = tuple(int(s) for s in shape), tuple(int(s) for s in stride)
        if self.offset < 0 or any(s < 0 for s in self.shape) or any(s < 0 for s in self.stride):
            raise CheckpointError(f"{storage.member}: a negative offset, size or stride")
        if all(self.shape):
            # the last element this view reaches must be inside the storage, or as_strided
            # would read memory that is not the file's
            last = self.offset + sum((n - 1) * s for n, s in zip(self.shape, self.stride))
            if last >= storage.numel:
                raise CheckpointError(f"{storage.member}: a view reaches element {last} of "
                                      f"{storage.numel}")

    def key(self) -> tuple:
        return (self.storage.member, self.offset, self.shape, self.stride)

    def realize(self) -> np.ndarray:
        flat = self.storage.flat()
        if not all(self.shape):
            return np.zeros(self.shape, dtype=flat.dtype)
        view = np.lib.stride_tricks.as_strided(
            flat[self.offset:], shape=self.shape,
            strides=tuple(s * flat.itemsize for s in self.stride), writeable=False)
        return np.array(view, copy=True)           # owned and writable, as torch.from_numpy wants


def _rebuild_tensor(storage, offset, size, stride, *_):
    return _Tensor(storage, offset, size, stride)


def _rebuild_parameter(data, requires_grad=None, backward_hooks=None, *_):
    return data                                     # an nn.Parameter is its tensor here


def _device(*args, **kwargs):
    return "torch.device(" + ", ".join(repr(a) for a in args) + ")"


class _Unpickler(pickle.Unpickler):
    """Only what a checkpoint's tensors and plain values need. Anything else is refused BY
    NAME, before it can be called - that is the whole point of not using pickle.load."""

    def __init__(self, file, zf, prefix: str, big_endian: bool):
        super().__init__(file)
        self.zf, self.prefix, self.big_endian = zf, prefix, big_endian
        self.storages: dict[str, _Storage] = {}

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        if module == "torch" and name == "device":
            return _device
        if module == "collections" and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        # numpy values in a training log (nnU-Net's), numpy >= 2 and older spellings both
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            from numpy._core import multiarray
            if name in ("scalar", "_reconstruct"):
                return getattr(multiarray, name)
        if module in ("numpy", "numpy.core", "numpy._core") and name == "ndarray":
            return np.ndarray
        if module == "numpy" and name == "dtype":
            return np.dtype
        if module == "numpy.dtypes" and name.endswith("DType"):
            return type(np.dtype(name[:-len("DType")].lower()))
        if module == "_codecs" and name == "encode":
            import codecs
            return codecs.encode
        raise CheckpointError(f"checkpoint pickle names {module}.{name}, which a checkpoint "
                              "of tensors and plain values does not need; refusing to load it")

    def persistent_load(self, pid):
        # ("storage", storage type, key, location, numel)
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise CheckpointError(f"unexpected persistent id {pid!r:.80}")
        _, stype, key, _location, numel = pid
        if not isinstance(stype, _StorageType):
            raise CheckpointError(f"storage {key!r} has no storage type")
        key = str(key)
        if key not in self.storages:
            self.storages[key] = _Storage(self.zf, f"{self.prefix}data/{key}", stype, numel,
                                          self.big_endian)
        return self.storages[key]


def _has_tensor(obj) -> bool:
    if isinstance(obj, _Tensor):
        return True
    if isinstance(obj, dict):
        return any(_has_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_tensor(v) for v in obj)
    return False


def _realize(obj, memo: dict):
    if isinstance(obj, _Tensor):
        k = obj.key()
        if k not in memo:                         # aliases of one view are one array object
            memo[k] = obj.realize()
        return memo[k]
    if isinstance(obj, dict):
        return type(obj)((k, _realize(v, memo)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return type(obj)(_realize(v, memo) for v in obj)
    return obj


def load(source, *, realize=("network_weights",)) -> dict:
    """The checkpoint's top-level dict: the entries named in ``realize`` with their tensors
    as numpy arrays, and every other entry that holds no tensor as it is (an nnU-Net
    checkpoint's ``trainer_name``, ``init_args``, ``inference_allowed_mirroring_axes``, its
    log). An entry holding tensors that was not asked for is left out, and its storages are
    never read.

    ``source`` is a path, a seekable binary file (an HTTP ``RangeFile`` reads only what is
    used) or an open ``ZipFile``. A view that occurs under two names (ResEnc's
    ``decoder.encoder.*``) comes back as ONE array object under both, as torch shares it."""
    if isinstance(source, zipfile.ZipFile):
        return _load_zip(source, realize)
    with zipfile.ZipFile(source) as zf:
        return _load_zip(zf, realize)


def _load_zip(zf: zipfile.ZipFile, realize) -> dict:
    names = zf.namelist()
    pkls = [n for n in names if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pkls) != 1:
        raise CheckpointError(f"not a torch zip checkpoint: {len(pkls)} data.pkl members")
    prefix = pkls[0][:-len("data.pkl")]
    order = (zf.read(prefix + "byteorder").decode().strip()
             if prefix + "byteorder" in names else "little")
    if order not in ("little", "big"):
        raise CheckpointError(f"byteorder {order!r}")
    up = _Unpickler(io.BytesIO(zf.read(pkls[0])), zf, prefix, order == "big")
    try:
        top = up.load()
    except CheckpointError:
        raise
    except Exception as e:                        # a truncated or garbled pickle
        raise CheckpointError(f"checkpoint pickle could not be read ({type(e).__name__}: {e})") from e
    if not isinstance(top, dict):
        raise CheckpointError(f"checkpoint holds a {type(top).__name__}, not a dict")
    memo: dict = {}
    out = {}
    for k, v in top.items():
        if k in realize:
            out[k] = _realize(v, memo)
        elif not _has_tensor(v):
            out[k] = v
    return out


def state_digest(state: dict) -> str:
    """A sha256 of a state dict's names, dtypes, shapes and values - not of any file's bytes,
    so a checkpoint re-saved by another torch, or read here instead of by torch, digests the
    same. Names in sorted order; each array little-endian and C-contiguous."""
    h = hashlib.sha256()
    for name in sorted(state):
        a = np.ascontiguousarray(state[name])
        a = a.astype(a.dtype.newbyteorder("<")) if a.dtype.byteorder == ">" else a
        h.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode("utf-8"))
        h.update(b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def stored_member(zf: zipfile.ZipFile, name: str) -> tuple[int, int]:
    """``(offset, size)`` of member ``name``'s bytes in the archive, for a member stored
    without compression - which is what lets a reader address a ``.pth`` inside a release zip
    with Range requests, and read only the storages it needs. ``zf`` may be over HTTP."""
    info = zf.getinfo(name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise CheckpointError(f"{name} is compressed in its archive; it cannot be read in place")
    fp = zf.fp
    fp.seek(info.header_offset)
    header = fp.read(30)
    if header[:4] != b"PK\x03\x04":
        raise CheckpointError(f"{name}: no local file header at {info.header_offset}")
    name_len = int.from_bytes(header[26:28], "little")
    extra_len = int.from_bytes(header[28:30], "little")
    return info.header_offset + 30 + name_len + extra_len, info.file_size
