"""Multi-fold nnU-Net ensembles with the ShuffleUp3d surgery on (2026-10-18).

Until that day `TorchModel._load_fold` raised NotImplementedError for any fold after the first
whenever the surgery was on - the default - so no multi-fold model ran on any device. No shipped
catalog had one; Steve Pieper's LNQ lymph-node models (ResEnc-M, 4-5 folds) were the first.

The oracle is nnU-Net's own predictor (`predict_logits_from_preprocessed_data`, mirroring off)
over the same folder and the same preprocessed tensor, not haversack's unsurgered path: a defect
in the fold loop that both haversack paths share (every fold loaded as fold 0, say) would make
them agree with each other and still be wrong.

The model is a tiny ResidualEncoderUNet - the LNQ models' architecture - built from its plans by
nnU-Net itself, with an anisotropic (1, 2, 2) upsampling beside an isotropic (2, 2, 2) one, so
the weight mapping is exercised on both shapes. Every fold's weights are distinct random draws.
"""
import json
import os
import unittest
from unittest import mock

import pytest
import torch

pytest.importorskip("nnunetv2")
pytest.importorskip("dynamic_network_architectures")

from haversack import cache
from haversack.network import TorchModel
from haversack.shuffleup import ShuffleUp3d, as_shuffled, surgered_state, swap_transposed

from conftest import device_names

FOLDER = "nnUNetTrainer__nnUNetPlans__3d_fullres"
ARCH = {
    "network_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
    "arch_kwargs": {
        "n_stages": 3, "features_per_stage": [4, 8, 8],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3]],
        "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2]],
        "n_blocks_per_stage": [1, 1, 1], "n_conv_per_stage_decoder": [1, 1],
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-05, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
    },
    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
}
RESAMPLE = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None}
PLANS = {
    "dataset_name": "Dataset990_Tiny", "plans_name": "nnUNetPlans",
    "original_median_spacing_after_transp": [2.0, 1.0, 1.0],
    "original_median_shape_after_transp": [24, 32, 32],
    "image_reader_writer": "SimpleITKIO",
    "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
    "experiment_planner_used": "ExperimentPlanner", "label_manager": "LabelManager",
    "foreground_intensity_properties_per_channel": {"0": {
        "max": 500.0, "mean": 40.0, "median": 40.0, "min": -300.0,
        "percentile_00_5": -50.0, "percentile_99_5": 150.0, "std": 35.0}},
    "configurations": {"3d_fullres": {
        "data_identifier": "nnUNetPlans_3d_fullres", "preprocessor_name": "DefaultPreprocessor",
        "batch_size": 2, "patch_size": [8, 16, 16], "median_image_size_in_voxels": [24, 32, 32],
        "spacing": [2.0, 1.0, 1.0], "normalization_schemes": ["CTNormalization"],
        "use_mask_for_norm": [False],
        "resampling_fn_data": "resample_data_or_seg_to_shape", "resampling_fn_data_kwargs": RESAMPLE,
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_seg_kwargs": dict(RESAMPLE, is_seg=True, order=1),
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities_kwargs": dict(RESAMPLE, order=1),
        "architecture": ARCH, "batch_dice": True}},
}
DATASET = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "a": 1, "b": 2},
           "numTraining": 1, "file_ending": ".nii.gz"}
FOLDS = (0, 1, 2)


def _network():
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    return get_network_from_plans(ARCH["network_class_name"], ARCH["arch_kwargs"],
                                  ARCH["_kw_requires_import"], 1, len(DATASET["labels"]),
                                  allow_init=True, deep_supervision=False)


def perturbed_network():
    """The tiny network with every parameter moved off its init, so norm affines and biases
    differ per draw too. In place, per PARAMETER: a ResidualEncoderUNet's state dict names its
    encoder twice (``encoder.*`` and ``decoder.encoder.*``, one tensor), so noise added per key
    would state two values for one weight."""
    net = _network()
    with torch.no_grad():
        for t in net.parameters():
            t.add_(0.3 * torch.randn_like(t))
    return net


def make_model_folder(root, folds=FOLDS):
    """A stock nnU-Net result folder with one distinct random checkpoint per fold."""
    folder = root / FOLDER
    folder.mkdir(parents=True)
    (folder / "plans.json").write_text(json.dumps(PLANS))
    (folder / "dataset.json").write_text(json.dumps(DATASET))
    for f in folds:
        torch.manual_seed(1000 + f)
        state = perturbed_network().state_dict()
        (folder / f"fold_{f}").mkdir()
        torch.save({"network_weights": state, "trainer_name": "nnUNetTrainer",
                    "init_args": {"configuration": "3d_fullres"},
                    "inference_allowed_mirroring_axes": (0, 1, 2)},
                   folder / f"fold_{f}" / "checkpoint_final.pth")
    return folder


def preprocessed(seed=7, shape=(1, 13, 37, 29)):
    """A normalized-looking input: not a multiple of the patch, so the window tiles and pads."""
    torch.manual_seed(seed)
    return torch.randn(shape)


def stock_logits(folder, folds, x):
    """nnU-Net's own ensemble, mirroring off, on the CPU in fp32 - the oracle."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        perform_everything_on_device=False, device=torch.device("cpu"),
                        verbose=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(str(folder), use_folds=folds,
                                           checkpoint_name="checkpoint_final.pth")
    return p.predict_logits_from_preprocessed_data(x.clone()).float()


def haversack_logits(folder, folds, x, *, device="cpu", dtype="fp32", surgery=True):
    m = TorchModel(folder, folds=folds, device=device, dtype=dtype, surgery=surgery,
                   accumulate="host", batch_size=1)
    return m, m.predict_logits(x.clone()).float().cpu()


class _Folder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        cls._tmp = tempfile.TemporaryDirectory()
        cls.folder = make_model_folder(Path(cls._tmp.name))
        cls.x = preprocessed()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


def within_surgery_rounding(got, want, ulps=4):
    """``got`` within ``ulps`` fp16 steps of ``want`` everywhere. The surgery is exact in
    arithmetic but not in rounding on every backend: on the CPU a 1x1 conv sums its input
    channels in another order than a transposed conv, 2.9e-6 apart in fp32 on this net
    (2026-10-18), and the fp16 accumulator turns that into a step now and then - a SINGLE
    surgered fold already differed from nnU-Net by up to 2 steps on 19 of 41,847 logits before
    multi-fold existed. Folds here differ by > 0.1, i.e. hundreds of steps."""
    step = torch.finfo(torch.float16).eps * want.abs().clamp_min(1.0)
    return bool(((got - want).abs() <= ulps * step).all())


def fold_network(folder, f):
    """An UNsurgered network holding fold ``f``'s checkpoint, as nnU-Net would load it."""
    net = _network().eval()
    ck = torch.load(folder / f"fold_{f}" / "checkpoint_final.pth", map_location="cpu",
                    weights_only=False)
    net.load_state_dict(ck["network_weights"])
    return net


class TestMultiFoldEnsemble(_Folder):
    def test_the_fixture_actually_has_transposed_convs_and_distinct_folds(self):
        # guards against a fixture that makes every assertion below vacuous
        m, _ = haversack_logits(self.folder, FOLDS, self.x)
        self.assertEqual(m.n_swapped, 2)
        one = [stock_logits(self.folder, (f,), self.x) for f in FOLDS]
        self.assertGreater((one[0] - one[1]).abs().max().item(), 0.1)
        self.assertGreater((one[1] - one[2]).abs().max().item(), 0.1)

    def test_each_fold_the_surgered_network_holds_is_that_folds_network(self):
        # the per-fold statement, in fp32 and before any fp16 accumulator: after loading fold f
        # the surgered network computes what fold f's checkpoint computes unsurgered
        m = TorchModel(self.folder, folds=FOLDS, device="cpu", dtype="fp32")
        patch = torch.randn(1, 1, 8, 16, 16, generator=torch.Generator().manual_seed(5))
        for f in (1, 2, 0, 2):
            m._load_fold(FOLDS.index(f))
            with torch.no_grad():
                got, want = m.net(patch), fold_network(self.folder, f)(patch)
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5, msg=f"fold {f}")
        # and the checkpoints' ConvTranspose3d tensors are not kept alive beside the mapped ones
        for state in m.predictor.list_of_parameters:
            self.assertFalse([k for k in state if k.endswith("transpconvs.0.weight")])

    def test_a_surgered_ensemble_is_nnunets_ensemble_up_to_the_surgerys_rounding(self):
        want = stock_logits(self.folder, FOLDS, self.x)
        _, got = haversack_logits(self.folder, FOLDS, self.x, surgery=True)
        self.assertEqual(got.shape, want.shape)
        self.assertTrue(within_surgery_rounding(got, want),
                        f"max |diff| {(got - want).abs().max().item()}")

    def test_an_unsurgered_ensemble_is_nnunets_ensemble_bit_for_bit(self):
        # the fold loop both paths share, held to the oracle exactly where nothing rounds apart
        want = stock_logits(self.folder, FOLDS, self.x)
        _, got = haversack_logits(self.folder, FOLDS, self.x, surgery=False)
        self.assertTrue(torch.equal(got, want), f"max |diff| {(got - want).abs().max().item()}")

    def test_the_held_fold_does_not_leak_into_the_next_volume(self):
        # a call ends holding the last fold; the next must start from fold 0 again. The same
        # input through the same (cached) model is bitwise the same answer on the third call.
        y = preprocessed(seed=8)
        want = {k: stock_logits(self.folder, (0, 2), v) for k, v in (("x", self.x), ("y", y))}
        m = TorchModel(self.folder, folds=(0, 2), device="cpu", dtype="fp32",
                       accumulate="host", batch_size=1)
        got = [m.predict_logits(v.clone()).float() for v in (self.x, y, self.x)]
        self.assertTrue(within_surgery_rounding(got[0], want["x"]))
        self.assertTrue(within_surgery_rounding(got[1], want["y"]))
        self.assertTrue(torch.equal(got[0], got[2]))

    def test_single_fold_runs_as_before_and_never_reloads(self):
        # before 2026-10-18 a surgered single fold ran the network built at load and never
        # loaded a state again; an unsurgered one reloaded fold 0 on every call (the same bytes)
        for surgery in (True, False):
            want = stock_logits(self.folder, (1,), self.x)
            m = TorchModel(self.folder, folds=(1,), device="cpu", dtype="fp32", surgery=surgery,
                           accumulate="host", batch_size=1)
            calls = []
            load = m.net.load_state_dict
            m.net.load_state_dict = lambda *a, **k: (calls.append(1), load(*a, **k))[1]
            got = [m.predict_logits(self.x.clone()).float() for _ in range(2)]
            self.assertEqual(calls, [], f"surgery={surgery}")
            self.assertTrue(torch.equal(got[0], got[1]))
            if surgery:
                self.assertTrue(within_surgery_rounding(got[0], want))
            else:
                self.assertTrue(torch.equal(got[0], want))


class TestHeldFold(_Folder):
    """What `_held` promises, asked directly: within predict_logits no fold is asked for twice
    in a row, so only a repeat shows the tracker is kept, and only a failed load shows it is
    cleared first (both survived the first mutation round, found by review 2026-10-18)."""

    def _counting(self, m):
        calls = []
        load = m.net.load_state_dict
        m.net.load_state_dict = lambda *a, **k: (calls.append(1), load(*a, **k))[1]
        return calls

    def test_loading_the_fold_already_held_is_free(self):
        m = TorchModel(self.folder, folds=FOLDS, device="cpu", dtype="fp32")
        calls = self._counting(m)
        m._load_fold(1)
        m._load_fold(1)
        self.assertEqual(len(calls), 1)

    def test_a_failed_load_leaves_no_fold_believed_held(self):
        # a strict load copies what matches, then raises: the network holds a mixture, and the
        # tracker must not go on naming the fold it held before
        m = TorchModel(self.folder, folds=FOLDS, device="cpu", dtype="fp32")
        good = m.fold_params[1]
        bad = dict(good)
        bad.pop(next(k for k in bad if k.endswith("conv.bias")))
        m.fold_params[1] = bad
        with self.assertRaises(RuntimeError):
            m._load_fold(1)
        m.fold_params[1] = good
        m._load_fold(0)
        patch = torch.randn(1, 1, 8, 16, 16, generator=torch.Generator().manual_seed(5))
        with torch.no_grad():
            got, want = m.net(patch), fold_network(self.folder, 0)(patch)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


class TestUnderNnunetCompile(_Folder):
    """nnU-Net wraps the network in torch.compile when `nnUNet_compile` is set, and the wrapper
    names every module and state entry `_orig_mod.<name>`. The first version of this change mapped
    and loaded through the wrapper and failed EVERY surgered model there, single folds too (found
    by review, 2026-10-18); multi-fold with surgery off failed there on main as well.

    The folds are checked through the module INSIDE the wrapper, never by calling the compiled
    network: that would run inductor, which wants a C compiler and minutes, and what is under
    test is which weights the module holds, not what the compiler makes of them."""

    def test_folds_load_into_the_module_inside_the_wrapper(self):
        from torch._dynamo.eval_frame import OptimizedModule
        patch = torch.randn(1, 1, 8, 16, 16, generator=torch.Generator().manual_seed(5))
        for surgery in (True, False):
            with mock.patch.dict(os.environ, {"nnUNet_compile": "1"}):
                m = TorchModel(self.folder, folds=FOLDS, device="cpu", dtype="fp32",
                               surgery=surgery)
            self.assertIsInstance(m.net, OptimizedModule)          # else this tests nothing
            self.assertEqual(m.n_swapped, 2 if surgery else 0)
            for f in (1, 2, 0):
                m._load_fold(FOLDS.index(f))
                with torch.no_grad():
                    got, want = m.net._orig_mod(patch), fold_network(self.folder, f)(patch)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5,
                                           msg=f"surgery={surgery} fold {f}")


@pytest.mark.parametrize("device", [d for d in device_names() if d != "cpu"])
def test_a_surgered_ensemble_in_half_precision_on_each_device(device, tmp_path):
    """fp16 on the GPU (MPS refuses ConvTranspose3d in half; the surgery is what runs it there)
    against nnU-Net's fp32 CPU ensemble. A half-precision network moves these logits (|l| <~ 3)
    by up to 0.05 on MPS (2026-10-18); fold 0 alone is ~10x further from the ensemble."""
    folder = make_model_folder(tmp_path)
    x = preprocessed()
    want = stock_logits(folder, FOLDS, x)
    _, got = haversack_logits(folder, FOLDS, x, device=device, dtype="fp16")
    tol = 0.02 * want.abs().max().item() + 0.02
    assert (got - want).abs().max().item() <= tol
    top2 = want.topk(2, dim=0).values
    clear = (top2[0] - top2[1]) > 2 * tol
    assert clear.float().mean().item() > 0.5
    assert torch.equal(got.argmax(0)[clear], want.argmax(0)[clear])
    # and the tolerance is tight enough that fold 0's weights for every fold would fail it
    alone = stock_logits(folder, (0,), x)
    assert (alone - want).abs().max().item() > 3 * tol


class TestSurgeredState(unittest.TestCase):
    def test_it_is_what_the_surgery_builds_from_the_same_weights(self):
        torch.manual_seed(3)
        net = perturbed_network()
        state = {k: v.clone() for k, v in net.state_dict().items()}
        swap_transposed(net)
        mapped = surgered_state(state, net)
        have = net.state_dict()
        self.assertEqual(set(mapped), set(have))
        for k in have:
            self.assertTrue(torch.equal(mapped[k], have[k]), k)

    def test_untouched_entries_are_the_same_tensors(self):
        net = _network()
        state = net.state_dict()
        swap_transposed(net)
        mapped = surgered_state(state, net)
        shared = [k for k in state if k in mapped]
        self.assertTrue(shared)
        self.assertTrue(all(mapped[k] is state[k] for k in shared))

    def test_a_checkpoint_of_another_shape_is_refused_by_name(self):
        net = _network()
        state = net.state_dict()
        swap_transposed(net)
        name = next(n for n, m in net.named_modules() if isinstance(m, ShuffleUp3d))
        bad = dict(state)
        bad[f"{name}.weight"] = bad[f"{name}.weight"][..., :1]
        with self.assertRaisesRegex(ValueError, name):
            surgered_state(bad, net)
        del bad[f"{name}.weight"]
        with self.assertRaisesRegex(ValueError, name):
            surgered_state(bad, net)

    def test_a_checkpoint_with_other_output_channels_is_refused_by_name(self):
        # the kernel test above leaves the output-channel half of the check unasked
        net = _network()
        state = net.state_dict()
        swap_transposed(net)
        name = next(n for n, m in net.named_modules() if isinstance(m, ShuffleUp3d))
        bad = dict(state)
        bad[f"{name}.weight"] = bad[f"{name}.weight"][:, :1]
        with self.assertRaisesRegex(ValueError, name):
            surgered_state(bad, net)

    def test_the_mapping_without_a_bias(self):
        t = torch.nn.ConvTranspose3d(3, 5, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
        w, b = as_shuffled(t.weight)
        self.assertIsNone(b)
        x = torch.randn(1, 3, 2, 3, 4)
        s = ShuffleUp3d(t)
        self.assertTrue(torch.equal(s.conv.weight, w))
        torch.testing.assert_close(s(x), t(x), rtol=0, atol=1e-6)


class TestModelCacheKeysEveryChoice(unittest.TestCase):
    """`surgery` is NOT in ModelCache's key, deliberately: the cache never takes it, so every model
    it builds is surgered (the TorchModel default), and no request through it can ask otherwise.
    What would break that is a knob added to `get` and forgotten in `_key` - a model built one way
    served for a request made another. So the two signatures are held equal."""

    def test_get_and_key_take_the_same_choices(self):
        import inspect
        get = set(inspect.signature(cache.ModelCache.get).parameters) - {"self"}
        key = set(inspect.signature(cache.ModelCache._key).parameters)
        self.assertEqual(get, key)
        self.assertNotIn("surgery", get)
