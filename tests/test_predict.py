import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

from PIL import Image


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeProcessor:
    def __call__(self, images, return_tensors):
        assert isinstance(images, Image.Image)
        assert return_tensors == "pt"
        return FakeBatch({"pixel_values": "fake"})


class FakeConfig:
    id2label = {0: "cat", 1: "dog"}


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, _index):
        return self.values


class FakeProbabilities:
    def __init__(self, values):
        self.values = values

    def argmax(self):
        return 1

    def __getitem__(self, index):
        return self.values[index]


class FakeOutputs:
    logits = FakeTensor([0.1, 0.9])


class FakeModel:
    def __init__(self):
        self.config = FakeConfig()
        self.device_received = None
        self.eval_called = False

    def to(self, device):
        self.device_received = device
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, **_kwargs):
        return FakeOutputs()


def import_predict_module(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.device = lambda *_args, **_kwargs: "cpu"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.no_grad = lambda: nullcontext()
    fake_torch.softmax = lambda _logits, dim: [FakeProbabilities([0.2, 0.8])]

    class PlaceholderProcessor:
        @staticmethod
        def from_pretrained(_model_dir):
            raise AssertionError("Processor loader should be patched in the test")

    class PlaceholderModel:
        @staticmethod
        def from_pretrained(_model_dir):
            raise AssertionError("Model loader should be patched in the test")

    fake_transformers = ModuleType("transformers")
    fake_transformers.ViTForImageClassification = PlaceholderModel
    fake_transformers.ViTImageProcessor = PlaceholderProcessor

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("src.inference.predict", None)
    return importlib.import_module("src.inference.predict")


def test_load_model_uses_pretrained_artifacts(monkeypatch):
    predict_module = import_predict_module(monkeypatch)
    model = FakeModel()
    processor = FakeProcessor()

    monkeypatch.setattr(
        predict_module,
        "ensure_model_dir",
        lambda model_dir, repo_id=None: model_dir,
    )

    def fake_processor_from_pretrained(model_dir):
        assert model_dir.replace("\\", "/") == "models/vit_catsdogs"
        return processor

    def fake_model_from_pretrained(model_dir):
        assert model_dir.replace("\\", "/") == "models/vit_catsdogs"
        return model

    monkeypatch.setattr(
        predict_module.ViTImageProcessor,
        "from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        predict_module.ViTForImageClassification,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    loaded_model, loaded_processor = predict_module.load_model()

    assert loaded_model is model
    assert loaded_processor is processor
    assert model.eval_called is True
    assert model.device_received is not None


def test_predict_image_accepts_pil_input(monkeypatch):
    predict_module = import_predict_module(monkeypatch)
    image = Image.new("RGB", (32, 32), color="white")
    processor = FakeProcessor()
    model = FakeModel()

    label, confidence = predict_module.predict_image(image, model, processor)

    assert label == "dog"
    assert confidence == 0.8
