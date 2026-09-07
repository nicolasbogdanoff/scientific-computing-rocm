import pytest

from scientific_computing_rocm import backend_report, choose_device


def test_report_is_serializable_and_conservative():
    report = backend_report()
    assert isinstance(report, dict)
    assert isinstance(report["python_version"], str)
    assert report["backend"] in {"cpu", "cuda", "rocm"}
    assert "device_name" in report
    assert choose_device(prefer_accelerator=False) == "cpu"


def test_required_accelerator_does_not_silently_fallback():
    with pytest.raises(RuntimeError, match="required"):
        choose_device(prefer_accelerator=False, require_accelerator=True)
