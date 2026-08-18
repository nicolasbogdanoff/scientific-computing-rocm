from scientific_computing_rocm import backend_report, choose_device


def test_report_is_serializable_and_conservative():
    report = backend_report()
    assert isinstance(report, dict)
    assert report["backend"] in {"cpu", "cuda", "rocm"}
    assert choose_device(prefer_accelerator=False) == "cpu"
