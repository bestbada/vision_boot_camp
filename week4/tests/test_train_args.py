from unittest.mock import MagicMock, patch


def _run_train(augment, tmp_path, monkeypatch):
    # 의미: 실제 학습 없이 model.train()에 넘어간 인자만 확인
    # 사용 이유: GPU 학습은 무거우므로 "설정이 올바르게 전달되는가"만 가볍게 검증
    monkeypatch.chdir(tmp_path)
    fake_best = tmp_path / "best.pt"
    fake_best.write_bytes(b"fake")

    mock_model = MagicMock()
    mock_model.trainer.best = fake_best
    with patch("src.train.YOLO", return_value=mock_model):
        from src.train import train_model
        fake_dataset = MagicMock()
        fake_dataset.location = "dataset"
        run_name = train_model(fake_dataset, epochs=1, batch=16, augment=augment)

    _, kwargs = mock_model.train.call_args
    return run_name, kwargs


def test_train_without_augment(tmp_path, monkeypatch):
    run_name, kwargs = _run_train(False, tmp_path, monkeypatch)
    assert run_name == "batch16"
    assert "copy_paste" not in kwargs
    assert "fl_gamma" not in kwargs
    assert (tmp_path / "weights/batch16/best.pt").exists()


def test_train_with_augment(tmp_path, monkeypatch):
    run_name, kwargs = _run_train(True, tmp_path, monkeypatch)
    assert run_name == "batch16_augmented"
    assert kwargs["copy_paste"] == 0.3
    assert kwargs["mixup"] == 0.1
    assert kwargs["cls"] == 1.0
    assert (tmp_path / "weights/batch16_augmented/best.pt").exists()
