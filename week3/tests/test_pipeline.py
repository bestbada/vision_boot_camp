import time
import pytest
from src.evaluate import evaluate_model
from src.infer import find_latest_weight, run_inference
import src.infer as infer_module
from config.config import MODEL_WEIGHT_PATH


# ── evaluate.py 관련 테스트 ──────────────────────────────

def test_evaluate_model_invalid_weight_path():
    # 존재하지 않는 weight 경로를 주면 명확한 예외가 나야 함
    with pytest.raises(Exception):
        evaluate_model("weights/존재하지_않는_파일.pt", "dataset/data.yaml")


# ── infer.py 관련 테스트 ──────────────────────────────────

def test_run_inference_no_input_images(tmp_path, monkeypatch):
    # input_data 폴더에 이미지가 하나도 없을 때 IndexError가 나야 함
    monkeypatch.setattr(infer_module, "INPUT_DATA_DIR", str(tmp_path))

    with pytest.raises(IndexError):
        run_inference()


def test_find_latest_weight_picks_most_recent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    old_dir = tmp_path / "runs/detect/train/weights"
    new_dir = tmp_path / "runs/detect/train-5/weights"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    old_weight = old_dir / "best.pt"
    new_weight = new_dir / "best.pt"
    old_weight.write_text("old")
    time.sleep(0.01)
    new_weight.write_text("new")

    result = find_latest_weight()

    # find_latest_weight()가 반환하는 값은 상대경로이므로,
    # 절대경로(new_weight)와 직접 비교하지 않고 os.path.samefile로
    # "같은 파일을 가리키는지"만 확인

    import os
    assert os.path.samefile(result, new_weight), (
        f"가장 최근 폴더(train-5)를 가리켜야 하는데 {result}가 반환됨"
    )

def test_find_latest_weight_falls_back_when_no_train_dirs(tmp_path, monkeypatch):
    # train* 폴더가 하나도 없으면 기본 경로(MODEL_WEIGHT_PATH)로 대체돼야 함
    monkeypatch.chdir(tmp_path)

    result = find_latest_weight()

    assert result == MODEL_WEIGHT_PATH
