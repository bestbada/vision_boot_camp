import pytest
from src.evaluate import evaluate_model
from src.infer import run_inference
import src.infer as infer_module
from config.config import get_weight_path


# ── evaluate.py 관련 테스트 ──────────────────────────────

def test_evaluate_model_invalid_weight_path():
    with pytest.raises(Exception):
        evaluate_model("weights/존재하지_않는_파일.pt", "dataset/data.yaml")


def test_get_weight_path_builds_correct_path():
    # 의미: 배치 번호를 넣으면 올바른 경로 문자열을 만들어내는지 확인
    # 사용 이유: config.py의 get_weight_path()가 오타 없이 일관된 규칙으로
    #           경로를 생성하는지 검증하기 위함
    assert get_weight_path(16) == "weights/batch16/best.pt"
    assert get_weight_path(32) == "weights/batch32/best.pt"


# ── infer.py 관련 테스트 ──────────────────────────────────

def test_run_inference_no_input_images(tmp_path, monkeypatch):
    # 의미: input_data 폴더에 이미지가 하나도 없을 때 IndexError가 나야 함
    monkeypatch.setattr(infer_module, "INPUT_DATA_DIR", str(tmp_path))

    with pytest.raises(IndexError):
        run_inference("weights/batch16/best.pt")


def test_run_inference_invalid_weight_path(tmp_path, monkeypatch):
    # 의미: 존재하지 않는 weight 경로를 주면 명확한 예외가 나야 함
    # 사용 이유: run_inference가 이제 weight_path를 파라미터로 받으므로,
    #           이 경로가 잘못됐을 때도 예외 처리가 되는지 확인
    monkeypatch.setattr(infer_module, "INPUT_DATA_DIR", "input_data")

    with pytest.raises(Exception):
        run_inference("weights/존재하지_않는_배치/best.pt")
