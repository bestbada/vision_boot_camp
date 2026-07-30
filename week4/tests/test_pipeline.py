import pytest
from src.evaluate import evaluate_model
from src.infer import run_inference
import src.infer as infer_module
from config.config import get_weight_path
from unittest.mock import MagicMock, patch


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
    

def test_get_weight_path_accepts_full_run_name():
    # 의미: 숫자가 아니라 완전한 실행 이름이 오면 그대로 사용해야 함
    assert get_weight_path("batch16_augmented") == "weights/batch16_augmented/best.pt"
    assert get_weight_path("batch16_fl0.0") == "weights/batch16_fl0.0/best.pt"


# ── infer.py 관련 테스트 ──────────────────────────────────

def test_run_inference_no_input_images(tmp_path, monkeypatch):
    # 의미: input_data 폴더에 이미지가 하나도 없을 때 IndexError가 나야 함
    # 사용 이유: weight_path는 이제 이 시점까지 도달하지 않으므로
    #           실제로 존재하지 않는 더미 경로를 넣어도 무방함
    monkeypatch.setattr(infer_module, "INPUT_DATA_DIR", str(tmp_path))

    with pytest.raises(IndexError):
        run_inference("weights/아무_경로여도_상관없음/best.pt")

def test_run_inference_invalid_weight_path(tmp_path, monkeypatch):
    # 의미: 존재하지 않는 weight 경로를 주면 명확한 예외가 나야 함
    # 사용 이유: run_inference가 이제 weight_path를 파라미터로 받으므로,
    #           이 경로가 잘못됐을 때도 예외 처리가 되는지 확인
    monkeypatch.setattr(infer_module, "INPUT_DATA_DIR", "input_data")

    with pytest.raises(Exception):
        run_inference("weights/존재하지_않는_배치/best.pt")
        
def test_train_model_passes_fl_gamma():
    # 의미: 실제 학습을 돌리지 않고, train_model()이 fl_gamma를
    #      model.train()에 올바른 값으로 전달하는지만 검증
    # 사용 이유: GPU 학습은 무거워서 로컬 유닛 테스트로 매번 돌리기 부적합하므로,
    #           "파라미터 전달이 맞는가"라는 좁은 범위만 가볍게 확인
    mock_model = MagicMock()
    with patch("src.train.YOLO", return_value=mock_model):
        from src.train import train_model
        fake_dataset = MagicMock()
        fake_dataset.location = "dataset"

        train_model(fake_dataset, epochs=30, batch=16, fl_gamma=1.5)

        _, kwargs = mock_model.train.call_args
        assert kwargs["fl_gamma"] == 1.5
        assert kwargs["batch"] == 16
