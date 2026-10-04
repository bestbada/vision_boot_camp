import os
from dotenv import load_dotenv

# 의미: .env 파일을 읽어와 환경변수로 등록함
# 사용 이유: API 키처럼 민감한 정보를 코드에서 분리하기 위함
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "tomatoes-g0kjm"
PROJECT = "tomatoes-detection-ml1e1"
VERSION = 1

def get_weight_path(run_name):
    # 의미: "16"처럼 숫자만 오면 "batch16"으로 자동 변환하고,
    #      "batch16_augmented"처럼 이미 완전한 이름이 오면 그대로 사용
    # 사용 이유: 배치 실험(batch8/16/32)과 이름이 붙은 실험(batch16_augmented 등)을
    #           하나의 함수로 동시에 지원하기 위함
    run_name = str(run_name)
    if run_name.isdigit():
        run_name = f"batch{run_name}"
    return f"weights/{run_name}/best.pt"
    
def normalize_run_name(run_name):
    run_name = str(run_name)
    if run_name.isdigit():
        return f"batch{run_name}"
    return run_name

def get_weight_path(run_name):
    return f"weights/{normalize_run_name(run_name)}/best.pt"

DATA_YAML_PATH = "dataset/data.yaml"

INPUT_DATA_DIR = "input_data"
OUTPUT_DATA_DIR = "output"

# 의미: 클래스별로 다른 confidence 임계값을 중앙에서 관리
# 사용 이유: Colab에서 확인한 값을 로컬 코드에도 그대로 반영하되,
#           하드코딩 대신 한 곳에서 관리해 나중에 값 조정이 쉽도록 함
CLASS_CONF_THRESHOLDS = {
    "tomato_half_ripe": 0.25,
    "tomato_ripe": 0.4,
    "tomato_unripe": 0.35,
}
DEFAULT_CONF = 0.5
