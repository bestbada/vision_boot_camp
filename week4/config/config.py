import os
from dotenv import load_dotenv

# 의미: .env 파일을 읽어와 환경변수로 등록함
# 사용 이유: API 키처럼 민감한 정보를 코드에서 분리하기 위함
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "tomatoes-g0kjm"
PROJECT = "tomatoes-detection-ml1e1"
VERSION = 1

def get_weight_path(batch_size):
    # 의미: 배치 크기를 인자로 받아 해당 폴더의 best.pt 경로를 돌려줌
    # 사용 이유: 배치별로 config.py를 매번 손으로 고치지 않기 위함
    return f"weights/batch{batch_size}/best.pt"

DATA_YAML_PATH = "dataset/data.yaml"

INPUT_DATA_DIR = "input_data"
OUTPUT_DATA_DIR = "output"
