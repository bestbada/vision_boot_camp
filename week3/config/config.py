import os
from dotenv import load_dotenv

# 의미: .env 파일을 읽어와 환경변수로 등록함
# 사용 이유: API 키처럼 민감한 정보를 코드에서 분리하기 위함
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "tomatoes-g0kjm"
PROJECT = "tomatoes-detection-ml1e1"
VERSION = 1

MODEL_WEIGHT_PATH = "weights/best.pt"
DATA_YAML_PATH = "dataset/data.yaml"                    # 추가
TRAIN_RESULTS_CSV = "runs/detect/train-5/results.csv"   # 추가

INPUT_DATA_DIR = "input_data"
OUTPUT_DATA_DIR = "output"
