import json
from collections import defaultdict
from glob import glob
from difflib import SequenceMatcher

#https://colab.research.google.com/drive/1jkeoeG7atT7kxbrwT_HZLNahWwXyc99t#scrollTo=d12Ilrzwhzr6
"""
1. 파일 준비:
   - 'ensemble/results_soft' 안에 앙상블하고 싶은 모든 'nbest_predictions.json' 파일들을 넣습니다.

2. 유사도 설정:
   - 'similar' 함수에서 유사도 임계값을 조정합니다.
   - 예: SequenceMatcher(None, a, b).ratio() > 0.8
   - 0.8은 유사도 임계값입니다. 이 값을 높이면 더 엄격하게, 낮추면 더 관대하게 답변들을 그룹화합니다.
   - 값의 범위: 0.0 (완전히 다름) ~ 1.0 (완전히 동일)

3. 무응답 임계값 설정:
   - 'NO_ANSWER_THRESHOLD' 값을 설정합니다.
   - 최종 선택된 답변의 확률이 이 임계값보다 낮으면 빈 문자열("")을 반환합니다.
   - 예: NO_ANSWER_THRESHOLD = 0.5
     - 이 경우, 최종 답변의 확률이 50% 미만이면 답변을 비웁니다.

4. 코드 실행:
   - 설정을 마친 후 코드를 실행합니다.
   - 코드는 자동으로 폴더 내의 모든 JSON 파일을 읽어 앙상블을 수행합니다.

5. 결과 확인:
   - 앙상블 결과는 'final_soft_predictions.json' 파일로 저장됩니다.
   - 이 파일에는 각 질문에 대한 최종 답변이 포함되어 있습니다.
"""
# 가중치 변경 가능
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.8

# 예측 파일들을 로드합니다.
prediction_files = glob("./results_soft/*.json")
ensemble_predictions = defaultdict(list)

# 각 prediction 파일을 읽어와서 ensemble_predictions에 추가합니다.
for file_name in prediction_files:
    with open(file_name, "r") as file:
        predictions = json.load(file)
        for key, value in predictions.items():
            if isinstance(value, list):
                ensemble_predictions[key].extend(value)
            else:
                ensemble_predictions[key].append({"text": value, "probability": 1.0})

# Soft voting을 사용하여 각 예측 결과에 대한 가중 평균을 계산합니다.
final_predictions = {}
NO_ANSWER_THRESHOLD = 1.0  # 적절한 임계값 설정 필요

for key, predictions in ensemble_predictions.items():
    grouped_predictions = defaultdict(list)
    
    # 유사한 답변들을 그룹화합니다.
    for pred in predictions:
        if isinstance(pred, dict) and "text" in pred:
            text = pred["text"]
        elif isinstance(pred, str):
            text = pred
        else:
            continue  # 예상치 못한 형식의 예측은 건너뜁니다.
        
        added = False
        for group_key in grouped_predictions:
            if similar(text, group_key):
                grouped_predictions[group_key].append(pred)
                added = True
                break
        if not added:
            grouped_predictions[text].append(pred)
    
    # 그룹별로 점수를 계산합니다.
    group_scores = {}
    for group_key, group_preds in grouped_predictions.items():
        group_scores[group_key] = sum(p.get("probability", 1.0) for p in group_preds)
    
    # 가장 높은 점수를 가진 그룹을 선택합니다.
    if group_scores:
        best_text = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_text]
    else:
        best_text = ""
        best_score = 0
    
    # No-answer 처리
    if best_score < NO_ANSWER_THRESHOLD:
        final_predictions[key] = {"text": "", "score": best_score}
    else:
        final_predictions[key] = {"text": best_text, "score": best_score}

# 결과를 출력 형식에 맞게 변환합니다.
converted_data = {key: value["text"] for key, value in final_predictions.items()}

# 결과를 출력합니다.
print(json.dumps(converted_data, indent=4, ensure_ascii=False))

# 결과를 파일로 저장합니다.
json.dump(converted_data, open("final_soft_predictions.json", "w"), indent=4, ensure_ascii=False)
