import json
from collections import defaultdict
from glob import glob
from difflib import SequenceMatcher
#nbest_predictions들을 result 파일에 넣습니다.
#비슷한 단어들은 가중치를 더합니다.
#가장 확률이 높은 답을 답변으로 정합니다.
#https://colab.research.google.com/drive/1jkeoeG7atT7kxbrwT_HZLNahWwXyc99t#scrollTo=d12Ilrzwhzr6


# 가중치 변경 가능
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio() > 0.5

# 예측 파일들을 로드합니다.
prediction_files = glob("./results/*.json")
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
NO_ANSWER_THRESHOLD = -5.0  # 적절한 임계값 설정 필요

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
json.dump(converted_data, open("final_predictions.json", "w"), indent=4, ensure_ascii=False)
