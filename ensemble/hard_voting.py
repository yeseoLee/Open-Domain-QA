import json
from glob import glob
from collections import Counter, OrderedDict
"""
# 하드 보팅 앙상블 사용 방법

1. 파일 준비:
   - 'ensemble/results_hard' 폴더 안에 앙상블하고 싶은 모든 'predictions.json' 파일들을 넣습니다.

2. 우선순위 설정:
   - 'priority_order' 리스트에 모델의 우선순위를 정의합니다.
   - 예: priority_order = ['predictions1.json', 'predictions2.json', 'predictions3.json']
   - 리스트의 앞쪽에 있는 모델일수록 높은 우선순위를 가집니다.

3. 코드 실행:
   - 설정을 마친 후 코드를 실행합니다.
   - 코드는 자동으로 폴더 내의 모든 JSON 파일을 읽어 앙상블을 수행합니다.

4. 하드 보팅 과정:
   - 각 질문에 대해 모든 모델의 답변을 수집합니다.
   - 가장 많이 나온 답변(들)을 선택합니다.
   - 동점인 경우, 우선순위가 가장 높은 모델의 답변을 선택합니다.

5. 결과 확인:
   - 앙상블 결과는 'final_hard_predictions.json' 파일로 저장됩니다.
   - 이 파일에는 각 질문에 대한 최종 답변이 포함되어 있습니다.

주의: 모델의 우선순위는 각 모델의 성능이나 특성을 고려하여 신중히 결정해야 합니다. 
우선순위 설정에 따라 최종 결과가 크게 달라질 수 있습니다.
"""
# 우선순위 직접 정의
priority_order = ['predictions(1).json', 'predictions(2).json', 'predictions(3).json']

def hard_voting_with_priority(predictions, priority_order):
    result = {}
    for question_id in predictions[0].keys():
        answers = [pred[question_id] for pred in predictions if question_id in pred]
        answer_counts = Counter(answer for answer in answers if answer)
        
        if answer_counts:
            # 가장 많이 나온 답변들을 찾습니다
            max_count = max(answer_counts.values())
            top_answers = [ans for ans, count in answer_counts.items() if count == max_count]
            
            if len(top_answers) == 1:
                # 하나의 답변이 가장 많다면 그것을 선택
                result[question_id] = top_answers[0]
            else:
                # 동점이라면 우선순위가 높은 모델의 답변을 선택
                for model in priority_order:
                    model_index = next((i for i, pred in enumerate(predictions) if pred.get('filename') == model), None)
                    if model_index is not None:
                        model_answer = predictions[model_index].get(question_id, "")
                        if model_answer in top_answers:
                            result[question_id] = model_answer
                            break
                else:
                    # 우선순위 모델에서도 답을 찾지 못했다면 첫 번째 top answer 선택
                    result[question_id] = top_answers[0]
        else:
            result[question_id] = ""
    return result

# 예측 파일들을 로드합니다.
prediction_files = glob("./results_hard/*.json")
predictions = []

# 각 prediction 파일을 읽어와서 predictions 리스트에 추가합니다.
for file_name in prediction_files:
    with open(file_name, "r", encoding='utf-8') as file:
        prediction = json.load(file)
        predictions.append(prediction)

# 하드 보팅을 수행합니다.
final_predictions = hard_voting_with_priority(predictions, priority_order)

# 결과를 출력합니다.
print(json.dumps(final_predictions, indent=4, ensure_ascii=False))

# 결과를 파일로 저장합니다.
with open("final_hard_predictions.json", "w", encoding='utf-8') as f:
    json.dump(final_predictions, f, indent=4, ensure_ascii=False)
