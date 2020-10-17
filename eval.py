import six
import string
import re
import collections
def normalize_answer_v1(s):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
  prediction_tokens = normalize_answer_v1(prediction).split()
  ground_truth_tokens = normalize_answer_v1(ground_truth).split()
  common = (
      collections.Counter(prediction_tokens)
      & collections.Counter(ground_truth_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def exact_match_score(prediction, ground_truth):
  return (normalize_answer_v1(prediction) == normalize_answer_v1(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)


def evaluate_v1(dataset, predictions):
  dataset =dataset["data"]
  f1 = exact_match = total = 0
  for article in dataset:
    for paragraph in article["paragraphs"]:
      for qa in paragraph["qas"]:
        total += 1
        if str((qa["id"])) not in predictions:
          message = ("Unanswered question " + six.ensure_str(str(qa["id"])) +
                     "  will receive score 0.")
          #print(message, file=sys.stderr)
          continue
        ground_truths = [x["text"] for x in qa["answers"]]
        # ground_truths = list(map(lambda x: x["text"], qa["answers"]))
        predictionList = predictions[str(qa["id"])]
        prediction = predictionList[0]['text']
        exact_match += metric_max_over_ground_truths(exact_match_score,
                                                     prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

  exact_match = 100.0 * exact_match / total
  f1 = 100.0 * f1 / total

  return {"exact_match": exact_match, "f1": f1}

if __name__ == '__main__':

  dataset = {
              'data': [{'paragraphs': [{'qas': [{
    'question': 'Covid-19 chủ yếu lây qua đường nào ?',
    'id': 27527,
    'answers': [{
        'answer_id': 25239,
        'document_id': 37254,
        'question_id': 27527,
        'text': "Vi-rút g"
            ,
        'answer_start': 0,
        }],
    'is_impossible': False,
    }],
        "context": "Vi-rút gây bệnh COVID-19 chủ yếu lây truyền qua các giọt bắn văng ra khi người nhiễm bệnh ho, hắt hơi hoặc thở ra \
         Những giọt bắn này quá nặng nên không thể bay lơ lửng trong không khí và nhanh chóng rơi xuống sàn nhà hoặc các bề mặt \
Bạn có thể bị nhiễm bệnh khi hít phải vi-rút nếu đang ở gần người nhiễm COVID-19 hoặc chạm vào bề mặt có vi-rút, rồi lại chạm tay vào mắt, mũi hoặc miệng",
          "document_id": 37255}]}]}

  prediction = {
    '27527': [
        {
            "text": "Vi-rút gây bệnh COVID-19",
            "probability": 0.7198556977431992,
            "start_logit": 8.594310760498047,
            "end_logit": -0.35643261671066284
        },
          ]
        }


  print(evaluate_v1(dataset,prediction))