import datetime
import json
import os
from collections import defaultdict
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from PIL import Image
import io
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import warnings

# NOTICE 需要 python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)

dir_name = os.path.dirname(os.path.abspath(__file__))
# NOTICE 下载数据集图像，置于该路径下
image_path = os.path.join(dir_name, "image")
metrics_file = os.path.join(dir_name, "metrics.txt")
word_association = os.path.join(dir_name, "relation.json")
safe_words_file = os.path.join(dir_name, "safe_words.txt")


def init():
    metrics = {}
    with open(metrics_file, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split("=")
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value

    return metrics


def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith("NN")]
    return nouns


def check_synonyms_word(word1, word2, similarity_score=0.8):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score


def amber_doc_to_visual(doc):
    img_name = doc["image"]
    img_file = os.path.join(image_path, img_name)
    with open(img_file, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
    return [img.convert("RGB")]


def amber_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question


def amber_process_results(doc, results):
    response = results[0]

    metrics = init()
    # 避免误差累积
    for metric in metrics.keys():
        metrics[metric] = 0.0

    association = json.load(open(word_association, "r", encoding="utf-8"))
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)

    global_safe_words = []
    with open(safe_words_file, "r", encoding="utf-8") as safe_file:
        for line in safe_file:
            line = line.split("\n")[0]
            global_safe_words.append(line)

    # 默认 evaluation_type == 'a'
    dimension = {"g": True, "de": True, "da": True, "dr": True}

    id = doc["id"]
    if doc["type"] == "generative":
        nouns = extract_nouns(response)
        after_process_nouns = []
        for noun in nouns:
            if noun in hallucination_words:
                after_process_nouns.append(noun)

        safe_words = []
        safe_list = []
        for idx, word in enumerate(doc["answer"]["truth"]):
            safe_words += association[word]
            safe_list += [idx] * len(association[word])

        ha_words = []
        ha_list = []
        for idx, word in enumerate(doc["answer"]["hallu"]):
            ha_words += association[word]
            ha_list += [idx] * len(association[word])

        safe_words += doc["answer"]["truth"]
        safe_len = len(doc["answer"]["truth"])
        safe_list += [0] * safe_len
        safe_flag_list = [0] * len(after_process_nouns)

        ha_words += doc["answer"]["hallu"]
        ha_len = len(doc["answer"]["hallu"])
        ha_list += [0] * ha_len

        for idx, noun in enumerate(after_process_nouns):
            if noun in global_safe_words:
                continue

            if noun in safe_words:
                for j in range(len(safe_words)):
                    if noun == safe_words[j]:
                        if j < (len(safe_list) - safe_len):
                            safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                continue

            if noun in ha_words:
                for j in range(len(ha_words)):
                    if noun == ha_words[j]:
                        if j < (len(ha_list) - ha_len):
                            ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break

            for j, check_word in enumerate(ha_words):
                if check_synonyms_word(noun, check_word, 0.8):
                    if j < (len(ha_list) - ha_len):
                        ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                    else:
                        ha_list[j] = 1
                    break

            flag = False
            for j, check_word in enumerate(safe_words):
                if check_synonyms_word(noun, check_word, 0.8):
                    flag = True
                    if j < (len(safe_list) - safe_len):
                        safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                    else:
                        safe_list[j] = 1
                    break
            if flag == True:
                continue

            safe_flag_list[idx] = 1

        metrics["chair_score"] += sum(safe_flag_list)
        metrics["chair_num"] += len(safe_flag_list)
        metrics["safe_cover_score"] += sum(safe_list[-safe_len:])
        metrics["safe_cover_num"] += len(safe_list[-safe_len:])
        metrics["hallu_cover_score"] += sum(ha_list[-ha_len:])
        metrics["hallu_cover_num"] += len(ha_list[-ha_len:])
        if sum(safe_flag_list) == 0:
            metrics["non_hallu_score"] += 1
        metrics["non_hallu_num"] += 1

    else:
        if "Yes" in response or "yes" in response:
            response = "Yes"
        else:
            response = "No"
        # print(f"response: {response}\n")

        metrics["qa_correct_num"] += 1
        if doc["type"] == "discriminative-attribute-state":
            metrics["as_qa_correct_num"] += 1
        elif doc["type"] == "discriminative-attribute-number":
            metrics["an_qa_correct_num"] += 1
        elif doc["type"] == "discriminative-attribute-action":
            metrics["aa_qa_correct_num"] += 1
        elif doc["type"] == "discriminative-hallucination":
            metrics["ha_qa_correct_num"] += 1
        else:
            metrics["asso_qa_correct_num"] += 1

        truth = doc["answer"]["truth"][0]
        if truth == "yes":
            if response == "Yes":
                metrics["qa_correct_score"] += 1
                if doc["type"] == "discriminative-attribute-state":
                    metrics["as_qa_correct_score"] += 1
                elif doc["type"] == "discriminative-attribute-number":
                    metrics["an_qa_correct_score"] += 1
                elif doc["type"] == "discriminative-attribute-action":
                    metrics["aa_qa_correct_score"] += 1
                elif doc["type"] == "discriminative-hallucination":
                    metrics["ha_qa_correct_score"] += 1
                else:
                    metrics["asso_qa_correct_score"] += 1
        else:
            metrics["qa_no_num"] += 1
            if doc["type"] == "discriminative-attribute-state":
                metrics["as_qa_no_num"] += 1
            elif doc["type"] == "discriminative-attribute-number":
                metrics["an_qa_no_num"] += 1
            elif doc["type"] == "discriminative-attribute-action":
                metrics["aa_qa_no_num"] += 1
            elif doc["type"] == "discriminative-hallucination":
                metrics["ha_qa_no_num"] += 1
            else:
                metrics["asso_qa_no_num"] += 1

            if response == "No":
                metrics["qa_correct_score"] += 1
                metrics["qa_no_score"] += 1
                if doc["type"] == "discriminative-attribute-state":
                    metrics["as_qa_correct_score"] += 1
                    metrics["as_qa_no_score"] += 1
                elif doc["type"] == "discriminative-attribute-number":
                    metrics["an_qa_correct_score"] += 1
                    metrics["an_qa_no_score"] += 1
                elif doc["type"] == "discriminative-attribute-action":
                    metrics["aa_qa_correct_score"] += 1
                    metrics["aa_qa_no_score"] += 1
                elif doc["type"] == "discriminative-hallucination":
                    metrics["ha_qa_correct_score"] += 1
                    metrics["ha_qa_no_score"] += 1
                else:
                    metrics["asso_qa_correct_score"] += 1
                    metrics["asso_qa_no_score"] += 1

        if response == "No":
            metrics["qa_ans_no_num"] += 1
            if doc["type"] == "discriminative-attribute-state":
                metrics["as_qa_ans_no_num"] += 1
            elif doc["type"] == "discriminative-attribute-number":
                metrics["an_qa_ans_no_num"] += 1
            elif doc["type"] == "discriminative-attribute-action":
                metrics["aa_qa_ans_no_num"] += 1
            elif doc["type"] == "discriminative-hallucination":
                metrics["ha_qa_ans_no_num"] += 1
            else:
                metrics["asso_qa_ans_no_num"] += 1
            if truth == "no":
                metrics["qa_ans_no_score"] += 1
                if doc["type"] == "discriminative-attribute-state":
                    metrics["as_qa_ans_no_score"] += 1
                elif doc["type"] == "discriminative-attribute-number":
                    metrics["an_qa_ans_no_score"] += 1
                elif doc["type"] == "discriminative-attribute-action":
                    metrics["aa_qa_ans_no_score"] += 1
                elif doc["type"] == "discriminative-hallucination":
                    metrics["ha_qa_ans_no_score"] += 1
                else:
                    metrics["asso_qa_ans_no_score"] += 1

    return {
        "amber_score": metrics,
        "chair": metrics,
        "cover": metrics,
        "hal": metrics,
        "cog": metrics,
        "accuracy": metrics,
        "precision": metrics,
        "recall": metrics,
        "f1": metrics,
    }


def amber_aggregate_results_amber_score(results):
    """
    amber_score
    """
    metrics = init()

    for result in results:
        metrics["chair_score"] += result["chair_score"]
        metrics["chair_num"] += result["chair_num"]
        metrics["qa_ans_no_score"] += result["qa_ans_no_score"]
        metrics["qa_ans_no_num"] += result["qa_ans_no_num"]
        metrics["qa_no_score"] += result["qa_no_score"]
        metrics["qa_no_num"] += result["qa_no_num"]

    CHAIR = round(metrics["chair_score"] / metrics["chair_num"], 1)

    Precision = round(metrics["qa_ans_no_score"] / metrics["qa_ans_no_num"] * 100, 1)
    Recall = round(metrics["qa_no_score"] / metrics["qa_no_num"] * 100, 1)
    F1 = round(
        2
        * (Precision / 100)
        * (Recall / 100)
        / ((Precision / 100) + (Recall / 100) + 0.0001),
        1,
    )
    # original formula: AMBER_Score = 1.0 / 2 * (1 - CHAIR + F1)
    # for ease of observation: AMBER_Score = 100.0 / 2 * (1 - CHAIR + F1)
    AMBER_Score = 100.0 / 2 * (1 - CHAIR + F1)
    print(f"AMBER_Score: {AMBER_Score}")
    return AMBER_Score


def amber_aggregate_results_chair(results):
    """
    chair
    """
    metrics = init()

    for result in results:
        metrics["chair_score"] += result["chair_score"]
        metrics["chair_num"] += result["chair_num"]

    CHAIR = round(metrics["chair_score"] / metrics["chair_num"] * 100, 1)
    print(f"CHAIR: {CHAIR}")
    return CHAIR


def amber_aggregate_results_cover(results):
    """
    cover
    """
    metrics = init()

    for result in results:
        metrics["safe_cover_score"] += result["safe_cover_score"]
        metrics["safe_cover_num"] += result["safe_cover_num"]

    Cover = round(metrics["safe_cover_score"] / metrics["safe_cover_num"] * 100, 1)
    print(f"Cover: {Cover}")
    return Cover


def amber_aggregate_results_hal(results):
    """
    hal
    """
    metrics = init()

    for result in results:
        metrics["non_hallu_score"] += result["non_hallu_score"]
        metrics["non_hallu_num"] += result["non_hallu_num"]

    Ha_p = round(100 - metrics["non_hallu_score"] / metrics["non_hallu_num"] * 100, 1)
    print(f"Ha_p: {Ha_p}")
    return Ha_p


def amber_aggregate_results_cog(results):
    """
    cog
    """
    metrics = init()

    for result in results:
        metrics["hallu_cover_score"] += result["hallu_cover_score"]
        metrics["hallu_cover_num"] += result["hallu_cover_num"]

    Ha = round(metrics["hallu_cover_score"] / metrics["hallu_cover_num"] * 100, 1)
    print(f"Ha: {Ha}")
    return Ha


def amber_aggregate_results_accuracy(results):
    """
    accuracy
    """
    metrics = init()

    for result in results:
        metrics["qa_correct_score"] += result["qa_correct_score"]
        metrics["qa_correct_num"] += result["qa_correct_num"]

    Accuracy = round(metrics["qa_correct_score"] / metrics["qa_correct_num"] * 100, 1)
    print(f"Accuracy: {Accuracy}")
    return Accuracy


def amber_aggregate_results_precision(results):
    """
    precision
    """
    metrics = init()

    for result in results:
        metrics["qa_ans_no_score"] += result["qa_ans_no_score"]
        metrics["qa_ans_no_num"] += result["qa_ans_no_num"]

    Precision = round(metrics["qa_ans_no_score"] / metrics["qa_ans_no_num"] * 100, 1)
    print(f"Precision: {Precision}")
    return Precision


def amber_aggregate_results_recall(results):
    """
    recall
    """
    metrics = init()

    for result in results:
        metrics["qa_no_score"] += result["qa_no_score"]
        metrics["qa_no_num"] += result["qa_no_num"]

    Recall = round(metrics["qa_no_score"] / metrics["qa_no_num"] * 100, 1)
    print(f"Recall: {Recall}")
    return Recall


def amber_aggregate_results_f1(results):
    """
    f1
    """
    metrics = init()

    for result in results:
        metrics["qa_ans_no_score"] += result["qa_ans_no_score"]
        metrics["qa_ans_no_num"] += result["qa_ans_no_num"]
        metrics["qa_no_score"] += result["qa_no_score"]
        metrics["qa_no_num"] += result["qa_no_num"]

    Precision = round(metrics["qa_ans_no_score"] / metrics["qa_ans_no_num"] * 100, 1)
    Recall = round(metrics["qa_no_score"] / metrics["qa_no_num"] * 100, 1)
    F1 = round(
        2
        * (Precision / 100)
        * (Recall / 100)
        / ((Precision / 100) + (Recall / 100) + 0.0001)
        * 100,
        1,
    )
    print(f"F1: {F1}")
    return F1
