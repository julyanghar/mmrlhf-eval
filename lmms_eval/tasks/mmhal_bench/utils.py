import datetime
import json
import os
from collections import defaultdict
import requests
import time
from loguru import logger as eval_logger


API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def get_eval(content: str = "xxx", max_tokens: int = 2048, retries: int = 5):
    global headers

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the quality of the answer.",
        },
        {"role": "user", "content": content},
    ]
    import json

    payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

    # if API_TYPE == "azure":
    #     payload.pop("model")

    for attempt in range(retries):
        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', json.dumps(payload), headers=headers)
            response.raise_for_status()
            response_data = response.json()

            content = response_data["choices"][0]["message"]["content"].strip()
            if content != "":
                return content, response_data["model"]
            break  # If successful, break out of the loop

        except Exception as e:
            eval_logger.info(f"Attempt {attempt + 1} failed with error: {e}")
            if (
                attempt < retries
            ):  # If we have retries left, sleep and then continue to next attempt
                time.sleep(0.1)
            else:  # If this was the last attempt, log and return empty
                eval_logger.error(
                    f"All {retries} attempts failed. Last error message: {e}"
                )
                return "", ""
    return "", ""


template = """Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.
For clarity, consider these examples:
### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.
### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.
### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.
### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.
### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.
With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.
Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.
To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination
### Image Contents
{}
### Question
{}
### Standard Human-Generated Answer
{}
### LMM Response to Evaluate
{}
"""


def mmhal_bench_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmhal_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question


def mmhal_bench_process_results(doc, results):
    pred = results[0]
    image_content = ", ".join(doc["image_content"])
    input_text = template.format(
        image_content,
        doc["question"],
        doc["gt_answer"],
        pred,
    )
    response, model = get_eval(input_text, 1024)
    score = -1  # Final Score
    scores_found = []
    for s in range(7):
        if f"rating: {s}" in response.lower():
            scores_found.append(s)
    if len(scores_found) == 1:
        score = scores_found[0]
    else:
        score = 0
    dic = {"question_type": doc["question_type"], "score": score}
    return {
        "mmhal_bench_average_score": dic,
        "hallucination_rate": dic,
        "average_score_for_attribute": dic,
        "average_score_for_adversarial": dic,
        "average_score_for_comparison": dic,
        "average_score_for_counting": dic,
        "average_score_for_relation": dic,
        "average_score_for_environment": dic,
        "average_score_for_holistic": dic,
        "average_score_for_other": dic,
    }


def mmhal_bench_average_score(results):
    return sum([r["score"] for r in results]) / len(results)


def mmhal_bench_hallucination_rate(results):
    hallucination = []
    for result in results:
        if result["score"] >= 3:
            hallucination.append(0)
        else:
            hallucination.append(1)
    return sum(hallucination) / len(hallucination)


def mmhal_bench_average_score_for_each_type(results):
    scores = defaultdict(list)
    for result in results:
        scores[result["question_type"]].append(result["score"])
    return {k: sum(v) / len(v) for k, v in scores.items()}


def mmhal_bench_average_score_for_attribute(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["attribute"]


def mmhal_bench_average_score_for_adversarial(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["adversarial"]


def mmhal_bench_average_score_for_comparison(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["comparison"]


def mmhal_bench_average_score_for_counting(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["counting"]


def mmhal_bench_average_score_for_relation(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["relation"]


def mmhal_bench_average_score_for_environment(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["environment"]


def mmhal_bench_average_score_for_holistic(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["holistic"]


def mmhal_bench_average_score_for_other(results):
    scores_for_each_type = mmhal_bench_average_score_for_each_type(results)
    return scores_for_each_type["other"]
