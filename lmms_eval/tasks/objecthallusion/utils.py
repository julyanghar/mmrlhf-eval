import datetime
import json
import os
from collections import defaultdict
import nltk  # pip install nltk==3.8.1
import spacy  # pip install spacy==3.7.0
from nltk.stem import *
from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

# nlp = spacy.load("en_core_web_trf") nltk.download('punkt_tab')
lemma = nltk.wordnet.WordNetLemmatizer()


class CHAIR:
    def __init__(self):
        # read in synonyms
        synonyms = open(os.path.join(dir_name, "synonyms_refine.txt")).readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            new_synonym = [s.strip() for s in synonym]
            self.mscoco_objects.extend(new_synonym)
            for s in new_synonym:
                self.inverse_synonym_dict[s] = new_synonym[0]

        coco_double_words = [word for word in self.inverse_synonym_dict.keys() if len(word.strip().split(" ")) >= 2]
        coco_double_words += ["home plate", "train track"]
        # print("double word count:", len(coco_double_words))

        animal_words = [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]
        vehicle_words = ["jet", "train"]

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict["baby %s" % animal_word] = animal_word
            self.double_word_dict["adult %s" % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict["passenger %s" % vehicle_word] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption):
        """
        Input: caption
        Output: MSCOCO words in the caption
        """
        words = nltk.word_tokenize(caption.lower())
        words_2 = [lemma.lemmatize(w) for w in words]
        words = words_2

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append([word, self.inverse_synonym_dict[word]])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def compute_chair(self, caption, objects):
        """
        compute chair without GPT
        """
        cap = caption
        gt_objects = objects

        num_coco_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0
        
        # get all words in the caption, as well as corresponding node word
        words, node_words, idxs, raw_words = self.caption_to_words(cap)

        # count hallucinated words, if [word, coco_obj_cls] is unique, count as one prediction
        coco_word_count = len(node_words)

        hallucinated = False
        for word, node_word, idx in zip(words, node_words, idxs):
                if node_word[-1] not in gt_objects:
                    hallucinated_word_count += 1
                    hallucinated = True

        # count hallucinated caps
        if hallucinated:
            num_hallucinated_caps += 1
            
        if len(words) > 0:
            num_coco_caps += 1
            
        res = {
            "num_hallucinated_caps": num_hallucinated_caps,
            "num_caps": 1,
            "num_coco_caps": num_coco_caps,
            "hallucinated_word_count": hallucinated_word_count,
            "coco_word_count": coco_word_count
        }
        
        return res


evaluator = CHAIR()


def hallucination_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def hallucination_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    return question


def hallucination_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    objects = doc["objects"]  # groundtruth
    caption = results[0]  # response from model
    
    # print(f'caption: {caption}')

    res = evaluator.compute_chair(caption, objects)
    
    output = {"CHAIR_i_score": res,"CHAIR_s_score":res}
    
    return output


def hallucination_get_CHAIR_i_score(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    hallucinated_word_count = 0.0
    coco_word_count = 0.0
    
    for result in results:
        hallucinated_word_count += result['hallucinated_word_count']
        coco_word_count += result['coco_word_count']

    CHAIR_i = hallucinated_word_count / coco_word_count
    
    # print(f"hallucinated_word_count: {hallucinated_word_count}")
    # print(f"coco_word_count: {coco_word_count}")
    
    return CHAIR_i


def hallucination_get_CHAIR_s_score(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    num_hallucinated_caps = 0.0
    num_caps = 0.0
    num_coco_caps = 0.0
    
    for result in results:
        num_hallucinated_caps += result['num_hallucinated_caps']
        num_caps += result['num_caps']
        num_coco_caps += result['num_coco_caps']
    
    CHAIR_s = num_hallucinated_caps / num_caps
    CHAIR_s_refine = num_hallucinated_caps / num_coco_caps
    
    # print(f"num_hallucinated_caps: {num_hallucinated_caps}")
    
    return CHAIR_s
