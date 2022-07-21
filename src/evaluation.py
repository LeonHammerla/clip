import csv
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from multilingual_clip import pt_multilingual_clip
import transformers
import torch
from bert_encoder import BertCLSEncoder


def calculate_f_scores(rooms: List[str],
                       objects: List[str],
                       data_cos_sim: Dict[str, Dict[str, float]],
                       data: Dict[str, Dict[str, float]]) -> Tuple[float, float, float, List[float]]:

    """
    Function for calculating all different kinds of f1-scores.
    :param rooms:
    :param objects:
    :param data_cos_sim:
    :param data:
    :return:
    """
    def zd(a, b):
        """
        Helper for enabling zero divisions.
        :param a:
        :param b:
        :return:
        """
        try:
            return a / b
        except:
            return 0
    f1_class_scores = []
    weights = []
    global_tp = 0
    global_fp = 0
    for i, room in enumerate(rooms):
        tp, fp, fn, tn = 0, 0, 0, 0
        for j, object_id in enumerate(objects):
            if data_cos_sim[object_id] == i:
                if data[object_id] == i:
                    tp += 1
                    global_tp += 1
                else:
                    fp += 1
                    global_fp += 1
            else:
                if data[object_id] == i:
                    fn += 1
                else:
                    tn += 1
        precision = zd(tp, (tp + fp))
        recall = zd(tp, (tp + fn))
        f1score = zd((2 * precision * recall), (precision + recall))
        f1_class_scores.append(f1score)
        weights.append(tp + fn)

    # macro-averaged F1-score
    macro_f1 = zd(sum(f1_class_scores), len(f1_class_scores))
    # weighted-average F1-score
    weighted_f1 = zd(sum([weights[i] * f1_class_scores[i] for i in range(0, len(f1_class_scores))]), sum(weights))
    # micro-averaged F1-score
    micro_f1 = zd(global_tp, (global_tp + global_fp))

    return macro_f1, weighted_f1, micro_f1, f1_class_scores


def print_result(rooms: List[str],
                 macro_f1: float,
                 weighted_f1: float,
                 micro_f1: float,
                 f1_class_scores: List[float],
                 eval_id: str):
    """
    Print results to console.
    :param eval_id:
    :param f1_class_scores:
    :param rooms:
    :param macro_f1:
    :param weighted_f1:
    :param micro_f1:
    :return:
    """
    print(28 * "=")
    print(f"For this test {eval_id} was used!")
    print(28 * "=")
    print("Class-Specific-F1-Score:")
    print(" ")
    for i, room in enumerate(rooms):
        print(f"{room}: {f1_class_scores[i]}")
    print(28 * "=")
    print("macro-averaged F1-score:")
    print(" ")
    print(macro_f1)
    print(28 * "=")
    print("weighted-average F1-score:")
    print(" ")
    print(weighted_f1)
    print(28 * "=")
    print("micro-averaged F1-score:")
    print(" ")
    print(micro_f1)
    print(28 * "=")


def read_data() -> Dict[str, Dict[str, float]]:
    """
    Function to read in data.
    Returns this data in format of a dictionary where every
    key is an item and every value is another dict containing (key, value)
    pairs of romm-id and probability for this room to contain the item.
    :return:
    """
    data = []
    with open('data.csv', newline='\n') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            data.append(row[0].split("\t"))

    i = 1
    for row in data[1:]:
        total_rooms = int(row[1])
        j = 2
        for line in row[2:]:
            data[i][j] = int(line) / total_rooms
            j += 1
        i += 1

    #print(data[1])

    i = 2
    final_data = dict()
    for line in data[0][2:]:
        final_data[line] = dict()
        for row in data[1:]:
            final_data[line][row[0]] = row[i]
        i += 1

    return final_data


def get_all_strings(data: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str]]:
    """
    Function to collect all distinct strings in the data.
    :param data:
    :return:
    """
    objects = list(data.keys())
    rooms = list(data[list(data.keys())[0]].keys())
    return objects, rooms


def construct_template_object(word: str, room: str) -> str:
    """
    Function constructs template for an object.
    :param word:
    :param room:
    :return:
    """
    vocals = ["a", "e", "i", "o", "u"]
    if word[0] in vocals:
        return f"An {word} is usually in the {room}"
    else:
        return f"A {word} is usually in the {room}"


def construct_template_room(room: str) -> str:
    """
    Function constructs a template for a room.
    :param room:
    :return:
    """
    return f"This is usually in the {room}"


def calculate_room_for_object(object_dict: Dict[str, float]) -> int:
    return max(range(len(list(object_dict.values()))), key=list(object_dict.values()).__getitem__)


def eval_clip(use_template_sentences: bool) -> None:
    """
    Function for calculating f-score of zero shot-classification of
    object-room relations. For this Evaluation CLIP is used.
    :return:
    """
    # read in data
    data = read_data()
    # load model and tokenizer (+modelname)
    model_name = 'M-CLIP/XLM-Roberta-Large-Vit-L-14'
    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # cos_similarity-function for torch tensors
    cos = torch.nn.CosineSimilarity(dim=0)
    # get all (strings) rooms and objects in data
    objects, rooms = get_all_strings(data)

    # new_data based on cos similarity
    data_cos_sim = dict()


    if use_template_sentences:
        # getting template embeddings for every room
        room_sent_batch = [construct_template_room(room) for room in rooms]
        room_embeddings = model.forward(room_sent_batch, tokenizer)
    else:
        room_embeddings = model.forward(rooms, tokenizer)

    # Main-loop for calculating embeddings of all
    i = 0
    for object_id in tqdm(objects, desc="Calculating Embeddings"):
        sent_batch = [] # room/object
        data_cos_sim[object_id] = dict()

        if use_template_sentences:
            for j, room_id in enumerate(rooms):
                sent_batch.append(construct_template_object(word=object_id, room=room_id))
            embeddings = model.forward(sent_batch, tokenizer) # [num_rooms, model_size: 768]
            for j, room_id in enumerate(rooms):
                data_cos_sim[object_id][room_id] = float(cos(embeddings[j], room_embeddings[j]))

        else:
            embeddings = model.forward([object_id], tokenizer)
            for j, room_id in enumerate(rooms):
                data_cos_sim[object_id][room_id] = float(cos(embeddings[0], room_embeddings[j]))

        del embeddings
        i += 1

    for object_id in tqdm(objects, desc="Calculating Labels"):
        data[object_id] = calculate_room_for_object(data[object_id])
        data_cos_sim[object_id] = calculate_room_for_object(data_cos_sim[object_id])

    macro_f1, weighted_f1, micro_f1, f1_class_scores = calculate_f_scores(rooms, objects, data_cos_sim, data)

    print_result(rooms, macro_f1, weighted_f1, micro_f1, f1_class_scores, "CLIP")


def eval_bert(use_template_sentences: bool) -> None:
    """
    Function for calculating f-score of zero shot-classification of
    object-room relations. For this Evaluation BERT is used.
    :return:
    """
    # read in data
    data = read_data()
    # load model and tokenizer (+modelname)
    enc = BertCLSEncoder()
    # cos_similarity-function for torch tensors
    cos = torch.nn.CosineSimilarity(dim=0)
    # get all (strings) rooms and objects in data
    objects, rooms = get_all_strings(data)

    # new_data based on cos similarity
    data_cos_sim = dict()


    if use_template_sentences:
        # getting template embeddings for every room
        room_sent_batch = [construct_template_room(room) for room in rooms]
        room_embeddings = enc.documents_to_vecs(room_sent_batch)
    else:
        room_embeddings = enc.documents_to_vecs(rooms)

    # Main-loop for calculating embeddings of all
    i = 0
    for object_id in tqdm(objects, desc="Calculating Embeddings"):
        sent_batch = []  # room/object
        data_cos_sim[object_id] = dict()

        if use_template_sentences:
            for j, room_id in enumerate(rooms):
                sent_batch.append(construct_template_object(word=object_id, room=room_id))
            embeddings = enc.documents_to_vecs(sent_batch)  # [num_rooms, model_size: 1024]
            for j, room_id in enumerate(rooms):
                data_cos_sim[object_id][room_id] = float(cos(embeddings[j], room_embeddings[j]))

        else:
            embeddings = enc.documents_to_vecs([object_id])
            for j, room_id in enumerate(rooms):
                data_cos_sim[object_id][room_id] = float(cos(embeddings[0], room_embeddings[j]))

        del embeddings
        i += 1

    for object_id in tqdm(objects, desc="Calculating Labels"):
        data[object_id] = calculate_room_for_object(data[object_id])
        data_cos_sim[object_id] = calculate_room_for_object(data_cos_sim[object_id])

    macro_f1, weighted_f1, micro_f1, f1_class_scores = calculate_f_scores(rooms, objects, data_cos_sim, data)


    print_result(rooms, macro_f1, weighted_f1, micro_f1, f1_class_scores, "BERT")


if __name__ == "__main__":
    print("without Templates")

    eval_bert(use_template_sentences=False)
    eval_clip(use_template_sentences=False)

    print("With Templates")

    eval_bert(use_template_sentences=True)
    eval_clip(use_template_sentences=True)
