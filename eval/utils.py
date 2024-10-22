from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from dataset.RefCOCO import RefCOCODataset
import json
import os
from torch.utils.data import DataLoader


def eval_object_classification(dataset, data, config):
    json_results = []
    coco_data = json.load(open(os.path.join(config['DATA_FOLDER'], "COCO2017", "annotations", f"instances_val2017.json")))
    coco_eval_mapping = {c['name']:c['id'] for c in coco_data['categories']}

    class_names = sorted(dataset.class_counts.keys())
    mapping = {class_names[i]:i for i in range(len(class_names))}

    gt = []
    predictions = []
    class_amounts = {}
    class_correct = {}
    mistakes = {}
    total = 0
    correct = 0
    for key in data:
        answer = data[key]["answer"]
        prediction = data[key]["prediction"]
        # if prediction not in mapping:
        #     #print(prediction)
        #     prediction = answer
        bbox = dataset.entries[key]['bbox']
        d = {}
        d["image_id"] = int(dataset.entries[key]['id'])
        d["category_id"] = coco_eval_mapping.get(prediction, 0)
        d['score'] = 0.99
        d['bbox'] = dataset.entries[key]['bbox']

        json_results.append(d)
        gt.append(mapping[answer])
        predictions.append(mapping.get(prediction, 0))
        if answer not in class_amounts:
            class_amounts[answer] = 0
            class_correct[answer] = 0
            mistakes[answer] = {}


        if answer.lower() in  prediction.lower():
            correct += 1
            class_correct[answer] += 1
        else:
            if prediction not in mistakes[answer]:
                mistakes[answer][prediction] = 0
            mistakes[answer][prediction] += 1
        class_amounts[answer] += 1
        total += 1

    out_file = open("./outputs/test.json", "w") 
    json.dump(json_results, out_file) 
    print(f"Performance")
    print(f"Overall Accuracy {correct / total:.3}")
    top_5_frequent_classes = sorted(class_amounts, key = lambda x: class_amounts[x], reverse=True)
    print(f"Top 5 Class Performance")
    for i, c in enumerate(top_5_frequent_classes):
        #percentage = class_amounts[c] / sum(class_amounts.values())
        performance = class_correct[c] / class_amounts[c]
        #most_common_mistake = sorted(mistakes[c].keys(), key = lambda x:mistakes[c][x], reverse=True)[0]
        print(f"{c}: {round(performance, 3)}")
        #print(f"'{c}' class which is {percentage * 100:.3}% of data: {performance:.2}. Most common mistake is {most_common_mistake}") 
    print(f"Overall Accuracy {correct / total:.3}")

def eval_captioning(dataset, data):
    print("Collecting Reference Sentences ...")
    gts = {}
    res = {}
    train_loader = DataLoader(dataset, 1, shuffle=False, num_workers=2, collate_fn=dataset.collate_fn)
    for i, batch in enumerate(train_loader):
        references = batch['refs'][0]
        d = [{"caption": x} for x in references]

        gts[i] = d

        d = [{"caption": x} for x in [data[i]["prediction"]]]
        res[i] = d

    # =================================================
    # Set up scorers
    # =================================================
    print('Tokenization...')
    tokenizer = PTBTokenizer()
    gts  = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # =================================================
    # Set up scorers
    # =================================================
    print('Setting up scorers...')
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]

    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        print('Computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        print(method, score)