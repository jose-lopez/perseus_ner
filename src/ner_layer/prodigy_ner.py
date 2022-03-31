'''
Created on 25 mar. 2022

@author: jose-lopez
'''

import json
import sys

import spacy


def load_jsonl(path):
    data = []
    lines = 0
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            lines += 1
            print(lines)
            data.append(json.loads(line))
    return data


if __name__ == '__main__':

    path = sys.argv[1].split("=")[1]
    prodigy_annotations = load_jsonl(path)
    examples = ((eg["text"], eg) for eg in prodigy_annotations)
    nlp = spacy.load("grc_ud_perseus_lge")
    for doc, eg in nlp.pipe(examples, as_tuples=True):
        doc.ents = [doc.char_span(s["start"], s["end"], s["label"])
                    for s in eg["spans"]]
        iob_tags = [
            f'{t.ent_iob_}-{t.ent_type_}' if t.ent_iob_ else "O" for t in doc]
        for w in doc:
            print(f'{w.text}\t{w.ent_type_}\t{w.ent_iob_}')
