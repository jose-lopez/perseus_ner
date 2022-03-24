# -*- coding: utf-8 -*-

'''
Created on 17 mar. 2022

@author: jose-lopez
'''
import json
import math
import random
import sys
import time

from about_time import about_time
from alive_progress import alive_bar
from alive_progress import alive_it
from spacy.matcher import Matcher
from spacy.tokens import Span, DocBin
import spacy


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            data.append(json.loads(line))
    return data


def setting_patterns(patterns, matcher):

    current_line = 1

    labels = ["GOD", "PERSON", "PLACE", "GROUP"]

    for name_pattern in patterns:
        if name_pattern["label"] in labels:
            matcher.add(name_pattern["label"], [name_pattern["pattern"]])
            current_line += 1
        else:
            print("There is an error on label value at line: {}".format(
                current_line))
            sys.exit()


def report_entities_json(documents, arguments, with_entities):

    if with_entities:
        path = "reports/examples_ner_pos.json"
    else:
        path = "reports/examples_ner_empthy.json"

    for argument in arguments[1:]:
        name = argument.split("=")[0]
        value = argument.split("=")[1]

        if name == "--number_of_sentences" and value.isdecimal:
            sentences_number = int(value)
        elif name == "--percentage" and value.isdecimal:
            percentage = int(value) / 100
        else:
            print("Please check the argument at the line of commands")
            print("the syntax is: --number_of_sentences=<value>")
            sys.exit()

    if len(documents) > 0:

        required_sentences = math.floor(sentences_number * percentage)

        if with_entities:

            if required_sentences >= len(documents):
                print(
                    f'The required sentences with entities required ({required_sentences}) is greater than the availables ({len(documents)}). '
                    f'Reporting anyways...')

        else:
            required_sentences = sentences_number - required_sentences

            if required_sentences >= len(documents):
                print(
                    f'The required sentences without entities required ({required_sentences}) is greater than the availables ({len(documents)}). '
                    f'Reporting anyways...')

        num_of_sentences = 1

        file = open(
            path, 'w', encoding="utf8")

        file.write("[" + "\n")

        docs = []

        for doc in documents:

            docs.append(doc)

            spans = []

            for ent in doc.ents:

                spans.append([ent.start, ent.end, ent.label_])

            spans_string = f'{spans}'.replace("'", '"')

            example = f'["{doc.text}", {{ "entities": {spans_string} }}]'
            # print(example)

            if num_of_sentences < required_sentences and num_of_sentences < len(documents):
                file.write("    " + example + "," + "\n")
                num_of_sentences += 1
            else:
                file.write("    " + example + "\n")
                break

        file.write("]")

        file.close()

        return docs

    else:
        if with_entities:
            print("There isn't any positive NER examples to print ")
        else:
            print("There isn't any empthy NER examples to print ")

        sys.exit()


def token_from_span_in(spans, current_span):

    already_present = False

    current_span_tokens = [t.i for t in current_span]

    for span in spans:

        span_tokens = [t.i for t in span]

        for token_index in span_tokens:
            if token_index in current_span_tokens:
                already_present = True

        if already_present is True:
            break

    return already_present


def tagging_ner_docs(sentences, matcher):

    with_entities = []
    without_entities = []

    for doc in nlp.pipe(sentences):

        matches = matcher(doc)
        doc.ents = []
        spans = []
        """print([(doc[start:end], nlp.vocab.strings[match_id])
               for match_id, start, end in matches])"""
        for match_id, start, end in matches:
            current_span = Span(
                doc, start, end, label=nlp.vocab.strings[match_id])
            if not token_from_span_in(spans, current_span):
                spans.append(current_span)

        doc.ents = spans
        # print([(ent.text, ent.label_) for ent in doc.ents])

        if matches:
            with_entities.append(doc)
        else:
            without_entities.append(doc)

    return (with_entities, without_entities)


if __name__ == '__main__':

    # CORPUS_PATH = "data/text.txt"
    CORPUS_PATH = "data/corpus_sm.txt"
    NAMES_PATTERNS_PATH = "data/names_patterns.jsonl"
    GROUP_PATTERN_PATH = "data/group_pattern.jsonl"

    print("\n" + "\n")
    print("Loading the corpus...")
    with open(CORPUS_PATH, 'r', encoding="utf8") as f:
        SENTENCES = [line.strip() for line in alive_it(f.readlines())]
    print(".. done" + "\n")

    with about_time() as t_total:
        with about_time() as t1:
            print("Loading the model...")
            nlp = spacy.load("grc_ud_perseus_lge")
            # nlp = spacy.load("en_core_web_md")
            print(".. done" + "\n")

            # print(nlp.pipe_names)
            # print(nlp.pipeline)
        with about_time() as t2:
            print("Loading the entities' patterns...")
            matcher = Matcher(nlp.vocab)
            names_patterns = load_jsonl(NAMES_PATTERNS_PATH)
            group_patterns = load_jsonl(GROUP_PATTERN_PATH)
            setting_patterns(names_patterns, matcher)
            setting_patterns(group_patterns, matcher)
            print(".. done" + "\n")

        with about_time() as t3:
            print("Creating the tagged NER examples from the corpus ")
            with_entities, without_entities = tagging_ner_docs(
                SENTENCES, matcher)
            DOCS_SIZE = len(with_entities) + len(without_entities)
            print(f'Corpus size: {DOCS_SIZE}')
            print(
                f'Numer of docs with entities: {len(with_entities)}:{DOCS_SIZE}')
            print(
                f'Numer of docs without entities: {len(without_entities)}:{DOCS_SIZE}')
            print(".. done" + "\n")

        with about_time() as t4:
            print("Ramdomizing the NER examples ....")
            random.shuffle(with_entities)
            random.shuffle(without_entities)
            print(".. done" + "\n")

        with about_time() as t5:
            print(
                "Reporting the NER examples files for manual evaluation (examples_ner_pos.json, examples_ner_empthy.json) ....")
            pos_docs = report_entities_json(with_entities, sys.argv, True)
            empthy_docs = report_entities_json(
                without_entities, sys.argv, False)
            print(
                f'Reported NER examples with entities: {len(pos_docs)}:{len(with_entities)}')
            print(
                f'Reported empthy NER examples for evaluation: {len(empthy_docs)}:{len(without_entities)}')
            print(".. done" + "\n")

        with about_time() as t6:
            print(
                "Creating the files for the NER's layer training (train.spacy) and evaluation (eval.spacy)")
            # Create and save a collection of training docs
            all_ner_examples = pos_docs + empthy_docs
            random.shuffle(all_ner_examples)
            random.shuffle(all_ner_examples)
            random.shuffle(all_ner_examples)

            train_docbin = DocBin(docs=all_ner_examples[:len(pos_docs)])
            train_docbin.to_disk("./data/train.spacy")

            # Create and save a collection of evaluation docs
            eval_docbin = DocBin(docs=all_ner_examples[len(pos_docs):])
            eval_docbin.to_disk("./data/eval.spacy")

            print(".. done" + "\n")

    print(f'percentage1 = {t1.duration / t_total.duration}')
    print(f'percentage2 = {t2.duration / t_total.duration}')
    print(f'percentage3 = {t3.duration / t_total.duration}')
    print(f'percentage4 = {t4.duration / t_total.duration}')
    print(f'percentage5 = {t3.duration / t_total.duration}')
    print(f'percentage6 = {t4.duration / t_total.duration}')
