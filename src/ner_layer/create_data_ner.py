# -*- coding: utf-8 -*-

'''
Created on 17 mar. 2022

@author: jose-lopez
'''
from pathlib import Path
import copy
import json
import math
import random
import sys

from about_time import about_time
from alive_progress import alive_bar
from alive_progress import alive_it
from spacy.matcher import Matcher
from spacy.tokens import Span, DocBin
import spacy


def load_jsonl(path):
    data = []
    lines = 0
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            if not line == "\n":
                # lines += 1
                # print(lines)
                data.append(json.loads(line))
    return data


def setting_patterns(patterns, matcher):

    current_line = 1

    labels = ["GOD", "PERSON", "PLACE", "GROUP", "WORK"]

    for name_pattern in patterns:

        if name_pattern["label"] in labels:
            matcher.add(name_pattern["label"], [name_pattern["pattern"]])
            current_line += 1
        else:
            print("There is an error on label value at line: {}".format(
                current_line))
            sys.exit()


def get_arguments(arguments):

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

    return sentences_number, percentage


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


def report_entities(docs, percentage, sentences_number,
                    corpus_length, proportion, report):

    PATH_NER_ENTITIES = "reports/examples_ner_entities.jsonl"

    docs = define_sample(docs, percentage, sentences_number,
                         corpus_length, PATH_NER_ENTITIES, proportion, report)

    file = open(
        PATH_NER_ENTITIES, 'w', encoding="utf8")

    for doc in docs:

        ents_counter = 1
        spans_ = ""
        spans = []

        for ent in doc.ents:
            s_text = doc[0:ent.end].text
            start = len(s_text) - doc[ent.start].__len__()
            end = start + doc[ent.start].__len__()
            spans.append([start, end, ent.label_])

        for span in spans:

            label = span[2].replace("'", '"')
            # print(label)

            """print(
                f'{{"start":{span[0]},"end":{span[1]},"label":"{label}"}}')
            """

            if len(doc.ents) > ents_counter:
                spans_ = spans_ + \
                    f'{{"start":{span[0]},"end":{span[1]},"label":"{label}"}},'
                ents_counter += 1
            else:
                spans_ = spans_ + \
                    f'{{"start":{span[0]},"end":{span[1]},"label":"{label}"}}'

        if not doc.ents:
            spans_ = ""

        file.write(f'{{"text":"{doc.text}", "spans":[{spans_}]}}' + "\n")

    file.close()


def from_corpus(CORPUS_PATH, sentences_number):

    corpus_length = 0

    files_ = [str(x) for x in Path(CORPUS_PATH).glob("**/*.txt")]

    if files_:

        for file_path_ in files_:
            with open(file_path_, 'r', encoding="utf8") as f:
                sentences = list(f.readlines())

            corpus_length += len(sentences)

    else:
        print(f'Not files at {CORPUS_PATH}')
        sys.exit()

    if sentences_number < corpus_length:
        proportion = sentences_number / corpus_length
    else:
        proportion = 1.0

    return corpus_length, files_, proportion


def define_sample(docs, percentage, sentences_number, corpus_length, file_name, proportion, report):

    if report:
        samples = sentences_number
    else:
        samples = len(docs)

    pos_entities = math.ceil(samples * percentage)
    neg_entities = samples - pos_entities

    pos = 0
    neg = 0
    entities_sample = []

    for doc in docs:

        if doc.ents:
            if pos < pos_entities:
                entities_sample.append(doc)
                pos += 1
        elif neg < neg_entities:
            entities_sample.append(doc)
            neg += 1

        if pos + neg == samples:
            break

    if pos < pos_entities:
        print(
            f'File: {file_name} -> The required sentences with entities ({pos_entities}) is greater than the available ({pos}).'
            f'Reporting anyways...')
    elif neg < neg_entities:
        print(
            f'The required sentences without entities ({neg_entities}) is greater than the available ({neg})).'
            f'Reporting anyways...')

    print(f'Number of sampled sentences with entities  {pos} | {samples}')
    print(f'Number of sampled sentences without entities {neg} | {samples}')

    return entities_sample


def getting_ner_examples(files, sentences_number, corpus_length, percentage, proportion, matcher):

    files_counter = 1
    examples = []
    total_with_entities = 0
    total_without_entities = 0

    for file_path in files:

        file_name = file_path.split("/")[2]

        with open(file_path, 'r', encoding="utf8") as fl:
            SENTENCES = [line.strip() for line in fl.readlines()]

        print(
            f'Defining the tagged NER examples for the corpus file -> {file_name}: {files_counter} | {len(files)}')

        print("randomizing.....")
        random.shuffle(SENTENCES)

        sentences_to_tag = math.ceil(len(SENTENCES) * proportion)

        if sentences_to_tag >= 100:
            SENTENCES = SENTENCES[:100]

        print("tagging.....")
        print(f'sentences to tag: {len(SENTENCES)}')
        with_entities, without_entities = tagging_ner_docs(
            SENTENCES, matcher)

        entities = with_entities + without_entities
        random.shuffle(entities)
        entities_sample = define_sample(
            entities, percentage, sentences_number, corpus_length, file_name, proportion, False)
        examples += entities_sample

        total_with_entities += len(with_entities)
        total_without_entities += len(without_entities)

        files_counter += 1

    print(
        f'Total of sampled sentences with entities....: {total_with_entities}')
    print(
        f'Total of sampled sentences without entities....: {total_without_entities}')

    random.shuffle(examples)

    return examples


if __name__ == '__main__':

    CORPUS_PATH = "data/corpus"

    PATTERNS_PATH = "data/patterns2.1.jsonl"
    # PATTERNS_PATH = "data/names_patterns_en.jsonl"

    sentences_number, percentage = get_arguments(sys.argv)

    print("\n" + "\n")
    print(">>>>>>> Starting the entities tagging...........")
    print("\n" + "\n")

    print("Loading the model...")
    nlp = spacy.load("grc_ud_proiel_lg")
    # nlp = spacy.load("en_core_web_sm")
    print(".. done" + "\n")

    print("Loading the entities' patterns...")
    matcher = Matcher(nlp.vocab)
    patterns = load_jsonl(PATTERNS_PATH)
    setting_patterns(patterns, matcher)
    print(".. done" + "\n")

    print("Processing the corpus for NER tagging.......")
    # Total of sentences in the corpus and the proportion of sentences required
    corpus_length, files, proportion = from_corpus(
        CORPUS_PATH, sentences_number)
    # getting the required ner examples
    ner_examples = getting_ner_examples(
        files, sentences_number, corpus_length, percentage, proportion, matcher)
    print(".. done" + "\n")

    print(
        "Reporting the required or available tagged sentences (examples_ner_entities.jsonl)..... ")
    report_entities(ner_examples, percentage,
                    sentences_number, corpus_length, proportion, True)
    print(".. done" + "\n" + "\n")

    print(">>>>>>> Entities tagging finished...........")
