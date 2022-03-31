'''
Created on 28 mar. 2022

@author: jose-lopez
'''

import json
import math
import random
import sys
import time


def load_jsonl(path):
    data = []
    lines = 0
    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            lines += 1
            print(lines)
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


def report_entities(entities):

    path_entities = "reports/examples_ner_entities.jsonl"

    file = open(
        path_entities, 'w', encoding="utf8")

    for entity in entities:

        text = entity[0]
        entity_spans = entity[1]["entities"]

        spans = []

        for span in entity_spans:
            span_dict = f'"start":{span[0]},"end":{span[1]},"label":{span[2]}]'
            spans.append(span_dict)

        example = f'{"text":{text},"spans":{spans}}'

        file.write(example + "\n")

    file.close()
