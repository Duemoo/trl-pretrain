import spacy

def ner_spacy(text: str) -> list:
    output = []
    # Faster and smaller pipeline, but less accurate
    # to use "en_core_web_sm", you should download the model first typing "python -m spacy download en_core_web_sm" in terminal
    # ner_model = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    # Larger and slower pipeline, but more accurate
    # to use "en_core_web_trf", you should download the model first typing "python -m spacy download en_core_web_trf" in terminal
    ner_model = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    ner_output = ner_model(text)
    # print(dir(text1))

    for word in ner_output.ents:
        # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
        output.append((word.ents[0], word.label_))
    return output