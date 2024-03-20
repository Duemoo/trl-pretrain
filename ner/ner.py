import spacy

def ner_in_batch_spacy(text: str) -> dict:
    output = {}
    # To use spaCy in GPU
    spacy.prefer_gpu()
    # Faster and smaller pipeline, but less accurate
    # to use "en_core_web_sm", you should download the model first typing "python -m spacy download en_core_web_sm" in terminal
    ner_model = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    # Larger and slower pipeline, but more accurate
    # to use "en_core_web_trf", you should download the model first typing "python -m spacy download en_core_web_trf" in terminal
    # ner_model = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    ner_model.max_length = 5000000 
    ner_output = ner_model(text)
    # print(dir(text1))

    for word in ner_output.ents:
        # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
        if len(word.ents) > 1:
            raise Exception(f"{word.ents}")
        
        entity = str(word.ents[0])
        label = str(word.label_)
        
        if not entity in output or label != output[entity]:
            output[entity] = label
        
        # if str(word.ents[0]) in output.keys() and str(word.label_) == output[str(word.ents[0])]["label"]:
        #     output[str(word.ents[0])]['count'] += 1
        # else:
        #     output[str(word.ents[0])] = {"label": str(word.label_), "step": [step_idx]}
    return output


if __name__=="__main__":
    t = "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century."
    print(ner_in_batch_spacy(t))