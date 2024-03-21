import spacy
import time
import multiprocessing as mp
import parmap
from functools import partial
import itertools
from tqdm import tqdm
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import torch
        

def organize_output(ner_output, output):
    for word in ner_output.ents:
        # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
        if len(word.ents) > 1:
            raise Exception(f"{word.ents}")
        
        entity = str(word.ents[0])
        label = str(word.label_)
        
        if not entity in output or label != output[entity]:
            output[entity] = label
    return


def organize_flair_output(sentence, output):
    for entity_dict in sentence['entities']:
        entity = entity_dict['text']
        ner_tag = entity_dict['labels'][0]['value']
        if not entity in output or ner_tag != output[entity]:
            output[entity] = ner_tag
    return


def ner_in_batch_spacy(texts: list, per_document=False, is_gpu_version=False) -> dict:
    output = {}
    # To use spaCy in GPU
    # ValueError("Cannot convert non-numpy from base Ops class") 발생 -> 아직 해결 불가능
    # flag = spacy.prefer_gpu()
    # assert flag
    # Faster and smaller pipeline, but less accurate
    # to use "en_core_web_sm", you should download the model first typing "python -m spacy download en_core_web_sm" in terminal
    ner_model = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    # Larger and slower pipeline, but more accurate
    # to use "en_core_web_trf", you should download the model first typing "python -m spacy download en_core_web_trf" in terminal
    # ner_model = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    # ner_outputs is generator()
    ner_outputs = ner_model.pipe(texts)
    
    if per_document:
        for i, ner_output in enumerate(ner_outputs):
            sub_dict = {}
            for word in ner_output.ents:
                # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
                if len(word.ents) > 1:
                    raise Exception(f"{word.ents}")
                
                entity = str(word.ents[0])
                label = str(word.label_)
                
                if not entity in output or label != sub_dict[entity]:
                    sub_dict[entity] = label
            output[i] = sub_dict
        return output
    else:
        manager = mp.Manager()
        output = manager.dict()
        if is_gpu_version:
            flair.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tagger = SequenceTagger.load("flair/ner-english-fast")
            # start = time.time()
            sentences = [Sentence(text) for text in texts]
            # end = time.time()
            # print(f"Make Sentence time : {end-start}")
            # start = time.time()
            # output is still List
            tagger.predict(sentences, verbose=True, mini_batch_size=128)
            # end = time.time()
            # print(f"tagger.predict() time : {end-start}")
            # start = time.time()
            sentences = [sentence.to_dict() for sentence in sentences]
            # end = time.time()
            # print(f"Make Dict time : {end-start}")
            # start = time.time()
            parmap.map(organize_flair_output, sentences, output=output, pm_pbar=True, pm_processes=20)
            # end = time.time()
        else:
            # start = time.time()
            parmap.map(organize_output, list(ner_outputs), output, pm_pbar=True, pm_processes=10)
            # end = time.time()
            # print(f"Option 2 sorting time : {end-start}")


        return dict(output)


if __name__=="__main__":
    t = ["Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important Albert Einstein Albert Einstein contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century." for i in range(5)]
    t.append("Hello my name is Jinho")
    t.append("I'm majoring AI in KAIST")
    start = time.time()
    print(ner_in_batch_spacy(t, per_document=False))
    end = time.time()
    print(end-start)