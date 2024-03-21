import spacy
import time
import multiprocessing as mp
import parmap
from functools import partial
import itertools
from tqdm import tqdm


class GenwithLen:
    def __init__(self, generator, *args, **kwargs):
        self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    def __len__(self):
        self.generator, new_generator = itertools.tee(self.generator)
        return sum(1 for _ in new_generator)
        

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

def organize_output1(entity, label, output):        
    if not entity in output or label != output[entity]:
        output[entity] = label
    return


def ner_in_batch_spacy(texts: list, per_document=False) -> dict:
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
    ner_model.max_length = 10000000
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
        # Option1
        # start = time.time()
        # for ner_output in tqdm(ner_outputs):
        #     parmap.starmap(organize_output1, [(str(word.ents[0]), str(word.label_)) for word in ner_output.ents], output, pm_processes=10)
        # end = time.time()
        # print(f"Option 1 time : {end-start}")
        # # Option 2
        # start = time.time()
        # parmap.map(organize_output, list(ner_outputs), output, pm_pbar=True, pm_processes=10)
        # end = time.time()
        # print(f"Option 2 time : {end-start}")
        # # Option 3
        output = {}
        start = time.time()
        for ner_output in ner_outputs:
            for word in ner_output.ents:
                # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
                if len(word.ents) > 1:
                    raise Exception(f"{word.ents}")
                
                entity = str(word.ents[0])
                label = str(word.label_)
                
                if not entity in output or label != output[entity]:
                    output[entity] = label
        end = time.time()
        print(f"Option 3 time : {end-start}")
        
    # for ner_output in ner_outputs:
    #     print(len(list(ner_output.ents)), end=" ")
    #     for word in ner_output.ents:
    #         # word.ents가 하나의 element만 포함하는지 확인할 필요 있음
    #         if len(word.ents) > 1:
    #             raise Exception(f"{word.ents}")
            
    #         entity = str(word.ents[0])
    #         label = str(word.label_)
            
    #         if not entity in output or label != output[entity]:
    #             output[entity] = label
            
    #         # if str(word.ents[0]) in output.keys() and str(word.label_) == output[str(word.ents[0])]["label"]:
            #     output[str(word.ents[0])]['count'] += 1
            # else:
            #     output[str(word.ents[0])] = {"label": str(word.label_), "step": [step_idx]}
        # print(output)
    # return output


if __name__=="__main__":
    t = ["Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, Einstein also made important Albert Einstein Albert Einstein contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century." for i in range(5)]
    t.append("Hello my name is Jinho")
    t.append("I'm majoring AI in KAIST")
    print(t)
    start = time.time()
    print(ner_in_batch_spacy(t, per_document=True))
    end = time.time()
    print(end-start)