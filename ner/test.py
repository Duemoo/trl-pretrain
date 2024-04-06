import os, glob, json

FILE_PATH = os.path.realpath(__file__)

def merge_multiple_json_file():
    merge_list = []
    
    for file_path in glob.glob(os.path.join(os.path.dirname(FILE_PATH), "results/entity/7b/yanghwa bridge*.json")):
        print(file_path)
        with open(file_path, "r") as f:
            data = json.load(f)
            merge_list.extend(data)
            
    result = sorted(merge_list, key=lambda d: d["step"])
    
    with open(os.path.join(os.path.dirname(FILE_PATH), "results/entity/7b/yanghwa bridge-20000-30000.json"), 'w') as f:
        json.dump(result, f, indent=4)



def main():
    # entity_list = ['cheng chau', "Eunsol Choi", "Minjoon Seo", "ulleungdo", "Chujado", "Geochang", "KAIST", "yanghwa bridge", "Diptyque", "goesan", "Luke Zettlemoyer", "samcheok"]
    
    entity_list = ['cheonjamun']
    
    for entity in entity_list:
        refined_data = []
        with open(os.path.join(os.path.dirname(FILE_PATH), f"results/entity/7b/{entity}-20000-30000.json"), "r") as f:
            data = json.load(f)
            
        for instance in data:
            passage = ""
            document_list = instance["passage"].split("<|endoftext|>")
            for document in document_list:
                if entity.lower() in document.lower():
                    passage = passage + document + "<|endoftext|>"
            refined_data.append({"entity": instance["entity"], "step": instance["step"], "passage": passage})
            
        with open(os.path.join(os.path.dirname(FILE_PATH), f"results/entity/7b/{entity}-20000-30000_refined.json"), 'w') as f:
                json.dump(refined_data, f, indent=4)


if __name__=="__main__":
    main()
    