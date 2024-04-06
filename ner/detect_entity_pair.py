import argparse, json, os, glob


FILE_PATH = os.path.realpath(__file__)

def main(args):
    json_path = glob.glob(os.path.join(os.path.dirname(FILE_PATH), f"results/entity/7b/{args.root_entity}*_refined.json"))
    assert len(json_path) == 1, "Before run, you have to merge multiple json file"
    json_path = json_path[0]
    start_idx = os.path.splitext(os.path.basename(json_path))[0].split("-")[1]
    end_idx = os.path.splitext(os.path.basename(json_path))[0].split("-")[2].split('_')[0]
    with open(json_path, "r") as f:
        data = json.load(f)
    
    result = []
    for step in data:
        for candidate in args.candidates:
            if candidate in step["passage"]:
                result.append({"entity_pair": f"{step['entity']} & {candidate}", "step": step["step"], "passage": step["passage"]})

    for candidate in args.candidates:
        entity_result = [d for d in result if d["entity_pair"]==f"{args.root_entity} & {candidate}"]
        with open(os.path.join(os.path.dirname(FILE_PATH), f'results/entity_pair/{args.root_entity}-{candidate}-{start_idx}-{end_idx}.json'), 'w') as f:
            json.dump(entity_result, f, indent=4)




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=64, help="number of processes")
    parser.add_argument("--root_entity", type=str, help="entity that you want find entity pair")
    parser.add_argument("--candidates", type=str, nargs='+', help="entity to search")
    
    args = parser.parse_args()
    
    main(args)