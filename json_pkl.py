import pickle
import json
import json5
def save_json(data, json_file_path):
    # Save to JSON file
    if len(json_file_path)>0:
        fl=open(json_file_path,"w", encoding="utf-8")
        json.dump(data, fl, indent=4, ensure_ascii=False)
        fl.close()



def save_pkl(data,file_path):
        with open(file_path, 'wb') as file:
             pickle.dump(data, file)




def read_pkl(file_path):

        with open(file_path, 'rb') as file:
            return pickle.load(file)


def read_json(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            return json.load(file)
    except:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Use json5.load() instead of json.load()
            return json5.load(file)