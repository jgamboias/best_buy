import json

def read_file(path):
    
    f = open(path)
    data = json.load(f)
    f.close()
    
    return data