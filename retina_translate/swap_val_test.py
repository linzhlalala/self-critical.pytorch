import json

def main():
    #cut it
    original = json.load(open("../retina_wp/dataset_retina3.json"))
    
    new_list = original["images"]
    for record in new_list:
        if record['split'] == 'val':
            record['split'] = 'test'
        elif record['split'] == 'test':
            record['split'] = 'val'            
    with open("../retina_wp/dataset_retina4.json","w") as output:
        json.dump({"images":new_list,'dataset':"retina4"},output)

if __name__ == "__main__":
    main()