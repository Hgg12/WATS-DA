import json
import re

char_at_position = []
for i in range(1400):
    with open('tracker/BAN/train_dataset/Wildlife2024_1/result/'+str(i)+'.json') as file:
        # read the json file
        content = file.read()
        pattern = r'\}(\r?\n)\{'
        content = re.sub(pattern,'};\n{',content)
        dictionaries = content.split(';')
    for dictionary in dictionaries:
        data = json.loads(dictionary)
        with open('tracker/BAN/train_dataset/Wildlife_1/result/list.json', 'a') as file:
                json.dump(data, file, indent=4, sort_keys=True)
                file.write('\n')
