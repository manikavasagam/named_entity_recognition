import spacy
import json


modelfile = input("Enter your Model Name: ")

nlp = spacy.load(modelfile)

output = {}

while True:
    #Test your text
    test_text = input("Enter your testing text: ")
    doc = nlp(test_text)
    id = 1
    for ent in doc.ents:
        data = {'field': '', 'start_index':'', 'end_index':'', 'type':''}
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
        data['field'] = ent.text
        data['start_index'] = ent.start_char
        data['end_index'] = ent.end_char
        data['type'] = ent.label_

        field_id = "field" + str(id)
        id = id + 1

        output[field_id] = data
    
    
    with open('result.json', 'w') as outfile:
        json.dump(output, outfile)   