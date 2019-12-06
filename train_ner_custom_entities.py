import spacy
import random
import json
import pandas as pd

TRAIN_DATA = []

def train_spacy(data,iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       
    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
            print(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.35,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


def generate_train_data(input_data_file, input_entities_file):

    train_data = []

    #input_data_file = 'C:\\Users\\manikavasagam.p\\Desktop\\AWS\\data\\Custom entity\\simple\\rawdata.csv'
    #input_entities_file = 'C:\\Users\\manikavasagam.p\\Desktop\\AWS\\data\\Custom entity\\simple\\entities.csv'

    df_data = pd.read_csv(input_data_file)
    df_entities = pd.read_csv(input_entities_file)


    data_index = 0
    while data_index < df_data.shape[0]:
        data = df_data['Requirement'][data_index]
        data_index = data_index + 1

        ents = []
        entities_index = 0
        while entities_index < df_entities.shape[0]:
            
            entity = df_entities['Text'][entities_index]
            entity_type = df_entities['Type'][entities_index]
            index = data.find(entity)
            if index >= 0:
                ent = (index, index+len(entity), entity_type)
                ents.append(ent)

            entities_index = entities_index + 1

        train_data.append((data.lower(),{'entities':ents}))

        with open("traindata.txt",'w') as write:
	        write.write(str(train_data))
    
    return train_data



"""
filename = input("Enter your train data filename : ")
print(filename)

with open(filename) as train_data:
	train = json.load(train_data)

TRAIN_DATA = []
for data in train:
	ents = [tuple(entity) for entity in data['entities']]
	TRAIN_DATA.append((data['content'],{'entities':ents}))


with open('{}'.format(filename.replace('json','txt')),'w') as write:
	write.write(str(TRAIN_DATA))
"""

input_data_file = input("Enter your train data filename : ")
input_entities_file = input("Enter your entities filename : ")


TRAIN_DATA = generate_train_data(input_data_file, input_entities_file)


prdnlp = train_spacy(TRAIN_DATA, 10)

# Save our trained Model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text.lower())
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
 

#fp=open("result.json", 'w') # output file
#json.dump(doc.ents, fp)
#fp.write('\n')