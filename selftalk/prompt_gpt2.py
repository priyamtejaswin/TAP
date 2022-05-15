from transformers import pipeline, set_seed
from tqdm import tqdm
import json
import os

print("Testing...")
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print(generator("Look at the image, and tell me: what is the brand of phone in the image?", max_length=30, num_return_sequences=5))

import sys
sys.exit()

questions = []

with open('/usr0/home/ptejaswi/Download/TextVQA_0.5.1_train.json') as fp:
    data = json.loads(fp.read())
    print("Train keys", data.keys())
    print("Train items", len(data['data']))

for x in tqdm(data['data']):
    pass

print("--sample--")
print(x)
print("--end--")

for x in tqdm(data['data']):
    text = x['question']
    qid = x['question_id']
    sname = x['set_name']

    gen = generator(text, max_length=30, num_return_sequences=1)[0]['generated_text']
    new = text + ' ' + gen.lower()
    questions.append({'question': text, 'question_id': qid, 'wgpt2': new, 'set_name': sname})

with open('/usr0/home/ptejaswi/Download/temp_gpt_output.json', 'w') as fp:
    fp.write(json.dumps(questions))

with open('/usr0/home/ptejaswi/Download/TextVQA_0.5.1_val.json') as fp:
    data = json.loads(fp.read())
    print("Val keys", data.keys())
    print("Val items", len(data['data']))

for x in tqdm(data['data']):
    text = x['question']
    qid = x['question_id']
    sname = x['set_name']

    gen = generator(text, max_length=30, num_return_sequences=1)[0]['generated_text']
    new = text + ' ' + gen.lower()
    questions.append({'question': text, 'question_id': qid, 'wgpt2': new, 'set_name': sname})

with open('/usr0/home/ptejaswi/TAP/selftalk/qgpt2.json', 'w') as fp:
    fp.write(json.dumps(questions))

print("Done.")

