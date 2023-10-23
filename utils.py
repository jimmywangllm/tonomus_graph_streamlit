import requests
from transformers import AutoTokenizer, AutoModel
import torch as torch
import torch.nn.functional as F
import pickle
import pandas as pd
from scipy import spatial
import numpy as np

LLM_URL = "http://37.224.68.132:27044/generate"

system_prompt = f"You are a large language model named AraMUS developed by TONOMUS, NEOM to chat with the users. Only respond to the last instruction. Your response should be short and abstract, less than 64 words. Do not try to continue the conversation by generating more instructions. Stop generating more responses and return the output when you produce the first [INST]. Stop generating more responses when the current generated response is longer than 64 tokens."


def chat_respond(
    user_input,
    messages = [],
    ):

    try:

        # query similar question and answers
        top_questions = requests.post(
            'http://37.224.68.132:26170/tonomus_llm/question_search',
            json = {"query": user_input},    
        ).json()['results']
        
        top_answers = requests.post(
            'http://37.224.68.132:26170/tonomus_llm/answer_search',
            json = {"query": user_input},    
        ).json()['results']

        # build the base prompts from the questions and answers
        base_qa_parirs_prompt = []        
        for r in top_answers[0:5] + top_questions[0:5]:
            if r['score'] > 0.4:
                base_qa_parirs_prompt.append(f"[INST] {r['Question']} [/INST]\n{r['Answer']}")      

        for m in messages[-20:]:
            if m['role'] == 'user':
                base_qa_parirs_prompt.append(f"[INST] {m['content']} [/INST]")
            else:
                base_qa_parirs_prompt.append(f"{m['content']}")

        base_qa_parirs_prompt = '\n'.join(base_qa_parirs_prompt)

        #print(f'\n\n\n>>>>{base_qa_parirs_prompt}\n\n\n')

        # build the main prompt
        prompt = f"""
        <<SYS>>
        {system_prompt.strip()}
        <</SYS>>
        
        {base_qa_parirs_prompt.strip()}
        [INST] {user_input.strip()} [/INST]
        """.strip()

        # send to the llama-2 model 
        
        response = requests.post(
            'http://37.224.68.132:27027/tonomus_llm/llama2_generate',
            json  = {"prompt": prompt}
        )               
        
        response = response.json()['response']
        response = response.split('[INST')[0].strip()
         
        return response

    except:
        return None


def ner_entity(text):
    print('Get LLM extract Entity !')
    tmp_sys = "Please extract the entities from the input. Just give the entities, do not answer anything else. Example: Input: What is Aramus? Answer: Aramus."
    prompt = f"""<<SYS>>
        {tmp_sys.strip()}
        <</SYS>>
    """.strip()
    info_post = {"instruction": f" {prompt} ", "input": f" [INST] {text} [/INST]", 
                 "parameters": {"top_p": 0.9, "max_tokens": 512, "top_k": 50}}
    response = requests.post(LLM_URL, json=info_post)
    res = response.json()
    print('ner result : {}'.format(res))
    return res['generated_text'].split('\n')

def cos_sim_top(all_embedding, entity, top_k=1):
    sim_list = []
    entity_embedding = encoder.encode(entity, 512)[0]
    for i in all_embedding:
        res_sim = 1 - spatial.distance.cosine(i[0], entity_embedding)
        sim_list.append(res_sim)
    sim_list = np.array(sim_list)
    return np.argsort(-sim_list)[:top_k]


class TextEncoder:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)

    def encode(self, text: str, max_length: int):
        encoded_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        encoded_input = encoded_input.to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        # Perform pooling
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def graph_data_prepare(need_return=0):
    

    triplets = pd.read_excel('triplets.xlsx')
    triplets = triplets.to_dict('records')
    entity_sub_ob_link_dict = {}
    entity_list = []
    embeddings_list = []
    for num, i in enumerate(triplets):
        if i['subject'] not in entity_sub_ob_link_dict.keys():
            entity_sub_ob_link_dict[i['subject']] = [num]
        else:
            entity_sub_ob_link_dict[i['subject']].append(num) 
        if i['object'] not in entity_sub_ob_link_dict.keys():
            entity_sub_ob_link_dict[i['object']] = [num]
        else:
            entity_sub_ob_link_dict[i['object']].append(num) 
        entity_list.append(i['subject'])
        entity_list.append(i['object'])
        
        encode_sub = encoder.encode(i['subject'], 512)
        encode_ob = encoder.encode(i['object'], 512)
        embeddings_list.append(encode_sub)
        embeddings_list.append(encode_ob)

        with open('./entity_data/embedding_data.pkl', 'wb') as f:
            pickle.dump(embeddings_list, f)
        with open('./entity_data/entity_list.pkl', 'wb') as f:
            pickle.dump(entity_list, f)
        with open('./entity_data/entity_sub_ob_link_dict.pkl', 'wb') as f:
            pickle.dump(entity_sub_ob_link_dict, f)

    if need_return:
        return entity_sub_ob_link_dict, entity_list, embeddings_list

encoder = TextEncoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5')


if __name__ == "__main__":
    # encoder = TextEncoder('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
    # embeddings = encoder.encode('Hello', 512)
    # print(embeddings)
    graph_data_prepare()