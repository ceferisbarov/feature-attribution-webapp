import os
import torch
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from openai import OpenAI
from together import Together

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

tg_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
model.eval()

bge = AutoTokenizer.from_pretrained("BAAI/bge-base-en")
def get_local_embeddings(texts):
    texts = [text.replace("\n", " ") for text in texts]
    # Tokenize sentences
    encoded_input = tokenizer(texts, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings

# Download RoBERTa already finetuned for MNLI
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()  # disable dropout for evaluation

gpt_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

def ask(content):
    chat_completion = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="gpt-3.5-turbo",
        seed=69
    )

    return chat_completion.choices[0].message.content

def compare(seq1, seq2):
    tokens = roberta.encode(seq1, seq2)
    out = roberta.predict('mnli', tokens)
    diff = out[0][0]
    
    return diff.item()

def scale_to_255(values):
    min_val = min(values)
    max_val = max(values)
    
    # Normalize values to 0-1 range
    normalized_values = [(x - min_val) / (max_val - min_val) for x in values]
    
    # Scale normalized values to 0-255 range
    scaled_values = [int(x * 255) for x in normalized_values]
    
    return scaled_values

def get_embeddings(texts, model='togethercomputer/m2-bert-80M-8k-retrieval'):
    texts = [text.replace("\n", " ") for text in texts]
    outputs = tg_client.embeddings.create(model=model, input=texts)
    return [outputs.data[i].embedding for i in range(len(texts))]

def get_red_code(intensity):
    """
    Get the ANSI escape code for the red color with a given intensity.
    Intensity should be between 0 and 255.
    """
    return f'\033[38;2;{intensity};0;0m'
