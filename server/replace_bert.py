from transformers import pipeline
from utils import compare, ask, scale_to_255, get_red_code

from copy import copy

pipe = pipeline("fill-mask", model="FacebookAI/roberta-base") 

MASK_ID = 50264

def get_importance_values(prompt, stopwords):
    diff_list = []

    original_response = ask(prompt)
    diff_list = []
    words = prompt.split()

    for i in range(len(words)):
        temp_words = copy(words)
        temp_words[i] = "<mask>"
        masked_prompt = " ".join(temp_words)
        options = pipe(masked_prompt)
        new_prompt = masked_prompt.replace("<mask>", options[0]["token_str"])
        new_response = ask(new_prompt)

        diff_list.append(compare(original_response, new_response))

    diff_list = scale_to_255(diff_list)

    formatted_text = ""
    results = []
    for word, diff in zip(prompt.split(), diff_list):
        red_code = get_red_code(diff)
        # formatted_text += f"{red_code}{word}\033[0m "
        results.append((word, diff))

    return results
