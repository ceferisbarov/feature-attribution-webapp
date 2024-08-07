def n_gram_precision(candidate, references, n):
    def get_ngrams(sequence, n):
        return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
    
    candidate_ngrams = get_ngrams(candidate, n)
    reference_ngrams = [get_ngrams(ref, n) for ref in references]

    match_count = 0
    total_count = len(candidate_ngrams)

    reference_count = {}
    for reference in reference_ngrams:
        for ngram in reference:
            if ngram in reference_count:
                reference_count[ngram] += 1
            else:
                reference_count[ngram] = 1
    
    for ngram in candidate_ngrams:
        if ngram in reference_count and reference_count[ngram] > 0:
            match_count += 1
            reference_count[ngram] -= 1

    if total_count == 0:
        return 0

    return match_count / total_count

def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min((abs(len(ref) - c), len(ref)) for ref in references)[1]
    
    if c > r:
        return 1
    else:
        return (c / r) ** 0.5

def bleu_score(candidate, references, max_n=4):
    precisions = [n_gram_precision(candidate, references, n) for n in range(1, max_n+1)]
    brevity = brevity_penalty(candidate, references)
    
    if min(precisions) > 0:
        score = brevity * (sum(precisions) / max_n)
    else:
        score = 0.0
    
    return score

# Example usage:
candidate = "the cat is on the mat".split()
references = [
    "there is a cat on the mat".split(),
    "the cat is playing on the mat".split(),
    "my name is Jafar".split()
]

print(bleu_score(candidate, references, max_n=4))
