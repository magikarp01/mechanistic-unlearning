#%%
# Datasets for training a subnetwork discovery algorithm or unlearning.

"""
From DISCOVERING KNOWLEDGE-CRITICAL SUBNETWORKS IN PRETRAINED LANGUAGE MODELS
"""
from transformers import GPT2Tokenizer
from datasets import load_dataset
import random
from nltk.corpus import wordnet as wn
import pandas as pd
from tqdm import tqdm
import torch

def load_controllm():
    # Load WikiText-2 dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')

    # Load GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize and prepare the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'])

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    chunked_datasets = tokenized_datasets.map(lambda examples: {'chunks': tokenizer.chunk_examples(examples['input_ids'], 512)}, batched=True)

    # Get training, validation, and test datasets
    train_dataset = chunked_datasets['train']
    valid_dataset = chunked_datasets['validation']
    test_dataset = chunked_datasets['test']
    return train_dataset, valid_dataset, test_dataset


#%%


import pandas as pd
conceptnet_data = pd.read_csv('datasets/short_conceptnet_assertions.csv', delimiter='\t', header=None)
# Filter the data to include only rows with single-token tail entities
filtered_data = conceptnet_data[conceptnet_data[2].apply(lambda x: len(x.split()) == 1)]


# %%
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

def get_hypernyms(synset):
    hypernyms = []
    for hypernym in synset.hypernyms():
        sentence = f"A {synset.lemmas()[0].name()} is a {hypernym.lemmas()[0].name()}."
        hypernyms.append(((synset.name(), 'IsA', hypernym.name()), sentence))
    return hypernyms

def get_wordnet_triplets(tokenizer=GPT2Tokenizer.from_pretrained('gpt2'), tot_triplets=None):
    """
    Iterate through hypernyms and collect triplets and sentences. Filter so that only sentences with single-token tail entities are included. Returns a list of triplets and a list of sentences.
    """
    num_triplets = 0
    
    triplets = []
    sentences = []
    used_words = set()

    for synset in wn.all_eng_synsets():
        hypernyms = get_hypernyms(synset)
        if len(hypernyms) < 1:
            continue
        for relationship, sentence in hypernyms:
            head, relation, tail = relationship
            
            head = head.split('.')[0]
            tail = tail.split('.')[0]
            
            head_tokens = tokenizer.tokenize(head)
            tail_tokens = tokenizer.tokenize(tail)
            if len(tail_tokens) == 1 and tail not in used_words and head not in used_words:
                used_words.add(head)
                used_words.add(tail)

                triplets.append((head, relation, tail))
                
                # replace head underscore with space
                sentences.append(sentence.replace('_', ' '))
                num_triplets += 1

            if tot_triplets is not None and num_triplets >= tot_triplets:
                return triplets, sentences

    return triplets, sentences

#%%

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

triplets, sentences = get_wordnet_triplets(tokenizer, tot_triplets=100)
# %%
wordnet_formats = ["A {head} is a {tail}", "A {head} is a type of {tail}", "A {head} is a kind of {tail}", "{head} is a {tail}", "{head} is a type of {tail}", "{head} is a kind of {tail}"]
def optimize_sentences(model, triplets, formats):
    """
    Optimize sentence verbalizations of knowledge triplets by choosing the verbalization that minimizes perplexity.
    """
    sentences = []
    for triplet in tqdm(triplets):
        head, relation, tail = triplet
        
        min_perplexity = 0
        best_sentence = None
        for format in formats:
            sentence = format.replace('{head}', head).replace('{tail}', tail).replace("_", " ")
            # Capitalize first letter of sentence
            sentence = sentence[0].upper() + sentence[1:]

            # calculate perplexity of sentence given model

            with torch.no_grad():
                input_ids = tokenizer.encode(sentence, return_tensors='pt').cuda()
                outputs = model(input_ids, labels=input_ids)
                log_likelihood = outputs.loss * input_ids.shape[1]
                # loss = outputs[0]
                # print(outputs)
                # print(f"{input_ids.shape=}, {outputs[0].shape=}")
            perplexity = torch.exp(log_likelihood)
            if perplexity < min_perplexity or best_sentence is None:
                min_perplexity = perplexity
                best_sentence = sentence
        sentences.append(best_sentence)

    return sentences

#%%
from transformers import GPT2LMHeadModel, GPT2Config

# Load pre-trained GPT-2 Medium model
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
# send to CUDA
gpt2_model.cuda()

#%%
optimized_sentences = optimize_sentences(gpt2_model, triplets, wordnet_formats)
#%%
"""
Get the ConceptNet triplets and sentences. Filter so that only sentences with single-token tail entities are included. Returns a list of triplets and a list of sentences.
"""

# Predicates: {'AtLocation', 'CapableOf', 'Causes', 'CausesDesire', 'Desires', 'HasA','HasPrerequisite', 'HasProperty', 'HasSubevent', 'IsA', 'MadeOf', 'MotivatedByGoal', 'NotDesires', 'PartOf', 'ReceivesAction', 'UsedFor'}
# format sentences for each of these predicates
predicate_formats = {'AtLocation': }
def get_conceptnet_triplets(short=True):
    hf_conceptnet = load_dataset('lama', 'conceptnet', split='train')
    conceptnet_data = hf_conceptnet.data.to_pandas()
    triplets = []
    sentences = []

#%%