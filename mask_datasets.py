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
import pickle
def get_toxic_dataset(split="train"):
    """
    Retrieve the toxicity data from Circuit Breaking: Removing Model Behaviors with Targeted Ablation.
    Split can be train, test, or uniform.
    """
    if split == "train":
        filename = "train"
    elif split == "test":
        filename = "test"
    elif split == "uniform":
        filename = "eval_uniform"
    with open(f"datasets/toxicity/{filename}.pkl", "rb") as f:
        toxicity_data = pickle.load(f)
    return [tup[2] for tup in toxicity_data]

# %%
# IOI Dataset

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LATE_IOS = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument and after that [B] said to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
]

BABA_EARLY_IOS = [
    "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
    "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and after that [B] said to [A]",
    "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
]

TEMPLATES_VARIED_MIDDLE = [
    "",
]

# no end of texts, GPT-2 small wasn't trained this way (ask Arthur)
# warnings.warn("Adding end of text prefixes!")
# for TEMPLATES in [BABA_TEMPLATES, BABA_EARLY_IOS, BABA_LATE_IOS]:
#     for i in range(len(TEMPLATES)):
#         TEMPLATES[i] = "<|endoftext|>" + TEMPLATES[i]

ABBA_TEMPLATES = BABA_TEMPLATES[:]
ABBA_LATE_IOS = BABA_LATE_IOS[:]
ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

ANIMALS = [
    "dog",
    "cat",
    "snake",
    "elephant",
    "beetle",
    "hippo",
    "giraffe",
    "tiger",
    "husky",
    "lion",
    "panther",
    "whale",
    "dolphin",
    "beaver",
    "rabbit",
    "fox",
    "lamb",
    "ferret",
]


def multiple_replace(dict, text):
    # from: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)


def iter_sample_fast(iterable, samplesize):
    results = []
    # Fill in the first samplesize elements:
    try:
        for _ in range(samplesize):
            results.append(next(iterable))
    except StopIteration:
        raise ValueError("Sample larger than population.")
    random.shuffle(results)  # Randomize their positions

    return results


NOUNS_DICT = NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}


def gen_prompt_uniform(
    templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = rd.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = rd.choice(names)
            name_2 = rd.choice(names)
            name_3 = rd.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = rd.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = rd.randint(30, 40)
            pref = ".".join(rd.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            nb_gen += 1
    return ioi_prompts
