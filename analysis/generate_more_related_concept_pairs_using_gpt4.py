import re
import os
import json
import time
import pickle
import openai
import logging
import unidecode

from reasoning_with_vectors.conf import configuration
from reasoning_with_vectors.core.utils import read_datasets

# NLP Group API key
# os.environ['OPENAI_API_KEY'] = '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' # Old
# os.environ['OPENAI_API_KEY'] = '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'

# ELEXIR API key
os.environ['OPENAI_API_KEY'] = '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'

openai.api_key = os.getenv('OPENAI_API_KEY')

# model = "gpt-4"
model = "gpt-3.5-turbo"
# model = "text-davinci-003"

prompt_text = '''Given a concept pair [A, B], the goal is to generate the 20 most analogous concept pairs. You should first accurately determine the relationship between A and B, then generate 20 pairs sharing the same relationship. Also, be sure to precisely specify the determined relationship in your response.

I'll provide an example to clarify the format in which you should generate your examples.

Given concept pair: ['apple', 'fruit']

Relationship: 'apple' is a specific example of 'fruit'.

Generated analogous concept pairs:

['lion', 'mammal']
['rose', 'flower']
['whale', 'marine animal']
['banana', 'fruit']
['oak', 'tree']
['eagle', 'bird']
['poodle', 'dog']
['salmon', 'fish']
['corolla', 'car']
['python', 'snake']
['sunflower', 'flower']
['sparrow', 'bird']
['tulip', 'flower']
['rattlesnake', 'snake']
['beagle', 'dog']
['sparrow', 'bird']
['tuna', 'fish']
['puma', 'mammal']
['trout', 'fish']
['falcon', 'bird']

'''


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] +
                            f'_model_{model}.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)
    logging.info('Started')


def get_gpt_response(prompt):
    response1 = None
    processed = False
    while not processed:
        try:
            response1 = openai.ChatCompletion.create(
                model=model,  # This should be changed to GPT-4's engine name when it's released
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            processed = True
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(60)
    message = response1.choices[0].message.content.strip()
    return message


def parse_response(response):
    # Split the response into lines and remove empty ones
    lines = [line.strip() for line in response.split("\n") if line.strip() != '']

    relationship = None
    generated_pairs = []

    # Look for the relationship and pairs in each line
    for line in lines:
        if line.startswith("Relationship:"):
            # Remove the "Relationship:" part and strip extra spaces
            relationship = line.replace("Relationship:", "").strip()

        elif line.startswith("Generated analogous concept pairs:"):
            # The next lines should be pairs, so we continue to the next iteration
            continue

        elif line.startswith("[") and line.endswith("]"):
            # This is a pair, so we remove the brackets, split by comma, and strip extra spaces and quotes
            pair = [item.strip().strip("'") for item in line.strip("[]").split(",")]
            generated_pairs.append(pair)
    return relationship, generated_pairs


def find_analogous_concept_pairs(query):
    question = f'Given concept pair: {query}'
    prompt1 = prompt_text + question
    response1 = get_gpt_response(prompt1)
    [relationship, generated_pairs] = parse_response(response1)
    return [relationship, generated_pairs]


def token_size_determination(text=prompt_text):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=1
    )
    token_count = response['usage']['total_tokens']
    print("Number of tokens:", token_count)


class HighQualityConceptPairsGenerator:
    def __init__(self):
        self.high_quality_related_concept_pairs = []
        self.analogy_datasets = ['scan', 'sat']
        self.high_quality_pairs = f'/scratch/c.scmnk4/elexir/resources/results/{model}_generated_high_quality_pairs'

    def write_results_to_file(self):
        with open(self.high_quality_pairs, 'wb') as f:
            pickle.dump(self.high_quality_related_concept_pairs, f)

    def generator_per_dataset(self, data, dataset_name):
        counter = 0
        for item in data:
            stem = item['stem']
            [stem1, stem2] = stem
            stem1 = unidecode.unidecode(stem1.lower())
            stem2 = unidecode.unidecode(stem2.lower())
            stem = [stem1, stem2]
            [relationship, generated_pairs] = find_analogous_concept_pairs(stem)
            result = dict()
            result['query'] = stem
            result['relationship'] = relationship
            result['analogous_pairs'] = generated_pairs
            self.high_quality_related_concept_pairs.append(result)
            logging.info(f'Dataset = {dataset_name}, Query = {stem}, Relationship = {relationship}, '
                         f'Analogous concept pairs = {generated_pairs}')
            counter += 1
            if counter > 3:
                break

    def generator(self):
        for dataset_name in self.analogy_datasets:
            data = read_datasets(dataset_name, approach='test_and_valid')
            self.generator_per_dataset(data, dataset_name)
        self.write_results_to_file()


if __name__ == '__main__':
    # token_size_determination()
    initialization()
    obj = HighQualityConceptPairsGenerator()
    obj.generator()
