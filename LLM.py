import re
import os
import json
import time
import openai
import logging
import unidecode

from utils import read_datasets


class Configurations:
    logging_folder = "logs"
    experimental_results = "/scratch/c.scmnk4/elexir/resources/results"
    error_analysis_pos_file = "/scratch/c.scmnk4/elexir/resources/results/gpt-3.5-turbo_correct_predictions.txt"
    error_analysis_neg_file = "/scratch/c.scmnk4/elexir/resources/results/gpt-3.5-turbo_incorrect_predictions.txt"
    analogy_datasets = ['ekar', 'scan', 'sat', 'u2', 'u4', 'bats', 'google']
    # analogy_datasets = ['ekar']


# NLP Group API key
# os.environ['OPENAI_API_KEY'] = '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' # Old
# os.environ['OPENAI_API_KEY'] = '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'

# ELEXIR API key
os.environ['OPENAI_API_KEY'] = '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'

openai.api_key = os.getenv('OPENAI_API_KEY')

# model = "gpt-4"
model = "gpt-3.5-turbo"
# model = "text-davinci-003"

prompt_text = '''
Many standardized tests, including high school entrance exams, the SATs, civil service exams, the GREs, and others, use analogy questions to test both logic and reasoning skills and word knowledge. These questions ask test takers to identify relationships between pairs of words. 

In order to solve analogy questions, you must first have a clear understanding of the words' definitions and then use that understanding to determine how the words are related. The key to solving an analogy question is to precisely describe the relationship between the pair of words and then apply the same relationship to determine which word completes the analogy. Most analogy questions rely on your ability to deduce the correct relationship between words and to draw logical conclusions about the possible answer choices. 

The relationships that are found in analogy questions fall into several general types. 

1) Part to Whole. In this type of question, a pair of words consists of a part and a whole. For example, spoke : wheel. A spoke is part of a wheel. 

2) Type and Category. These questions use pairs of words in which one word is a specific type in a general category. For example, orange : citrus. An orange is a type of citrus. 

3) Degree of Intensity. These questions test your ability to discern nuance of meaning among pairs of words. For example, shower : monsoon. A shower is light rainfall and a monsoon is heavy rainfall. 

4) Function. These questions pair words that are related through function. For example, hammer : build. A hammer is used to build. 

5) Manner. This type of analogy describes the manner, way, or style by which an action is accomplished. For example, shamble : walk. Shamble means to walk in an awkward manner. 

6) Symbol or representation. These questions pair words in which one word is the symbol of the other. For example, dove : peace. A dove is a symbol of peace. 

7) Action and significance. In this type of analogy one word describes an action and the other word indicates the significance of the action. For example, cry : sorrow. To cry signifies sorrow

Analogy questions can also be used to test word knowledge and factual content. Word knowledge questions are generally pairs of synonyms or pairs of antonyms. Factual content questions demand a certain level of general knowledge, and cannot be deduced from the relationship alone.

Given the word pair, your aim is to choose the word pair from choices that is analogously most similar. Also, give an explanation. The explanation should be precise. I will show some examples then you will have to do it yourself. 

Query = ['banana', 'peel']; Choices = [['egg', 'crack'], ['carrot', 'uproot'], ['apple', 'core'], ['bread', 'slice'], ['corn', 'husk']]

Answer: choice number 4; Explanation: A banana has a peel that can be removed, and corn has a husk that can be removed.

Query = ['birds', 'wings']; Choices = [['moose', 'antlers'], ['camel', 'hump'], ['spider', 'legs'], ['alligator', 'tail'], ['cat', 'whiskers']]

Answer: choice number 2; Explanation: Birds have wings, and spiders have legs.

Query = ['berate', 'criticize']; Choices = [['goad', 'urge'], ['accuse', 'apologize'], ['regret', 'remember'], ['betray', 'follow'], ['evaluate', 'praise']]

Answer: choice number 0; Explanation: To berate is to criticize, and to goad is to urge.

Now, answer the following questions: 

'''

small_prompt_text = '''
In this task, you're to identify the pair of words that best completes an analogy based on a given query. The key to solving this is by understanding the relationship between the words in the query and then finding a pair in the choices that share the same relationship. Additionally, provide a precise explanation for your answer.

I will show you some examples that specify the format of the questions and the format in which you will have to generate your answers. After that you will have to do yourself.

Query = ['banana', 'peel']; Choices = [['egg', 'crack'], ['carrot', 'uproot'], ['apple', 'core'], ['bread', 'slice'], ['corn', 'husk']]

Answer: choice number 5; Explanation: A banana has a peel that can be removed, and corn has a husk that can be removed.

Query = ['birds', 'wings']; Choices = [['moose', 'antlers'], ['camel', 'hump'], ['spider', 'legs'], ['alligator', 'tail'], ['cat', 'whiskers']]

Answer: choice number 3; Explanation: Birds have wings, and spiders have legs.

'''


def calculate_accuracy(actual_answers, predicted_answers):
    n = len(actual_answers)
    count = 0
    for i in range(0, n):
        if actual_answers[i] == predicted_answers[i]:
            count = count + 1
    accuracy = (count / float(n)) * 100
    return accuracy


def dump_results(dump, filename):
    f = open(filename, 'a')
    for item in dump:
        f.write(item + '\n')
    f.close()


def initilization():
    log_file = os.path.join(Configurations.logging_folder,
                            os.path.splitext(os.path.basename(__file__))[0] + '-' + model + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


def get_gpt_response(prompt1):
    response1 = openai.ChatCompletion.create(
        engine=model,  # This should be changed to GPT-4's engine name when it's released
        prompt=prompt1,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response1.choices[0].text.strip()
    return message


def get_gpt_3_5_response(prompt):
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


def parse_response(input_str):
    # Extract the answer
    ans_pattern = r'choice number (\d+)'
    ans_match = re.search(ans_pattern, input_str, flags=re.IGNORECASE)
    ans = int(ans_match.group(1)) if ans_match else None

    # Extract the explanation
    exp_pattern = r'Explanation: (.*)'
    exp_match = re.search(exp_pattern, input_str, flags=re.IGNORECASE)
    exp = exp_match.group(1) if exp_match else None

    return ans, exp


def solve_analogies(query, choices):
    question = 'Query = ' + str(query) + '; Choices = ' + str(choices)
    prompt1 = prompt_text + question
    response1 = get_gpt_3_5_response(prompt1)
    [ans, exp] = parse_response(response1)
    return [ans, exp]


def solve_analogies_using_small_prompt(query, choices):
    question = 'Query = ' + str(query) + '; Choices = ' + str(choices)
    prompt1 = small_prompt_text + question
    response1 = get_gpt_3_5_response(prompt1)
    [ans, exp] = parse_response(response1)
    if ans is None:
        result = [ans, exp]
    else:
        result = [ans - 1, exp]
    return result


# def solve_analogies_via_gpt_3_5(query, choices):
#     question = 'Query = ' + str(query) + '; Choices = ' + str(choices)
#     response1 = get_gpt_3_5_response(question)
#     [ans, exp] = parse_response(response1)
#     return [ans, exp]


pos_file = None
neg_file = None


def open_error_files():
    global pos_file, neg_file
    m = model
    pos_file = open(Configurations.error_analysis_pos_file, 'w')
    pos_file.write('Correct Predictions, Method = %s\n' % m)
    pos_file.write('===================\n\n\n')
    neg_file = open(Configurations.error_analysis_neg_file, 'w')
    neg_file.write('Incorrect Predictions, Method = %s\n' % m)
    neg_file.write('===================\n\n\n')


def close_error_files():
    global pos_file, neg_file
    pos_file.close()
    neg_file.close()


def token_size_determination(text=prompt_text):
    text = small_prompt_text
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=text,
        max_tokens=1
    )
    token_count = response['usage']['total_tokens']
    print("Number of tokens:", token_count)


def write_error_file(dataset_name, stem, choice, answer, predicted_answer, explanation):
    if answer == predicted_answer:
        file = pos_file
    else:
        file = neg_file
    file.write('\n\nDataset = %s, Query = %s, Choices = %s, Actual answer = %s, Predicted answer = %s, Explanation = '
               '%s\n\n' % (dataset_name, str(stem), str(choice), str(answer), str(predicted_answer), explanation))
    file.flush()
    os.fsync(file)


def evaluate(data, dataset_name):
    method = model
    data_len = len(data)
    actual_answers = []
    predicted_answers = []
    dump = []
    counter = 0
    for item in data:
        stem = item['stem']
        [stem1, stem2] = stem
        stem1 = unidecode.unidecode(stem1.lower())
        stem2 = unidecode.unidecode(stem2.lower())
        stem = [stem1, stem2]
        choice = item['choice']
        temp = []
        for ch in choice:
            [ch1, ch2] = ch
            ch1 = unidecode.unidecode(ch1.lower())
            ch2 = unidecode.unidecode(ch2.lower())
            ch = [ch1, ch2]
            temp.append(ch)
        choice = temp
        answer = item['answer']

        # [predicted_answer, explanation] = solve_analogies(stem, choice)

        [predicted_answer, explanation] = solve_analogies_using_small_prompt(stem, choice)

        predicted_answers.append(predicted_answer)
        actual_answers.append(answer)
        write_error_file(dataset_name, stem, choice, answer, predicted_answer, explanation)
        item['predicted_answer'] = predicted_answer
        dump.append(json.dumps(item))
        counter = counter + 1
        if counter % 100 == 1:
            logging.info('Evaluation Progress: Dataset = %s, Method = %s, %s out of %s processed',
                         dataset_name, method, str(counter), str(data_len))
        logging.info('Dataset = %s, Stem = %s, Choices = %s, Actual answer = %s, Predicted answer = %s, Accuracy till '
                     'now = %s, Explanation = %s', dataset_name, str(stem), str(choice), str(answer),
                     str(predicted_answer), str(calculate_accuracy(actual_answers, predicted_answers)), explanation)
    if len(data) != len(actual_answers):
        raise Exception('List mismatch')
    else:
        count = 0
        for i in range(0, len(data)):
            if actual_answers[i] == predicted_answers[i]:
                count = count + 1
        accuracy = (count / float(len(data))) * 100
        x = {'accuracy': accuracy, 'method': method}
        dump.append(json.dumps(x))
    return dump


if __name__ == '__main__':
    # token_size_determination()
    initilization()
    open_error_files()
    for analogy_dataset in Configurations.analogy_datasets:
        the_data = read_datasets(analogy_dataset, approach='test_and_valid')
        the_dump = evaluate(the_data, analogy_dataset)
        result_file = os.path.join(Configurations.experimental_results, analogy_dataset + '_result.json')
        dump_results(the_dump, result_file)
    close_error_files()
