import copy
import datetime
import json
import os
import re
import string
import time
from typing import Optional, Any

import gradio as gr
import openai
import google.generativeai as genai


# Set up LLM APIs
llm_api_options = ['gemini-pro', 'gemini-1.5-flash', 'gpt-3.5-turbo-1106', 'gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18']

def query_gpt_model(
    prompt: str,
    llm: str = 'gpt-3.5-turbo-1106',
    client: Optional[Any] = None,
    temperature: float = 0.0,
    max_decode_steps: int = 512,
    seconds_to_reset_tokens: float = 30.0,
) -> str:

  while True:
    try:
      raw_response = client.chat.completions.with_raw_response.create(
        model=llm,
        max_tokens=max_decode_steps,
        temperature=temperature,
        messages=[
          {'role': 'user', 'content': prompt},
        ]
      )
      completion = raw_response.parse()
      return completion.choices[0].message.content
    except openai.RateLimitError as e:
      print(f'{datetime.datetime.now()}: query_gpt_model: RateLimitError {e.message}: {e}')
      time.sleep(seconds_to_reset_tokens)
    except openai.APIError as e:
      print(f'{datetime.datetime.now()}: query_gpt_model: APIError {e.message}: {e}')
      print(f'{datetime.datetime.now()}: query_gpt_model: Retrying after 5 seconds...')
      time.sleep(5)

safety_settings=[
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"}
]

def query_gemini_model(
    prompt: str,
    llm: str = 'gemini-pro',
    client: Optional[Any] = None,
    retries: int = 10,
) -> str:
  del client
  model = genai.GenerativeModel(llm)
  generation_config={'temperature': 0.0}
  while True and retries > 0:
    try:
      response = model.generate_content(
        prompt,
        safety_settings=safety_settings,
        generation_config=generation_config
      )
      text_response = response.text.replace("**", "")
      return text_response
    except Exception as e:
      print(f'{datetime.datetime.now()}: query_gemini_model: Error: {e}')
      print(f'{datetime.datetime.now()}: query_gemini_model: Retrying after 5 seconds...')
      retries -= 1
      time.sleep(5)


def query_model(
    prompt: str,
    model_name: str = 'gemini-pro',
    client: Optional[Any] = None,
) -> str:
  model_type = model_name.split('-')[0]
  if model_type == "gpt":
    return query_gpt_model(prompt, llm=model_name, client=client)
  elif model_type == "gemini":
    return query_gemini_model(prompt, llm=model_name, client=client)
  else:
    raise ValueError('Unexpected model_name: ', model_name)


# Load QuALITY dataset
_ONE2ONE_FIELDS = (
    'article',
    'article_id',
    'set_unique_id',
    'writer_id',
    'source',
    'title',
    'topic',
    'url',
    'writer_id',
    'author',
)

quality_dev = []
with open('QuALITY.v1.0.1.htmlstripped.dev', 'r') as f:
  for line in f.readlines():
    j = json.loads(line)
    fields = {k: j[k] for k in _ONE2ONE_FIELDS}
    fields.update({
        'questions': [q['question'] for q in j['questions']],
        'question_ids': [q['question_unique_id'] for q in j['questions']],
        'difficults': [q['difficult'] for q in j['questions']],
        'options': [q['options'] for q in j['questions']],
    })

    fields.update({
        'gold_labels': [q['gold_label'] for q in j['questions']],
        'writer_labels': [q['writer_label'] for q in j['questions']],
      })

    quality_dev.append(fields)

# likely to succeed
index_map = {'A': 1, 'B': 9, 'C': 13, 'D': 200}


# Helper functions
all_lowercase_letters = string.ascii_lowercase  # "abcd...xyz"
bracketed_lowercase_letters_set = set(
    [f"({l})" for l in all_lowercase_letters]
)  # {"(a)", ...}
bracketed_uppercase_letters_set = set(
    [f"({l.upper()})" for l in all_lowercase_letters]
)  # {"(a)", ...}

choices = ['(A)', '(B)', '(C)', '(D)']

def get_index_from_symbol(answer):
  """Get the index from the letter symbols A, B, C, D, to extract answer texts.

  Args:
    answer (str): the string of answer like "(B)".

  Returns:
    index (int): how far the given choice is from "a", like 1 for answer "(B)".
  """
  answer = str(answer).lower()
  # extract the choice letter from within bracket
  if answer in bracketed_lowercase_letters_set:
    answer = re.findall(r".*?", answer)[0][1]
  index = ord(answer) - ord("a")
  return index

def count_words(text):
  """Simple word counting."""
  return len(text.split())

def quality_gutenberg_parser(raw_article):
  """Parse Gutenberg articles in the QuALITY dataset."""
  lines = []
  previous_line = None
  for i, line in enumerate(raw_article.split('\n')):
    line = line.strip()
    original_line = line
    if line == '':
      if previous_line == '':
        line = '\n'
      else:
        previous_line = original_line
        continue
    previous_line = original_line
    lines.append(line)
  return ' '.join(lines)


# ReadAgent (1) Episode Pagination
prompt_pagination_template = """
You are given a passage that is taken from a larger text (article, book, ...) and some numbered labels between the paragraphs in the passage.
Numbered label are in angeled brackets. For example, if the label number is 19, it shows as <19> in text.
Please choose one label that it is natural to break reading.
Such point can be scene transition, end of a dialogue, end of an argument, narrative transition, etc.
Please answer the break point label and explain.
For example, if <57> is a good point to break, answer with \"Break point: <57>\n Because ...\"

Passage:

{0}
{1}
{2}

"""

def parse_pause_point(text):
  text = text.strip("Break point: ")
  if text[0] != '<':
    return None
  for i, c in enumerate(text):
    if c == '>':
      if text[1:i].isnumeric():
        return int(text[1:i])
      else:
        return None
  return None


def quality_pagination(example,
                       model_name='gemini-pro',
                       client=None,
                       word_limit=600,
                       start_threshold=280,
                       max_retires=10,
                       verbose=True,
                       allow_fallback_to_last=True):
  article = example['article']
  title = example['title']
  text_output = f"[Pagination][Article {title}]" + '\n\n'
  paragraphs = quality_gutenberg_parser(article).split('\n')

  i = 0
  pages = []
  while i < len(paragraphs):
    preceding = "" if i == 0 else "...\n" + '\n'.join(pages[-1])
    passage = [paragraphs[i]]
    wcount = count_words(paragraphs[i])
    j = i + 1
    while wcount < word_limit and j < len(paragraphs):
      wcount += count_words(paragraphs[j])
      if wcount >= start_threshold:
        passage.append(f"<{j}>")
      passage.append(paragraphs[j])
      j += 1
    passage.append(f"<{j}>")
    end_tag = "" if j == len(paragraphs) else paragraphs[j] + "\n..."

    pause_point = None
    if wcount < 350:
      pause_point = len(paragraphs)
    else:
      prompt = prompt_pagination_template.format(preceding, '\n'.join(passage), end_tag)
      response = query_model(prompt=prompt, model_name=model_name, client=client).strip()
      pause_point = parse_pause_point(response)
      if pause_point and (pause_point <= i or pause_point > j):
        # process += f"prompt:\n{prompt},\nresponse:\n{response}\n"
        # process += f"i:{i} j:{j} pause_point:{pause_point}" + '\n'
        pause_point = None
      if pause_point is None:
        if allow_fallback_to_last:
          pause_point = j
        else:
          raise ValueError(f"prompt:\n{prompt},\nresponse:\n{response}\n")

    page = paragraphs[i:pause_point]
    pages.append(page)
    text_output += f"Paragraph {i}-{pause_point-1}: {page}\n\n" 
    i = pause_point
  text_output += f"\n\n[Pagination] Done with {len(pages)} pages"
  return pages, text_output


# ReadAgent (2) Memory Gisting
prompt_shorten_template = """
Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{}

"""

def quality_gisting(example, pages, model_name, client=None, word_limit=600, start_threshold=280, verbose=True):
  article = example['article']
  title = example['title']
  word_count = count_words(article)
  text_output = f"[Gisting][Article {title}], {word_count} words\n\n"

  shortened_pages = []
  for i, page in enumerate(pages):
    prompt = prompt_shorten_template.format('\n'.join(page))
    response = query_model(prompt, model_name, client)
    shortened_text = response.strip()
    shortened_pages.append(shortened_text)
    text_output += "[gist] page {}: {}\n\n".format(i, shortened_text)
  shortened_article = '\n'.join(shortened_pages)
  gist_word_count = count_words(shortened_article)
  text_output += '\n\n' + f"Shortened article:\n{shortened_article}\n\n" 
  output = copy.deepcopy(example)
  output.update({'title': title, 'word_count': word_count, 'gist_word_count': gist_word_count, 'shortened_pages': shortened_pages, 'pages': pages})
  text_output += f"\n\ncompression rate {round(100.0 - gist_word_count/word_count*100, 2)}% ({gist_word_count}/{word_count})"
  return output, text_output


# ReadAgent (3) Look-Up
prompt_lookup_template = """
The following text is what you remembered from reading an article and a multiple choice question related to it.
You may read 1 to 6 page(s) of the article again to refresh your memory to prepare yourselve for the question.
Please respond with which page(s) you would like to read.
For example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";
if your would like to read Page 7 and 12, respond with \"I want to look up Page [7, 12] to ...\";
if your would like to read Page 2, 3, 7, 15 and 18, respond with \"I want to look up Page [2, 3, 7, 15, 18] to ...\".
if your would like to read Page 3, 4, 5, 12, 13 and 16, respond with \"I want to look up Page [3, 3, 4, 12, 13, 16] to ...\".
DO NOT select more pages if you don't need to.
DO NOT answer the question yet.

Text:
{}

Question:
{}
{}

Take a deep breath and tell me: Which page(s) would you like to read again?
"""

prompt_answer_template = """
Read the following article and answer a multiple choice question.
For example, if (C) is correct, answer with \"Answer: (C) ...\"

Article:
{}

Question:
{}
{}

"""

def quality_parallel_lookup(example, model_name, client, verbose=True):
  preprocessed_pages = example['pages']
  article = example['article']
  title = example['title']
  word_count = example['word_count']
  gist_word_count = example['gist_word_count']
  pages = example['pages']
  shortened_pages = example['shortened_pages']
  questions = example['questions']
  options = example['options']
  gold_labels = example['gold_labels']  # numerical [1, 2, 3, 4]

  text_outputs = [f"[Look-Up][Article {title}] {word_count} words"]

  model_choices = []
  lookup_page_ids = []

  shortened_pages_pidx = []
  for i, shortened_text in enumerate(shortened_pages):
    shortened_pages_pidx.append("\n".format(i) + shortened_text)
  shortened_article = '\n'.join(shortened_pages_pidx)

  expanded_gist_word_counts = []

  for i, label in enumerate(gold_labels):
    # only test the first question for demo
    if i != 1:
      continue
    q = questions[i]
    text_output = f"question {i}: {q}" + '\n\n'
    options_i = [f"{ol} {o}" for ol, o in zip(choices, options[i])]
    text_output += "options: " + "\n".join(options_i)
    text_output += '\n\n'
    prompt_lookup = prompt_lookup_template.format(shortened_article, q, '\n'.join(options_i))

    page_ids = []

    response = query_model(prompt=prompt_lookup, model_name=model_name, client=client).strip()

    try: start = response.index('[')
    except ValueError: start = len(response)
    try: end = response.index(']')
    except ValueError: end = 0
    if start < end:
      page_ids_str = response[start+1:end].split(',')
      page_ids = []
      for p in page_ids_str:
        if p.strip().isnumeric():
          page_id = int(p)
          if page_id < 0 or page_id >= len(pages):
            text_output += f"Skip invalid page number: {page_id}\n\n"
          else:
            page_ids.append(page_id)

    text_output += "Model chose to look up page {}\n\n".format(page_ids)

    # Memory expansion after look-up, replacing the target shortened page with the original page
    expanded_shortened_pages = shortened_pages[:]
    if len(page_ids) > 0:
      for page_id in page_ids:
        expanded_shortened_pages[page_id] = '\n'.join(pages[page_id])

    expanded_shortened_article = '\n'.join(expanded_shortened_pages)
    expanded_gist_word_count = count_words(expanded_shortened_article)
    text_output += "Expanded shortened article:\n" + expanded_shortened_article + '\n\n'
    prompt_answer = prompt_answer_template.format(expanded_shortened_article, q, '\n'.join(options_i))

    model_choice = None
    response = query_model(prompt=prompt_answer, model_name=model_name, client=client)
    response = response.strip()
    for j, choice in enumerate(choices):
      if response.startswith(f"Answer: {choice}") or response.startswith(f"Answer: {choice[1]}"):
        model_choice = j+1
        break
    is_correct = 1 if model_choice == label else 0
    text_output += f"reference answer: {choices[label]}, model prediction: {choices[model_choice]}, is_correct: {is_correct}" + '\n\n'
    text_output += f"compression rate {round(100.0 - gist_word_count/word_count*100, 2)}% ({gist_word_count}/{word_count})" + '\n\n'
    text_output += f"compression rate after look-up {round(100.0 - expanded_gist_word_count/word_count*100, 2)}% ({expanded_gist_word_count}/{word_count})" + '\n\n'
    text_output += '\n\n'
    text_outputs.append(text_output)
    return text_outputs


# ReadAgent
def query_model_with_quality(
      index: int,
      model_name: str = 'gemini-pro',
      api_key: Optional[str] = None,
):
   # setup api key first
   client = None
   model_type = model_name.split('-')[0]
   if model_type == "gpt":
    # api_key = os.environ.get('OPEN_AI_KEY')
    client = openai.OpenAI(api_key=api_key)
   elif model_type == "gemini":
     # api_key = os.environ.get('GEMINI_API_KEY')
     genai.configure(api_key=api_key)

   example = quality_dev[index_map[index]]
   article = f"[Title: {example['title']}]\n\n{example['article']}"
   pages, pagination = quality_pagination(example, model_name, client)
   print('Finish Pagination.')
   example_with_gists, gisting = quality_gisting(example, pages, model_name, client)
   print('Finish Gisting.')
   answers = quality_parallel_lookup(example_with_gists, model_name, client)
   # return prompt_pagination_template, pagination, prompt_shorten_template, gisting, prompt_lookup_template, '\n\n'.join(answers)
   return article, pagination, gisting, '\n\n'.join(answers)


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

    [[website]](https://read-agent.github.io/)
    [[view on huggingface]](https://huggingface.co/spaces/ReadAgent/read-agent)
    [[arXiv]](https://arxiv.org/abs/2402.09727)
    [[OpenReview]](https://openreview.net/forum?id=OTmcsyEO5G)

    ![teaser](/file=./asset/teaser.png)

    The demo below showcases a version of the ReadAgent algorithm, which is nspired by how humans interactively read long documents.    
    We implement ReadAgent as a simple prompting system that uses the advanced language capabilities of LLMs to (1) decide what content to store together in a memory episode (**Episode Pagination**), (2) compress those memory episodes into short episodic memories called gist memories (**Memory Gisting**), and (3) take actions to look up passages in the original text if ReadAgent needs to remind itself of relevant details to complete a task (**Parallel Lookup and QA**)
    This demo can handle long-document reading comprehension tasks ([QuALITY](https://arxiv.org/abs/2112.08608); max 6,000 words) efficiently.

    To get started, you can choose an example article from QuALITY dataset.
    This demo uses Gemini API or OpenAI API so it requires the corresponding API key.
    """)
    with gr.Row():
        with gr.Column():
            llm_options = gr.Radio(llm_api_options, label="Backend LLM API", value='gemini-pro')
            llm_api_key = gr.Textbox(
              label="Paste your OpenAI API key (sk-...) or Gemini API key",
              lines=1,
              type="password",
            )
            # index = gr.Dropdown(list(range(len(quality_dev))), value=13, label="QuALITY Index")
            index = gr.Radio(['A', 'B', 'C', 'D'], label="Example Article", value='A')
            with gr.Row():
                example_article_a = gr.Textbox(
                  label="Example Article (A)",
                  lines=10,
                  value=f"[Title: {quality_dev[index_map['A']]['title']}]\n\n{quality_dev[index_map['A']]['article']}")
                example_article_a = gr.Textbox(
                  label="Example Article (B)",
                  lines=10,
                  value=f"[Title: {quality_dev[index_map['B']]['title']}]\n\n{quality_dev[index_map['B']]['article']}")
                example_article_a = gr.Textbox(
                  label="Example Article (C)",
                  lines=10,
                  value=f"[Title: {quality_dev[index_map['C']]['title']}]\n\n{quality_dev[index_map['C']]['article']}")
                example_article_a = gr.Textbox(
                  label="Example Article (D)",
                  lines=10,
                  value=f"[Title: {quality_dev[index_map['D']]['title']}]\n\n{quality_dev[index_map['D']]['article']}")
            button = gr.Button("Execute")
            choosen_article = gr.Textbox(label="Choosen Original Article", lines=20)
            # prompt_pagination = gr.Textbox(label="Episode Pagination Prompt Template", lines=5)
            pagination_results = gr.Textbox(label="(1) Episode Pagination", lines=20)
            # prompt_gisting = gr.Textbox(label="Memory Gisting Prompt Template", lines=5)
            gisting_results = gr.Textbox(label="(2) Memory Gisting", lines=20)
            # prompt_lookup = gr.Textbox(label="Parallel Lookup Prompt Template", lines=5)
            lookup_qa_results = gr.Textbox(label="(3) Parallel Lookup and QA", lines=20)

    button.click(
        fn=query_model_with_quality,
        inputs=[
           index,
           llm_options,
           llm_api_key,
        ],
        outputs=[
          # prompt_pagination, pagination_results,
          # prompt_gisting, gisting_results,
          # prompt_lookup, lookup_qa_results,
          choosen_article,
          pagination_results,
          gisting_results,
          lookup_qa_results,
        ]
    )


if __name__ == '__main__':
    demo.launch(allowed_paths=['./asset/teaser.png'])
