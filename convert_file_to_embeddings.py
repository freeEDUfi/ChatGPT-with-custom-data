################################################################################
### Step 1 Imports + Functions (the Step numbering scheme is kept as reference to the original code from openai tutorial)
################################################################################
import os.path

import pandas as pd
import tiktoken
import openai

import PyPDF2
import re
from nltk.tokenize import sent_tokenize
from openai.embeddings_utils import get_embedding

################################################################################
### Utility Functions
################################################################################
def read_pdf_file(pdf_file):
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        text = ""

        for page in range(num_pages):
            page_obj = pdf_reader.pages[page]
            text += page_obj.extract_text()

    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode('utf-8').decode('unicode_escape')

    return text

def read_text_file(text_file):
    with open(text_file, 'r') as file:
        text = file.read()

    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_other_artifacts(serie):
    serie = serie.str.replace('\\x00', 'fi', regex=False)
    serie = serie.str.replace('\\ue934', ' ', regex=False)
    serie = serie.str.replace('\\ue974', ' ', regex=False)
    return serie

def remove_newlines(serie):
    serie = serie.str.replace("\\'", "'", regex=False)
    serie = serie.str.replace('\\n', ' ', regex=False)
    serie = serie.str.replace('  ', ' ', regex=False)
    serie = serie.str.replace('  ', ' ', regex=False)
    return serie

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = 500):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

################################################################################
### Step 1: Sequence Start
# Get text from input files, append whole text as a list item
################################################################################
def generate_embeddings(api_key, input, is_text_file=False, is_pdf_file=False):
    openai.api_key = api_key

    if is_text_file:
        text = [('txt_file', read_text_file(input))]
    elif is_pdf_file:
        text = [('pdf_file', read_pdf_file(input))]
    else:
        text = [('raw_text', input)]
        
    # Create a dataframe from the list of texts
    df = pd.DataFrame(text, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    # This is for pdf files
    df['text'] = remove_other_artifacts(df.text)

    ################################################################################
    ### Step 2 Tokenize the Scraped text
    ################################################################################
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    ################################################################################
    ### Step 3 Splitting/shortening the text in the scraped files
    ################################################################################
    # max_tokens controls how long the 'text' value in each row of the embeddings dataframe will be, effectively affecting the total number of rows in the df
    max_tokens = 500

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():
        print(row[1])
        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], max_tokens)

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

    ################################################################################
    ### Step 4 Remake the Dataframe with the text value 'split-to-many'/shortened, resulting in more rows with shorter texts
    ################################################################################

    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    ################################################################################
    ### Step 5 Run each row of the data frame into openAI embeddings generator using ada-002
    ################################################################################
    # Note that you may run into rate limit issues depending on how many files you try to embed
    # Please check out our rate limit guide to learn more on how to handle this: https://platform.openai.com/docs/guides/rate-limits

    print("Generating embeddings...")
    import time

    # Initialize an empty list to store the embeddings
    embeddings = []

    # OpenAI API has a request limit of 60 per minute, so add a delay of 30s every 50 rows/requests
    # Loop through each row of the dataframe
    for i, row in df.iterrows():
        # Call the OpenAI API to get the embedding for the current row
        embedding = get_embedding(text=row['text'], engine='text-embedding-ada-002')

        # Append the embedding to the list
        embeddings.append(embedding)

        # Add a delay after every 30 rows
        if i % 50 == 0 and i > 0:
            # print(f"row index = {i}, sleeping for 30s..")
            time.sleep(30)
            # print("Sleep finished. Continuing..")

    # Assign the embeddings to a new column in the dataframe
    df['embeddings'] = embeddings
    
    # print(f"Saving embeddings to {embeddings_file}..")
    # df.to_json(embeddings_file)

    print("Done.. -------------------------------\n")
    return df

# generate_embeddings("Course_PDFs")