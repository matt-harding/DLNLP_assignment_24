import os
from datasets import load_dataset, get_dataset_split_names
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from Utils import TextPreprocessor

# https://huggingface.co/datasets/yahoo_answers_topics
DATASET = 'yahoo_answers_topics'
DATA_DIRECTORY = 'Datasets'

'''
    Step 1: Fetch dataset & Preprocess
'''
for split in get_dataset_split_names(DATASET):
    # load the split of the dataset
    dataset = load_dataset(DATASET, split=split)
    # save the split of the dataset to the download directory as json linzes file
    dataset.to_json(os.path.join(DATA_DIRECTORY, f"{split}.jsonl"))

'''
    Step 2: Preprocess data
'''
for split in get_dataset_split_names(DATASET):
    print('Processing ' + split)
    df = pd.read_json('Datasets/' + split + '.jsonl', lines=True)

    clean_df = pd.DataFrame(columns=["question_text", "question_topic"])
    
    for index in tqdm(range(len(df)), total=len(df)):
        row = df.iloc[index]
        clean_question = TextPreprocessor.clean_text(row['question_title'] + ' ' + row['question_content'])
        new_row = pd.Series({"question_text": clean_question,
                "question_topic": row['topic']})
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
    
    clean_df.to_csv(os.path.join(DATA_DIRECTORY, f"clean_{split}.csv")) 
