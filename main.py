import os
from datasets import load_dataset, get_dataset_split_names
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from Utils import TextPreprocessor

# https://huggingface.co/datasets/yahoo_answers_topics
DATASET = 'yahoo_answers_topics'
DATA_DIRECTORY = 'Datasets'
TOPIC_DICTIONARY = {
    0: 'Society & Culture',
    1: 'Science & Mathematics',
    2: 'Health',
    3: 'Education & Reference',
    4: 'Computers & Internet',
    5: 'Sports',
    6: 'Business & Finance',
    7: 'Entertainment & Music',
    8: 'Family & Relationships',
    9: 'Politics & Government'
}

'''
    Step 1: Fetch dataset & Preprocess
'''
# for split in get_dataset_split_names(DATASET):
#     # load the split of the dataset
#     dataset = load_dataset(DATASET, split=split)
#     # save the split of the dataset to the download directory as json linzes file
#     dataset.to_json(os.path.join(DATA_DIRECTORY, f"{split}.jsonl"))

'''
    Step 2: Preprocess data
'''
# for split in get_dataset_split_names(DATASET):
#     print('Processing ' + split)
#     df = pd.read_json('Datasets/' + split + '.jsonl', lines=True)

#     clean_df = pd.DataFrame(columns=["question_text", "question_topic"])

#     count = 0
    
#     for index in tqdm(range(len(df)), total=len(df)):
#         if count >= 100:
#             break
#         row = df.iloc[index]
#         clean_question = TextPreprocessor.clean_text(row['question_title'] + ' ' + row['question_content'])
#         new_row = pd.Series({"question_text": clean_question,
#                 "question_topic": row['topic']})
#         clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
#         count += 1
    
#     clean_df.to_csv(os.path.join(DATA_DIRECTORY, f"clean_{split}.csv")) 


'''
    Step 3: Baseline Bag-of-Words model
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import numpy as np

train_df = pd.read_csv(os.path.join(DATA_DIRECTORY, f"clean_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIRECTORY, f"clean_test.csv"))

vectorizer = CountVectorizer(max_features=5000)  # Limit number of features to stop dataset being too sparse
train_question_dtm = vectorizer.fit_transform(train_df['question_text'])  # create Document-Term Matrix (DTM)
train_topics = train_df['question_topic']  # No need to convert to NumPy array
test_question_dtm = vectorizer.transform(test_df['question_text'])
test_topics = test_df['question_topic']  # No need to convert to NumPy array

# Naive Bayes Model
naive_bayes_model = MultinomialNB().fit(train_question_dtm, train_topics)

# Calculate ROC-AUC score
y_preds = naive_bayes_model.predict_proba(test_question_dtm)
aucs = []
for class_idx in range(10):  # Assuming 10 classes (0-9)
    aucs.append(roc_auc_score(test_topics == class_idx, y_preds[:, class_idx], multi_class='ovr'))
mean_auc = np.mean(aucs)

print(f"Mean ROC-AUC score: {mean_auc:.3f}")

# Sanity check
####################
# validate_question_dtm = vectorizer.transform(['lose lot money stock one year carry negative capital gain forward work many year carry forward'])

# validate_predict = naive_bayes_model.predict_proba(validate_question_dtm)

# max_prob_index = np.argmax(validate_predict[0])

# print("Predict:" + TOPIC_DICTIONARY[max_prob_index])
