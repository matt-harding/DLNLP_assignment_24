import os
from sagemaker.huggingface.model import HuggingFaceModel
from datasets import load_dataset, get_dataset_split_names
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import numpy as np
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

    count = 0
    
    for index in tqdm(range(len(df)), total=len(df)):
        if count >= 100:
            break
        row = df.iloc[index]
        clean_question = TextPreprocessor.clean_text(row['question_title'] + ' ' + row['question_content'])
        new_row = pd.Series({"question_text": clean_question,
                "question_topic": row['topic']})
        clean_df = pd.concat([clean_df, pd.DataFrame([new_row])], ignore_index=True)
        count += 1
    
    clean_df.to_csv(os.path.join(DATA_DIRECTORY, f"clean_{split}.csv")) 


train_df = pd.read_csv(os.path.join(DATA_DIRECTORY, f"clean_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIRECTORY, f"clean_test.csv"))

'''
    Step 3: Baseline Bag-of-Words model
'''
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

'''
    Step 4: Zero-shot model
    Please note this code was run on a Sagemaker notebook so not all imports are explicitly declared
'''
# Hub model configuration <https://huggingface.co/models>
hub = {
  'HF_MODEL_ID':'facebook/bart-large-mnli', # model_id from hf.co/models
  'HF_TASK':'zero-shot-classification'                           # NLP task you want to use for predictions
}

# Create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,                                                # configuration for loading model from Hub
   role='REDACTED',                                              # IAM role with permissions to create an endpoint
   transformers_version="4.26",                             # Transformers version used
   pytorch_version="1.13",                                  # PyTorch version used
   py_version='py39',                                      # Python version used
)

# Deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.p3.16xlarge"
)

# Reverse the label_dic to map integer labels to text labels
reverse_label_dic = {value: key for key, value in label_dic.items()}

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

for index in tqdm(range(len(df)), total=len(df)):
    row = df.iloc[index]
    data_zero_shot = {
        "inputs": row['question_text'],
        "parameters": {
            "candidate_labels": list(label_dic.keys()),
            "multi_label": False
        }
    }
    prediction = predictor.predict(data_zero_shot)

    # Get the true label
    true_label_int = row['question_topic']
    true_label_text = reverse_label_dic[true_label_int]

    # Convert the true label to a one-hot encoded vector
    true_label_one_hot = [0] * len(label_dic)
    true_label_one_hot[true_label_int] = 1

    # Reorder the predicted probabilities to match the order of label_dic
    ordered_scores = [prediction['scores'][prediction['labels'].index(label)] for label in label_dic.keys()]

    # Append the true label and predicted probabilities to the lists
    true_labels.append(true_label_one_hot)
    predicted_probs.append(ordered_scores)


# Calculate the AUC score
auc_score = roc_auc_score(true_labels, predicted_probs, multi_class='ovr')
print(f"AUC score: {auc_score}")

