import re
import pandas as pd

# Text preprocessing libraries
from bs4 import BeautifulSoup
from urlextract import URLExtract
import phonenumbers


train_ratio = 0.85

with open("./labeled_CEAS_08_preprocessed_shuffled.csv", "r", encoding="utf-8") as fh:
    df = pd.read_csv(fh)

# Split test set (15%)
train_split_index = int(train_ratio*len(df))
train_df = df.iloc[:train_split_index, :].copy()
test_df = df.iloc[train_split_index:, :].copy()

test_df = test_df.drop("label", axis=1)



url_extractor = URLExtract()

def preprocess_text(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text(separator=' ')

    # Replace URLs using urlextract library
    urls = url_extractor.find_urls(text)
    for url in urls:
        text = text.replace(url, '<URL>')

    # Replace email addresses with special token
    text = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '<EMAIL>', text)

    # Replace phone numbers using phonenumbers library
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, None):
            text = text.replace(match.raw_string, '<PHONE>')
    except:
        pass

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply preprocessing to each text column
print("Preprocessing text data")
for col in test_df.columns:
    train_df[col] = train_df[col].fillna('').astype(str)
    test_df[col] = test_df[col].fillna('').astype(str)

    train_df[col] = train_df[col].apply(preprocess_text)
    test_df[col] = test_df[col].apply(preprocess_text)




with open("../baselines/train.csv", "w", newline='', encoding="utf-8") as fh:
    train_df.to_csv(fh, index=False, na_rep='')

with open("../baselines/test.csv", "w", newline='', encoding="utf-8") as fh:
    test_df.to_csv(fh, index=False, na_rep='')