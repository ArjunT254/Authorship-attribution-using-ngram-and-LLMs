import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.lm.preprocessing import pad_both_ends, flatten,padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.lm import KneserNeyInterpolated
from nltk.util import ngrams
from nltk.lm.models import StupidBackoff
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import argparse
from huggingface_hub import notebook_login
import time
import torch
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import pandas as pd
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import random
from datasets import Dataset, DatasetDict

def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def create_dataset(author_dict):
  full_list_text=[]
  full_list_label=[]
  for file_path,author in author_dict.items():
    with open(file_path, 'r') as file:
        lines = file.readlines()
        text_data=''.join(lines)
    text_without_newlines = text_data.replace('\n', ' ')
    sentences = sent_tokenize(text_without_newlines)
    #list_data=list_format(sentences,author)

    full_list_text += sentences
    full_list_label += [ author for _ in range(len(sentences))]
  random.seed(42)
  combined = list(zip(full_list_text, full_list_label))
  random.shuffle(combined)
  text_data, author_data = zip(*combined)

  total_length = len(text_data)
  train_length = int(0.8 * total_length)
  val_length = int(0.1 * total_length)
  test_length = total_length - train_length-val_length

  # Split the original list
  train_list_text = text_data[:train_length]
  val_list_text = text_data[train_length:(train_length+val_length)]
  test_list_text = text_data[train_length+val_length:]

  train_list_label=author_data[:train_length]
  val_list_label=author_data[train_length:(train_length+val_length)]
  test_list_label=author_data[train_length+val_length:]

  test_data = {
    'text': test_list_text,
    'label': test_list_label}

  train_data = {
    'text': train_list_text,
    'label': train_list_label}

  val_data = {
    'text': val_list_text,
    'label': val_list_label}

  train_dataset = Dataset.from_dict(train_data)
  test_dataset = Dataset.from_dict(test_data)
  val_dataset=Dataset.from_dict(val_data)

  dataset_dict = DatasetDict({
      'train': train_dataset,
      'test':test_dataset,
      'validation': val_dataset})


  return dataset_dict

def train_test_split(file_path,total_text):
    with open(file_path, 'r',encoding='utf-8') as file:
      text = file.read()
    # Tokenize the text into words
    custom_punctuation = "!@#$%^&*()_+{}[]|\\:;\"',“/<>’”‘-?"
    words = word_tokenize(text)

    # Convert words to lowercase
    words = [word.lower() for word in words]

    # Remove punctuation
    words = [word for word in words if word not in custom_punctuation]

    #Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    preprocessed_text = " ".join(words)
    total_lines = len(preprocessed_text)
    ninety_percent = int(total_lines * 0.9)
    ten_percent = int(total_lines * 0.1)
    first_90_percent = ''.join(preprocessed_text[:ninety_percent])
    last_10_percent =''.join(preprocessed_text[-ten_percent:])
    total_text+=first_90_percent+' '+last_10_percent
    return first_90_percent,last_10_percent,total_text


def train_text(text_data):
  sentences = sent_tokenize(text_data)
  n=2
  # Tokenize each sentence into words
  tokenized_text = [word_tokenize(sentence) for sentence in sentences]
  
  train, vocab = padded_everygram_pipeline(n, tokenized_text)


  lm = KneserNeyInterpolated(order=2)
  lm.fit(train, vocab)


  return lm

def classify(test_data,author):
  sample_count=0
  auth_count=0
  sentences_test = sent_tokenize(test_data)
  i=0
  while i <len(sentences_test):
    sentence=sentences_test[i]
    sample_count += 1
    temp=list(pad_both_ends(word_tokenize(sentence), n=2))
    padded_bigrams = [(temp[i], temp[i+1]) for i in range(1, len(temp)-1, 2)]
    perp1 = lm_austen.perplexity(padded_bigrams)
    perp2 = lm_dickens.perplexity(padded_bigrams) 
    perp3 = lm_tolstoy.perplexity(padded_bigrams)
    perp4 = lm_wilde.perplexity(padded_bigrams)
    perp=[perp1,perp2,perp3,perp4]
    min_element = min(perp)
    min_index = perp.index(min_element)
    i+=10
    if min_index==author:
      auth_count += 1
    else:
      print('Incorrect Prediction:',sentence)
      print('Actual Author:',id2label[author],' Predicted Author:',id2label[min_index])
  return sample_count,auth_count

def main(args):
    model=args.approach
    if(args.test==None):
      key=False
    else:
      key=True
      print('Test set extracting')
    file_path_auten='austen_utf8.txt'
    file_path_dickens='dickens_utf8.txt'
    file_path_tolstoy='tolstoy_utf8.txt'
    file_path_wilde='wilde_utf8.txt'
    if model=='generative':

      txt=''
      train_text_austen,test_text_austen,txt= train_test_split(file_path_auten,txt)
      train_text_dickens,test_text_dickens,txt= train_test_split(file_path_dickens,txt)
      train_text_tolstoy,test_text_tolstoy,txt= train_test_split(file_path_tolstoy,txt)
      train_text_wilde,test_text_wilde,txt= train_test_split(file_path_wilde,txt)


      lm_austen=train_text(train_text_austen)
      lm_dickens=train_text(train_text_dickens)
      lm_tolstoy=train_text(train_text_tolstoy)
      lm_wilde=train_text(train_text_wilde)


      def classify(test_data,author):
        if key==False:
          print('Training....')
        sample_count=0
        auth_count=0
        sentences_test = sent_tokenize(test_data)
        i=0
        while i <len(sentences_test):
          sentence=sentences_test[i]
          sample_count += 1
          temp=list(pad_both_ends(word_tokenize(sentence), n=2))
          padded_bigrams = [(temp[i], temp[i+1]) for i in range(1, len(temp)-1, 2)]
          perp1 = lm_austen.perplexity(padded_bigrams)
          perp2 = lm_dickens.perplexity(padded_bigrams) 
          perp3 = lm_tolstoy.perplexity(padded_bigrams)
          perp4 = lm_wilde.perplexity(padded_bigrams)
          perp=[perp1,perp2,perp3,perp4]
          min_element = min(perp)
          min_index = perp.index(min_element)
          i+=10
          if min_index==author:
            auth_count += 1
          if key==True:
            print("Predicted Author")
            print(id2label[min_index])
          # else:                                   #Used to get the incorrect predictions
          #   print('Incorrect Prediction:',sentence)
          #   print('Actual Author:',id2label[author],' Predicted Author:',id2label[min_index])
        return sample_count,auth_count

      id2label = {0:"austen",1:"dickens",2:"tolstoy",3:"wilde"}

      sample_count1,auth_count1=classify(test_text_austen,0)
      sample_count2,auth_count2=classify(test_text_dickens,1)
      sample_count3,auth_count3=classify(test_text_tolstoy,2)
      sample_count4,auth_count4=classify(test_text_wilde,3)
      
      if key==False:

        print('Austen',auth_count1*100/sample_count1)
        print('Dicken',auth_count2*100/sample_count2)
        print('Tolstoy',auth_count3*100/sample_count3)
        print('Wilde',auth_count4*100/sample_count4)

    elif model=='discriminative':
     print('Discriminative')
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
     token='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'
     author_dict={file_path_auten:0,file_path_dickens:1,file_path_tolstoy:2,file_path_wilde:3}
     #notebook_login(token=token,write_permission=True)

     author_dataset=create_dataset(author_dict)

     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
     tokenized_dataset = author_dataset.map(preprocess_function, batched=True)
     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
     accuracy = evaluate.load("accuracy")

     id2label = {0:"austen",1:"dickens",2:"tolstoy",3:"wilde"}
     label2id = {"austen":0,"dickens":1,"tolstoy":2,"wilde":3}
     model_name='distilbert-base-uncased'
     num_labels=4

     model = AutoModelForSequenceClassification.from_pretrained(
     model_name,
     num_labels=num_labels,
     id2label=id2label,
     label2id=label2id
     )

     training_args = TrainingArguments(
     output_dir='Ngram_classifier',
     learning_rate=2e-5,
     per_device_train_batch_size=32,
     per_device_eval_batch_size=32,
     num_train_epochs=2,
     weight_decay=0.01,
     evaluation_strategy='epoch',
     save_strategy='epoch',
     load_best_model_at_end=True,
     )
     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=tokenized_dataset['train'],
         eval_dataset=tokenized_dataset['validation'],
         tokenizer=tokenizer,
         data_collator=data_collator,
         compute_metrics=compute_metrics,
     )

     test=author_dataset['validation']

     if key==False:
       trainer.train()

       trainer.push_to_hub()
       output_df=pd.DataFrame()
       model_name='Stonekraken/Ngram_classifier'
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForSequenceClassification.from_pretrained(model_name)

       total_austen=0
       correct_austen=0
       total_wilde=0
       correct_wilde=0
       total_dickens=0
       correct_dickens=0
       total_tolstoy=0
       correct_tolstoy=0

       for i in range(len(test)):
         text=test['text'][i]
         inputs = tokenizer(text,return_tensors="pt")
         with torch.no_grad():
           logits = model(**inputs).logits
         predicted_class_id = logits.argmax().item()

         if test['label'][i]==0:
           total_austen+=1
           if predicted_class_id==0:
             correct_austen+=1
         elif test['label'][i]==1:
           total_wilde+=1
           if predicted_class_id==1:
             correct_wilde+=1
         elif test['label'][i]==2:
           total_dickens+=1
           if predicted_class_id==2:
             correct_dickens+=1
         elif test['label'][i]==3:
           total_tolstoy+=1
           if predicted_class_id==3:
             correct_tolstoy+=1

       print('Austen Accuracy',correct_austen*100/total_austen)
       print('wilde Accuracy',correct_wilde*100/total_wilde)
       print('dickens Accuracy',correct_dickens*100/total_dickens)
       print('tolstoy Accuracy',correct_tolstoy*100/total_tolstoy)

     if key==True:
       test=author_dataset['test']
       model_name='Stonekraken/Ngram_classifier'
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForSequenceClassification.from_pretrained(model_name)
       for i in range(len(test)):
         text=test['text'][i]
         inputs = tokenizer(text,return_tensors="pt")
         with torch.no_grad():
           logits = model(**inputs).logits
         predicted_class_id = logits.argmax().item()
         print('Predicted Author for line',i+1,' :',model.config.id2label[predicted_class_id])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Author Classification')
    parser.add_argument('author_list', type=str, help='File containing list of author files')
    parser.add_argument('-approach', choices=['generative', 'discriminative'], required=True)
    parser.add_argument('-test', type=str, help='Test file for classification')
    args = parser.parse_args()
    main(args)