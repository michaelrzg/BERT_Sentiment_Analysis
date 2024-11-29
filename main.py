# BERT for sentiment analysis of amazon reivews
# michael rizig

from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf


# returns data list of tuples (review,label)
def load_data():
  train = []
  test = []
  file = open("dataset/train_formatted.csv")
  for line in file:
    l = line.split(",")
    l[1] = l[1].replace("\n","")
    train.append((l[1],l[0]))
  file = open("dataset/test_formatted.csv")
  for line in file:
    l = line.split(",")
    l[1] = l[1].replace("\n","")
    test.append((l[1],l[0]))
  return [train,test]

data = load_data()