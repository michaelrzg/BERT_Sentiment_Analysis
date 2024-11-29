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

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label
# remember when we talked about the BertTokenizer
def encode(review):
  return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = 512, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )


def encode_all(dataset):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    for review, label in dataset:

        bert_input = encode(review)
        input_ids_list.append(bert_input['input_ids'])

        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])

    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)


# main ---------------------------------------------------------------------------------------------------

# initilize bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
learning_rate = 2e-5
number_of_epochs = 1
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
batch_size = 6
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# load data
print("Loading data..")
data = load_data()

# encode train dataset
ds_train_encoded = encode_all(data[0]).shuffle(10000).batch(batch_size)

# encode test dataset
ds_test_encoded = encode_all(data[1]).batch(batch_size)

# Finally, we are training our model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)

# Example of testing, you will need to modify that part to accomodate the full test
test_sentence = "This is a really good movie. I loved it and will watch again"

# don't forget to tokenize your test inputs
predict_input = tokenizer.encode(test_sentence, truncation=True, padding=True, return_tensors="tf")

tf_output = model.predict(predict_input)[0]
tf_prediction = tf.nn.softmax(tf_output, axis=1)

labels = ['Negative','Positive'] #(0:negative, 1:positive)
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
print(labels[label[0]])