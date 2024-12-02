
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def convert_example_to_feature(review):
  return tokenizer.encode_plus(review,
                add_special_tokens = True, # add [CLS], [SEP]
                max_length = max_length, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
              )

max_length = 512

batch_size = 1

def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label


def encode_examples(ds, limit=-1):
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []

  if (limit > 0):
      ds = ds.take(limit)

  for review, label in ds:

    bert_input = convert_example_to_feature(review)
    input_ids_list.append(bert_input['input_ids'])

    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])

  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def load_data():
  train = []
  test = []
  file = open("dataset/train_formatted.csv",encoding="utf8")
  for line in file:
    l = line.split(",")
    l[1] = l[1].replace("\n","")
    train.append([l[1],l[0]])
  file = open("dataset/test_formatted.csv",encoding="utf8")
  for line in file:
    l = line.split(",")
    l[1] = l[1].replace("\n","")
    test.append([l[1],l[0]])
  return [train,test]

[ds_train, ds_test] = load_data()
for batch in ds_test:
  batch[1] = tf.strings.to_number(batch[1])
for batch in ds_train:
  batch[1] = tf.strings.to_number(batch[1])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


learning_rate = 2e-5

number_of_epochs = 1

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
print("Compiling model..")
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print("Encoding Training Dataset..")
ds_train_encoded = encode_examples(ds_train[:50000]).shuffle(10000).batch(batch_size)

print("Encoding Testing Dataset.. ")
ds_test_encoded = encode_examples(ds_test[:10000]).batch(batch_size)

print("Training model..")
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_test_encoded)
print("Training Complete. \nSaving...")

model.save_pretrained("trained")