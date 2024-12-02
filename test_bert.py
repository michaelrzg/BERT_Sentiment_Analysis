from transformers import TFBertForSequenceClassification, BertTokenizer
from transformers import BertTokenizer
import tensorflow as tf
# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("trained")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def load_data():
    test = []
    file = open("dataset/test_formatted.csv",encoding="utf8")
    for line in file:
        l = line.split(",")
        l[1] = l[1].replace("\n","")
        test.append([l[1],l[0]])
    return test

def runtest(data):
    fp=0
    fn=0
    tp=0
    tn=0
    correct=0
    for line in data:
        input = tokenizer.encode(line[0], truncation=True, padding=True, return_tensors="tf")
        output = model.predict(input)[0]
        prediction = tf.nn.softmax(output, axis=1)
        label = tf.argmax(prediction, axis=1)
        label = label.numpy()
        if label[0] == 0:
            if int(line[1]) ==0:
                tn+=1
                correct+=1
            else:
                fn+=1
        else:
            if int(line[1]) ==0:
                fp+=1
            else:
                correct+=1
                tp+=1
    return [correct,tp,tn,fp,fn]



data = load_data()
correct,tp,tn,fp,fn = runtest(data[:1000])
accuracy = correct/1000
precision = tp / (tp+fp)
recall = tp / (tp+fn)
f1score = (2 * precision * recall) / (precision + recall) 

print(f"""Accuracy: {accuracy}
True Positives: {tp}
False Positives: {fp}
True Negatives: {tn}
False Negatives: {fn}
Precision: {precision}
Recall: {recall}
F1 Score: {f1score}
""")