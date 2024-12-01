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
            if line[1] ==0:
                tn+=1
                correct+=1
            else:
                fn+=1
        else:
            if line[1] ==0:
                fp+=1
            else:
                correct+=1
                tp+=1
    return [correct,tp,tn,fp,fn]

def evaluate_model(model, dataset, threshold=0.5):
    """
    Evaluate the model on the dataset and calculate accuracy, TP, FP, TN, FN.
    
    Args:
        model: The trained Keras model.
        dataset: List of tuples (sample, label), where:
                 - sample: Input sample for the model.
                 - label: Ground truth label (0 or 1 for binary classification).
        threshold: Threshold to classify probabilities into binary predictions (default=0.5).
    
    Returns:
        A dictionary containing accuracy, TP, FP, TN, FN.
    """
    # Prepare data for prediction
    x_data = np.array([sample for sample, _ in dataset])
    y_true = np.array([label for _, label in dataset])
    
    # Get predictions
    y_pred_probs = model.predict(x_data)
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Compute metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
    fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
    tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
    fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives
    
    accuracy = (tp + tn) / len(y_true)  # Accuracy
    
    # Return metrics
    return  [correct,tp,tn,fp,fn]           
data = load_data()
correct,tp,tn,fp,fn = runtest(data)
accuracy = correct/len(data)
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