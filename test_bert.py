from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

# Load the model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("trained")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

def load_data():
    test = []
    with open("dataset/test_formatted.csv", encoding="utf8") as file:
        for line in file:
            l = line.strip().split(",")
            text, label = l[1], int(l[0])  # Ensure the label is an integer
            test.append([text, label])
    return test

def test(data):
   
    texts = [item[0] for item in data]
    true_labels = np.array([item[1] for item in data])
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="tf")

    logits = model.predict(inputs).logits  
    probabilities = tf.nn.softmax(logits, axis=1)  
    predicted_labels = tf.argmax(probabilities, axis=1).numpy() 

    # Calculate metrics
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))  
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))  
    tn = np.sum((predicted_labels == 0) & (true_labels == 0)) 
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))  

    correct = tp + tn
    accuracy = correct / len(true_labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

# Load the dataset
data = load_data()

# Run the evaluation
metrics = test(data)

# Print the results
print(f"""Accuracy: {metrics['accuracy']:.4f}
True Positives: {metrics['true_positives']}
False Positives: {metrics['false_positives']}
True Negatives: {metrics['true_negatives']}
False Negatives: {metrics['false_negatives']}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1_score']:.4f}
""")
