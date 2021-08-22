from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import torch


def get_probability(text):
    """Computes sentiment analysis confidence score for a text

    Args:
        string: text to be reviewed

    Returns:
        array: confidence score for positive case

    """
    
    inputs = tokenizer(text, return_tensors="pt",truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.forward(**inputs, return_dict=True)
    probabilities = (F.softmax(outputs.logits, dim=1))
    
    prob_positive = probabilities.data.numpy()[0][1]
    
    return prob_positive

model = DistilBertForSequenceClassification.from_pretrained('lschneidpro/distilbert_uncased_imdb')
tokenizer = DistilBertTokenizerFast.from_pretrained('lschneidpro/distilbert_uncased_imdb')

