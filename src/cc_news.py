from datasets import load_dataset
from gensim.utils import simple_preprocess

def dataset_cc_news():
    """Method to get cc_news dataset that is provided by Huggingface.

    Return:
    text -- cc_news dataset
    """
    #Load cc_news dataset from datasets provided by Huggingface
    dataset = load_dataset('cc_news', split = 'train')

    #Keep only the text from the dataset
    text = dataset['text']
    return text


def data_preprocess(df):
    """Method to preprocess the text by applying a simple_preprocess function provided by pandas.
       Given a text, it gets tokenized and removes punctuations and special characters

    Keyword arguments:
    df -- dataset, dataframe

    Return:
    proc_text -- processed dataset, text
    """
    #apply simple preprocessing(tokenization, removal of punctuations and special characters) on text
    proc_text = df[0].apply(simple_preprocess)
    return proc_text
