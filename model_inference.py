import pandas as pd
import numpy as np


def generate_predictions(model, tokenizer, row:pd.Series, device="cpu", num_return_sequences=5, verbose=True):
    """
    Given a row of test dataframe, generate predictions
    Args:
        model (transformers.AutoAdapterModel)
        tokenizer (transformers.AutoTokenizer or T5Tokenizer)
        row (pd.Series): a row of a pd.DataFrame
        device (str): device the model is on "cpu" or "cuda" 
        num_return_sequences (int): number of predictions to make
        verbose (bool): whether to print outputs
    Returns:
        list of prediction strings
    """

    if verbose is True:
        print('Input: ', row['in'])
    
    to_model = 'paraphrase: ' + row['in']

    # sentence = 'paraphrase: We should go to the movies today because it is raining.'
    encoding = tokenizer(row['in'], return_tensors="pt")
    
    # Push tensors to device
    for key, value in encoding.items():
        encoding[key] = value.to(device)
    
    # Generate
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    out = model.generate(input_ids=input_ids, do_sample=True, attention_mask=attention_masks, max_length=512,
                        top_k=250, top_p=0.99, early_stopping=True, num_return_sequences=num_return_sequences)
    
    # Decode generated predictions
    predictions = []
    for p in out:
        pred = tokenizer.decode(p, skip_special_tokens=True)
        predictions.append(pred)
        if verbose is True:
            print('Prediction: ', pred)
        
    if verbose is True:
        print('Expected: ', row['expected'], "\n")

    return predictions


def inference(model, tokenizer, df_test, num_examples=1, device="cpu", verbose=True) -> pd.DataFrame:
    """
    Generate and print output for a few of the test dataset 
    Args:
        model (transformers.AutoAdapterModel)
        tokenizer (transformers.AutoTokenizer or T5Tokenizer)
        df_test (pd.DataFrame)
        num_examples (int): number of test sentences to generate paraphrase for
        device (str): device the model is on "cpu" or "cuda"
    Returns:
        df_test_head_copy (pd.DataFrame): a copy of the first num_examples rows of the df_test
            dataframe with the predictions appended as additional columns pred1, pred2, pred3, ...
    """

    num_return_sequences = 5
    
    df_test_head_copy = df_test.head(num_examples).copy(deep=True)

    predictions = df_test_head_copy.apply(lambda row: 
        generate_predictions(model, tokenizer, row, 
            device=device, num_return_sequences=num_return_sequences, 
            verbose=verbose
        ), 
        axis=1
    )

    df_test_head_copy[[f"pred{i}" for i in range(1,num_return_sequences+1)]] = pd.DataFrame(np.vstack(predictions))

    return df_test_head_copy

