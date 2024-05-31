device = "cpu"


def convert_examples_to_features(example_batch):
    # Preprocessing the dialogue
    input_encodings = tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

    # Preprocessing the summary considering them as target
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length = 128, truncation = True )

    # Returning the preprocessed data as dict which is a requirement
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }


def generate_batch_sized_chunks(list_of_elements, batch_size):
    
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements."""

    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def calculate_metric_on_test_ds(dataset,
                                metric,
                                model,
                                tokenizer,
                                batch_size=16,
                                device=device,
                                column_text="article",
                                column_summary="highlights"):

    # Generating batches of input and output
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    # Iterating through batches
    # Using tqdm for progress bar 
    for article_batch, target_batch in tqdm(
        # Zipping input and target
        zip(article_batches, target_batches), total=len(article_batches)):

        # Tokenizing a batch
        inputs = tokenizer(article_batch, max_length=1024,  truncation=True,
                        padding="max_length", return_tensors="pt")

        # Calculating the summary
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device),
                         length_penalty=0.8, num_beams=8, max_length=128)
        
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''

        # Finally, we decode the generated texts, replace the token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True) for s in summaries]

        decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    # Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score