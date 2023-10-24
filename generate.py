# utils.py

import pandas as pd
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def corpus_directory_to_dataframe(corpus_directory, scope='file'):
    """
    Returns a DataFrame with all corpus text and corresponding filenames.
    :param corpus_directory: directory that contains all corpus files
    :param scope: options are 'file' or 'lines'; level at which to label and return text
    :return: a DataFrame with all text and the file it came from
    """
    corpus_lists = []
    for file in os.listdir(corpus_directory):
        with open(corpus_directory + "/" + file, errors='ignore', encoding='utf8') as current_file:
            if scope == 'file':
                corpus_lists.append([current_file.read(), str(corpus_directory + "/" + file)])
            elif 'line' in scope:
                for line in current_file.readlines():
                    corpus_lists.append([line, str(corpus_directory + "/" + file)])
    corpus = pd.DataFrame(corpus_lists, columns=['text', 'file'])
    return corpus


def corpus_dataframe_to_list(corpus_dataframe):
    """Converts corpus DataFrame to a list of texts/strings."""
    return corpus_dataframe['text'].to_list()


def tokenize_corpus(corpus_list, pretrained_model='gpt2', truncation_setting=True, padding_level='longest',
                    maximum_length=512, return_tensors_setting='pt'):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    # Add a new pad token to the tokenizer
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_dataset = tokenizer.batch_encode_plus(
        corpus_list,
        truncation=truncation_setting,
        padding=padding_level,
        max_length=maximum_length,
        return_tensors=return_tensors_setting
    )
    return tokenized_dataset


def train_llm(tokenized_dataset, pretrained_model='gpt2', batch_size=8, num_epochs=25, sequence_length=512):
    # define and configure model
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    # prepare the input tensors for fine-tuning
    input_ids = tokenized_dataset['input_ids']
    attention_mask = tokenized_dataset['attention_mask']
    # define and configure optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(num_epochs):
        for i in range(0, len(input_ids), batch_size):
            # Prepare the batch
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_mask = attention_mask[i:i + batch_size]

            # Check if the batch size matches the sequence length
            if len(batch_input_ids) != sequence_length:
                continue  # Skip this batch if the size doesn't match

            # Verify the shape of the input tensors
            assert batch_input_ids.shape == (batch_size, sequence_length)
            assert batch_attention_mask.shape == (batch_size, sequence_length)

            # Forward pass
            outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_input_ids)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress
            print(f"Epoch: {epoch + 1}, Batch: {i // batch_size + 1}, Loss: {loss.item()}")
    return model


def generate_text(model, prompt, pretrained_model='gpt2', number_responses=5, max_response_length=10):
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate candidate formulae
    output = model.generate(
        input_ids,
        max_length=max_response_length,
        num_return_sequences=number_responses,
        temperature=1.0,
        num_beams=number_responses,
        early_stopping=True
    )

    # Process and store the generated candidates
    generated_candidates = []
    for i in range(output.shape[0]):
        generated_candidates.append([prompt, tokenizer.decode(output[i], skip_special_tokens=True)])
    return generated_candidates


def generate_text_from_prompts(model, prompt_file, output_file, pretrained_model_param, number_responses_param,
                               max_response_length_param):
    """
    Generates text for each prompt and writes a .csv file with the results.
    :param number_responses_param: number of responses to generate for each prompt
    :param pretrained_model_param: the pretrained model type to use
    :param max_response_length_param: maximum response length for generated text
    :param model: trained model to use for generation
    :param prompt_file: filename where all the prompts live, each prompt on one line
    """
    with open(prompt_file, errors='ignore', encoding='utf8') as current_file:
        prompts = current_file.readlines()
    all_generated_text = []
    generated_text_columns = ['prompt', 'generated_text']
    for prompt in prompts:
        generated_from_prompt = generate_text(model, prompt, pretrained_model=pretrained_model_param,
                                              number_responses=number_responses_param,
                                              max_response_length=max_response_length_param)
        for generated in generated_from_prompt:
            all_generated_text.append(generated)
    generated_df = pd.DataFrame(all_generated_text, columns=generated_text_columns)
    print(generated_df)
    generated_df.to_csv(output_file)


def main():
    user = os.getenv('USER')
    corpus_dir = '/scratch/users/{}/llm_test/corpus/'.format(user)
    prompt_file = '/scratch/users/{}/llm_test/prompt.txt'.format(user)
    output_file = '/scratch/users/{}/llm_test/output.csv'.format(user)
    print("Expecting to find corpus files in " + corpus_dir)
    print("Expecting to find prompt file at " + prompt_file)

    pretrained_gpt_model = 'gpt2'
    # tokenization settings
    truncation_setting_tokenization = True
    padding_level_tokenization = 'longest'
    max_length_sequence = 512
    return_tensors = 'pt'
    # model settings
    model_batch_size = 8
    number_epochs = 25
    sequence_length_model = 512
    # generate settings
    number_responses_to_generate = 5
    maximum_response_length = 50

    print(
        "Training model with the following parameters. To change these parameters, modify the main function of "
        "generate.py.")
    print()
    print("Pretrained GPT model type: ", pretrained_gpt_model)
    print("Tokenization truncation setting: ", str(truncation_setting_tokenization))
    print("Tokenization padding level: ", padding_level_tokenization)
    print("Maximum tokenized sequence length: ", str(max_length_sequence))
    print("Return tensors tokenization setting: ", return_tensors)
    print("Model batch size: ", str(model_batch_size))
    print("Number training epochs: ", str(number_epochs))
    print("Training sequence length: ", str(sequence_length_model))
    print("Number of responses to generate (per prompt): ", str(number_responses_to_generate))
    print("Maximum length of generated response: ", str(maximum_response_length))
    print()
    print("Unpacking and tokenizing dataset...")
    corpus_df = corpus_directory_to_dataframe('corpus', scope='lines')
    corpus_ls = corpus_dataframe_to_list(corpus_df)

    tokenized_dataset = tokenize_corpus(corpus_ls, pretrained_model=pretrained_gpt_model,
                                        truncation_setting=truncation_setting_tokenization,
                                        padding_level=padding_level_tokenization,
                                        maximum_length=max_length_sequence, return_tensors_setting=return_tensors)
    print("Training model...")
    model = train_llm(tokenized_dataset, pretrained_model=pretrained_gpt_model, batch_size=model_batch_size,
                      num_epochs=number_epochs, sequence_length=sequence_length_model)
    print("Generating text from prompts...")

    generate_text_from_prompts(model, prompt_file, output_file, pretrained_gpt_model, number_responses_to_generate,
                               maximum_response_length)


if __name__ == "__main__":
    main()
