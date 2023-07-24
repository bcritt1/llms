import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Input

# Txt directory
import os
corpusdir = '/farmshare/learning/data/emerson/'.format(user)
corpus = []
for infile in os.listdir(corpusdir):
  with open(corpusdir+infile, errors='ignore') as fin:
    corpus.append(fin.read())


#list input
#corpus = [
#    "Sing, O Muse, of the wrath of Achilles",
#    "Tell me, O Muse, of that hero's journey",
#    "Rosy-fingered Dawn appeared on the horizon",
#    "Swift-footed Achilles unleashed his rage",
#    "Hector, breaker of horses, fought valiantly",
#    "The wine-dark sea stretched out before them",
#    "Aegis-bearing Zeus thundered from the heavens",
#    "Noble Odysseus, master of stratagems",
#    "The shield of Achilles shimmered in the sun",
#    "Swift ships sailed across the wine-dark sea"
#]


# Tokenization

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Add a new pad token to the tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

tokenized_dataset = tokenizer.batch_encode_plus(
    corpus,
    truncation=True,
    padding='longest',
    max_length=512,
    return_tensors='pt'
)


# Model Configuration

model = GPT2LMHeadModel.from_pretrained('gpt2')


# Fine-tuning

# Prepare the input tensors for fine-tuning
input_ids = tokenized_dataset['input_ids']
attention_mask = tokenized_dataset['attention_mask']

# Fine-tuning parameters
batch_size = 8
num_epochs = 25
sequence_length = 512

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()

for epoch in range(num_epochs):
    for i in range(0, len(input_ids), batch_size):
        # Prepare the batch
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]

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
        print(f"Epoch: {epoch+1}, Batch: {i//batch_size+1}, Loss: {loss.item()}")

# Text Generation

# Set the model to evaluation mode
model.eval()

# Provide a prompt or seed text for generation
prompt = "Sing"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')
num_candidates = 5

# Generate candidate formulae
output = model.generate(
    input_ids,
    max_length=10,
    num_return_sequences=num_candidates,
    temperature=1.0,
    num_beams=num_candidates,
    early_stopping=True
)

# Process and store the generated candidates
generated_candidates = []
for i in range(output.shape[0]):
    generated_candidates.append(tokenizer.decode(output[i], skip_special_tokens=True))

# Write the generated candidates to a text file
user = os.getenv('USER')
with open('/scratch/users/{}/outputs/generated_candidates.txt'.format(user), 'w') as file:
    for candidate in generated_candidates:
        file.write(candidate + '\n')
