import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Input

# Txt directory
import os
user = os.getenv('USER')
corpusdir = '/scratch/users/{}/corpus/'.format(user)
corpus = []
for infile in os.listdir(corpusdir):
  with open(corpusdir+infile, errors='ignore') as fin:
    corpus.append(fin.read())


# List input
# Define the dataset_formulae and oversampling factor
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
#    "Swift ships sailed across the wine-dark sea",
#    "Sing, O Muse, of the great deeds of heroes",
#    "Achilles, the son of Peleus, renowned warrior",
#    "Brave Hector, defender of Troy, faced his foe",
#    "Ares, the god of war, looked upon the battlefield",
#    "In the halls of Olympus, the gods convened",
#    "The golden fleece, sought by Jason and his crew",
#    "Penelope, faithful wife of Odysseus, waited patiently",
#    "The land of the Cyclopes, a place of danger and awe",
#    "With his bow and arrow, Apollo struck his enemies",
#    "The winds carried the ships across the vast sea",
#    "Mighty Zeus, ruler of gods and men, watched",
#    "The Sirens' enchanting song lured sailors to their doom",
#    "Poseidon, god of the sea, stirred up a mighty storm",
#    "The Trojan War, a conflict that lasted for years",
#    "Helen of Troy, the face that launched a thousand ships",
#    "The Lotus Eaters, whose flowers induced forgetfulness",
#    "In the land of the dead, shades roamed the shadows",
#    "The hero's journey, a path filled with trials and triumphs",
#    "The lyre of Orpheus, whose music moved even stones",
#    "With his cunning, Odysseus devised a plan to escape",
#    "The land of the Phaeacians, a haven for the weary",
#    "The wrath of Poseidon, a formidable obstacle",
#    "In the underworld, souls awaited their final judgment"
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

# Generate text
output = model.generate(input_ids, attention_mask=torch.ones_like(input_ids), max_length=10, num_return_sequences=1, temperature=1.0)


# Decode the generated output to text
generated_formulae = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

# Write generated formulae to a file
output_file = "generated_formulae_successful.txt"

with open(output_file, 'w', encoding='utf-8') as file:
    file.write("Generated Homeric Formulae:\n")
    for formula in generated_formulae:
        file.write(formula + '\n')

print("Generated Homeric formulae saved to", output_file)

