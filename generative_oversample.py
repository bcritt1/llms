import torch
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained GPT-2 model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Oversampling function
def oversample_dataset(dataset, oversampling_factor):
    return dataset * oversampling_factor

# Define the dataset_formulae and oversampling factor
dataset_formulae = [
    "Sing, O Muse, of the wrath of Achilles",
    "Tell me, O Muse, of that hero's journey",
    "Rosy-fingered Dawn appeared on the horizon",
    "Swift-footed Achilles unleashed his rage",
    "Hector, breaker of horses, fought valiantly",
    "The wine-dark sea stretched out before them",
    "Aegis-bearing Zeus thundered from the heavens",
    "Noble Odysseus, master of stratagems",
    "The shield of Achilles shimmered in the sun",
    "Swift ships sailed across the wine-dark sea",
    "Sing, O Muse, of the great deeds of heroes",
    "Achilles, the son of Peleus, renowned warrior",
    "Brave Hector, defender of Troy, faced his foe",
    "Ares, the god of war, looked upon the battlefield",
    "In the halls of Olympus, the gods convened",
    "The golden fleece, sought by Jason and his crew",
    "Penelope, faithful wife of Odysseus, waited patiently",
    "The land of the Cyclopes, a place of danger and awe",
    "With his bow and arrow, Apollo struck his enemies",
    "The winds carried the ships across the vast sea",
    "Mighty Zeus, ruler of gods and men, watched",
    "The Sirens' enchanting song lured sailors to their doom",
    "Poseidon, god of the sea, stirred up a mighty storm",
    "The Trojan War, a conflict that lasted for years",
    "Helen of Troy, the face that launched a thousand ships",
    "The Lotus Eaters, whose flowers induced forgetfulness",
    "In the land of the dead, shades roamed the shadows",
    "The hero's journey, a path filled with trials and triumphs",
    "The lyre of Orpheus, whose music moved even stones",
    "With his cunning, Odysseus devised a plan to escape",
    "The land of the Phaeacians, a haven for the weary",
    "The wrath of Poseidon, a formidable obstacle",
    "In the underworld, souls awaited their final judgment"
]

oversampling_factor = 2  # Adjust the oversampling factor as needed

# Shuffle the dataset
random.shuffle(dataset_formulae)

# Oversample the dataset
oversampled_dataset = oversample_dataset(dataset_formulae, oversampling_factor)

# Generate candidate formulae
num_candidates = 5  # Specify the number of candidate formulae to generate

# Open the output file for writing
user = os.getenv('USER')
with open('/scratch/users/{}/outputs/oversampled_candidates.txt'.format(user), 'w') as file:
    for prompt in oversampled_dataset:
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Define the attention mask tensor
        attention_mask = torch.ones_like(input_ids).to(device)

        # Generate candidate formulae
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=45,  # Increase max_length to 15 or greater
            num_return_sequences=num_candidates,
            temperature=1.0,
            num_beams=num_candidates,
            early_stopping=True
        )

        # Process and store the generated candidates
        generated_candidates = []
        for i in range(output.shape[0]):
            generated_candidates.append(tokenizer.decode(output[i], skip_special_tokens=True))

        # Write the generated candidates to the output file
        file.write(f'Prompt: {prompt}\n')
        file.write('Generated Candidates:\n')
        for candidate in generated_candidates:
            file.write(candidate + '\n')
        file.write('\n')

