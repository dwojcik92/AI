from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast, Trainer, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import get_cosine_schedule_with_warmup
import gc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

vocab_size = 50304
gpt2_config = GPT2Config(vocab_size=vocab_size,n_layer=4, n_head=4, n_positions=1024)
model = GPT2LMHeadModel(config=gpt2_config)
tokenizer = GPT2TokenizerFast(vocab_file='tokenizer/vocab.json', merges_file='tokenizer/merges.txt')
tokenizer.pad_token = tokenizer.eos_token

def tokenize(element):
    outputs = tokenizer(
        element[0]["text"],
        truncation=True,
        padding=True,
        max_length=gpt2_config.n_positions,
        return_tensors="pt",
    )
    # Pad the sequences to length n_positions
    input_ids_padded = torch.nn.functional.pad(outputs['input_ids'], (0, gpt2_config.n_positions - len(outputs['input_ids'])))
    attention_mask = torch.ones_like(input_ids_padded)
    return {"input_ids": input_ids_padded, "attention_mask": attention_mask}

dataset = load_dataset("Skylion007/openwebtext")

dataloader = DataLoader(dataset=dataset['train'], 
                        collate_fn=tokenize, 
                        batch_size=16, 
                        pin_memory=False, 
                        )

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model, tokenizer, data loader and the args here as shown in the code above.
model.train() # Set the model to training mode.
model = model.to(device)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=5.e-5, amsgrad=True) # Define an optimizer.
# Create a learning rate scheduler that starts with a low learning rate and increases it over time.
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100,
    num_training_steps=len(dataset['train'])
)

loss_values = []
steps = []


fig, ax = plt.subplots(figsize=(2, 2))

global_step = 0
loop = tqdm(dataloader, leave=True)
plt.ion()
for batch in loop:
    input_ids = batch["input_ids"].to(device) # Move the data to the correct device (CPU/GPU/MPS).
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids) # Forward pass.
    
    loss = outputs.loss
    
    loss.backward() # Backward pass.

    optimizer.step() # Update the parameters.
    lr_scheduler.step()
    
    # Manually delete input_ids and attention_mask to free up memory
    del input_ids
    del attention_mask  
    gc.collect()
    torch.mps.empty_cache()
    
    optimizer.zero_grad() # Reset gradients to zero for the next iteration.
    
    loop.set_description_str(f"Loss: {loss.item():.5f}, lr = {lr_scheduler.get_last_lr()[0]:.2e}")  

    loss_values.append(loss.item())
    steps.append(global_step)
    global_step += 1

    # Update the plot after each step
    ax.clear()
    ax.plot(steps, loss_values)
    ax.set_title('Live Loss Plot')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.relim()
    ax.set_yscale('log')
    ax.autoscale_view()
    
    fig.canvas.draw()  # Update the display to reflect changes
    fig.canvas.flush_events()  # Ensure events are flushed
    
    plt.pause(0.1)  # Small