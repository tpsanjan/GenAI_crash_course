# Fine tuning language model demo

# !pip install transformers datasets peft hf_xet

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, PeftType

# Let's load the IMDB dataset
# - Contains IMDB reviews with negative (0) or positive (1) labels
# - Can be used for sentiment analysis
dataset = load_dataset("imdb")
print('Dataset-len = ',len(dataset),' & Type(Data-set) = ',type(dataset))

# Instead of using the full dataset, we will use only a small subset
train_subset = dataset["train"].shuffle(seed=42).select(range(500))
test_subset = dataset["test"].shuffle(seed=42).select(range(100))

# Prepare the pre-trained language model and tokenizer using the BERT model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
 
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_train = train_subset.map(tokenize_function, batched=True)
tokenized_test = test_subset.map(tokenize_function, batched=True)
print('Successfully tokenised the dataset!')

'''
# Gearing up for training the language model
# - To speed up or in case of OOM:
# -- lower the per_device_train_batch_size from 8 to 4
# -- set fp16 = True (Default is False) to use 16-bit instead of 32-bit
# -- try setting no_cuda = True (Default is False) to avoid using cuda

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,  
    weight_decay=0.01,
)

# Start the full fine-tuning process
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)
 
trainer.train()
trainer.evaluate()

'''

# Now let's try the PEFT approach ...
print('\n\n\n Running the PEFT module \n\n\n')

# Configuring LoRA while downloading the PEFT model
peft_config = LoraConfig(
    peft_type=PeftType.LORA,
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
peft_model = get_peft_model(model, peft_config)

# Setting up the model training params
training_args = TrainingArguments(
    output_dir="./peft_results",
    eval_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,          # per_device_train_batch_size = 8
    num_train_epochs=1,
)

# Fine-tuning the model using PEFT
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)
 
trainer.train()
trainer.evaluate()
''