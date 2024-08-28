from transformers import RobertaModel, RobertaTokenizer

# Load pre-trained model and tokenizer
model = RobertaModel.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize input
input_ids = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]

# Get the model outputs
outputs = model(input_ids)

# Access the last hidden state
last_hidden_state = outputs.last_hidden_state

print(last_hidden_state.shape)  # Shape: (batch_size, sequence_length, hidden_size)
print(model.config.vocab_size)
