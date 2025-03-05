from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import onnx
from dotenv import dotenv_values

# Load environment variables as a dictionary
config = dotenv_values(".env")
# Load the model and tokenizer
model_name = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")
onnx_model_name = config.get("ONNX_NAME", 'tinybert_model')
max_length = int(config.get("MAX_TOKENS_LENGTH", 512))
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get the BERT model
bert_model = model.bert

# Create a wrapper class to output the pooler
class BertWithPooler(torch.nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        return outputs[0], outputs[1]  # Last hidden state and pooler output

wrapped_model = BertWithPooler(bert_model)
wrapped_model.eval()

dummy_text = "This is a sample input that will be padded or truncated to 512 tokens. " * 20
dummy_input = tokenizer(
    dummy_text, 
    return_tensors="pt", 
    max_length=512, 
    truncation=True, 
    padding='max_length'
)

# Export the model to ONNX format
onnx_model_path = "data/tinybert_model.onnx"

# Create the export arguments dictionary with both outputs
export_args = {
    "f": onnx_model_path,
    "input_names": ["input_ids", "attention_mask"],
    "output_names": ["last_hidden_state", "pooler_output"],
    "dynamic_axes": {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"}
    },
    "opset_version": 14,
    "do_constant_folding": True
}

with torch.no_grad():
    inputs = (dummy_input["input_ids"], dummy_input["attention_mask"])
    try:
        torch.onnx.export(wrapped_model, inputs, **export_args)
        print(f"Model converted to ONNX format and saved as {onnx_model_path}")
        
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful")
        
        tokenizer.save_pretrained("data/tokenizer")
        print("Tokenizer saved as 'data/tokenizer' directory")
     
        test_input = tokenizer(
            "Testing pooler output", 
            return_tensors="pt", 
            max_length=max_length, 
            truncation=True, 
            padding='max_length'
        )
        print(f'test_input shape {test_input["input_ids"].shape}')
        with torch.no_grad():
            pytorch_outputs = wrapped_model(test_input["input_ids"], test_input["attention_mask"])
            print("\nPyTorch pooler output shape:", pytorch_outputs[1].shape)
            
    except Exception as e:
        print(f"Error during export: {str(e)}")