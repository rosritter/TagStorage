from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import onnx
from dotenv import dotenv_values

# Load environment variables as a dictionary
config = dotenv_values(".env")
# Load the model and tokenizer
model_name = config.get("MODEL_NAME", "huawei-noah/TinyBERT_General_4L_312D")
onnx_model_name = config.get("ONNX_NAME", 'tinybert_model')
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

# Wrap the model
wrapped_model = BertWithPooler(bert_model)
wrapped_model.eval()

# Create a dummy input for ONNX conversion
dummy_input = tokenizer("This is a sample input", return_tensors="pt")

# Export the model to ONNX format
onnx_model_path = f"data/{onnx_model_name}.onnx"

# Create the export arguments dictionary with both outputs
export_args = {
    "f": onnx_model_path,
    "input_names": ["input_ids", "attention_mask"],
    "output_names": ["last_hidden_state", "pooler_output"],
    "dynamic_axes": {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "last_hidden_state": {0: "batch_size"},
        "pooler_output": {0: "batch_size"}
    },
    "opset_version": 14,
    "do_constant_folding": True
}

# Try export with training mode off and error checking
with torch.no_grad():
    inputs = (dummy_input["input_ids"], dummy_input["attention_mask"])
    try:
        torch.onnx.export(wrapped_model, inputs, **export_args)
        print(f"Model converted to ONNX format and saved as {onnx_model_path}")
        
        # Verify the ONNX model
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful")
        
        # Save the tokenizer
        tokenizer.save_pretrained("data/tokenizer")
        print("Tokenizer saved as 'data/tokenizer' directory")
        
        # Verify outputs
        test_input = tokenizer("Testing pooler output", return_tensors="pt")
        with torch.no_grad():
            pytorch_outputs = wrapped_model(test_input["input_ids"], test_input["attention_mask"])
            print("\nPyTorch pooler output shape:", pytorch_outputs[1].shape)
            
    except Exception as e:
        print(f"Error during export: {str(e)}")