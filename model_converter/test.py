import onnxruntime
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import Dict, List, Tuple

class ModelTester:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("data/tokenizer")
        self.ort_session = onnxruntime.InferenceSession("data/tinybert_model.onnx")
        self.pytorch_model = AutoModelForSequenceClassification.from_pretrained(
            "huawei-noah/TinyBERT_General_4L_312D"
        ).bert
        self.pytorch_model.eval()
        
        # Get the input shape from the ONNX model
        self.max_length = self.ort_session.get_inputs()[0].shape[1]
        print(f"ONNX model max sequence length: {self.max_length}")
        
        # Print ONNX model output names
        output_names = [output.name for output in self.ort_session.get_outputs()]
        print(f"ONNX model output names: {output_names}")

    def prepare_input(self, text: str) -> Dict:
        """Tokenize input text and prepare for model inference."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

    def run_pytorch_inference(self, inputs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with PyTorch model."""
        with torch.no_grad():
            outputs = self.pytorch_model(**inputs)
            # Return both last hidden state and pooler output
            return (
                outputs.last_hidden_state.numpy(),
                outputs.pooler_output.numpy()
            )

    def run_onnx_inference(self, inputs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with ONNX model."""
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        # Get both outputs from ONNX model
        last_hidden_state, pooler_output = self.ort_session.run(None, ort_inputs)
        return last_hidden_state, pooler_output

    def compare_outputs(self, 
                       pytorch_hidden: np.ndarray, 
                       pytorch_pooler: np.ndarray,
                       onnx_hidden: np.ndarray, 
                       onnx_pooler: np.ndarray) -> Tuple[bool, bool]:
        """Compare PyTorch and ONNX model outputs."""
        hidden_match = np.allclose(pytorch_hidden, onnx_hidden, rtol=1e-02, atol=1e-03)
        pooler_match = np.allclose(pytorch_pooler, onnx_pooler, rtol=1e-02, atol=1e-03)
        return hidden_match, pooler_match

    def test_model(self, texts: List[str]):
        """Run comprehensive model test."""
        for text in texts:
            print("\nTesting:", text)
            try:
                inputs = self.prepare_input(text)
                
                print(f"Input shapes - ids: {inputs['input_ids'].shape}, "
                      f"mask: {inputs['attention_mask'].shape}")
                
                # Run both models
                pytorch_hidden, pytorch_pooler = self.run_pytorch_inference(inputs)
                onnx_hidden, onnx_pooler = self.run_onnx_inference(inputs)
                
                # Compare outputs
                hidden_match, pooler_match = self.compare_outputs(
                    pytorch_hidden, pytorch_pooler,
                    onnx_hidden, onnx_pooler
                )
                
                print("\nOutput Shapes:")
                print(f"PyTorch hidden state: {pytorch_hidden.shape}")
                print(f"PyTorch pooler: {pytorch_pooler.shape}")
                print(f"ONNX hidden state: {onnx_hidden.shape}")
                print(f"ONNX pooler: {onnx_pooler.shape}")
                
                print("\nComparison Results:")
                print(f"Hidden states match: {hidden_match}")
                print(f"Pooler outputs match: {pooler_match}")
                
                if not hidden_match or not pooler_match:
                    print("\nMax Differences:")
                    print(f"Hidden state max diff: {np.max(np.abs(pytorch_hidden - onnx_hidden))}")
                    print(f"Pooler max diff: {np.max(np.abs(pytorch_pooler - onnx_pooler))}")
                
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    # Test sentences
    test_sentences = [
        "This product is amazing!",
        "I'm very disappointed with the service.",
        "The quality is acceptable but could be better.",
    ]
    
    try:
        tester = ModelTester()
        print("\n=== Running Model Tests ===")
        tester.test_model(test_sentences)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise  # Re-raise the exception for debugging