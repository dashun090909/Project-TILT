# model.py - Dashun Feng
# Produces an TensorFlow AllenAI longerformer base model, TILTRegression.

import torch.nn
from transformers import LongformerModel

# Define TILTRegression class, which inherits from torch.nn.Module
class TILTRegression(torch.nn.Module):
    def __init__(self):
        # Call the constructor of the parent class (torch.nn.Module)
        super(TILTRegression, self).__init__()
        
        # Load the AllenAI Longformer model
        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        
        # Define a linear regression head that maps the [CLS] token embedding to a single output
        self.regressor = torch.nn.Linear(self.model.config.hidden_size, 1)

    # Defines the data forward pass
    def forward(self, input_ids, attention_mask):
        # Run the input through the Longformer model to get hidden states
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the CLS token representation (first token's embedding) from the last hidden layer
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Pass it through the regression head and remove the last dimension (from [batch_size, 1] to [batch_size])
        return self.regressor(cls_output).squeeze(-1)