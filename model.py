import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=9, d_model=128, nhead=8, 
                 dim_feedforward=1024, num_layers=4, dropout=0.2, 
                 sequence_length=24):
        super().__init__()
        
        # -------------------------------
        # Input Projection Layer
        # -------------------------------
        # This linear layer projects the input features from 'input_dim' to 'd_model' dimensions.
        # Input shape: (batch_size, sequence_length, input_dim)
        # Output shape: (batch_size, sequence_length, d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # -------------------------------
        # Positional Embeddings
        # -------------------------------
        # Positional embeddings are added to the input to provide the model with information about the position of each element in the sequence.
        # Shape: (1, sequence_length, d_model) - broadcasted across the batch dimension during addition.
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, sequence_length, d_model)
        )
        
        # -------------------------------
        # Transformer Encoder Layers
        # -------------------------------
        # A stack of Transformer encoder layers. Each layer consists of multi-head self-attention and feedforward neural networks.
        # The number of layers is determined by 'num_layers'.
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # -------------------------------
        # Layer Normalization
        # -------------------------------
        # Applies layer normalization to stabilize and accelerate the training process.
        # Shape: (batch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # -------------------------------
        # Adaptive Average Pooling
        # -------------------------------
        # Reduces the sequence dimension to 1 by computing the average across the sequence.
        # Before pooling: (batch_size, d_model, sequence_length)
        # After pooling: (batch_size, d_model, 1) -> squeezed to (batch_size, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # -------------------------------
        # Output Layer
        # -------------------------------
        # Final linear layer that maps the 'd_model' features to 2 outputs: mean and variance.
        # Shape: (batch_size, 2)
        self.output = nn.Linear(d_model, 2)  # Predict mean and variance
    
    def forward(self, x):
        """
        Forward pass of the TransformerModel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), containing mean and variance.
        """
        # -------------------------------
        # Input Projection
        # -------------------------------
        # Project the input features to 'd_model' dimensions.
        # Input shape: (batch_size, sequence_length, input_dim)
        # Output shape: (batch_size, sequence_length, d_model)
        x = self.input_proj(x)
        
        # -------------------------------
        # Adding Positional Embeddings
        # -------------------------------
        # Incorporate positional information by adding positional embeddings to the input projections.
        # Broadcasting positional embeddings across the batch dimension.
        # Shape remains: (batch_size, sequence_length, d_model)
        x = x + self.positional_embeddings
        
        # -------------------------------
        # Permute for Transformer Encoder
        # -------------------------------
        # Transformer encoder expects input shape as (sequence_length, batch_size, d_model).
        # Rearrange dimensions accordingly.
        # Before permute: (batch_size, sequence_length, d_model)
        # After permute: (sequence_length, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # -------------------------------
        # Transformer Encoder Layers
        # -------------------------------
        # Pass the data through each Transformer encoder layer sequentially.
        # Each layer maintains the shape: (sequence_length, batch_size, d_model)
        for layer in self.encoder_layers:
            x = layer(x)
        
        # -------------------------------
        # Permute Back to Batch First
        # -------------------------------
        # After encoding, rearrange dimensions back to (batch_size, d_model, sequence_length)
        # to prepare for pooling.
        # Before permute: (sequence_length, batch_size, d_model)
        # After permute: (batch_size, d_model, sequence_length)
        x = x.permute(1, 2, 0)
        
        # -------------------------------
        # Adaptive Average Pooling
        # -------------------------------
        # Apply adaptive average pooling to reduce the sequence dimension to 1.
        # This aggregates information across the entire sequence.
        # Before pooling: (batch_size, d_model, sequence_length)
        # After pooling: (batch_size, d_model, 1)
        x = self.pool(x)
        
        # -------------------------------
        # Squeeze the Last Dimension
        # -------------------------------
        # Remove the last dimension (sequence_length reduced to 1) to get shape (batch_size, d_model)
        x = x.squeeze(-1)
        
        # -------------------------------
        # Layer Normalization
        # -------------------------------
        # Normalize the features to stabilize training.
        # Shape: (batch_size, d_model)
        x = self.norm(x)
        
        # -------------------------------
        # Split into Mean and Variance
        # -------------------------------
        # The output layer predicts two values for each sample: mean and variance.
        # Mean is taken directly from the first output neuron.
        # Variance is obtained by applying a softplus activation to ensure it's positive, plus a small epsilon for numerical stability.
        # Shapes:
        # - mean: (batch_size,)
        # - var: (batch_size,)
        mean = x[:, 0]  # First output neuron for mean
        var = torch.nn.functional.softplus(x[:, 1]) + 1e-6  # Second output neuron for variance
        
        # -------------------------------
        # Stack Mean and Variance
        # -------------------------------
        # Combine mean and variance into a single tensor with shape (batch_size, 2)
        output = torch.stack([mean, var], dim=1)
        
        return output

if __name__ == '__main__':
    # -------------------------------
    # Model Instantiation
    # -------------------------------
    # Create an instance of the TransformerModel with default parameters.
    model = TransformerModel()
    
    # -------------------------------
    # Create a Random Input Tensor
    # -------------------------------
    # Generate a random tensor to simulate a batch of input data.
    # Shape: (batch_size=32, sequence_length=24, input_dim=9)
    x = torch.randn(32, 24, 9)
    
    # -------------------------------
    # Forward Pass
    # -------------------------------
    # Pass the input tensor through the model to obtain predictions.
    # Output shape: (batch_size=32, 2) - containing mean and variance for each sample in the batch.
    y = model(x)
    
    # -------------------------------
    # Print the Output Shape
    # -------------------------------
    print(y.shape)  # Expected output: torch.Size([32, 2])
