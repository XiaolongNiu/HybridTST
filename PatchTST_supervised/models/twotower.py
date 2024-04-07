import torch.nn as nn

class DualTowerModel(nn.Module):
    def __init__(self, patchtst_model, transformer_model):
        super(DualTowerModel, self).__init__()
        self.patchtst_model = patchtst_model
        self.transformer_model = transformer_model

    def forward(self, patchtst_input, transformer_input):
        # Forward pass through PatchTST tower
        patchtst_output = self.patchtst_model(patchtst_input)

        # Forward pass through Transformer tower
        transformer_output = self.transformer_model(*transformer_input)

        # Combine outputs (you can modify this part based on your specific task)
        combined_output = torch.cat((patchtst_output, transformer_output), dim=-1)

        return combined_output

# Assuming you have instances of your PatchTST and Transformer models
patchtst_model_instance = ModelPatchTST(configs_patchtst)
transformer_model_instance = ModelTransformer(configs_transformer)

# Creating the DualTowerModel
dual_tower_model = DualTowerModel(patchtst_model_instance, transformer_model_instance)


import torch.nn as nn

class DualTowerModel(nn.Module):
    def __init__(self, patchtst_model, transformer_model):
        super(DualTowerModel, self).__init__()
        self.patchtst_model = patchtst_model
        self.transformer_model = transformer_model

    def forward(self, patchtst_input, transformer_input):
        # Forward pass through PatchTST tower
        patchtst_output = self.patchtst_model(patchtst_input)

        # Forward pass through Transformer tower
        transformer_output = self.transformer_model(*transformer_input)

        # Combine outputs (you can modify this part based on your specific task)
        combined_output = torch.cat((patchtst_output, transformer_output), dim=-1)

        return combined_output

# Assuming you have instances of your PatchTST and Transformer models
patchtst_model_instance = PatchTST.Model(configs_patchtst)  # Replace with your actual configuration
transformer_model_instance = Transformer.Model(configs_transformer)  # Replace with your actual configuration

# Creating the DualTowerModel
dual_tower_model = DualTowerModel(patchtst_model_instance, transformer_model_instance)
