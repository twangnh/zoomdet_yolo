import torch

# # Step 1: Load the original model
original_model_state = torch.load('/root/autodl-tmp/double_conv_e2e_train/epoch_4.pth')
new_model_state = {}
for key in original_model_state['state_dict']:
    if 'saliency_network' in key:
        new_model_state[key.replace('grid_generator.saliency_network.', '')] = original_model_state['state_dict'][key]

print(new_model_state)
torch.save(new_model_state, '/root/autodl-tmp/e2e.pth')