import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights
from vit_pytorch import ViT
from transformers import ViTModel as HF_ViTModel, ViTConfig
from vit_pytorch.vit_for_small_dataset import ViT as ViT_small
from vit_pytorch.cct import CCT


class ChannelProjector(nn.Module):
    """
    Project an arbitrary number of channels to 3 channels.
    """
    # Dimensionality reduction
    def __init__(self, input_channels, output_channels=3):
        super(ChannelProjector, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        if self.input_channels == self.output_channels:
            return x
        return self.conv(x)


class ResNetFeatureExtractor(nn.Module):
    """Pretrained ResNet-18 feature extractor"""

    def __init__(self, in_channels=128, freeze=False):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet18(progress=True) #weights=ResNet18_Weights.DEFAULT, 

        # resnet = resnet34()
        
        self.hidden_dim = resnet.fc.in_features
        
        # Modify the first layer to accept arbitrary channels
        # Save the original weight for potential initialization strategy
        original_conv1 = resnet.conv1
        
        # Create a new conv1 layer with the desired input channels
        resnet.conv1 = nn.Conv2d(
            in_channels, 
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )
        
        # Remove the last dense layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        if freeze:
            # Freeze all parameters
            for param in resnet.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class HSResNet18(nn.Module):
    def __init__(self, input_channels, num_outputs, output_channels=3):
        super(HSResNet18, self).__init__()
        self.channel_projector = ChannelProjector(input_channels, output_channels)
        self.feature_extractor = ResNetFeatureExtractor(in_channels=output_channels)
        
        self.regression_head = nn.Sequential(
            nn.Linear(self.feature_extractor.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        x = self.channel_projector(x)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.regression_head(x)
        return x


class HSResNet34(nn.Module):
    def __init__(self, input_channels, num_outputs, output_channels=3, image_size=224, patch_size=16):
        super(HSResNet34, self).__init__()
        self.channel_projector = ChannelProjector(input_channels, output_channels) 
        self.feature_extractor = ResNetFeatureExtractor(in_channels=output_channels) 
        
        self.dropout = nn.Dropout(0.5)
        self.regression_head = nn.Linear(self.feature_extractor.hidden_dim, num_outputs)

    def forward(self, x):
        x = self.channel_projector(x)
        features = self.feature_extractor(x)
        # Flatten the features tensor before applying dropout and regression
        x = torch.flatten(features, 1)
        x = self.dropout(x)
        x = self.regression_head(x)
        return x

class ViTModel(nn.Module):
    def __init__(self, input_channels, num_outputs, image_size=512, patch_size=16, output_channels=3):
        super(ViTModel, self).__init__()
        self.channel_projector = ChannelProjector(input_channels, output_channels)
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1024,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.,
            emb_dropout=0.,
            channels=output_channels
        )
        
    #     # Shared layers
    #     self.shared_layers = nn.Sequential(
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Dropout(0.3)
    #     )
        
    #     # Create individual branches for each output
    #     self.output_branches = nn.ModuleList()
    #     for i in range(num_outputs):
    #         branch = nn.Sequential(
    #             nn.Linear(512, 256),
    #             nn.ReLU(),
    #             nn.Dropout(0.2),  # Add dropout here
    #             nn.Linear(256, 128),
    #             nn.ReLU(),
    #             nn.Dropout(0.1),  # Add dropout here too
    #             nn.Linear(128, 64),  # Add an extra layer
    #             nn.ReLU(),
    #             nn.Linear(64, 1)
    #         )
    #         self.output_branches.append(branch)

    # def forward(self, x):
    #     x = self.channel_projector(x)
    #     x = self.vit(x)
    #     x = self.shared_layers(x)
        
    #     # Process each branch and collect outputs
    #     outputs = []
    #     for branch in self.output_branches:
    #         outputs.append(branch(x))
        
    #     # Concatenate outputs into a single tensor [batch, num_outputs]
    #     return torch.cat(outputs, dim=1)layer

        # self.fc_block = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, num_outputs) # Output layer
        # )
        
        self.fc_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])

        #Relu activation function
        self.relu = nn.ReLU()
        
        self.output_layer = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = self.channel_projector(x)
        x = self.vit(x)
        # x = self.fc_block(x)
        
        # Apply blocks with residual connections
        # identity = x
        for i, block in enumerate(self.fc_block):
            if i == 0 or identity.shape[1] != x.shape[1]:
                identity = x
            else:
                x = x + identity  # Residual connection
                identity = x
            x = block(x)
        
        x = self.output_layer(x)
        return x

class ViTModelSmall(nn.Module):
    def __init__(self, input_channels, num_outputs, image_size=512, patch_size=16, output_channels=3):
        super(ViTModelSmall, self).__init__()
        self.channel_projector = ChannelProjector(input_channels,output_channels)
        self.vit = ViT_small(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        hidden_size = 1000  # Define hidden_size based on the ViT_small configuration

        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, num_outputs)
        )

        #Relu activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.channel_projector(x)
        x = self.vit(x)
        x = self.fc_block(x)
        return x
    
class ViTModelCCT(nn.Module):
    def __init__(self, input_channels, num_outputs, image_size=512, patch_size=16, output_channels=3):
        super(ViTModelCCT, self).__init__()
        self.channel_projector = ChannelProjector(input_channels, output_channels)

        self.vit = CCT(
            img_size = (image_size, image_size),
            n_input_channels = output_channels,
            embedding_dim = 512,           # Increased from 384
            n_conv_layers = 3,             # Increased from 2
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 16,               # Increased from 14
            num_heads = 8,                 # Increased from 6
            mlp_ratio = 4.,                # Increased from 3
            num_classes = 500,
            positional_embedding = 'learnable',
            dropout = 0.1,                 # Added dropout
        )

        # self.fc_block = nn.Sequential(
        #     nn.Linear(500, num_outputs)
        # )
        
        self.fc_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(500, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        
        self.output_layer = nn.Linear(128, num_outputs)


    def forward(self, x):
        x = self.channel_projector(x)
        x = self.vit(x)
        # x = self.fc_block(x)
        
        # Apply blocks with residual connections
        identity = x
        for i, block in enumerate(self.fc_block):
            if i == 0 or identity.shape[1] != x.shape[1]:
                identity = x
            else:
                x = x + identity  # Residual connection
                identity = x
            x = block(x)
        
        x = self.output_layer(x)
        return x 
    
class PretrainedViTModel(nn.Module):
    def __init__(self, input_channels, num_outputs, image_size=512, output_channels=3):
        super(PretrainedViTModel, self).__init__()
        # If the input isn't already 3 channels, project it down.
        self.channel_projector = ChannelProjector(input_channels, output_channels) if input_channels != output_channels else nn.Identity()

        # Load the default configuration and update the image size.
        
        # check architecture
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.image_size = image_size  # Set to 512
        # When using a different image size, the positional embeddings will be interpolated.
        self.hf_vit = HF_ViTModel.from_pretrained("google/vit-base-patch16-224", config=config, ignore_mismatched_sizes=True)
        
        #! freeze the model params
        for param in self.hf_vit.parameters():
            param.requires_grad = False
        
        hidden_size = self.hf_vit.config.hidden_size
        
        #! Define a custom fully connected head to map the ViT output to your desired outputs.
        # hidden_size = 768
        
        # make this simpler
        self.fc_block = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        # Project the input channels if needed.
        x = self.channel_projector(x)
        # The HF ViT expects inputs in shape [batch, channels, height, width].
        outputs = self.hf_vit(pixel_values=x)
        # Use the pooled output (corresponding to the [CLS] token) for downstream tasks.
        pooled_output = outputs.pooler_output
        x = self.fc_block(pooled_output)
        return x