import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algos.dropout import FixableDropout
from src.algos.bbb_layers import BBBConv2d, BBBLinear
from src.algos.rank1 import Rank1Linear, Rank1Conv2D

class DenseNetBlock(nn.Module):
    def __init__(self, in_features, growth_rate, layers):
        super().__init__()
        self.convs = nn.ModuleList([])
        for i in range(layers):
            self.convs.append(nn.Conv2d(in_features + i * growth_rate, growth_rate, 3, padding=1))
        self.down_conv = nn.Conv2d(in_features + layers * growth_rate, in_features + layers * growth_rate, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, input):
        outputs = input
        for conv in self.convs:
            outputs = torch.cat([outputs, conv(F.relu(outputs, inplace=True))], dim=1)
        return self.pool(self.down_conv(outputs))

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
     
    def forward(self, input):
        return self.conv2(F.relu(self.conv1(F.relu(input, inplace=True)), inplace=True)) + input

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder_unit = ResidualConvUnit(features)
        self.joint_unit = ResidualConvUnit(features)

    def forward(self, encoder_input, decoder_input):
        return F.interpolate(self.joint_unit(decoder_input + self.encoder_unit(encoder_input)), scale_factor=2, mode="bilinear", align_corners=True)

class Interpolate(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return F.interpolate(input, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)

class MidasTiramisu(nn.Module):
    def __init__(self, features=256):
        super().__init__()

        self.in_conv = nn.Conv2d(3, 64, 3, padding=1)

        self.encode1 = DenseNetBlock(64, 16, 4)
        self.encode2 = DenseNetBlock(128, 16, 5)
        self.encode3 = DenseNetBlock(208, 16, 7)
        self.encode4 = DenseNetBlock(320, 16, 10) # ==> 480

        self.trans1 = nn.Conv2d(480, features, 3, padding=1)
        self.trans2 = nn.Conv2d(320, features, 3, padding=1)
        self.trans3 = nn.Conv2d(208, features, 3, padding=1)
        self.trans4 = nn.Conv2d(128, features, 3, padding=1)

        self.fuse1 = nn.Sequential(ResidualConvUnit(features), Interpolate(2))
        self.fuse2 = FeatureFusionBlock(features)
        self.fuse3 = FeatureFusionBlock(features)
        self.fuse4 = FeatureFusionBlock(features)

        self.out = nn.Sequential(
            nn.Conv2d(features, 128, 3, padding=1),
            #Interpolate(2),
            nn.ReLU(True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 1, padding=0),
            nn.ReLU(True)
        )

    def forward(self, input):
        input = self.in_conv(input)

        enc1 = self.encode1(input)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)
        
        dec1 = self.fuse1(self.trans1(enc4))
        dec2 = self.fuse2(self.trans2(enc3), dec1)
        dec3 = self.fuse3(self.trans3(enc2), dec2)
        dec4 = self.fuse4(self.trans4(enc1), dec3)

        return self.out(dec4)

##################################################

def get_conv(in_channels, out_channels, kernel_size, stride, padding=0, bias=True, config={"type": "plain"}):
    if config["type"] == "variational":
        return BBBConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, weight_prior=config["prior"], bias_prior=config["prior"])
    elif config["type"] == "rank1":
        return Rank1Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, prior=config["prior"], components=config["components"])
    elif config["type"] == "plain":
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    else:
        raise ValueError(f"Unknown convolution layer type '{config['type']}")

def get_linear(in_features, out_features, config):
    if config["type"] == "variational":
        return BBBLinear(in_features, out_features, weight_prior=config["prior"], bias_prior=config["prior"])
    elif config["type"] == "rank1":
        return Rank1Linear(in_features, out_features, prior=config["prior"], components=config["components"])
    elif config["type"] == "plain":
        return nn.Linear(in_features, out_features)
    else:
        raise ValueError(f"Unknown linear layer type '{config['type']}")

def get_drop(config):
    if "dropout_p" in config:
        return FixableDropout(p=config["dropout_p"])
    else:
        return nn.Identity()

def get_norm(in_features, config):
    return nn.BatchNorm2d(in_features, track_running_stats=config.get("track_running_stats", True))

class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate, bn_size, config):
        super().__init__()
        self.norm1 = get_norm(in_features, config)
        self.conv1 = get_conv(in_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False, config=config["conv"])
        self.drop1 = get_drop(config)

        self.norm2 = get_norm(bn_size * growth_rate, config)
        self.conv2 = get_conv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False, config=config["conv"])
        self.drop2 = get_drop(config)

    def forward(self, inputs):
        bn_out = self.drop1(self.conv1(F.relu(self.norm1(torch.cat(inputs, 1)))))
        return self.drop2(self.conv2(F.relu(self.norm2(bn_out))))

class DoubleDenseBlock(nn.Module):
    def __init__(self, layers, in_features, bn_size, growth_rate, config):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            layer = DenseLayer(in_features + i * growth_rate, growth_rate, bn_size, config)
            self.layers.append(layer)
        
    def forward(self, input):
        features = [input]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, 1)

class Transition(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        self.norm = get_norm(in_features, config)
        self.conv = get_conv(in_features, out_features, kernel_size=1, stride=1, bias=False, config=config["conv"])
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        return self.pool(self.conv(F.relu(self.norm(input))))

class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, in_channels, init_features, bn_size, config={"linear": {"type": "plain"}, "conv": {"type": "plain"}}):
        super().__init__()
        self.in_block = nn.Sequential(
            get_conv(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False, config=config["conv"]),
            get_norm(init_features, config),
            nn.ReLU(),
            get_drop(config),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.features = nn.Sequential()
        feature_count = init_features
        for i, layers in enumerate(block_config):
            block = DoubleDenseBlock(layers, feature_count, bn_size, growth_rate, config)
            self.features.append(block)
            feature_count += layers * growth_rate
            if i != len(block_config) - 1:
                transition = Transition(feature_count, feature_count // 2, config)
                self.features.append(transition)
                feature_count //= 2
        self.features.append(get_norm(feature_count, config))

        self.out_features = feature_count

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        init_features = self.in_block(input)
        features = self.features(init_features)
        output = F.adaptive_avg_pool2d(F.relu(features), (1, 1))
        return output

class ClassificationHead(nn.Module):
    def __init__(self, in_features, classes, config={"linear": {"type": "plain"}}):
        super().__init__()
        self.head = get_linear(in_features, classes, config["linear"])

    def forward(self, input):
        return self.head(torch.flatten(input, 1))