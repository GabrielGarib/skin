
import torch
import torch.nn as nn
import torchvision.models as models
import torchextractor as tx

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from efficientnet_pytorch import EfficientNet


def get_model(model_name, n_classes=8, pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model      = None
    input_size = 0
    weights    = None

    if 'resnet' in model_name:
        model_n = model_name[6:]
        if pretrained:
            weights = f'ResNet{model_n}_Weights.DEFAULT'

        if model_name == 'resnet18':
            model = models.resnet18(weights=weights)
        elif model_name == 'resnet34':
            model = models.resnet34(weights=weights)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=weights)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=weights)
        elif model_name == 'resnet152':
            model = models.resnet152(weights=weights)
        else:
            raise Exception('Resnet model must be resnet18, resnet34, resnet50, resnet101 or resnet152')

        n_feats = model.fc.in_features
        input_size     = 224
        model.fc       = nn.Linear(n_feats, n_classes)
        feats_layer    = -4

    elif 'resnext' in model_name:
        if pretrained:
            weights = f'DEFAULT'

        if model_name == 'resnext50':
            model = models.resnext50_32x4d(weights=weights)
        elif model_name == 'resnext101':
            model = models.resnext101_32x8d(weights=weights)

        n_feats = model.fc.in_features
        input_size     = 224
        model.fc       = nn.Linear(n_feats, n_classes)
        feats_layer    = -4
        

    elif 'effnet' in model_name:
        model_n = model_name[-1]
        full_model_name = f'efficientnet-b{model_n}'

        if pretrained:
            model = EfficientNet.from_pretrained(full_model_name, num_classes=n_classes)
        else:
            model = EfficientNet.from_name(full_model_name, num_classes=n_classes)

        n_feats        = model._fc.in_features
        input_size     = EfficientNet.get_image_size(full_model_name)
        feats_layer    = None


    elif 'vgg' in model_name:
        if pretrained:
            weights = 'DEFAULT'

        if model_name == 'vgg11':
            model = models.vgg11(weights=weights)
        elif model_name == 'vgg13':
            model = models.vgg13(weights=weights)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=weights)
        elif model_name == 'vgg19':
            model = models.vgg19(weights=weights)

        n_feats             = 512
        n_class_feats       = model.classifier[6].in_features

        input_size          = 224
        model.classifier[6] = nn.Linear(n_class_feats, n_classes)
        feats_layer = -9

    elif model_name == 'alexnet':
        if pretrained:
            weights = 'AlexNet_Weights.DEFAULT'
        model = models.alexnet(weights=weights)

        # n_feats             = model.classifier[6].in_features
        n_feats             = 256
        n_class_feats       = model.classifier[6].in_features

        input_size          = 224
        model.classifier[6] = nn.Linear(n_class_feats, n_classes)
        feats_layer = -10

    elif 'vit' in model_name:
        vit_type = model_name[4]
        vit_num  = model_name[6:8]
        if pretrained:
            weights = f'ViT_{vit_type.upper()}_{vit_num}_Weights.IMAGENET1K_V1'

        if model_name == 'vit_b_16':
            model = models.vit_b_16(weights=weights)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(weights=weights)
        elif model_name == 'vit_l_16':
            model = models.vit_l_16(weights=weights)
        elif model_name == 'vit_l_32':
            model = models.vit_l_32(weights=weights)
        elif model_name == 'vit_h_14':
            weights = 'DEFAULT'
            model = models.vit_h_14(weights=weights)
        else:
            raise Exception('ViT model must be vit_b_16, vit_b_32, vit_l_16, vit_l_32, or vit_h_14')
            
        n_feats          = model.hidden_dim
        model.heads.head = nn.Linear(n_feats, n_classes)
        input_size  = 224
        feats_layer =  -3
        
    else:
        print('Invalid model name, exiting...')
        #exit()

    model.name           = model_name
    model.n_feats        = n_feats
    model.input_size     = input_size
    model.feats_layer    = feats_layer

    return model


class BaseMetaModel(nn.Module):
    def __init__(self, model):

        super().__init__()
        self.model       = model
        self.name        = self.model.name 
        self.n_feats     = self.model.n_feats
        self.input_size  = self.model.input_size
        self.feats_layer = self.model.feats_layer
        
        if 'effnet' in self.name:
            self.extract_features = self.model.extract_features

    def forward(self, img, metadata=None):
        return self.model(img)
    
    
class FeatureExtractor(nn.Module):
    def __init__(self, model, n_classes=8):
        super().__init__()
        self.model = model
        self.model_name = model.name

        if 'effnet' not in self.model_name:
            train_nodes, eval_nodes = get_graph_node_names(self.model)
            
            self.layer_name = eval_nodes[self.model.feats_layer]
            return_nodes    = [self.layer_name]
            #print(return_nodes)
            self.features   = create_feature_extractor(self.model, return_nodes=return_nodes)#nn.Sequential(*list(self.model.children())[:-2])

    def forward(self, x, metadata=None):
        batch_size = x.shape[0]
        if 'effnet' in self.model_name:
            x  = self.model.extract_features(x)
        elif 'vit' in self.model_name:
            x = self.features(x, metadata)[self.layer_name].permute((0, 2, 1))[:, :, :-1].reshape((batch_size, self.model.n_feats, 7, 7))
        else:
            x  = self.features(x, metadata)[self.layer_name]
            #x  = self.features(x)

        return x
    
    
class Passer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, metadata=None):
        return feats.float()
    
    
class MetaNet(nn.Module):
    """
    Fusing Metadata and Dermoscopy Images for Skin Disease Diagnosis - https://ieeexplore.ieee.org/document/9098645
    """
    def __init__(self, n_feats, n_metadata, hidden=256):
        super(MetaNet, self).__init__()
        
        self.metaprocesser = nn.Sequential(
            nn.Conv2d(n_metadata, hidden, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden, n_feats, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, feat_maps, metadata):
        x = self.metaprocesser(metadata.unsqueeze(-1).unsqueeze(-1).float())
        x = feat_maps * x
        return x
    
    
class MetaBlock(nn.Module):
    """
    Implementing the Metadata Processing Block (MetaBlock)
    """
    def __init__(self, n_feats, n_metadata):
        super().__init__()
        self.fb = nn.Sequential(nn.Linear(n_metadata, n_feats), nn.BatchNorm1d(n_feats))
        self.gb = nn.Sequential(nn.Linear(n_metadata, n_feats), nn.BatchNorm1d(n_feats))

    def forward(self, feats, metadata):
        t1 = self.fb(metadata.float()).unsqueeze(-1).unsqueeze(-1)
        t2 = self.gb(metadata.float()).unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(torch.tanh(feats * t1) + t2)
        return x
    
    
class FusionBlock(nn.Module):
    def __init__(self, n_feats, n_metadata, fusion_method='concat', n_reducer_block=256, p_dropout=.5):
        super().__init__()

        self.avg_pool      = nn.AvgPool2d(kernel_size=7)
        self.fusion_method = fusion_method

        if n_reducer_block > 0:
            self.reducer_block = ReducerBLock(n_reducer_block=n_reducer_block, p_dropout=p_dropout, n_feats=n_feats)
        else:
            self.reducer_block = None

        if self.fusion_method == 'metanet':
            self.fusion = MetaNet(n_feats, n_metadata)
        elif self.fusion_method == 'metablock':
            self.fusion = MetaBlock(n_feats, n_metadata)
        else:
            self.fusion = Passer()

        # Exceptions
        if n_metadata > 0 and fusion_method == None:
            raise Exception('Provide a fusion method (concat, metanet, metablock)')
        if n_metadata == 0 and fusion_method != None:
            raise Exception(f'Provide metadata for fusion method: {fusion_method}')

    def forward(self, feats, metadata):
        x = self.fusion(feats, metadata) # batch_size x n_feats x 7 x 7

        x = self.avg_pool(x)             # batch_size x n_feats x 1 x 1
        x = x.view(x.size(0), -1)        # batch_size x n_feats (flatting)

        if self.reducer_block is not None:
            x = self.reducer_block(x)    # batch_size x n_reducer_block

        if self.fusion_method == 'concat':
            x = torch.cat([x, metadata.float()], dim=1) # concatenation
        return x
    

class ReducerBLock(nn.Module):
    def __init__(self, n_reducer_block=256, p_dropout=0.5, n_feats=1024):
        super().__init__()
        self.reducer_block = nn.Sequential(
                nn.Linear(n_feats, n_reducer_block),
                nn.BatchNorm1d(n_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        
    def forward(self, x):
        return self.reducer_block(x)
    
    
class MetaModel(nn.Module):

    def __init__(self, model, n_classes, n_metadata=0, fusion_method='concat', n_reducer_block=256,
                 p_dropout=0.5, freeze=True):

        super().__init__()

        self.model             = model
        self.n_classes         = n_classes
        self.n_metadata        = n_metadata
        self.fusion_method     = fusion_method
        self.n_reducer_block   = n_reducer_block
        self.model_name        = model.name
        self.n_feats           = model.n_feats
        self.feature_extractor = FeatureExtractor(model, n_classes=n_classes)
        self.fusion_method     = fusion_method
        self.fusion_block      = FusionBlock(self.n_feats, n_metadata, fusion_method=fusion_method, n_reducer_block=n_reducer_block, p_dropout=p_dropout)

        self.n_final           = self.get_n_final()
        self.classifier        = nn.Linear(self.n_final, n_classes)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, img, metadata):
        x = self.feature_extractor(img)       # feats:  batch_size * n_feats * 7 * 7
        x = self.fusion_block(x, metadata)    # vector: batch_size * n_reducer_block | batch_size * (n_reducer_block + n_metadata)

        return self.classifier(x)


    def get_n_final(self):
        if self.n_reducer_block > 0:
            n_model_out = self.n_reducer_block
        else:
            n_model_out = self.n_feats

        if self.fusion_method == 'concat':
            n_final = n_model_out + self.n_metadata
        else:
            n_final = n_model_out

        return n_final