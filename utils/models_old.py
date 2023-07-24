import torchvision.models as models
import torchvision.transforms as transforms

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
        

    elif 'effnet' in model_name:
        model_n = model_name[-1]
        full_model_name = f'efficientnet-b{model_n}'

        if pretrained:
            model = EfficientNet.from_pretrained(full_model_name, num_classes=n_classes)
        else:
            model = EfficientNet.from_name(full_model_name, num_classes=n_classes)

        n_feats = model._fc.in_features
        input_size     = EfficientNet.get_image_size(full_model_name)


    elif model_name == 'alexnet':
        if pretrained:
            weights = 'AlexNet_Weights.DEFAULT'
        model = models.alexnet(weights=weights)

        n_feats      = model.classifier[6].in_features
        input_size          = 224
        model.classifier[6] = nn.Linear(n_feats, n_classes)

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
        input_size = 224
        
    else:
        print('Invalid model name, exiting...')
        exit()

    model.name           = model_name
    model.n_feats        = n_feats
    model.input_size     = input_size

    return model

class BaseMetaModel(nn.Module):

    def __init__(self, model):

        super().__init__()
        self.model      = model
        self.name       = self.model.name
        self.n_feats    = self.model.n_feats
        self.input_size = self.model.input_size

    def forward(self, img, metadata):
        return self.model(img)
    
class Passer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, feats, metadata):
        return feats.float()