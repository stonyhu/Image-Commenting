import torch
import torch.nn as nn
from torchvision.models import resnet152
from torchvision.models import resnet50
from .attention import SelfAttention


class CNN(nn.Module):
    def __init__(self, finetune_start_layer=6):
        super().__init__()
        cnn = resnet50(pretrained=True)
        self.cnn_head = nn.Sequential(*list(cnn.children())[:finetune_start_layer])
        self.cnn_tail = nn.Sequential(*list(cnn.children())[finetune_start_layer:-2])
        self.attention = SelfAttention(2048)
        # self.cnn_tail.apply(self._random_init_weights)
        self.tune = False
        for p in self.cnn_head.parameters():
            p.requires_grad = False

    def _random_init_weights(self, m):
        if type(m) == nn.Conv2d:
            m.weight.data.normal_(0, 0.01)

    def tuning(self, flag):
        self.tune = flag

    def forward(self, image):
        """
        :param image: torch.Tensor [bsz x 3 x 224 x 224]
        :return: torch.Tensor [bsz x 2048 x 7 x 7]
        """
        is_train = self.tune and self.training
        self.cnn_head.train(False)
        self.cnn_tail.train(is_train)
        state = self.cnn_head(image)
        with torch.set_grad_enabled(is_train):
            state = self.cnn_tail(state)
        state = state.view(-1, 2048, 49).transpose(1, 2)  # [bsz x 49 x 2048]
        v = self.attention(state)  # [bsz x 2048]
        return v
