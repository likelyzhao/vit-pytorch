import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F


MIN_NUM_PATCHES = 16

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim/heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Encoder1DBlock(nn.Module):
    def __init__(self, input_shape, heads, mlp_dim, dtype=torch.float32, dropout_rate=0.1,attention_dropout_rate=0.1,deterministic=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.deterministic = deterministic
        self.input_shape = input_shape
        self.layer_norm_input = nn.LayerNorm(input_shape)
        self.layer_norm_out = nn.LayerNorm(input_shape)

        # self.layer_norm_input = nn.GroupNorm(1)
        # self.layer_norm_out = nn.GroupNorm(1)

        self.attention = MultiHeadDotProductAttention(input_shape, heads = heads)
        self.mlp = FeedForward(input_shape, mlp_dim, dropout_rate)
        self.drop_out_attention  = nn.Dropout(attention_dropout_rate)
    
    def forward(self, inputs):
        x = self.layer_norm_input(inputs)
        x = self.attention(x)
        x = self.drop_out_attention(x)
        x = x + inputs
        y = self.layer_norm_out(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, input_shape, num_layers, heads, mlp_dim, inputs_positions= None, dropout_rate=0.1, train=False):
        super().__init__()
        self.num_layers = num_layers 
        self.mlp_dim  = mlp_dim
        self.inputs_positions = inputs_positions
        self.dropout_rate = dropout_rate
        self.train_flag  = train
        self.encoder_norm = nn.LayerNorm(input_shape)
        # self.encoder_norm = nn.GroupNorm(1)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([Encoder1DBlock(input_shape,heads, mlp_dim)]))

    def forward(self, img, mask = None):
        x = img
        for layer in self.layers:
            x = layer[0](x)
        return self.encoder_norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, depth, heads, mlp_dim, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        hidden_size = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.embedding = nn.Conv2d(channels,hidden_size, patch_size, patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Encoder(hidden_size, depth, heads, mlp_dim, dropout_rate = dropout)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, img, mask = None):
        x = self.embedding(img)

        x = rearrange(x, 'b c h w  -> b (h w) c')
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def VIT_B16_224(**kwargs):
    input_size = 224
    patch_size = 16
    num_layers = 12
    num_classes = 1000
    if 'num_classes' in kwargs:
        num_classes = kwargs['num_classes']

    return ViT(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        depth = num_layers,
        heads = 12,
        mlp_dim = 3072,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    

if __name__ == '__main__':
    import torch
    input_size = 224

    v = VIT_B16_224()

    img = torch.randn(1, 3, input_size, input_size)
    preds = v(img) # (1, 1000)

    print(preds.flatten()[0:10])

    v.load_state_dict(torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth'))

    preds = v(img) # (1, 1000)

    print(preds.flatten()[0:10])
