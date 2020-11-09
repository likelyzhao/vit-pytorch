# vit-pytorch
vit model from tensorflow

## google vision transformer 

converted from https://github.com/google-research/vision_transformer


referenced from https://github.com/lucidrains/vit-pytorch

###  model table

|model type|input_size|pytorch_weights|
|---|---|---|
|ViT-B_16|224*224|[B_16-224 提取码: 9mrd](https://pan.baidu.com/s/1PDV8own0jOs_UyMsYPzymQ)|

### useage

```
from vit_pytorch import VIT_B16_224

model = VIT_B16_224()
model.load_state_dict(torch.load('imagenet21k+imagenet2012_ViT-B_16-224.pth'))

input_size = 224
img = torch.randn(1, 3, input_size, input_size)

preds = model(img) # (1, 1000)

```
## caution

this model using mean value of 127.5 and normlized with 127.5 the default normlize should replaced by

```
normalize_tf = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
```

it will reduce 5% cls accurecy in imagenet 1K cls task
