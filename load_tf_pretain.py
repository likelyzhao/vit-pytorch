import logging
import unittest

import jax
import jax.numpy as jnp

from vit_jax import checkpoint
from vit_jax import models
from vit_pytorch import ViT
import torch
import numpy as np

def tranpose_weight(input):
    weight_shape = input.shape
    if len(weight_shape) == 3:
        input = np.array(jnp.transpose(input, (1, 2, 0)))
        weight_shape_0 = weight_shape[1] * weight_shape[2]
        return input.reshape(weight_shape_0, weight_shape[0])
    else:
        # weight_shape_0 = weight_shape[1] 
        return np.array(jnp.transpose(input, (1,0)))
    # return input.transpose([weight_shape_0,weight_shape[0]])

model = models.KNOWN_MODELS['ViT-B_16'].partial(num_classes=1000)
_, params = model.init_by_shape(
    jax.random.PRNGKey(0),
    [((1, 224, 224, 3), jnp.float32)],
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# pretrain_tf_model = checkpoint.inspect_params(checkpoint.load('imagenet21k+imagenet2012_ViT-B_16-224.npz'),
#                                             params=params, logger= logger)

pretain_tf_model = checkpoint.load_pretrained(
    pretrained_path='imagenet21k+imagenet2012_ViT-B_16-224.npz',
    init_params=params,
    model_config=models.CONFIGS['ViT-B_16'],
    logger=logger)

def print_size(dict_):
    if isinstance(dict_, dict):
        for dic in dict_:
            print(dic, print_size(dict_[dic]))
    else:
        return str(dict_.shape)

# print(pretain_tf_model.keys())
input_size = 224
patch_size = 16
num_layers = 12
# print(pretain_tf_model.keys())
# print_size(pretain_tf_model['pre_logits'])

v = ViT(
    image_size = input_size,
    patch_size = patch_size,
    num_classes = 1000,
    depth = num_layers,
    heads = 12,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.1
)

print("Model's state_dict:")
for param_tensor in v.state_dict():
    print(param_tensor, "\t", v.state_dict()[param_tensor].size())

## copy embedding  
tf_dict = {}

embedding_weight_shape = pretain_tf_model['embedding']['kernel'].shape
embedding_weight = np.array(jnp.transpose( pretain_tf_model['embedding']['kernel'], (3,2,0,1)))
# embedding_weight = pretain_tf_model['embedding']['kernel'].reshape([embedding_weight_shape[3],embedding_weight_shape[2],embedding_weight_shape[1],embedding_weight_shape[0]])
tf_dict['embedding.weight'] = torch.from_numpy(embedding_weight) 
tf_dict['embedding.bias'] = torch.from_numpy(pretain_tf_model['embedding']['bias']) 

## copy mlp_head  

weight_shape = pretain_tf_model['head']['kernel'].shape
mlp_weight = tranpose_weight(pretain_tf_model['head']['kernel'])
tf_dict['mlp_head.weight'] = torch.from_numpy(mlp_weight) 
tf_dict['mlp_head.bias'] = torch.from_numpy(pretain_tf_model['head']['bias']) 

## copy pos_embedding
tf_dict['pos_embedding'] = torch.from_numpy(pretain_tf_model['Transformer']['posembed_input']['pos_embedding']) 
tf_dict['cls'] = torch.from_numpy(pretain_tf_model['cls']) 

## transformer.encoder_norm.weight
tf_dict['transformer.encoder_norm.weight'] = torch.from_numpy(pretain_tf_model['Transformer']['encoder_norm']['scale']) 
tf_dict['transformer.encoder_norm.bias'] = torch.from_numpy(pretain_tf_model['Transformer']['encoder_norm']['bias']) 

## attetion blocks
for i in range(num_layers):
    tf_key = 'encoderblock_{0}'.format(i)
    torch_key_prefix = 'transformer.layers.{0}.0'.format(i)
    # print(pretain_tf_model['Transformer'][tf_key].keys())
    # print_size(pretain_tf_model['Transformer'][tf_key])
    ## layernorm_0
    tf_dict[torch_key_prefix + '.layer_norm_input.weight' ] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['LayerNorm_0']['scale']) 
    tf_dict[torch_key_prefix + '.layer_norm_input.bias'] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['LayerNorm_0']['bias']) 
    ## LayerNorm_2
    tf_dict[torch_key_prefix + '.layer_norm_out.weight' ] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['LayerNorm_2']['scale']) 
    tf_dict[torch_key_prefix + '.layer_norm_out.bias'] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['LayerNorm_2']['bias']) 
    ## MlpBlock_3
    tf_dict[torch_key_prefix + '.mlp.net.0.weight' ] = torch.from_numpy(tranpose_weight(pretain_tf_model['Transformer'][tf_key]['MlpBlock_3']['Dense_0']['kernel'])) 
    tf_dict[torch_key_prefix + '.mlp.net.0.bias'] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['MlpBlock_3']['Dense_0']['bias']) 

    tf_dict[torch_key_prefix + '.mlp.net.3.weight' ] = torch.from_numpy(tranpose_weight(pretain_tf_model['Transformer'][tf_key]['MlpBlock_3']['Dense_1']['kernel'])) 
    tf_dict[torch_key_prefix + '.mlp.net.3.bias'] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['MlpBlock_3']['Dense_1']['bias'])   

    ## merge the attetion weights
    q_w = tranpose_weight(pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['query']['kernel'])
    k_w = tranpose_weight(pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['key']['kernel'])
    v_w = tranpose_weight(pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['value']['kernel'])

    qkv_w = np.array(jnp.concatenate([q_w, k_w, v_w], axis= 0))

    q_b = pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['query']['bias'].flatten()
    k_b = pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['key']['bias'].flatten()
    v_b = pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['value']['bias'].flatten()

    qkv_b = np.array(jnp.concatenate([q_b, k_b, v_b], axis= 0))
    tf_dict[torch_key_prefix + '.attention.to_qkv.weight' ] = torch.from_numpy(qkv_w) 
    tf_dict[torch_key_prefix + '.attention.to_qkv.bias'] = torch.from_numpy(qkv_b)   

    # out 
    weight_shape = pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['out']['kernel'].shape
    weight = pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['out']['kernel'].reshape(weight_shape[0]* weight_shape[1], weight_shape[2])
    weight = np.array(jnp.transpose(weight))
    tf_dict[torch_key_prefix + '.attention.to_out.0.weight' ] = torch.from_numpy(weight) 
    tf_dict[torch_key_prefix + '.attention.to_out.0.bias'] = torch.from_numpy(pretain_tf_model['Transformer'][tf_key]['MultiHeadDotProductAttention_1']['out']['bias']) 


img = torch.randn(1, 3, input_size, input_size)
mask = torch.ones(1, input_size//patch_size, input_size//patch_size).bool() # optional mask, designating which patch to attend to
preds = v(img, mask = mask) # (1, 1000)

print(preds.flatten()[0:10])

v.load_state_dict(tf_dict)

preds = v(img, mask = mask) # (1, 1000)

print(preds.flatten()[0:10])

# print(pretain_tf_model(img))

# torch.save
torch.save(v.state_dict(), "imagenet21k+imagenet2012_ViT-B_16-224.pth")