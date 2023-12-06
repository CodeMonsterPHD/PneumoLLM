import torch
import json
from phe_llm import ModelArgs, Tokenizer, Transformer
from phe_llm.mm_adapter import set_Clip_Adapter, set_MMAdapter
from pathlib import Path


def _load_and_redistribute_checkpoint(llama_model_path, model_name):
    # with open(Path(llama_model_path) / model_name / 'params.json') as f:
    with open(Path(llama_model_path) / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(llama_model_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + '/consolidated.00.pth', map_location="cpu")
        # checkpoint = torch.load(llama_model_path + model_name + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params
    checkpoints = (Path(llama_model_path) / model_name).glob('*.pth')
    checkpoints = sorted(checkpoints)
    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))
    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict
    return checkpoint, tokenizer, params

# write here
def phellm(args):
    llama_model_path =args.llama_model_path
    model_name = args.llm_model
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, model_name)
    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len, max_batch_size=32, hidden_proj=args.hidden_proj, drop_path=args.drop_path, **params
    )
    model_args.vocab_size = tokenizer.n_words
    model_args.RPO_K = args.RPO_K
    # load with GPU
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_args.multiscale=args.multiscale
    llama = Transformer(model_args)
    # delete language encoder
    del llama.backbone.transformer

    torch.set_default_tensor_type(torch.FloatTensor)
    if args.bits in ['4bit','8bit']:
        from util.quantization import quant_model_bnb
        llama.layers=quant_model_bnb(llama.layers,quant_bit=args.bits)
    llama.load_state_dict(checkpoint, strict=False)
    if args.adapter_type=='block' or args.adapter_type=='attn':
        # aa=0
        set_MMAdapter(llama,args.adapter_type,dim=args.adapter_dim,s=args.adapter_scale,t=args.temperature,gradient_checkpointing=args.gradient_checkpointing)
        set_Clip_Adapter(llama.backbone.visual,args.visual_adapter_type,dim=args.adapter_dim,s=args.adapter_scale,t=args.temperature)

    learnable_keys=['prompt', 'adapter']
    total=0.
    trainable_names=[]
    for name, param in llama.named_parameters():
        for key in learnable_keys:
            if key in name:
                param.requires_grad = True
                param.data = param.data.float()
                total += param.nelement()
                trainable_names.append(name)
            else:
                param.requires_grad = False
    print('  + Number of trainable params: %.10fM' % (total / 1e6))
    return llama

