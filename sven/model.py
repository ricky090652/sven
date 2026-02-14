import os
import torch
from typing import Optional, Tuple, Union, List
from transformers import AutoTokenizer, AutoConfig, logging, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions
from sven.hf import CodeGenForCausalLM, XGLMForCausalLM, GPT2LMHeadCustomModel, GPT2CustomConfig

class CodeGenPrefixCausalLM(CodeGenForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class IncoderPrefixLM(XGLMForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.d_model // config.attention_heads
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_layers):
                for _ in range(2):
                    param_size = (config.attention_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.num_layers):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_layers * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, use_cache=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1:]
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)
        # first step, decoder_cached_states are empty
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": None,
            "past_key_values": past,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            head_mask,
            cross_attn_head_mask,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class SantaPrefixLM(GPT2LMHeadCustomModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_embed_per_head = config.n_embd // config.n_head
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.n_layer):
                # mha
                for _ in range(2):
                    param_size = (config.n_head, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.zeros(param_size, requires_grad=True))
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        past = list()
        for i in range(self.config.n_layer):
            past.append(list())
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.n_layer * 2 + i * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            past[i].append(torch.stack(key_stack))
            past[i].append(torch.stack(val_stack))
        return past

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        else:
            control_ids = [kwargs['control_id']] * input_ids.shape[0]
            past = self.get_past_from_prefix(control_ids)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": None,
            "attention_mask": None,
            "token_type_ids": token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        control_id = None, # placeholder for passing checks of huggingface, actually unused in this function
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

class QwenPrefixCausalLM(Qwen2ForCausalLM):
    """Qwen2 model with prefix tuning support for security-aware code generation."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Qwen uses GQA: num_key_value_heads (2) < num_attention_heads (12)
        self.n_kv_heads = config.num_key_value_heads
        self.n_embed_per_head = config.hidden_size // config.num_attention_heads
        
        self.prefix_params = torch.nn.ParameterList()
        for _ in range(config.n_control):
            for _ in range(config.num_hidden_layers):
                for _ in range(2):  # key and value
                    # Use KV heads count for GQA compatibility
                    param_size = (self.n_kv_heads, config.n_prefix_token, self.n_embed_per_head)
                    param = torch.nn.Parameter(torch.randn(param_size, requires_grad=True) * 0.01)
                    self.prefix_params.append(param)
        self.dropout = torch.nn.Dropout(config.prefix_dropout)

    def get_past_from_prefix(self, control_ids):
        """Build past_key_values from prefix parameters using DynamicCache for new transformers."""
        cache = DynamicCache()
        
        for layer_idx in range(self.config.num_hidden_layers):
            key_stack, val_stack = [], []
            for control_id in control_ids:
                key_idx = control_id * self.config.num_hidden_layers * 2 + layer_idx * 2
                val_idx = key_idx + 1
                key = self.dropout(self.prefix_params[key_idx])
                val = self.dropout(self.prefix_params[val_idx])
                key_stack.append(key)
                val_stack.append(val)
            
            # Stack along batch dimension: (batch, n_kv_heads, seq_len, head_dim)
            layer_key = torch.stack(key_stack, dim=0)  # (batch, n_kv_heads, n_prefix, head_dim)
            layer_val = torch.stack(val_stack, dim=0)
            
            # Update the cache for this layer
            cache.update(layer_key, layer_val, layer_idx)
        
        return cache

    def generate(self, input_ids, **kwargs):
        """Override generate to extend model_kwargs with prefix info before the loop."""
        control_id = kwargs.pop('control_id', None)
        self._pending_control_id = control_id
        
        if control_id is not None:
            # Extend attention_mask with prefix ones in model_kwargs.
            # This ensures _update_model_kwargs_for_generation correctly
            # tracks the prefix tokens across ALL generation steps.
            attention_mask = kwargs.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            prefix_mask = torch.ones(
                input_ids.shape[0], self.config.n_prefix_token,
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            kwargs['attention_mask'] = torch.cat([prefix_mask, attention_mask], dim=1)
        
        try:
            result = super().generate(input_ids, **kwargs)
        finally:
            self._pending_control_id = None
        return result

    def _get_initial_cache_position(self, seq_length, device, model_kwargs):
        """Override to offset cache_position by n_prefix when prefix is active."""
        model_kwargs = super()._get_initial_cache_position(seq_length, device, model_kwargs)
        if getattr(self, '_pending_control_id', None) is not None:
            n_prefix = self.config.n_prefix_token
            model_kwargs["cache_position"] = model_kwargs["cache_position"] + n_prefix
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        cache_position = kwargs.get("cache_position")

        # Check if cache already has content (= subsequent step)
        has_cached = (
            past_key_values is not None
            and hasattr(past_key_values, 'get_seq_length')
            and past_key_values.get_seq_length() > 0
        )

        if has_cached:
            # Subsequent steps: only use the last token
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids[:, -1:]
            else:
                position_ids = None
        else:
            # First step: inject prefix into the cache
            control_id = getattr(self, '_pending_control_id', None)
            if control_id is not None:
                control_ids = [control_id] * input_ids.shape[0]
                prefix_cache = self.get_past_from_prefix(control_ids)
                if past_key_values is not None and hasattr(past_key_values, 'update'):
                    for layer_idx in range(len(prefix_cache)):
                        k, v = prefix_cache[layer_idx]
                        past_key_values.update(k, v, layer_idx)
                else:
                    past_key_values = prefix_cache

            # Set position_ids from cache_position (already offset by generate)
            if cache_position is not None:
                position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
            else:
                position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        control_id = None,  # placeholder for passing checks of huggingface
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

def model_from_pretrained(lm_path, model_type, config):
    kwargs = dict()
    if lm_path.startswith('Salesforce/codegen-'):
        if model_type == 'lm':
            model_class = CodeGenForCausalLM
        elif model_type == 'prefix':
            model_class = CodeGenPrefixCausalLM
        else:
            assert False
    elif lm_path.startswith('facebook/incoder-'):
        if config is not None:
            config.attention_dropout = 0.0
            config.dropout = 0.0
        if model_type == 'lm':
            model_class = XGLMForCausalLM
        elif model_type == 'prefix':
            model_class = IncoderPrefixLM
        else:
            assert False
    elif lm_path == 'bigcode/santacoder':
        kwargs['revision'] = 'mha'
        if config is not None:
            config.attn_pdrop = 0.0
            config.embd_pdrop = 0.0
            config.resid_pdrop = 0.0
        if model_type == 'lm':
            model_class = GPT2LMHeadCustomModel
        elif model_type == 'prefix':
            model_class = SantaPrefixLM
        else:
            assert False
    elif lm_path.startswith('Qwen/Qwen2.5-Coder'):
        # Enable BF16 and Flash Attention 2 for faster training
        kwargs['torch_dtype'] = torch.bfloat16
        kwargs['attn_implementation'] = 'flash_attention_2'
        if model_type == 'lm':
            model_class = Qwen2ForCausalLM
        elif model_type == 'prefix':
            model_class = QwenPrefixCausalLM
        else:
            assert False
    else:
        assert False

    if config is None:
        model = model_class.from_pretrained(lm_path, **kwargs)
    else:
        model = model_class.from_pretrained(lm_path, **kwargs, config=config)

    return model

def config_from_pretrained(lm_path, path):
    if lm_path == 'bigcode/santacoder':
        return GPT2CustomConfig.from_pretrained(path, revision='mha')
    elif lm_path.startswith('Qwen/Qwen2.5-Coder'):
        return AutoConfig.from_pretrained(path)
    else:
        return AutoConfig.from_pretrained(path)

def save_model(model, path, args):
    if type(model) in (CodeGenPrefixCausalLM, IncoderPrefixLM, SantaPrefixLM, QwenPrefixCausalLM):
        # For prefix models, only save the prefix parameters (not the full model)
        config_file = os.path.join(path)
        model.config.save_pretrained(config_file)
        prefix_file = os.path.join(path, 'pytorch_model.bin')
        state_dict = model.prefix_params.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        torch.save(state_dict, prefix_file)
        lm_path_file = os.path.join(path, 'lm.txt')
        with open(lm_path_file, 'w') as f:
            f.write(args.pretrain_dir)
    else:
        model.save_pretrained(path)

def load_model(model_type, path, is_training, args):
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_type == 'lm':
        config = config_from_pretrained(path, path)
        model = model_from_pretrained(path, model_type, config)
    elif model_type == 'prefix':
        if is_training:
            lm_path = path
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = args.n_prefix_token
            lm_config.prefix_dropout = args.dropout
            lm_config.n_control = 2
            model = model_from_pretrained(lm_path, model_type, lm_config)
            # Reinitialize prefix params for Qwen models (from_pretrained may corrupt initialization)
            # Also ensure they use BF16 dtype to match the model
            if lm_path.startswith('Qwen/'):
                with torch.no_grad():
                    for param in model.prefix_params:
                        param.data = (torch.randn_like(param) * 0.01).to(torch.bfloat16)
        else:
            lm_path_file = os.path.join(path, 'lm.txt')
            assert os.path.exists(lm_path_file)
            with open(lm_path_file) as f:
                lm_path = f.read()
            prefix_config = config_from_pretrained(lm_path, path)
            lm_config = config_from_pretrained(lm_path, lm_path)
            lm_config.n_prefix_token = prefix_config.n_prefix_token
            lm_config.prefix_dropout = prefix_config.prefix_dropout
            lm_config.n_control = prefix_config.n_control
            model = model_from_pretrained(lm_path, model_type, lm_config)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            model.prefix_params.load_state_dict(torch.load(prefix_file))
    else:
        assert False

    model.resize_token_embeddings(len(tokenizer))
    input_device = parallelize_model(model, args)
    return tokenizer, model, input_device

def parallelize_model(model, args):
    if args.n_gpu > 1:
        model.parallelize()
        input_device = model.transformer.first_device
    else:
        model.to(args.device)
        input_device = args.device
    return input_device