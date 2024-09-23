import math
from typing import Optional

import tensorlayerx as tlx
from tensorlayerx import nn

from .transform import is_nchw
from .act import get_activation


def pair(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def get_initializer(initializer_range: float = 0.02):
    return tlx.initializers.TruncatedNormal(stddev=initializer_range)


def shape_list(x):
    return tlx.get_tensor_shape(x)


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        hidden_size,
        initializer_range,
        hidden_dropout_prob,
        name="",
        data_format="channels_first",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.data_format = data_format
        self.patch_embeddings = PatchEmbeddings(
            image_size,
            patch_size,
            num_channels,
            hidden_size,
            initializer_range,
            name=name + "/patch_embeddings",
            data_format=data_format,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

        num_patches = self.patch_embeddings.num_patches
        self.cls_token = self._get_weights(
            shape=(1, 1, hidden_size),
            init=self.str_to_init("zeros"),
            trainable=True,
            var_name="cls_token",
            order=True,
        )
        self.position_embeddings = self._get_weights(
            shape=(1, num_patches + 1, hidden_size),
            init=self.str_to_init("zeros"),
            trainable=True,
            var_name="position_embeddings",
            order=True,
        )
        self.patch_size = patch_size

    def interpolate_pos_encoding(self, embeddings, height, width):
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        batch_size, seq_len, dim = shape_list(embeddings)
        npatch = seq_len - 1

        _, N, _ = shape_list(self.position_embeddings)
        N -= 1

        if npatch == N and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        h0 = height // self.patch_size
        w0 = width // self.patch_size
        patch_pos_embed = tlx.resize(
            tlx.reshape(
                patch_pos_embed, shape=(1, int(math.sqrt(N)), int(math.sqrt(N)), dim)
            ),
            output_size=(h0, w0),
            method="bicubic",
            antialias=False,
        )

        shape = shape_list(patch_pos_embed)
        assert h0 == shape[-3] and w0 == shape[-2]
        patch_pos_embed = tlx.reshape(tensor=patch_pos_embed, shape=(1, -1, dim))
        return tlx.concat([class_pos_embed, patch_pos_embed], axis=1)

    def forward(self, x, interpolate_pos_encoding=False):
        B, C, H, W = x.shape
        x = self.patch_embeddings(x, interpolate_pos_encoding)

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = tlx.tile(self.cls_token, [B, 1, 1])
        embeddings = tlx.concat([cls_tokens, x], axis=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, H, W)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        hidden_size,
        initializer_range,
        name="",
        data_format="channels_first",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        image_size = pair(image_size)
        patch_size = pair(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_channels = num_channels
        self.embed_dim = hidden_size
        self.data_format = data_format

        self.projection = nn.Conv2d(
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
            data_format="channels_first",
            b_init="zeros",
            W_init=get_initializer(initializer_range),
            name=name + "/projection",
            in_channels=3,
        )

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        if not is_nchw(self.data_format):
            pixel_values = tlx.transpose(pixel_values, (0, 3, 1, 2))
        B, C, H, W = pixel_values.shape

        if not interpolate_pos_encoding:
            iW, iH = self.image_size
            if (H, W) != (iH, iW):
                raise ValueError(
                    f"Input image size ({H}*{W}) doesn't match model ({iH}*{iW})."
                )

        projection = self.projection(pixel_values)

        # Change the 2D spatial dimensions to a single temporal dimension.
        # shape = (batch_size, num_patches, out_channels=embed_dim)
        num_patches = (W // self.patch_size[1]) * (H // self.patch_size[0])
        x = tlx.reshape(tensor=projection, shape=(B, num_patches, -1))
        return x


class ViTSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        initializer_range,
        attention_probs_dropout_prob,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number "
                f"of attention heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(
            out_features=self.all_head_size,
            W_init=get_initializer(initializer_range),
            name=name + "/query",
            in_features=hidden_size,
        )
        self.key = nn.Linear(
            out_features=self.all_head_size,
            W_init=get_initializer(initializer_range),
            name=name + "/key",
            in_features=hidden_size,
        )
        self.value = nn.Linear(
            out_features=self.all_head_size,
            W_init=get_initializer(initializer_range),
            name=name + "/value",
            in_features=hidden_size,
        )
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, tensor, batch_size):
        tensor = tlx.reshape(
            tensor=tensor,
            shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size),
        )

        return tlx.transpose(tensor, perm=[0, 2, 1, 3])

    def forward(
        self,
        hidden_states,
        head_mask,
    ):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch size, num_heads, seq_len_q, seq_len_k)
        key_layer = tlx.transpose(key_layer, (0, 1, 3, 2))
        attention_scores = query_layer @ key_layer
        attention_scores /= self.sqrt_att_head_size

        # Normalize the attention scores to probabilities.
        attention_probs = tlx.softmax(logits=attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = tlx.multiply(attention_probs, head_mask)

        attention_output = attention_probs @ value_layer
        attention_output = tlx.transpose(attention_output, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tlx.reshape(
            tensor=attention_output, shape=(batch_size, -1, self.all_head_size)
        )
        outputs = (attention_output, attention_probs)

        return outputs


class ViTSelfOutput(nn.Module):
    def __init__(
        self, hidden_size, initializer_range, hidden_dropout_prob, name="", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.dense = nn.Linear(
            out_features=hidden_size,
            W_init=get_initializer(initializer_range),
            name=name + "/dense",
            in_features=hidden_size,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        initializer_range,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.self_attention = ViTSelfAttention(
            hidden_size,
            num_attention_heads,
            initializer_range,
            attention_probs_dropout_prob,
            name=name + "/attention",
        )
        self.dense_output = ViTSelfOutput(
            hidden_size, initializer_range, hidden_dropout_prob, name=name + "/output"
        )

    def forward(
        self,
        input_tensor,
        head_mask,
    ):
        self_outputs = self.self_attention(input_tensor, head_mask=head_mask)
        attention_output = self.dense_output(self_outputs[0], input_tensor=input_tensor)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them

        return outputs


class ViTIntermediate(nn.Module):
    def __init__(
        self,
        intermediate_size,
        initializer_range,
        hidden_size,
        hidden_act,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense = nn.Linear(
            out_features=intermediate_size,
            W_init=get_initializer(initializer_range),
            name=name + "/dense",
            in_features=hidden_size,
        )

        if isinstance(hidden_act, str):
            self.intermediate_act_fn = get_activation(hidden_act)
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(
        self,
        hidden_size,
        initializer_range,
        intermediate_size,
        hidden_dropout_prob,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dense = nn.Linear(
            out_features=hidden_size,
            W_init=get_initializer(initializer_range),
            name=name + "/dense",
            in_features=intermediate_size,
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class ViTLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        initializer_range,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.attention = ViTAttention(
            hidden_size,
            num_attention_heads,
            initializer_range,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            name=name + "/attention",
        )
        self.intermediate = ViTIntermediate(
            intermediate_size,
            initializer_range,
            hidden_size,
            hidden_act,
            name=name + "/intermediate",
        )
        self.vit_output = ViTOutput(
            hidden_size,
            initializer_range,
            intermediate_size,
            hidden_dropout_prob,
            name=name + "/output",
        )

        self.layernorm_before = nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            name=name + "/layernorm_before",
        )
        self.layernorm_before.build([None, None, hidden_size])
        self.layernorm_after = nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            name=name + "/layernorm_after",
        )
        self.layernorm_after.build([None, None, hidden_size])

    def forward(
        self,
        hidden_states,
        head_mask,
    ):
        input_tensor = self.layernorm_before(hidden_states)
        attention_outputs = self.attention(
            input_tensor,
            head_mask=head_mask,
        )
        attention_output = attention_outputs[0]

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        intermediate_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.vit_output(intermediate_output, input_tensor=hidden_states)
        outputs = (layer_output,) + attention_outputs[
            1:
        ]  # add attentions if we output them

        return outputs


class ViTEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        initializer_range,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        num_hidden_layers,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.layers = [
                ViTLayer(
                    hidden_size,
                    num_attention_heads,
                    initializer_range,
                    attention_probs_dropout_prob,
                    hidden_dropout_prob,
                    intermediate_size,
                    hidden_act,
                    layer_norm_eps,
                    name=name + f"/layer_._{i}",
                )
                for i in range(num_hidden_layers)
            ]

    def forward(
        self,
        hidden_states,
        head_mask,
    ):
        all_hidden_states = ()

        for i, layer_module in enumerate(self.layers):
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                head_mask=head_mask[i],
            )
            hidden_states = layer_outputs[0]

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)


class ViTMainLayer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        hidden_size,
        initializer_range,
        hidden_dropout_prob,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        num_hidden_layers,
        add_pooling_layer=True,
        name="",
        data_format="channels_first",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.embeddings = ViTEmbeddings(
            image_size,
            patch_size,
            num_channels,
            hidden_size,
            initializer_range,
            hidden_dropout_prob,
            name=name + "/embeddings",
            data_format=data_format,
        )
        self.encoder = ViTEncoder(
            hidden_size,
            num_attention_heads,
            initializer_range,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
            intermediate_size,
            hidden_act,
            layer_norm_eps,
            num_hidden_layers,
            name=name + "/encoder",
        )
        self.layernorm = nn.LayerNorm(
            normalized_shape=hidden_size,
            epsilon=layer_norm_eps,
            name=name + "/layernorm",
        )
        self.layernorm.build([None, None, hidden_size])
        self.pooler = (
            ViTPooler(hidden_size, initializer_range, name=name + "/pooler")
            if add_pooling_layer
            else None
        )
        self.num_hidden_layers = num_hidden_layers

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs,
    ):
        embedding_output = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        head_mask = [None] * self.num_hidden_layers

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return (sequence_output, pooled_output) + encoder_outputs[1:]


class ViTModel(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_channels,
        hidden_size,
        initializer_range,
        hidden_dropout_prob,
        num_attention_heads,
        attention_probs_dropout_prob,
        intermediate_size,
        hidden_act,
        layer_norm_eps,
        num_hidden_layers,
        *inputs,
        data_format="channels_first",
        add_pooling_layer=True,
        name="",
        **kwargs,
    ):
        super().__init__(name=name, *inputs, **kwargs)

        self.vit = ViTMainLayer(
            image_size,
            patch_size,
            num_channels,
            hidden_size,
            initializer_range,
            hidden_dropout_prob,
            num_attention_heads,
            attention_probs_dropout_prob,
            intermediate_size,
            hidden_act,
            layer_norm_eps,
            num_hidden_layers,
            add_pooling_layer=add_pooling_layer,
            name="vit",
            data_format=data_format,
        )

    def forward(
        self,
        pixel_values,
        interpolate_pos_encoding=None,
        **kwargs,
    ):
        outputs = self.vit(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        return outputs


class ViTPooler(nn.Module):
    def __init__(self, hidden_size, initializer_range, name="", **kwargs):
        super().__init__(name=name, **kwargs)

        self.dense = nn.Linear(
            out_features=hidden_size,
            W_init=get_initializer(initializer_range),
            act="tanh",
            name=name + "/dense",
            in_features=hidden_size,
        )

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output
