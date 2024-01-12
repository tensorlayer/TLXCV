import numpy as np
import tensorlayerx as tlx
from scipy.optimize import linear_sum_assignment
from tensorlayerx import nn


class Detr(nn.Module):
    def __init__(
        self,
        num_queries=100,
        num_encoder_layers=6,
        num_decoder_layers=6,
        model_dim=256,
        backbone_bn_shape=64,
        backbone_layer1_bn_shape=(
            (64, 64, 256, 256), (64, 64, 256), (64, 64, 256)),
        backbone_layer2_bn_shape=(
            (128, 128, 512, 512),
            (128, 128, 512),
            (128, 128, 512),
            (128, 128, 512),
        ),
        backbone_layer3_bn_shape=(
            (256, 256, 1024, 1024),
            (256, 256, 1024),
            (256, 256, 1024),
            (256, 256, 1024),
            (256, 256, 1024),
            (256, 256, 1024),
        ),
        backbone_layer4_bn_shape=(
            (512, 512, 2048, 2048),
            (512, 512, 2048),
            (512, 512, 2048),
        ),
        return_intermediate_dec=True,
        num_classes=92,
        class_cost=1,
        bbox_cost=5,
        giou_cost=2,
        num_labels=91,
        dice_loss_coefficient=1,
        bbox_loss_coefficient=5,
        giou_loss_coefficient=2,
        eos_coefficient=0.1,
        auxiliary_loss=True,
        data_format="channels_first",
        name="detr",
    ):
        """
        :param num_queries: (:obj:`int`, `optional`, defaults to 100):
            Number of object queries, i.e. detection slots. This is the maximal number of objects
        :param num_encoder_layers: (:obj:`int`, `optional`, defaults to 6):
            Number of decoder layers.
        :param num_decoder_layers: (:obj:`int`, `optional`, defaults to 6):
            Number of encoder layers.
        :param model_dim: (:obj:`int`, `optional`, defaults to 256):
            Dimension of the layers.
        :param backbone_bn_shape: resnet bn shape
        :param backbone_layer1_bn_shape: resnet layer1 bn shape
        :param backbone_layer2_bn_shape: resnet layer2 bn shape
        :param backbone_layer3_bn_shape: resnet layer3 bn shape
        :param backbone_layer4_bn_shape: resnet layer4 bn shape
        :param return_intermediate_dec:
        :param num_classes: num of object classes + 1
        :param class_cost: (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the classification error in the Hungarian matching cost.
        :param bbox_cost: (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 error of the bounding box coordinates in the Hungarian matching cost.
        :param giou_cost: (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss of the bounding box in the Hungarian matching cost.
        :param num_labels: num of object classes
        :param dice_loss_coefficient: (:obj:`float`, `optional`, defaults to 1):
            Relative weight of the DICE/F-1 loss in the panoptic segmentation loss.
        :param bbox_loss_coefficient: (:obj:`float`, `optional`, defaults to 5):
            Relative weight of the L1 bounding box loss in the object detection loss.
        :param giou_loss_coefficient: (:obj:`float`, `optional`, defaults to 2):
            Relative weight of the generalized IoU loss in the object detection loss.
        :param eos_coefficient: (:obj:`float`, `optional`, defaults to 0.1):
            Relative classification weight of the 'no-object' class in the object detection loss.
        :param auxiliary_loss: (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether auxiliary decoding losses (loss at each decoder layer) are to be used.
        :param name:
        """
        super().__init__(name=name)
        self.num_queries = num_queries
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.model_dim = model_dim
        self.backbone_bn_shape = backbone_bn_shape
        self.backbone_layer1_bn_shape = backbone_layer1_bn_shape
        self.backbone_layer2_bn_shape = backbone_layer2_bn_shape
        self.backbone_layer3_bn_shape = backbone_layer3_bn_shape
        self.backbone_layer4_bn_shape = backbone_layer4_bn_shape
        self.return_intermediate_dec = return_intermediate_dec
        self.num_classes = num_classes
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.num_labels = num_labels
        self.dice_loss_coefficient = dice_loss_coefficient
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.eos_coefficient = eos_coefficient
        self.auxiliary_loss = auxiliary_loss
        self.data_format = data_format

        self.backbone = ResNet50Backbone(
            backbone_bn_shape,
            backbone_layer1_bn_shape,
            backbone_layer2_bn_shape,
            backbone_layer3_bn_shape,
            backbone_layer4_bn_shape,
            data_format=data_format,
            name=name + "/backbone",
        )
        self.input_proj = nn.Conv2d(
            in_channels=2048,
            out_channels=model_dim,
            kernel_size=(1, 1),
            data_format=data_format,
            name=name + "/input_proj",
        )
        self.query_embed = FixedEmbedding(
            (num_queries, model_dim), name=name + "/query_embed"
        )
        self.transformer = Transformer(
            model_dim=model_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            return_intermediate_dec=return_intermediate_dec,
            name=name + "/transformer",
        )

        self.pos_encoder = PositionEmbeddingSine(
            num_pos_features=model_dim // 2,
            normalize=True,
            name=name + "/position_embedding_sine",
            data_format=data_format,
        )

        self.class_embed = nn.Linear(
            in_features=model_dim, out_features=num_classes, name="class_embed"
        )

        self.bbox_embed_linear1 = nn.Linear(
            in_features=model_dim, out_features=model_dim, name="bbox_embed_linear1"
        )
        self.bbox_embed_linear2 = nn.Linear(
            in_features=model_dim, out_features=model_dim, name="bbox_embed_linear2"
        )
        self.bbox_embed_linear3 = nn.Linear(
            in_features=model_dim, out_features=4, name="bbox_embed_linear3"
        )
        self.activation = tlx.ReLU()

    def loss_fn(self, outputs, labels):
        logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        # First: create the matcher
        matcher = DetrHungarianMatcher(
            class_cost=self.class_cost,
            bbox_cost=self.bbox_cost,
            giou_cost=self.giou_cost,
        )
        # Second: create the criterion
        losses = ["labels", "boxes", "cardinality"]
        criterion = DetrLoss(
            matcher=matcher,
            num_classes=self.num_labels,
            eos_coef=self.eos_coefficient,
            losses=losses,
        )
        # Third: compute the losses, based on outputs and labels
        outputs_loss = {}
        outputs_loss["logits"] = logits
        outputs_loss["pred_boxes"] = pred_boxes
        if self.auxiliary_loss:
            outputs_loss["auxiliary_outputs"] = outputs["aux"]

        loss_dict = criterion(outputs_loss, labels)
        # Fourth: compute total loss, as a weighted sum of the various losses
        weight_dict = {
            "loss_ce": 1,
            "loss_bbox": self.bbox_loss_coefficient,
            "loss_giou": self.giou_loss_coefficient,
        }

        if self.auxiliary_loss:
            aux_weight_dict = {}
            for i in range(self.num_decoder_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        return loss

    def downsample_masks(self, masks, x):
        masks = tlx.cast(masks, tlx.float32)

        if self.data_format == "channels_first":
            masks = tlx.expand_dims(masks, axis=1)
            h1, w1 = tlx.get_tensor_shape(x)[2:4]
            h2, w2 = tlx.get_tensor_shape(masks)[2:4]
        else:
            masks = tlx.expand_dims(masks, axis=-1)
            h1, w1 = tlx.get_tensor_shape(x)[1:3]
            h2, w2 = tlx.get_tensor_shape(masks)[1:3]
        masks = tlx.Resize(
            scale=(h1 / h2, w1 / w2),
            method="nearest",
            antialias=False,
            data_format=self.data_format,
        )(masks)
        if self.data_format == "channels_first":
            masks = tlx.squeeze(masks, 1)
        else:
            masks = tlx.squeeze(masks, -1)

        masks = tlx.cast(masks, tlx.bool)
        return masks

    def forward(self, input, downsample_masks=True):
        images, pixel_mask = input["images"], input["pixel_mask"]

        feature_maps = self.backbone(images)
        x = feature_maps[-1]

        if pixel_mask is None or tlx.get_tensor_shape(x)[0] == 1:
            shape = tlx.get_tensor_shape(x)
            if self.data_format == "channels_first":
                del shape[1]
            else:
                del shape[-1]
            pixel_mask = tlx.ones(shape, tlx.bool)
        else:
            if downsample_masks:
                pixel_mask = self.downsample_masks(pixel_mask, x)

        pos_encoding = self.pos_encoder(pixel_mask)
        # feature_map = x
        projected_feature_map = self.input_proj(x)
        hs, memory = self.transformer(
            projected_feature_map, pixel_mask, self.query_embed(
                None), pos_encoding
        )

        transformer_output, memory, feature_maps, masks, projected_feature_map = (
            hs,
            memory,
            feature_maps,
            pixel_mask,
            projected_feature_map,
        )

        outputs_class = self.class_embed(transformer_output)
        box_ftmps = self.activation(
            self.bbox_embed_linear1(transformer_output))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tlx.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "memory": memory,
            "feature_maps": feature_maps,
            "masks": masks,
            "transformer_output": transformer_output,
            "projected_feature_map": projected_feature_map,
        }

        output["aux"] = []
        for i in range(0, self.num_decoder_layers - 1):
            out_class = outputs_class[i]
            pred_boxes = outputs_coord[i]
            output["aux"].append(
                {"logits": out_class, "pred_boxes": pred_boxes})

        return output


class FrozenBatchNorm2D(nn.Module):
    def __init__(
        self, backbone_bn_shape, eps=1e-5, data_format="channels_first", name=None
    ):
        super().__init__(name=name)
        self.eps = eps
        if data_format == "channels_first":
            shape = [1, backbone_bn_shape, 1, 1]
        else:
            shape = [backbone_bn_shape]

        self.weight = self._get_weights(
            var_name="weight",
            shape=shape,
            init=self.str_to_init("xavier_uniform"),
            trainable=False,
            order=True,
        )
        self.bias = self._get_weights(
            var_name="bias",
            shape=shape,
            init=self.str_to_init("xavier_uniform"),
            trainable=False,
            order=True,
        )
        self.running_mean = self._get_weights(
            var_name="running_mean",
            shape=shape,
            init=self.str_to_init("zeros"),
            trainable=False,
            order=True,
        )
        self.running_var = self._get_weights(
            var_name="running_var",
            shape=shape,
            init=self.str_to_init("ones"),
            trainable=False,
            order=True,
        )

    def forward(self, x):
        scale = self.weight * tlx.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift


class FixedEmbedding(nn.Module):
    def __init__(self, embed_shape, name=None):
        super().__init__(name=name)

        self.w = self._get_weights(
            var_name="kernel",
            shape=embed_shape,
            init=self.str_to_init("xavier_uniform"),
            trainable=True,
        )

    def forward(self, x=None):
        return self.w


class ResNetBase(nn.Module):
    def __init__(
        self,
        backbone_bn_shape,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)

        self.pad1 = nn.ZeroPad2d(
            ((3, 3), (3, 3)), data_format=data_format, name=name + "/pad1"
        )
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding="valid",
            b_init=None,
            data_format=data_format,
            name=name + "/conv1",
        )
        self.bn1 = FrozenBatchNorm2D(
            backbone_bn_shape, data_format=data_format, name=name + "/bn1"
        )
        self.relu = nn.ReLU()
        self.pad2 = nn.ZeroPad2d(
            ((1, 1), (1, 1)), data_format=data_format, name=name + "/pad2"
        )
        self.maxpool = nn.MaxPool2d(
            (3, 3), (2, 2), "valid", data_format=data_format)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad2(x)
        x = self.maxpool(x)

        xs = []
        x = self.layer1(x)
        xs.append(x)
        x = self.layer2(x)
        xs.append(x)
        x = self.layer3(x)
        xs.append(x)
        x = self.layer4(x)
        xs.append(x)
        return xs


class ResNet50Backbone(ResNetBase):
    def __init__(
        self,
        backbone_bn_shape,
        backbone_layer1_bn_shape,
        backbone_layer2_bn_shape,
        backbone_layer3_bn_shape,
        backbone_layer4_bn_shape,
        replace_stride_with_dilation=[False, False, False],
        data_format="channels_first",
        name=None,
    ):
        super().__init__(backbone_bn_shape, data_format, name)

        self.layer1 = ResidualBlock(
            num_bottlenecks=3,
            dim1=64,
            dim2=256,
            strides=1,
            first_in_channels=64,
            bottlenecks_bn_shape=backbone_layer1_bn_shape,
            replace_stride_with_dilation=False,
            data_format=data_format,
            name=name + "/layer1",
        )
        self.layer2 = ResidualBlock(
            num_bottlenecks=4,
            dim1=128,
            dim2=512,
            strides=2,
            first_in_channels=backbone_layer1_bn_shape[-1][-1],
            bottlenecks_bn_shape=backbone_layer2_bn_shape,
            replace_stride_with_dilation=replace_stride_with_dilation[0],
            data_format=data_format,
            name=name + "/layer2",
        )
        self.layer3 = ResidualBlock(
            num_bottlenecks=6,
            dim1=256,
            dim2=1024,
            strides=2,
            first_in_channels=backbone_layer2_bn_shape[-1][-1],
            bottlenecks_bn_shape=backbone_layer3_bn_shape,
            replace_stride_with_dilation=replace_stride_with_dilation[1],
            data_format=data_format,
            name=name + "/layer3",
        )
        self.layer4 = ResidualBlock(
            num_bottlenecks=3,
            dim1=512,
            dim2=2048,
            strides=2,
            first_in_channels=backbone_layer3_bn_shape[-1][-1],
            bottlenecks_bn_shape=backbone_layer4_bn_shape,
            replace_stride_with_dilation=replace_stride_with_dilation[2],
            data_format=data_format,
            name=name + "/layer4",
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        num_bottlenecks,
        dim1,
        dim2,
        bottlenecks_bn_shape,
        first_in_channels,
        strides=1,
        replace_stride_with_dilation=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)

        if replace_stride_with_dilation:
            strides = 1
            dilation = 2
        else:
            dilation = 1

        bottlenecks = [
            BottleNeck(
                dim1,
                dim2,
                bottlenecks_bn_shape[0],
                strides=strides,
                first_in_channels=first_in_channels,
                downsample=True,
                data_format=data_format,
                name=name + "/bottlenecks/0",
            )
        ]

        for idx in range(1, num_bottlenecks):
            bottlenecks.append(
                BottleNeck(
                    dim1,
                    dim2,
                    bottlenecks_bn_shape[idx],
                    dilation=dilation,
                    first_in_channels=bottlenecks_bn_shape[idx - 1][-1],
                    data_format=data_format,
                    name=name + "/bottlenecks/" + str(idx),
                )
            )
        self.bottlenecks = nn.Sequential(bottlenecks)

    def forward(self, x):
        for btn in self.bottlenecks:
            x = btn(x)
        return x


class BottleNeck(nn.Module):
    def __init__(
        self,
        dim1,
        dim2,
        bn_shape,
        first_in_channels,
        strides=1,
        dilation=1,
        downsample=False,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)
        self.downsample = downsample

        self.pad = nn.ZeroPad2d(
            ((dilation, dilation), (dilation, dilation)), data_format=data_format
        )
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            out_channels=dim1,
            kernel_size=(1, 1),
            in_channels=first_in_channels,
            padding="valid",
            b_init=None,
            data_format=data_format,
            name=name + "/conv1",
        )
        self.bn1 = FrozenBatchNorm2D(
            bn_shape[0], data_format=data_format, name=name + "/bn1"
        )

        self.conv2 = nn.Conv2d(
            in_channels=dim1,
            out_channels=dim1,
            kernel_size=(3, 3),
            stride=(strides, strides),
            padding="valid",
            dilation=(dilation, dilation),
            b_init=None,
            data_format=data_format,
            name=name + "/conv2",
        )
        self.bn2 = FrozenBatchNorm2D(
            bn_shape[1], data_format=data_format, name=name + "/bn2"
        )

        self.conv3 = nn.Conv2d(
            in_channels=dim1,
            out_channels=dim2,
            kernel_size=(1, 1),
            padding="valid",
            b_init=None,
            data_format=data_format,
            name=name + "/conv3",
        )
        self.bn3 = FrozenBatchNorm2D(
            bn_shape[2], data_format=data_format, name=name + "/bn3"
        )

        if self.downsample:
            self.downsample_conv = nn.Conv2d(
                in_channels=first_in_channels,
                out_channels=dim2,
                kernel_size=(1, 1),
                stride=(strides, strides),
                padding="valid",
                b_init=None,
                data_format=data_format,
                name=name + "/downsample_conv",
            )
            self.downsample_bn = FrozenBatchNorm2D(
                bn_shape[3], data_format=data_format, name=name +
                "/downsample_bn"
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample:
            identity = self.downsample_bn(self.downsample_conv(x))

        out += identity
        out = self.relu(out)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        name=None,
    ):
        super().__init__(name=name)
        self.model_dim = model_dim

        enc_norm = (
            nn.LayerNorm(model_dim, epsilon=1e-5, name=name + "/norm_pre")
            if normalize_before
            else None
        )
        self.encoder = TransformerEncoder(
            model_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            enc_norm,
            num_encoder_layers,
            name=name + "/encoder",
        )

        dec_norm = nn.LayerNorm(model_dim, epsilon=1e-5,
                                name=name + "/decoder/norm")
        dec_norm.build([None, None, None])
        dec_norm._forward_state = True
        self.decoder = TransformerDecoder(
            model_dim,
            num_heads,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
            dec_norm,
            num_decoder_layers,
            return_intermediate=return_intermediate_dec,
            name=name + "/decoder",
        )

    def forward(self, source, mask, query_encoding, pos_encoding):
        batch_size = tlx.get_tensor_shape(source)[0]
        source = tlx.reshape(source, [batch_size, -1, self.model_dim])
        source = tlx.transpose(source, [1, 0, 2])

        pos_encoding = tlx.reshape(
            pos_encoding, [batch_size, -1, self.model_dim])
        pos_encoding = tlx.transpose(pos_encoding, [1, 0, 2])

        query_encoding = tlx.expand_dims(query_encoding, axis=1)
        query_encoding = tlx.tile(query_encoding, [1, batch_size, 1])

        mask = tlx.reshape(mask, [batch_size, -1])

        target = tlx.zeros_like(query_encoding)

        memory = self.encoder(
            source, source_key_padding_mask=mask, pos_encoding=pos_encoding
        )
        hs = self.decoder(
            target,
            memory,
            memory_key_padding_mask=mask,
            pos_encoding=pos_encoding,
            query_encoding=query_encoding,
        )

        hs = tlx.transpose(hs, [0, 2, 1, 3])
        # memory = tlx.transpose(memory, [1, 0, 2])
        # memory = tlx.reshape(memory, [batch_size, rows, cols, self.model_dim])

        return hs, memory


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm=None,
        num_encoder_layers=6,
        name="encoder",
    ):
        super().__init__(name=name)

        enc_layers = [
            EncoderLayer(
                model_dim,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                name=name + "/enc_layers/%d" % i,
            )
            for i in range(num_encoder_layers)
        ]
        self.enc_layers = nn.Sequential(enc_layers)

        self.norm = norm

    def forward(self, x, mask=None, source_key_padding_mask=None, pos_encoding=None):
        for layer in self.enc_layers:
            x = layer(
                x,
                source_mask=mask,
                source_key_padding_mask=source_key_padding_mask,
                pos_encoding=pos_encoding,
            )

        if self.norm:
            x = self.norm(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        norm=None,
        num_decoder_layers=6,
        return_intermediate=False,
        name="decoder",
    ):
        super().__init__(name=name)

        dec_layers = [
            DecoderLayer(
                model_dim,
                num_heads,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
                name=name + "/dec_layers/%d" % i,
            )
            for i in range(num_decoder_layers)
        ]
        self.dec_layers = nn.Sequential(dec_layers)

        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        x,
        memory,
        target_mask=None,
        memory_mask=None,
        target_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_encoding=None,
        query_encoding=None,
    ):
        intermediate = []

        for layer in self.dec_layers:
            x = layer(
                x,
                memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_encoding=pos_encoding,
                query_encoding=query_encoding,
            )

            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(x))
                else:
                    intermediate.append(x)

        if self.return_intermediate:
            return tlx.stack(intermediate, axis=0)

        if self.norm:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        name="layer",
    ):
        super().__init__(name=name)

        self.self_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name=name + "/self_attn"
        )

        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()

        self.linear1 = nn.Linear(
            in_features=model_dim, out_features=dim_feedforward, name=name + "/linear1"
        )
        self.linear2 = nn.Linear(
            in_features=dim_feedforward, out_features=model_dim, name=name + "/linear2"
        )

        self.norm1 = nn.LayerNorm(
            model_dim, epsilon=1e-5, name=name + "/norm1")
        self.norm1.build([None, None, None])
        self.norm1._forward_state = True
        self.norm2 = nn.LayerNorm(
            model_dim, epsilon=1e-5, name=name + "/norm2")
        self.norm2.build([None, None, None])
        self.norm2._forward_state = True

    def forward(
        self, source, source_mask=None, source_key_padding_mask=None, pos_encoding=None
    ):
        if pos_encoding is None:
            query = key = source
        else:
            query = key = source + pos_encoding
        attn_source, _ = self.self_attn(
            (query, key, source),
            attn_mask=source_mask,
            key_padding_mask=source_key_padding_mask,
            # need_weights=False
        )
        source += self.dropout(attn_source)
        source = self.norm1(source)

        x = self.linear1(source)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        source += self.dropout(x)
        source = self.norm2(source)

        return source


class DecoderLayer(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        name="layer",
    ):
        super().__init__(name=name)

        self.self_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name=name + "/self_attn"
        )
        self.multihead_attn = MultiHeadAttention(
            model_dim, num_heads, dropout=dropout, name=name + "/multihead_attn"
        )

        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = nn.ReLU()

        self.linear1 = nn.Linear(
            in_features=model_dim, out_features=dim_feedforward, name=name + "/linear1"
        )
        self.linear2 = nn.Linear(
            in_features=dim_feedforward, out_features=model_dim, name=name + "/linear2"
        )

        self.norm1 = nn.LayerNorm(
            model_dim, epsilon=1e-5, name=name + "/norm1")
        self.norm1.build([None, None, None])
        self.norm1._forward_state = True
        self.norm2 = nn.LayerNorm(
            model_dim, epsilon=1e-5, name=name + "/norm2")
        self.norm2.build([None, None, None])
        self.norm2._forward_state = True
        self.norm3 = nn.LayerNorm(
            model_dim, epsilon=1e-5, name=name + "/norm3")
        self.norm3.build([None, None, None])
        self.norm3._forward_state = True

    def forward(
        self,
        target,
        memory,
        target_mask=None,
        memory_mask=None,
        target_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_encoding=None,
        query_encoding=None,
    ):
        query_tgt = key_tgt = target + query_encoding
        attn_target = self.self_attn(
            (query_tgt, key_tgt, target),
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
            need_weights=False,
        )
        target += self.dropout(attn_target)
        target = self.norm1(target)

        query_tgt = target + query_encoding
        key_mem = memory + pos_encoding

        attn_target2 = self.multihead_attn(
            (query_tgt, key_mem, memory),
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        target += self.dropout(attn_target2)
        target = self.norm2(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        target += self.dropout(x)
        target = self.norm3(target)

        return target


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.0, name="multihead_attn"):
        super().__init__(name=name)

        self.model_dim = model_dim
        self.num_heads = num_heads

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = nn.Dropout(dropout)

        in_dim = self.model_dim * 3

        self.in_proj_weight = self._get_weights(
            var_name="in_proj_weight",
            shape=(in_dim, self.model_dim),
            init=self.str_to_init("xavier_uniform"),
            trainable=True,
        )
        self.in_proj_bias = self._get_weights(
            var_name="in_proj_bias",
            shape=(in_dim,),
            init=self.str_to_init("xavier_uniform"),
            trainable=True,
        )
        self.out_proj_weight = self._get_weights(
            var_name="out_proj_weight",
            shape=(self.model_dim, self.model_dim),
            init=self.str_to_init("xavier_uniform"),
            trainable=True,
        )
        self.out_proj_bias = self._get_weights(
            var_name="out_proj_bias",
            shape=(self.model_dim,),
            init=self.str_to_init("xavier_uniform"),
            trainable=True,
        )

    def forward(self, inputs, attn_mask=None, key_padding_mask=None, need_weights=True):
        query, key, value = inputs

        batch_size = query.shape[1]
        target_len = query.shape[0]
        source_len = key.shape[0]

        W = self.in_proj_weight[: self.model_dim, :]
        b = self.in_proj_bias[: self.model_dim]
        WQ = tlx.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.model_dim: 2 * self.model_dim, :]
        b = self.in_proj_bias[self.model_dim: 2 * self.model_dim]
        WK = tlx.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2 * self.model_dim:, :]
        b = self.in_proj_bias[2 * self.model_dim:]
        WV = tlx.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tlx.reshape(WQ, [target_len, batch_size *
                         self.num_heads, self.head_dim])
        WQ = tlx.transpose(WQ, [1, 0, 2])

        WK = tlx.reshape(WK, [source_len, batch_size *
                         self.num_heads, self.head_dim])
        WK = tlx.transpose(WK, [1, 0, 2])

        WV = tlx.reshape(WV, [source_len, batch_size *
                         self.num_heads, self.head_dim])
        WV = tlx.transpose(WV, [1, 0, 2])

        attn_output_weights = tlx.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = tlx.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = tlx.matmul(attn_output_weights, WV)
        attn_output = tlx.transpose(attn_output, [1, 0, 2])
        attn_output = tlx.reshape(
            attn_output, [target_len, batch_size, self.model_dim])
        attn_output = (
            tlx.matmul(attn_output, self.out_proj_weight, transpose_b=True)
            + self.out_proj_bias
        )

        if need_weights:
            attn_output_weights = tlx.reshape(
                attn_output_weights,
                [batch_size, self.num_heads, target_len, source_len],
            )
            # Retrun the average weight over the heads
            avg_weights = tlx.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights

        return attn_output


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_features=64,
        temperature=10000,
        normalize=False,
        scale=None,
        eps=1e-6,
        data_format="channels_first",
        name=None,
    ):
        super().__init__(name=name)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps
        self.data_format = data_format

    def forward(self, mask):
        not_mask = tlx.cast(mask, tlx.float32)
        y_embed = tlx.cumsum(not_mask, axis=1)
        x_embed = tlx.cumsum(not_mask, axis=2)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = tlx.arange(self.num_pos_features, dtype=tlx.float32)
        dim_t = self.temperature ** (2 *
                                     tlx.floor(dim_t / 2) / self.num_pos_features)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = tlx.stack(
            [tlx.sin(pos_x[..., 0::2]), tlx.cos(pos_x[..., 1::2])], axis=4
        )

        pos_y = tlx.stack(
            [tlx.sin(pos_y[..., 0::2]), tlx.cos(pos_y[..., 1::2])], axis=4
        )

        shape = [pos_x.shape[i] for i in range(3)] + [-1]
        pos_x = tlx.reshape(pos_x, shape)
        pos_y = tlx.reshape(pos_y, shape)

        pos_emb = tlx.concat([pos_y, pos_x], axis=3)

        if self.data_format == "channels_first":
            pos_emb = tlx.transpose(pos_emb, (0, 3, 1, 2))
        return pos_emb


class DetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1
    ):
        """
        Creates the matcher.

        Params:
            class_cost: This is the relative weight of the classification error in the matching cost
            bbox_cost: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            giou_cost: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        assert (
            class_cost != 0 or bbox_cost != 0 or giou_cost != 0
        ), "All costs of the Matcher can't be 0"

    def forward(self, outputs, targets):
        """
        Performs the matching.

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                 objects in the target) containing the class labels "boxes": Tensor of dim [num_target_boxes, 4]
                 containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:

                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries, num_classes = tlx.get_tensor_shape(outputs["logits"])

        # We flatten to compute the cost matrices in a batch

        out_prob = tlx.reshape(outputs["logits"], [
                               bs * num_queries, num_classes])
        out_prob = tlx.softmax(
            out_prob, axis=-1
        )  # [batch_size * num_queries, num_classes]
        out_bbox = tlx.reshape(
            outputs["pred_boxes"], [bs * num_queries, 4]
        )  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = tlx.concat(
            [tlx.convert_to_tensor(v["class_labels"]) for v in targets], axis=0
        )
        tgt_bbox = tlx.concat(
            [tlx.convert_to_tensor(v["boxes"]) for v in targets], axis=0
        )

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -tlx.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        bbox_cost = cdist(out_bbox, tgt_bbox)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(
            center_to_corners_format(
                out_bbox), center_to_corners_format(tgt_bbox)
        )

        # Final cost matrix
        cost_matrix = (
            self.bbox_cost * bbox_cost
            + self.class_cost * class_cost
            + self.giou_cost * giou_cost
        )
        cost_matrix = tlx.reshape(cost_matrix, [bs, num_queries, -1])
        # cost_matrix = cost_matrix.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(tlx.split(cost_matrix, sizes, -1))
        ]
        return [
            (
                tlx.convert_to_tensor(i, dtype=tlx.int64),
                tlx.convert_to_tensor(j, dtype=tlx.int64),
            )
            for i, j in indices
        ]


class DetrLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        """
        Create the criterion.

        A note on the num_classes parameter (copied from original repo in detr.py): "the naming of the `num_classes`
        parameter of the criterion is somewhat misleading. it indeed corresponds to `max_obj_id + 1`, where max_obj_id
        is the maximum id for a class in your dataset. For example, COCO has a max_obj_id of 90, so we pass
        `num_classes` to be 91. As another example, for a dataset that has a single class with id 1, you should pass
        `num_classes` to be 2 (max_obj_id + 1). For more details on this, check the following discussion
        https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

        Parameters:
            matcher: module able to compute a matching between targets and proposals.
            num_classes: number of object categories, omitting the special no-object category.
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = np.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.empty_weight = tlx.convert_to_tensor(
            empty_weight, dtype=tlx.float32)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        assert "logits" in outputs, "No logits were found in the outputs"
        src_logits = outputs["logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = tlx.concat(
            [
                tlx.gather(tlx.convert_to_tensor(t["class_labels"]), J, axis=0)
                for t, (_, J) in zip(targets, indices)
            ],
            axis=0,
        )
        target_classes = np.ones(src_logits.shape[:2]) * self.num_classes
        target_classes[
            tlx.convert_to_numpy(idx[0]), tlx.convert_to_numpy(idx[1])
        ] = target_classes_o
        target_classes = tlx.convert_to_tensor(target_classes, dtype=tlx.int64)

        weight = tlx.stack(
            [
                tlx.gather(self.empty_weight, tlx.convert_to_tensor(c))
                for c in target_classes
            ]
        )
        loss = tlx.losses.softmax_cross_entropy_with_logits(
            tlx.reshape(src_logits, [-1, (self.num_classes + 1)]),
            tlx.reshape(target_classes, [-1]),
            reduction="none",
        )
        loss = tlx.reshape(loss, (src_logits.shape[0], -1))
        loss *= weight

        loss_ce = tlx.reduce_mean(loss)

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        tgt_lengths = tlx.convert_to_tensor(
            [len(v["class_labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)

        arg = tlx.argmax(logits, axis=-1)
        arg = arg != self.num_classes
        arg = tlx.cast(arg, tlx.int32)
        card_pred = tlx.reduce_sum(arg)

        card_pred = tlx.cast(card_pred, tlx.float32)
        tgt_lengths = tlx.cast(tgt_lengths, tlx.float32)
        card_err = tlx.abs(card_pred - tgt_lengths)
        card_err = tlx.reduce_sum(card_err)
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "No predicted boxes found in outputs"
        idx = self._get_src_permutation_idx(indices)
        idx = tlx.stack(idx, axis=0)
        idx = tlx.transpose(idx)
        src_boxes = tlx.gather_nd(outputs["pred_boxes"], idx)
        target_boxes = tlx.concat(
            [
                tlx.gather(tlx.convert_to_tensor(t["boxes"]), i, axis=0)
                for t, (_, i) in zip(targets, indices)
            ],
            axis=0,
        )

        l1_loss = tlx.abs(src_boxes - target_boxes)

        losses = {}
        losses["loss_bbox"] = tlx.reduce_sum(l1_loss) / num_boxes

        loss_giou = generalized_box_iou(
            center_to_corners_format(
                src_boxes), center_to_corners_format(target_boxes)
        )
        shape = tlx.get_tensor_shape(loss_giou)[0]
        shape = tlx.arange(shape)
        shape = tlx.stack([shape, shape])
        shape = tlx.transpose(shape)
        loss_giou = tlx.gather_nd(loss_giou, shape)

        loss_giou = 1 - loss_giou
        losses["loss_giou"] = tlx.reduce_sum(loss_giou) / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        assert "pred_masks" in outputs, "No predicted masks found in outputs"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_idx = tlx.stack(src_idx, axis=0)
        src_idx = tlx.transpose(src_idx)
        src_masks = tlx.gather_nd(src_masks, src_idx)
        masks = [t["masks"] for t in targets]

        target_masks, valid = nested_tensor_from_tensor_list(masks)
        tgt_idx = tlx.stack(tgt_idx, axis=0)
        tgt_idx = tlx.transpose(tgt_idx)
        target_masks = tlx.gather_nd(target_masks, tgt_idx)

        # upsample predictions to the target size
        target_masks_shape = tlx.get_tensor_shape(target_masks)
        src_masks = tlx.transpose(src_masks, perm=[1, 2, 0])
        src_masks = tlx.resize(
            src_masks,
            output_size=tuple(target_masks_shape[-2:]),
            method="bilinear",
            antialias=False,
        )
        src_masks = tlx.transpose(src_masks, perm=[2, 0, 1])
        src_masks_shape = tlx.get_tensor_shape(src_masks)
        src_masks = tlx.reshape(src_masks, [src_masks_shape[0], -1])

        target_masks = tlx.reshape(
            target_masks, tlx.get_tensor_shape(src_masks))
        target_masks = tlx.cast(target_masks, dtype=tlx.float32)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = tlx.concat(
            [tlx.ones_like(src) * i for i, (src, _) in enumerate(indices)], axis=0
        )
        src_idx = tlx.concat([src for (src, _) in indices], axis=0)
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = tlx.concat(
            [tlx.ones_like(tgt) * i for i, (_, tgt) in enumerate(indices)], axis=0
        )
        tgt_idx = tlx.concat([tgt for (_, tgt) in indices], axis=0)
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"Loss {loss} not supported"
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "auxiliary_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = tlx.convert_to_tensor([num_boxes], dtype=tlx.float32)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = tlx.where(num_boxes >= 1, num_boxes,
                              tlx.ones_like(num_boxes))

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(
                        loss, auxiliary_outputs, targets, indices, num_boxes
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.

    Returns:
        a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    # assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = tlx.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = tlx.maximum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = rb - lt
    wh = tlx.where(wh >= 0, wh, tlx.zeros_like(wh))
    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = tlx.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = tlx.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = rb - lt
    wh = tlx.where(wh >= 0, wh, tlx.zeros_like(wh))
    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def nested_tensor_from_tensor_list(tensor_list):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        tensors = []
        masks = []
        for img in tensor_list:
            img = tlx.convert_to_tensor(img)
            img_shape = tlx.get_tensor_shape(img)
            new_img = tlx.pad(
                img,
                paddings=[
                    [0, max_size[0] - img_shape[0]],
                    [0, max_size[1] - img_shape[1]],
                    [0, max_size[2] - img_shape[2]],
                ],
            )
            tensors.append(new_img)
            mask = tlx.zeros((img_shape[1], img_shape[2]))
            new_mask = tlx.pad(
                mask,
                paddings=[
                    [0, max_size[1] - img_shape[1]],
                    [0, max_size[2] - img_shape[2]],
                ],
                constant_values=1,
            )
            new_mask = tlx.cast(new_mask, tlx.bool)
            masks.append(new_mask)
        tensor = tlx.stack(tensors, axis=0)
        mask = tlx.stack(masks, axis=0)
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return tensor, mask


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = tlx.sigmoid(inputs)
    ce_loss = tlx.losses.binary_cross_entropy(prob, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return tlx.reduce_sum(tlx.reduce_mean(loss, axis=1)) / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = tlx.sigmoid(inputs)
    numerator = tlx.reduce_sum(2 * (inputs * targets), axis=1)
    denominator = tlx.reduce_sum(inputs, axis=-1) + \
        tlx.reduce_sum(targets, axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return tlx.reduce_sum(loss) / num_boxes


def cdist(box_a, box_b):
    A = tlx.get_tensor_shape(box_a)[0]  # Number of bbox in box_a
    B = tlx.get_tensor_shape(box_b)[0]  # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tlx.tile(tlx.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tlx.tile(tlx.expand_dims(box_b, axis=0), [A, 1, 1])
    return tlx.reduce_sum(tlx.abs(tiled_box_a - tiled_box_b), axis=-1)


class GroupNorm(nn.Module):
    def __init__(self, c, name=None):
        super().__init__(name=name)
        self.c = c

        self.gamma = self._get_weights(
            var_name="gamma", shape=[1, 1, 1, c], init=tlx.initializers.ones()
        )
        self.beta = self._get_weights(
            var_name="beta", shape=[1, 1, 1, c], init=tlx.initializers.zeros()
        )

    def forward(self, x, g=8, eps=1e-5):
        n, h, w, c = tlx.get_tensor_shape(x)
        g = tlx.minimum(g, c)

        x = tlx.reshape(x, [n, h, w, g, c // g])
        mean, var = tlx.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tlx.sqrt(var + eps)

        x = tlx.reshape(x, [n, h, w, c]) * self.gamma + self.beta
        return x


def center_to_corners_format(x):
    # x_c, y_c, w, h = x.unbind(-1)
    x_c = x[..., 0]
    y_c = x[..., 1]
    w = x[..., 2]
    h = x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tlx.stack(b, axis=-1)
