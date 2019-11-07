# class VGG16(_BaseInitializer):
#     """
#     Attempts to shoehorn VGG weights into a particular model.
#     Will only work if the structure of the new model somewhat resembles


#     Attributes:
#         fpath (str): location of the pretrained weights file
#         initializer (_BaseInitializer): backup initializer if the weights can
#             only be partially applied

#     Example:
#         >>> from netharn.initializers import *
#         >>> import netharn
#         >>> model = netharn.models.segnet.SegNet(n_classes=5)
#         >>> self = VGG16()
#         >>> self(model)

#         >>> model = netharn.models.UNet(n_classes=5, feature_scale=1)
#         >>> self = VGG16()
#         >>> self(model)
#     """
#     def __init__(self, initializer='KaimingUniform'):
#         if isinstance(initializer, str):
#             from netharn import initializers
#             initializer = getattr(initializers, initializer)()
#         self.initializer = initializer

#     def forward(self, model):
#         import torchvision

#         # do backup initialization first
#         self.initializer(model)

#         print('extracting VGG-16 params.')
#         print('Note your model should partially agree with VGG structure')
#         vgg16 = torchvision.models.vgg16(pretrained=True)
#         src_layers = [_layer for _layer in vgg16.features.children()
#                       if isinstance(_layer, nn.Conv2d)]

#         # see how the model best lines up
#         dst_layers = [_layer for _layer in trainable_layers(model)
#                       if isinstance(_layer, nn.Conv2d)]

#         def layer_incompatibility(src, dst):
#             """
#             Measure how compatible two layers are
#             """
#             si, so, sh, sw = src.weight.size()
#             di, do, dh, dw = dst.weight.size()

#             incompatibility = 0

#             # determine if the two layers are compatible
#             compatible = True
#             compatible &= (src.groups == dst.groups)
#             compatible &= (src.dilation == dst.dilation)
#             compatible &= (src.transposed == dst.transposed)
#             compatible &= src.bias.size() == dst.bias.size()
#             compatible &= (sh == dh and sw == dw)

#             def _tuplediff(t1, t2):
#                 return (np.array(t1) - np.array(t2)).sum()

#             incompat = []
#             incompat.append(_tuplediff(src.stride, dst.stride))
#             incompat.append(_tuplediff(src.padding, dst.padding))
#             incompat.append(_tuplediff(src.output_padding, dst.output_padding))

#             if si != di or so != do:
#                 # compatible = False
#                 incompat.append(abs(si - di))
#                 incompat.append(abs(so - do))

#             incompat_ = [s for s in incompat if s > 0]

#             if incompat_:
#                 incompatibility = np.prod([s + 1 for s in incompat_])
#             else:
#                 incompatibility = 0

#             if not compatible:
#                 incompatibility = float('inf')

#             return incompatibility

#         try:
#             # check for a perfect ordered alignment
#             aligned_layers = []
#             for src, dst in zip(src_layers, dst_layers):

#                 incompatibility = layer_incompatibility(src, dst)

#                 if incompatibility != 0:
#                     raise AssertionError('VGG16 is not perfectly compatible')

#                 aligned_layers.append((src, dst))
#         except AssertionError:
#             import itertools as it
#             print('VGG initialization is not perfect')

#             # TODO: solve a matching problem to get a partial assignment
#             src_idxs = list(range(len(src_layers)))
#             dst_idxs = list(range(len(dst_layers)))

#             cost = np.full((len(src_idxs), len(dst_idxs)), np.inf)

#             for sx, dx in it.product(src_idxs, dst_idxs):
#                 src = src_layers[sx]
#                 dst = dst_layers[dx]
#                 incompatibility = layer_incompatibility(src, dst)
#                 cost[sx, dx] = incompatibility

#             rxs, cxs = util.mincost_assignment(cost)

#             print('Alignment')
#             print('rxs = {!r}'.format(rxs))
#             print('cxs = {!r}'.format(cxs))

#             aligned_layers = [
#                 (src_layers[rx], dst_layers[cx])
#                 for rx, cx in zip(rxs, cxs)
#             ]
#             for src, dst in aligned_layers:
#                 print('src = {!r}'.format(src))
#                 print('dst = {!r}'.format(dst))
#                 print('-----')
#                 pass
#             print('Able to align {} / {} dst layers from {} src layers'.format(len(aligned_layers), len(dst_layers), len(src_layers)))
#             if not aligned_layers:
#                 raise

#         # Copy over weights based on the assignment
#         for src, dst in aligned_layers:
#             si, so, sh, sw = src.weight.size()
#             di, do, dh, dw = dst.weight.size()

#             # we can handle different size input output channels by just
#             # copying over as much as we can. We should probably assert that
#             # the spatial dimensions should be the same though.
#             mo = min(so, do)
#             mi = min(si, di)

#             # mb = min(dst.bias.size(), src.bias.size())
#             dst.weight.data[0:mi, 0:mo, :, :] = src.weight.data[0:mi, 0:mo, :, :]
#             dst.bias.data[:] = src.bias.data[:]

#     def history(self):
#         """
#         if available return the history of the model as well
#         """
#         return 'torchvision.models.vgg16(pretrained=True)'



# def shock_he(tensor):
#     """
#     Adds a very small he initial values to current tensor state.
#     Helps tensor achieve full rank in case it lost it.

#     DEPRICATE IN FAVOR OF ABSTRACT SHOCK

#     Example:
#         >>> tensor = torch.eye(3, 3)
#         >>> tensor[0, 0] = 0
#         >>> np.linalg.matrix_rank(tensor.cpu().numpy())
#         2
#         >>> shock_he(tensor)
#         >>> np.linalg.matrix_rank(tensor.cpu().numpy())
#         3
#     """
#     if isinstance(tensor, Variable) and torch.__version__.startswith('0.3'):
#         shock_he(tensor.data)
#         return tensor
#     else:
#         # prb = tensor.clone()
#         # he_normal(prb, gain)
#         # tensor += prb
#         # return tensor
#         shock(tensor, he_normal, funckw={})
#         # fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
#         # std = gain * np.sqrt(1.0 / fan_in)
#         # prb = torch.randn(tensor.shape) * std
#         # tensor += prb
#         return tensor


# def shock(tensor, func, scale=.0001, funckw={}):
#     if isinstance(tensor, Variable) and torch.__version__.startswith('0.3'):
#         shock(tensor.data, func, scale, funckw)
#         return tensor
#     else:
#         perterb = tensor.clone()
#         # Init the perterbation matrix with the desired method and down scale
#         func(perterb, **funckw)
#         perterb *= scale
#         # Shock the tensor by perterbing it
#         tensor += perterb
#         return tensor
# # def shock_outward(tensor, scale=.1, a_min=.01):
# #     """
# #     send weights away from zero
# #     """
# #     if isinstance(tensor, Variable):
# #         shock_outward(tensor.data, scale)
# #         return tensor
# #     else:
# #         std = max(torch.abs(tensor).max(), a_min) * scale
# #         # perterb outward
# #         offset = np.abs(torch.randn(tensor.shape) * std) * torch.sign(tensor)
# #         tensor += offset
# #         return tensor

# # TRAINABLE_LAYER_TYPES = [
# #     # Any module with a reset_parameters()
# #     torch.nn.modules.conv._ConvNd,
# #     torch.nn.modules.batchnorm._BatchNorm,
# #     torch.nn.modules.Linear,
# #     torch.nn.modules.Embedding,
# #     torch.nn.modules.EmbeddingBag,
# #     torch.nn.modules.GRUCell,

# # ]

# # for key, value in vars(torch.nn.modules).items():
# #     if hasattr(value, 'reset_parameters'):
# #         print(key)


