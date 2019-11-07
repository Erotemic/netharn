import numpy as np
import netharn as nh
import torch
import torchvision
import ubelt as ub  # NOQA


class DescriptorNetwork(nh.layers.Module):
    """
    Produces resnet50 + MLP descriptors

    Example:
        >>> from netharn.models.descriptor_network import *
        >>> input_shape = (4, 3, 32, 32)
        >>> self = DescriptorNetwork(input_shape=input_shape)
        >>> print(self)
        >>> shape = self.output_shape_for(input_shape)
        >>> print(ub.repr2(shape.hidden.shallow(2), nl=-1))
        {
            'dvecs': {
                'conv1': (4, 64, 16, 16),
                'bn1': (4, 64, 16, 16),
                'relu1': (4, 64, 16, 16),
                'maxpool': (4, 64, 8, 8),
                'layer1': (4, 256, 8, 8),
                'layer2': (4, 512, 4, 4),
                'layer3': (4, 1024, 2, 2),
                'layer4': (4, 2048, 1, 1),
                'avgpool': (4, 2048, 1, 1),
                'view': (4, 2048),
                'fc': (4, 128)
            }
        }
    """

    def __init__(self, branch=None, input_shape=(1, 3, 416, 416),
                 norm_desc=False, desc_size=128, hidden_channels=[1024, 1024]):
        """
        Example:
            >>> from netharn.models.descriptor_network import *
            >>> input_shape = (4, 3, 32, 32)
            >>> self = DescriptorNetwork(input_shape=input_shape)
            >>> nh.OutputShapeFor(self)._check_consistency(input_shape)
            {'dvecs': (4, 128)}
        """
        super(DescriptorNetwork, self).__init__()
        if branch is None:
            self.branch = torchvision.models.resnet50(pretrained=True)
        else:
            self.branch = branch
        assert isinstance(self.branch, torchvision.models.ResNet)
        self.norm_desc = norm_desc

        self.in_channels = input_shape[1]
        self.out_channels = desc_size

        if self.branch.conv1.in_channels != self.in_channels:
            prev = self.branch.conv1
            cls = prev.__class__
            self.branch.conv1 = cls(
                in_channels=self.in_channels,
                out_channels=prev.out_channels,
                kernel_size=prev.kernel_size,
                stride=prev.stride,
                padding=prev.padding,
                dilation=prev.dilation,
                groups=prev.groups,
                bias=prev.bias,
            )
            nh.initializers.nninit_base.load_partial_state(
                self.branch.conv1,
                prev.state_dict(),
                initializer=nh.initializers.KaimingNormal(),
                verbose=0,
            )

        # Note the advanced usage of output-shape-for
        if __debug__:
            prebranch_field = nh.ReceptiveFieldFor(self.branch)(
                input_shape=input_shape)
            prepool_receptive_field = prebranch_field.hidden.shallow(1)['layer4']
            input_dims = np.array(input_shape[-2:])
            if np.any(input_dims < prepool_receptive_field['stride']):
                print('Network is too deep OR input is to small')

        prebranch_shape = nh.OutputShapeFor(self.branch)(input_shape)
        prepool_shape = prebranch_shape.hidden.shallow(1)['layer4']

        # replace the last layer of resnet with a linear embedding to learn the
        # LP distance between pairs of images.
        # Also need to replace the pooling layer in case the input has a
        # different size.
        self.prepool_shape = prepool_shape
        pool_channels = prepool_shape[1]
        pool_dims = prepool_shape[2:]
        self.branch.avgpool = torch.nn.AvgPool2d(pool_dims, stride=1)

        # Check that the modification to the layer fixed the size
        postbranch_shape = nh.OutputShapeFor(self.branch)(input_shape)
        postpool_shape = postbranch_shape.hidden.shallow(1)['layer4']

        assert np.all(np.array(prepool_shape[1:]) > 0)
        assert np.all(np.array(postpool_shape[1:]) > 0)

        # Replace the final linear layer with an MLP head
        self.branch.fc = nh.layers.MultiLayerPerceptronNd(
            dim=0, in_channels=pool_channels,
            hidden_channels=hidden_channels,
            out_channels=desc_size
        )

    def forward(self, inputs):
        """
        Compute a resnet50 vector for each input and look at the LP-distance
        between the vectors.

        Example:
            >>> inputs = nh.XPU(None).move(torch.rand(4, 21, 224, 224))
            >>> self = DescriptorNetwork(input_shape=inputs.shape)
            >>> output = self(inputs)

        Ignore:
            >>> input1 = nh.XPU(None).move(torch.rand(1, 3, 416, 416))
            >>> input2 = nh.XPU(None).move(torch.rand(1, 3, 416, 416))
            >>> input_shape1 = input1.shape
            >>> self = DescriptorNetwork(input_shape=input2.shape[1:])
            >>> self(input1, input2)
        """
        dvecs = self.branch(inputs)
        if self.norm_desc:
            # LP normalize the vectors
            dvecs = torch.nn.functional.normalize(dvecs, p=2)
        output = {
            'dvecs': dvecs
        }
        return output

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden):
        """
        TODO: work towards a system where we dont even have to miror the
        forward function, we can simply introspect and construct the analytic
        computation of shape / receptive field.
        """
        hidden = _Hidden()
        dvecs = hidden['dvecs'] = _OutputFor(self.branch)(inputs)
        if self.norm_desc:
            # LP normalize the vectors
            dvecs = hidden['norm_dvecs'] = _OutputFor(
                torch.nn.functional.normalize)(dvecs, p=2)
        output = {
            'dvecs': dvecs
        }
        return _Output(output, hidden)

    def output_shape_for(self, input_shape):
        return self._analytic_forward(input_shape, nh.OutputShapeFor,
                                      nh.OutputShape, nh.HiddenShapes)

    def receptive_field_for(self, input_field):
        return self._analytic_forward(input_field, nh.ReceptiveFieldFor,
                                      nh.ReceptiveField, nh.HiddenFields)
