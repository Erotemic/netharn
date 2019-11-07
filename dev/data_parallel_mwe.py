import torch
import netharn as nh
import ubelt as ub
import scriptconfig as scfg  # NOQA


class DemoConfig(scfg.Config):
    default = {
        'xpu': scfg.Value('0,1', help='can also be cpu, cuda0')
    }


class DemoModel(nh.layers.Module):
    """
    A gadget for transforming multiple 2D signals into multiple output heads
    """

    def __init__(self):
        super().__init__()

        def DefaultConv(c, C=None):
            C = c if C is None else C
            return nh.layers.ConvNormNd(
                dim=2, in_channels=c, out_channels=C, kernel_size=3,
                padding=1, norm='batch', noli='relu')

        def DefaultMLP(c, C=None):
            C = c if C is None else C
            return nh.layers.MultiLayerPerceptronNd(
                dim=2, in_channels=c, hidden_channels=[], out_channels=C,
                bias=True, dropout=0, noli='relu', norm='batch')

        self.branch1 = DefaultConv(3)
        self.branch2 = DefaultConv(1)

        self.feature = DefaultMLP(4, 4)
        self.head1 = DefaultMLP(4, 3)
        self.head2 = DefaultMLP(4, 1)

    def forward(self, inputs):
        """
        Raw -> Branch -> Concat -> Features -> Head
        """
        b1 = self.branch1(inputs['rgb'])
        b2 = self.branch2(inputs['aux'])
        c = torch.cat([b1, b2], dim=1)
        f = self.feature(c)
        p1 = self.head1(f)
        p2 = self.head2(f)
        outputs = {'head1': p1, 'head2': p2}
        return outputs


def demo_mwe_issue_dict_parallel():
    config = DemoConfig()
    parser = config.argparse()
    config.update(parser.parse_known_args()[0].__dict__)

    self = DemoModel()
    inputs = {
        'rgb': torch.rand(2, 3, 3, 5),
        'aux': torch.rand(2, 1, 3, 5),
    }
    xpu = nh.XPU.coerce(config['xpu'])
    inputs = xpu.move(inputs)
    model = xpu.mount(self)
    print('xpu = {!r}'.format(xpu))
    print('model = {!r}'.format(model))
    print('inputs.Tshape = ' + ub.repr2(ub.map_vals(lambda x: x.shape, inputs), nl=1))
    outputs = model(inputs)
    print('outputs.Tshape = ' + ub.repr2(ub.map_vals(lambda x: x.shape, outputs), nl=1))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/data_parallel_mwe.py
    """
    demo_mwe_issue_dict_parallel()
