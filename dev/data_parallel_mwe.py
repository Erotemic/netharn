
def demo_mwe_issue_dict_parallel():
    import torch
    import netharn as nh
    import ubelt as ub

    class MyModel(nh.layers.Module):
        def forward(self, inputs):
            outputs = {
                'key1': inputs + 1,
                'key2': inputs.sum(),
            }
            return outputs

    raw_model = MyModel()
    inputs = torch.rand(2, 1, 3, 3)

    xpu = nh.XPU([0, 1])
    inputs = xpu.move(inputs)
    model = xpu.mount(raw_model)

    print('xpu = {!r}'.format(xpu))
    print('inputs = {!r}'.format(inputs))
    print('model = {!r}'.format(model))
    outputs = model(inputs)
    print('outputs = ' + ub.repr2(ub.map_vals(lambda x: x.shape, outputs), nl=1))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/dev/data_parallel_mwe.py
    """
    demo_mwe_issue_dict_parallel()
