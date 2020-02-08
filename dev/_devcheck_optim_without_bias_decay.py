
def main():
    import netharn as nh
    import ubelt as ub

    model = nh.layers.Sequential(*[
        nh.layers.ConvNormNd(2, 3, 1),
        # nh.layers.ConvNormNd(2, 1, 1),
        # nh.layers.ConvNormNd(2, 1, 1),
    ])

    params = dict(model.named_parameters())
    param_keys = set(params)
    key_groups = {}
    other_keys = param_keys.copy()
    if 1:
        key_groups['norm'] = {p for p in other_keys if p.endswith(('.norm.weight', '.norm.weight'))}
        other_keys -= key_groups['norm']
    if 1:
        key_groups['bias'] = {p for p in other_keys if p.endswith('.bias')}
        other_keys -= key_groups['bias']
    if 1:
        key_groups['weight']  = {p for p in other_keys if p.endswith('.weight')}
        other_keys -= key_groups['weight']
    key_groups['other'] = other_keys

    named_param_groups = {}
    for group_name, keys in key_groups.items():
        if keys:
            param_group = {}
            param_group['params'] = list(ub.dict_subset(params, keys).values())
            named_param_groups[group_name] = param_group

    if 'bias' in named_param_groups:
        named_param_groups['bias']['weight_decay'] = 0
    if 'norm' in named_param_groups:
        named_param_groups['norm']['weight_decay'] = 0

    import torch
    param_groups = list(named_param_groups.values())

    optim_defaults = {
        'lr': 1e-3,
        'weight_decay': 1e1,
    }
    optim = torch.optim.AdamW(param_groups, **optim_defaults)

    learn = True

    model = model.train(learn)
    import time

    with torch.set_grad_enabled(learn):
        for i in range(10000):

            if learn:
                optim.zero_grad()
            inputs = torch.rand(3, 3, 2, 2)
            outputs = model(inputs)
            target = outputs.data.detach()
            # target = target * 1.0001
            target = torch.rand(3, 1, 2, 2) * 1e-3
            # target.fill_(0)
            loss = ((outputs - target) ** 2).sum()

            if learn:
                loss.backward()
                optim.step()
                optim.zero_grad()
            # print(ub.repr2(named_param_groups, nl=2))
            state = model.state_dict()
            state = ub.dict_diff(state, params)
            time.sleep(0.01)
            print('loss = {!r}'.format(float(loss.item())))
            print('param_state = ' + ub.repr2(params) + '\n' +
                  'buffer_state = ' + ub.repr2(state, nl=3))

            time.sleep(0.1)
