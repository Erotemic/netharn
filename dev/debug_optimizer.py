

def debug_optimizer(harn, snapshot_state):
    """
    debuging an issue where the param groups were created in different orders
    each time.
    """
    if False:
        # DEBUG: check that all optimizer params exist in the model
        self = harn.optimizer
        state_dict = snapshot_state['optimizer_state_dict']
        for param_group in harn.optimizer.param_groups:
            print('-----')
            print(param_group['weight_decay'])
            print('-----')
            for p in param_group['params']:

                # Find the model param that correspond to this
                found = None
                for name, mp in harn.model.named_parameters():
                    if mp is p:
                        found = name
                        break

                assert found is not None
                print('found = {!r}'.format(found))

                state = self.state[p]
                if state:
                    avg_shape = tuple(state['exp_avg'].shape)
                    p_shape = tuple(p.shape)
                    if avg_shape == p_shape:
                        print('avg_shape = {!r}'.format(avg_shape))
                    else:
                        print('p_shape = {!r}'.format(p_shape))
                        print('avg_shape = {!r}'.format(avg_shape))

        if 0:
            self = harn.optimizer
            for param_group in harn.optimizer.param_groups:
                for p in param_group['params']:
                    print(p.grad is None)

            for n, mp in harn.model.named_parameters():
                assert mp.requires_grad
                if mp.grad is not None:
                    mp.grad.detach_()
                    mp.grad.zero_()

            batch = harn._demo_batch()
            outputs = harn.model(batch['im'])
            loss = outputs['class_energy'].mean()

            harn.optimizer.zero_grad()
            loss.backward()

            for param_group in harn.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is None:
                        found = None
                        for name, mp in harn.model.named_parameters():
                            if mp is p:
                                found = name
                                break
                        print('no grad for found = {!r}'.format(found))

            harn.optimizer.step()

        if 0:
            snapshot_state_old = harn.get_snapshot_state()
            torch.save(snapshot_state_old, 'foo.pt')
            snapshot_state = harn.xpu.load('foo.pt')

            prev_states = harn.prev_snapshots()
            snapshot_state = harn.xpu.load(prev_states[-1])

            snapshot_state_old['optimizer_state_dict']['state'].keys()
            snapshot_state['optimizer_state_dict']['state'].keys()
            state_dict = snapshot_state['optimizer_state_dict']

            for id, state in state_dict['state'].items():
                pass

            for group in self.param_groups:
                for param in group['params']:
                    print(param.shape)

            for group in state_dict['param_groups']:
                for paramid in group['params']:
                    state = state_dict['state'][paramid]
                    print(state['exp_avg'].shape)

