# flake8: noqa
import sys
import ubelt as ub
sys.path.append(ub.expandpath('~/code/netharn/examples'))
from mnist_matching import setup_harn

resnet_harn = setup_harn(arch='resnet').initialize()
simple_harn = setup_harn(arch='simple').initialize()

harn = simple_harn

batch_vali = harn._demo_batch(tag='vali')
batch_train = harn._demo_batch(tag='train')


batch = batch_train


inputs = batch['chip']
outputs = harn.model(inputs)
dvecs = outputs['dvecs']


output_shape = harn.model.module.output_shape_for(inputs.shape)
print(ub.repr2(output_shape.hidden, nl=-1))


labels = batch['cpu_nx']
print('labels = {}'.format(
    ub.repr2(ub.odict(sorted(ub.dict_hist(labels.numpy()).items())), nl=1)
))

labels = labels[0:8]
dvecs = dvecs[0:8]

info1 = harn.criterion.mine_negatives(dvecs, labels, num=1, mode='hard')
info2 = harn.criterion.mine_negatives(dvecs, labels, num=1, mode='consistent')
info3 = harn.criterion.hard_triples(dvecs, labels)
info4 = harn.criterion.hard_triples2(dvecs, labels)
