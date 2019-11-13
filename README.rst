
NetHarn - a PyTorch Network Harness
-----------------------------------

|GitlabCIPipeline| |GitlabCICoverage| |Pypi| |Downloads| 

NAME:
    NetHarn (pronounced "net-harn")
FRAMEWORK:
    PyTorch
FEATURES: 
    * hyperparameter tracking
    * training directory management
    * callback-based public API 
    * XPU - code abstraction for [cpu, gpu, multi-gpu].
    * single-file deployments (NEW in version ``0.1.0``).
    * reasonable test coverage using pytest and xdoctest
    * CI testing on appveyor and travis (note a few tests are failing due to minor issues)
    * A rich utility set
    * Extensions of PyTorch objects (e.g. critions, initializers, layers,
      optimizers, schedulers)
BUILTINS:
   - training loop boilerplate
   - snapshots / checkpoints
   - progress bars (backend_choices: [progiter, tqdm])
   - data provenance of training history in ``train_info.json``
   - tensorboard metric visualization (optional)
DESIGN PHILOSOPHY: 
   Avoid boilerplate, built-it yourself when you need to, and don't repeat yourself.
SLOGAN: 
    Rein and train.
USAGE PATTERNS:
    (1) Write code for a torch object  (i.e. Dataset, Model, Criterion, Initializer, and Scheduler) just as you normally would.
    (2) Inherit from the ``nh.FitHarn`` object, define ``run_batch``, ``on_batch``, ``on_epoch``, etc...
    (3) Create an instance of ``nh.HyperParams`` to specify your dataset, model, criterion, etc...
    (4) Create an instance of your ``FitHarn`` object with those hyperparameters.
    (5) Then execute its ``run`` method.
    (6) ???
    (7) profit
EXAMPLES:
    * ToyData2d classification with nh.models.ToyNet2d (see doctest in netharn/fit_harn.py:__DOC__:0)
    * MNIST digit classification with MnistNet (examples/mnist.py)
    * Cifar10 category classification with ResNet50 / dpn91 (examples/cifar.py)
    * Voc2007+2012 object detection with YOLOv2 (examples/yolo_voc.py)
    * IBEIS metric learning with SiameseLP (examples/siam_ibeis.py)
STABILITY:
   Mostly harmless. Most tests pass, the current failures are probably not
   critical. I'm able to use it on my machine (tm). In this early stage of
   development, there are still a few pain points. Issues and PRs welcome.
KNOWN BUGS:
   * The metrics for computing detection mAP / AP might not be correct.
   * The YOLO example gets to about 70% mAP (using Girshik's mAP code) whereas we should be hitting 74-76%
AUTHORS COMMENTS:
   * My MNIST, CIFAR, and VOC examples will download the data as needed.
   * I'm hoping we can publicly release a few privately developed features.
     They would take a non-trivial amount of developer time to reproduce. These
     features mostly have to do with exporting / deploying fit models.
   * My CIFAR example for ResNet50 achieves 95.72% accuracy, outperforming the
     best DPN92 result (95.16%) that I'm aware of.
     This result seems real, I do not believe I've made an error in measurement
     (but this has need been peer-reviewed so, caveat emptor).  I've reproduced
     this results a few times. You can use the code in examples/cifar.py to see
     if you can too (please tell me if you cannot). 
   * My YOLO example is based of of EAVise's excellent lightnet (https://gitlab.com/EAVISE/lightnet/) package.
   * I reimplemented the CocoAPI (see nh.data.coco_api), because I had some
     (probably minor) issue with the original implementation. I've extended it
     quite a bit, and I'd recommend using it.
   * My metric-learning example requires code that is not publicly available
     :(, so only those with access to a copy of the ibeis software more recent than
     is more
     recent than July 2017
     more recent
     than 2017) can use it without modification.
DEPENDENCIES:
    * torch
    * numpy
    * Cython
    * ubelt
    * xdoctest
    * ... (see requirements.txt)

Installation
============

In the future these instructions may actually be different than the developer
setup instructions, but for now they are the same.

.. code-block:: bash

    mkdir -p ~/code
    git clone git@github.com:Erotemic/netharn.git ~/code/netharn
    cd ~/code/netharn
    ./run_developer_setup.sh

Description
===========

Parameterized fit harnesses for PyTorch.

Trains models and keeps track of your hyperparameters.

This is a clean port of the good parts developed in my research repo: ``clab``. 

See the netharn/examples folder for example usage. The doctests are also a good
resource. It would be nice if we had better docs.

NetHarn is a research framework for training and deploying arbitrary PyTorch models.
It was designed for the purpose of minimizing training-loop boilerplate and tracking hyperparameters to
  encourage reproducible research.
NetHarn separates the problem of training a model into the following core hyperparameter components:
the datasets, model, criterion, initializer, optimizer, and learning rate scheduler.
Runs with different hyperparameters are automatically logged to separate directories which makes it simple
  to compare the results of two experiments.
NetHarn also has the ability to create a single-file deployment of a trained model
  that is independent of the system used to train it.
This makes it fast and simple for research results to be externally verified and moved into production.


.. |TravisOld| image:: https://img.shields.io/travis/Erotemic/netharn/master.svg?label=Travis%20CI
   :target: https://travis-ci.org/Erotemic/netharn
.. |Travis| image:: https://img.shields.io/travis/Erotemic/netharn.svg
   :target: https://travis-ci.org/Erotemic/netharn
.. |Codecov| image:: https://codecov.io/github/Erotemic/netharn/badge.svg?branch=master&service=github
   :target: https://codecov.io/github/Erotemic/netharn?branch=master
.. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/github/Erotemic/netharn?svg=True
   :target: https://ci.appveyor.com/project/Erotemic/netharn/branch/master
.. |Pypi| image:: https://img.shields.io/pypi/v/netharn.svg
   :target: https://pypi.python.org/pypi/netharn


Developer Setup:
================


In the future these instructions might be different from the install
instructions, but for now they are the same.

.. code-block:: bash

    sudo apt-get install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip

    mkdir -p ~/code
    git clone git@github.com:Erotemic/netharn.git ~/code/netharn
    cd ~/code/netharn

    ./run_developer_setup.sh


Example:
========

This following example is the doctest in ``netharn/fit_harn.py``. It
demonstrates how to use NetHarn to train a model to solve a toy problem.  

In this toy problem, we do not extend the nh.FitHarn object, so we are using
the default behavior of ``run_batch``. The default ``on_batch``, and
``on_epoch`` do nothing, so only loss will be the only measurement of
performance.

For further examples please see the examples directory. These example show how
to extend nh.FitHarn to measure performance wrt a particular problem.  The
MNIST and CIFAR examples are the most simple. The YOLO example is more complex.
The IBEIS example depends on non-public data / software, but can still be
useful to look at.  Its complexity is more than CIFAR but less than YOLO.


.. code-block:: python

    >>> import netharn as nh
    >>> hyper = nh.HyperParams(**{
    >>>     # ================
    >>>     # Environment Components
    >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/demo'),
    >>>     'nice'        : 'demo',
    >>>     'xpu'         : nh.XPU.cast('auto'),
    >>>     # workdir is a directory where intermediate results can be saved
    >>>     # nice symlinks <workdir>/fit/nice/<nice> -> ../runs/<hashid>
    >>>     # XPU auto select a gpu if idle and VRAM>6GB else a cpu
    >>>     # ================
    >>>     # Data Components
    >>>     'datasets'    : {  # dict of plain ol torch.data.Dataset instances
    >>>         'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>         'vali': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
    >>>         'test': nh.data.ToyData2d(size=3, border=1, n=128, rng=2),
    >>>     },
    >>>     'loaders'     : {'batch_size': 64}, # DataLoader instances or kw
    >>>     # ================
    >>>     # Algorithm Components
    >>>     # Note the (cls, kw) tuple formatting
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {
    >>>         'lr': 0.0001
    >>>     }),
    >>>     # focal loss is usually better than nh.criterions.CrossEntropyLoss
    >>>     'criterion'   : (nh.criterions.FocalLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {
    >>>         'param': 0,
    >>>     }),
    >>>     # these may receive an overhaul soon
    >>>     'scheduler'   : (nh.schedulers.ListedScheduler, {
    >>>         'points': {'lr': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001}},
    >>>         'interpolation': 'linear',
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': 10,
    >>>     }),
    >>>     # dynamics are a config option that modify the behavior of the main
    >>>     # training loop. These parameters effect the learned model.
    >>>     'dynamics'   : {'batch_step': 4},
    >>> })
    >>> harn = nh.FitHarn(hyper)
    >>> # non-algorithmic behavior configs (do not change learned models)
    >>> harn.config['prog_backend'] = 'progiter'  # alternative: 'tqdm'
    >>> # start training.
    >>> harn.initialize(reset='delete')
    >>> harn.run()  # note: run calls initialize it hasn't already been called.
    >>> # xdoc: +IGNORE_WANT

Running this code produes the following output:

.. code-block:: 

   RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR
   Symlink: /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum -> /home/joncrall/.cache/netharn/demo/_mru
   ... already exists
   Symlink: /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum -> /home/joncrall/.cache/netharn/demo/fit/nice/demo
   ... already exists
   ... and points to the right place
   INFO: Initializing tensorboard (dont forget to start the tensorboard server)
   INFO: Model has 824 parameters
   INFO: Mounting ToyNet2d model on GPU(0)
   INFO: Exported model topology to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/ToyNet2d_2a3f49.py
   INFO: Initializing model weights with: <netharn.initializers.nninit_core.KaimingNormal object at 0x7fc67eff0278>
   INFO:  * harn.train_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum'
   INFO:  * harn.nice_dpath  = '/home/joncrall/.cache/netharn/demo/fit/nice/demo'
   INFO: Snapshots will save to harn.snapshot_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots'
   INFO: ARGV:
       /home/joncrall/.local/conda/envs/py36/bin/python /home/joncrall/.local/conda/envs/py36/bin/ipython
   INFO: dont forget to start:
       tensorboard --logdir ~/.cache/netharn/demo/fit/nice
   INFO: === begin training 0 / 10 : demo ===
   epoch lr:0.0001 │ vloss is unevaluated  0/10... rate=0 Hz, eta=?, total=0:00:00, wall=19:36 EST
   train loss:0.173 │ 100.00% of 64x8... rate=11762.01 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   vali loss:0.170 │ 100.00% of 64x4... rate=9991.94 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   test loss:0.170 │ 100.00% of 64x4... rate=24809.37 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   INFO: === finish epoch 0 / 10 : demo ===
   epoch lr:0.00505 │ vloss: 0.1696 (n_bad=00, best=0.1696)  1/10... rate=1.24 Hz, eta=0:00:07, total=0:00:00, wall=19:36 EST
   train loss:0.175 │ 100.00% of 64x8... rate=13522.14 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   vali loss:0.167 │ 100.00% of 64x4... rate=23598.31 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   test loss:0.167 │ 100.00% of 64x4... rate=20354.22 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   INFO: === finish epoch 1 / 10 : demo ===
   epoch lr:0.01 │ vloss: 0.1685 (n_bad=00, best=0.1685)  2/10... rate=1.28 Hz, eta=0:00:06, total=0:00:01, wall=19:36 EST
   train loss:0.177 │ 100.00% of 64x8... rate=15723.99 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   vali loss:0.163 │ 100.00% of 64x4... rate=29375.56 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   test loss:0.163 │ 100.00% of 64x4... rate=29664.69 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   INFO: === finish epoch 2 / 10 : demo ===

   <JUST MORE OF THE SAME; REMOVED FOR BREVITY>

   epoch lr:0.001 │ vloss: 0.1552 (n_bad=00, best=0.1552)  9/10... rate=1.11 Hz, eta=0:00:00, total=0:00:08, wall=19:36 EST
   train loss:0.164 │ 100.00% of 64x8... rate=13795.93 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   vali loss:0.154 │ 100.00% of 64x4... rate=19796.72 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   test loss:0.154 │ 100.00% of 64x4... rate=21396.73 Hz, eta=0:00:00, total=0:00:00, wall=19:36 EST
   INFO: === finish epoch 9 / 10 : demo ===
   epoch lr:0.001 │ vloss: 0.1547 (n_bad=00, best=0.1547) 10/10... rate=1.13 Hz, eta=0:00:00, total=0:00:08, wall=19:36 EST




   INFO: Maximum harn.epoch reached, terminating ...
   INFO: 



   INFO: training completed
   INFO: harn.train_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum'
   INFO: harn.nice_dpath  = '/home/joncrall/.cache/netharn/demo/fit/nice/demo'
   INFO: view tensorboard results for this run via:
       tensorboard --logdir ~/.cache/netharn/demo/fit/nice
   [DEPLOYER] Deployed zipfpath=/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/deploy_ToyNet2d_lnejaaum_009_GAEYQT.zip
   INFO: wrote single-file deployment to: '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/deploy_ToyNet2d_lnejaaum_009_GAEYQT.zip'
   INFO: exiting fit harness.

Furthermore, if you were to run that code when `'--verbose' in sys.argv`, then
it would produce this more detailed description of what it was doing:

.. code-block:: 

   RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR
   Symlink: /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum -> /home/joncrall/.cache/netharn/demo/_mru
   ... already exists
   Symlink: /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum -> /home/joncrall/.cache/netharn/demo/fit/nice/demo
   ... already exists
   ... and points to the right place
   DEBUG: Initialized logging
   INFO: Initializing tensorboard (dont forget to start the tensorboard server)
   DEBUG: harn.train_info[hyper] = {
       'model': (
           'netharn.models.toynet.ToyNet2d',
           {
               'input_channels': 1,
               'num_classes': 2,
           },
       ),
       'initializer': (
           'netharn.initializers.nninit_core.KaimingNormal',
           {
               'mode': 'fan_in',
               'param': 0,
           },
       ),
       'optimizer': (
           'torch.optim.sgd.SGD',
           {
               'dampening': 0,
               'lr': 0.0001,
               'momentum': 0,
               'nesterov': False,
               'weight_decay': 0,
           },
       ),
       'scheduler': (
           'netharn.schedulers.scheduler_redesign.ListedScheduler',
           {
               'interpolation': 'linear',
               'optimizer': None,
               'points': {'lr': {0: 0.0001, 2: 0.01, 5: 0.015, 6: 0.005, 9: 0.001}},
           },
       ),
       'criterion': (
           'netharn.criterions.focal.FocalLoss',
           {
               'focus': 2,
               'ignore_index': -100,
               'reduce': None,
               'reduction': 'mean',
               'size_average': None,
               'weight': None,
           },
       ),
       'loader': (
           'torch.utils.data.dataloader.DataLoader',
           {
               'batch_size': 64,
           },
       ),
       'dynamics': (
           'Dynamics',
           {
               'batch_step': 4,
               'grad_norm_max': None,
           },
       ),
   }
   DEBUG: harn.hyper = <netharn.hyperparams.HyperParams object at 0x7fb19b4b8748>
   DEBUG: make XPU
   DEBUG: harn.xpu = <XPU(GPU(0)) at 0x7fb12af24668>
   DEBUG: Criterion: FocalLoss
   DEBUG: Optimizer: SGD
   DEBUG: Scheduler: ListedScheduler
   DEBUG: Making loaders
   DEBUG: Making model
   DEBUG: ToyNet2d(
     (layers): Sequential(
       (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
       (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (2): ReLU(inplace)
       (3): Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
       (4): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       (5): ReLU(inplace)
       (6): Conv2d(8, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
     )
     (softmax): Softmax()
   )
   INFO: Model has 824 parameters
   INFO: Mounting ToyNet2d model on GPU(0)
   DEBUG: Making initializer
   DEBUG: Move FocalLoss() model to GPU(0)
   DEBUG: Make optimizer
   DEBUG: Make scheduler
   DEBUG: Make monitor
   DEBUG: Make dynamics
   INFO: Exported model topology to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/ToyNet2d_2a3f49.py
   INFO: Initializing model weights with: <netharn.initializers.nninit_core.KaimingNormal object at 0x7fb129e732b0>
   DEBUG: calling harn.initializer=<netharn.initializers.nninit_core.KaimingNormal object at 0x7fb129e732b0>
   INFO:  * harn.train_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum'
   INFO:  * harn.nice_dpath  = '/home/joncrall/.cache/netharn/demo/fit/nice/demo'
   INFO: Snapshots will save to harn.snapshot_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots'
   INFO: ARGV:
       /home/joncrall/.local/conda/envs/py36/bin/python /home/joncrall/.local/conda/envs/py36/bin/ipython --verbose
   INFO: dont forget to start:
       tensorboard --logdir ~/.cache/netharn/demo/fit/nice
   INFO: === begin training 0 / 10 : demo ===
   DEBUG: epoch lr:0.0001 │ vloss is unevaluated
   epoch lr:0.0001 │ vloss is unevaluated  0/10... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 EST
   DEBUG: === start epoch 0 ===
   DEBUG: log_value(epoch lr, 0.0001, 0
   DEBUG: log_value(epoch momentum, 0, 0
   DEBUG: _run_epoch 0, tag=train, learn=True
   DEBUG:  * len(loader) = 8
   DEBUG:  * loader.batch_size = 64
   train loss:-1.000 │ 0.00% of 64x8... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=train, epoch=0
   train loss:0.224 │ 100.00% of 64x8... rate=12052.25 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(train epoch loss, 0.22378234565258026, 0
   DEBUG: Finished batch iteration for tag=train, epoch=0
   DEBUG: _run_epoch 0, tag=vali, learn=False
   DEBUG:  * len(loader) = 4
   DEBUG:  * loader.batch_size = 64
   vali loss:-1.000 │ 0.00% of 64x4... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=vali, epoch=0
   vali loss:0.175 │ 100.00% of 64x4... rate=23830.75 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(vali epoch loss, 0.1749105490744114, 0
   DEBUG: Finished batch iteration for tag=vali, epoch=0
   DEBUG: epoch lr:0.0001 │ vloss: 0.1749 (n_bad=00, best=0.1749)
   DEBUG: _run_epoch 0, tag=test, learn=False
   DEBUG:  * len(loader) = 4
   DEBUG:  * loader.batch_size = 64
   test loss:-1.000 │ 0.00% of 64x4... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=test, epoch=0
   test loss:0.176 │ 100.00% of 64x4... rate=28606.65 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(test epoch loss, 0.17605290189385414, 0
   DEBUG: Finished batch iteration for tag=test, epoch=0
   DEBUG: Saving snapshot to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots/_epoch_00000000.pt
   DEBUG: Snapshot saved to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots/_epoch_00000000.pt
   DEBUG: new best_snapshot /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots/_epoch_00000000.pt
   DEBUG: Plotting tensorboard data
   Populating the interactive namespace from numpy and matplotlib
   INFO: === finish epoch 0 / 10 : demo ===

   <JUST MORE OF THE SAME; REMOVED FOR BREVITY>

   INFO: === finish epoch 8 / 10 : demo ===
   DEBUG: epoch lr:0.001 │ vloss: 0.2146 (n_bad=08, best=0.1749)
   epoch lr:0.001 │ vloss: 0.2146 (n_bad=08, best=0.1749)  9/10... rate=1.20 Hz, eta=0:00:00, total=0:00:07, wall=19:56 EST
   DEBUG: === start epoch 9 ===
   DEBUG: log_value(epoch lr, 0.001, 9
   DEBUG: log_value(epoch momentum, 0, 9
   DEBUG: _run_epoch 9, tag=train, learn=True
   DEBUG:  * len(loader) = 8
   DEBUG:  * loader.batch_size = 64
   train loss:-1.000 │ 0.00% of 64x8... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=train, epoch=9
   train loss:0.207 │ 100.00% of 64x8... rate=13580.13 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(train epoch loss, 0.2070118673145771, 9
   DEBUG: Finished batch iteration for tag=train, epoch=9
   DEBUG: _run_epoch 9, tag=vali, learn=False
   DEBUG:  * len(loader) = 4
   DEBUG:  * loader.batch_size = 64
   vali loss:-1.000 │ 0.00% of 64x4... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=vali, epoch=9
   vali loss:0.215 │ 100.00% of 64x4... rate=29412.91 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(vali epoch loss, 0.21514184772968292, 9
   DEBUG: Finished batch iteration for tag=vali, epoch=9
   DEBUG: epoch lr:0.001 │ vloss: 0.2148 (n_bad=09, best=0.1749)
   DEBUG: _run_epoch 9, tag=test, learn=False
   DEBUG:  * len(loader) = 4
   DEBUG:  * loader.batch_size = 64
   test loss:-1.000 │ 0.00% of 64x4... rate=0 Hz, eta=?, total=0:00:00, wall=19:56 ESTDEBUG: Making batch iterator
   DEBUG: Starting batch iteration for tag=test, epoch=9
   test loss:0.216 │ 100.00% of 64x4... rate=25906.58 Hz, eta=0:00:00, total=0:00:00, wall=19:56 EST
   DEBUG: log_value(test epoch loss, 0.21618007868528366, 9
   DEBUG: Finished batch iteration for tag=test, epoch=9
   DEBUG: Saving snapshot to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots/_epoch_00000009.pt
   DEBUG: Snapshot saved to /home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/torch_snapshots/_epoch_00000009.pt
   DEBUG: Plotting tensorboard data
   INFO: === finish epoch 9 / 10 : demo ===
   DEBUG: epoch lr:0.001 │ vloss: 0.2148 (n_bad=09, best=0.1749)
   epoch lr:0.001 │ vloss: 0.2148 (n_bad=09, best=0.1749) 10/10... rate=1.21 Hz, eta=0:00:00, total=0:00:08, wall=19:56 EST




   INFO: Maximum harn.epoch reached, terminating ...
   INFO: 



   INFO: training completed
   INFO: harn.train_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum'
   INFO: harn.nice_dpath  = '/home/joncrall/.cache/netharn/demo/fit/nice/demo'
   INFO: view tensorboard results for this run via:
       tensorboard --logdir ~/.cache/netharn/demo/fit/nice
   [DEPLOYER] Deployed zipfpath=/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/deploy_ToyNet2d_lnejaaum_000_JWPNDC.zip
   INFO: wrote single-file deployment to: '/home/joncrall/.cache/netharn/demo/fit/runs/demo/lnejaaum/deploy_ToyNet2d_lnejaaum_000_JWPNDC.zip'
   INFO: exiting fit harness.


]
 
 
Acknowledgements:
=================
The authors would like to thank AFRL for their support of this work via `SBIR Contract FA8650-18-C-1075 <https://govtribe.com/award/federal-contract-award/definitive-contract-fa865018c1075>`_. This library is approved for public release via 88ABW.


.. |Pypi| image:: https://img.shields.io/pypi/v/netharn.svg
   :target: https://pypi.python.org/pypi/netharn

.. |Downloads| image:: https://img.shields.io/pypi/dm/netharn.svg
   :target: https://pypistats.org/packages/netharn

.. |ReadTheDocs| image:: https://readthedocs.org/projects/netharn/badge/?version=latest
    :target: http://netharn.readthedocs.io/en/latest/

.. # See: https://ci.appveyor.com/project/jon.crall/netharn/settings/badges
.. .. |Appveyor| image:: https://ci.appveyor.com/api/projects/status/py3s2d6tyfjc8lm3/branch/master?svg=true
.. :target: https://ci.appveyor.com/project/jon.crall/netharn/branch/master

.. |GitlabCIPipeline| image:: https://gitlab.kitware.com/computer-vision/netharn/badges/master/pipeline.svg
   :target: https://gitlab.kitware.com/computer-vision/netharn/-/jobs

.. |GitlabCICoverage| image:: https://gitlab.kitware.com/computer-vision/netharn/badges/master/coverage.svg?job=coverage
    :target: https://gitlab.kitware.com/computer-vision/netharn/commits/master
