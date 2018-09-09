|Travis| |Codecov| |Appveyor| |Pypi|


NetHarn - a PyTorch Network Harness
-----------------------------------

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

Parameterized fit and prediction harnesses for PyTorch.

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
    >>>         'test': nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
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
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .0001, 2: .01, 5: .015, 6: .005, 9: .001},
    >>>         'interpolate': True,
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {
    >>>         'max_epoch': 10,
    >>>     }),
    >>>     # dynamics are a config option that modify the behavior of the main
    >>>     # training loop. These parameters effect the learned model.
    >>>     'dynamics'   : {'batch_step': 4},
    >>> })
    >>> harn = FitHarn(hyper)
    >>> # non-algorithmic behavior configs (do not change learned models)
    >>> harn.config['prog_backend'] = 'tqdm'  # I prefer progiter (I may be biased)
    >>> # start training.
    >>> harn.initialize(reset='delete')
    >>> harn.run()  # note: run calls initialize it hasn't already been called.
    >>> # xdoc: +IGNORE_WANT

Running this code produes the following output:

.. code-block:: 

    RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR
    Symlink: /home/joncrall/.cache/netharn/demo/fit/runs/olqtvpde -> /home/joncrall/.cache/netharn/demo/fit/nice/demo
    .... already exists
    .... and points to the right place
    Initializing tensorboard (dont forget to start the tensorboard server)
    Model has 824 parameters
    Mounting ToyNet2d model on GPU(0)
    Initializing new model
     * harn.train_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/olqtvpde'
     * harn.nice_dpath = '/home/joncrall/.cache/netharn/demo/fit/nice/demo'
    Snapshots will save to harn.snapshot_dpath = '/home/joncrall/.cache/netharn/demo/fit/runs/olqtvpde/torch_snapshots'
    dont forget to start:
        tensorboard --logdir /home/joncrall/.cache/netharn/demo/fit/nice
    begin training
    epoch lr:0.001 │ vloss is unevaluated: 100%|███████████████████████| 10/10 [00:00<00:00, 15.11it/s, wall=Jul:07 EST]10 [00:00<?, ?it/s]
    train x64 │ loss:0.186 │: 100%|████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 276.93it/s, wall=Jul:07 EST]
    test x64 │ loss:0.159 │: 100%|█████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 482.91it/s, wall=Jul:07 EST]


