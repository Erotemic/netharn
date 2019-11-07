Version 0.1.6
==============
* Fix Python2 compatibility issues
* Add `api.py` containing code to help reduce netharn boilerplate by parsing a config dictionary.
* Fixed bug in `IgnoreLayerContext` preventing it from being used with `DataParallel`
* Add `api.py` containing code to help reduce netharn boilerplate by parsing a config dictionary.
* Remove deprecated `_to_var`
* Add new `ListedScheduler` which is able to modify multiple optimizer attributes including learning rate and momentum. 
* FitHarn now logs momentum by default in addition to learning rate
* Add variant of Leslie Smith's learning rate test
* `nh.util.ExpMovingAve` now has a bias-correction option.


Version 0.1.5
==============
* Switched to skbuild
* Bug fixes

Version 0.1.4
==============
* Ported `multi_plot` from KWIL 
* Scheduler states are now saved by default
* Netharn now dumps tensorboard plots every epoch by default
* The default `prepare_batch` now returns a dictionary with keys `input` and `label`.
* `FitHarn.config` can now specify `export_modules`, which will be modules to
  expand when running the pytorch exporter.
* Ported modifications from KWIL to `imwrite`, `stack_imges`, etc...
* Fix issue with relative imports in netharn exporter
* Refactored the exporter closure-extractor into its own file.
* Add `devices` to `nh.layers.Module`
* Deprecate `HiddenShapesFor`
* Move `HiddenShapesFor` functionality to `OutputShapeFor`
* Improve CIFAR example.
* Improve MNIST example.
* Rename internal variables of `nh.Monitor`
* Improve doc-strings for `nh.Monitor`
* Move folder functionality into `hyperparams`.

Version 0.1.3
==============
* Add (hacked-in) better `imgaug` hyperparameter logging. 
* Add verbose kwarg to `Pretrained.forward`
* Add `IgnoreLayerContext`
* `nh.util.DisableBatchNorm` renamed to  `nh.util.BatchNormContext`
* Add `nh.ReceptiveFieldFor`
* `train_info.json` now gets backed up if it would be overwritten


Version 0.1.2
==============
* Fix Python 2.7 bugs. 
* `nh.CocoAPI.show_image` now correctly clears the axis before drawing
* Add `_demo_epoch` function to `FitHarn` which runs a single epoch for testing purposes.
* Add new layers: `GaussianBlurNd`, `L2Norm`, `Permute`, `Conv1d_pad`, `Conv2d_pad`
* Focal loss no longer produces warnings with newer versions of torch.
* The `nh.util.group_items` utility will now default to the `ubelt` implementation for object and string arrays.
* Improve efficiency of `DataFrameArray.groupby`
* `nh.XPU` now supports `__eq__`
* `one_hot_embedding` now supports the `dim` keyword argument.
* Add `nh.XPU.raw` to access the raw underlying model.
* `nh.Pretrained` can now discover weights inside deployment files.
* Add `util_filesys` which has the function `get_file_info`.
* Fix bug in `FitHarn._check_divergence`
* Add dependency on `astunparse` to fix bug where exporter could not handle complex assignments
* `nh.Pretrained` initializer now only requires the path to the deploy zip-file. It can figure out which files in the deployment are the weights.
* `nh.CocoAPI` can now look up images by filename
* `nh.CocoAPI` can now delete categories by category name


Version 0.1.1
==============
* Deprecate and removed irrelevant parts of `CocoAPI`
* Remote annotations and categories now dynamically updates indexes `CocoAPI`
* Add remove categories to `CocoAPI`
* Add experimental `_build_hashid` to `CocoAPI`
* Fixed take in `ObjectList1D` in `CocoAPI`
* Add compress to `ObjectList1D` in `CocoAPI`
* Add `hidden_shape_for`
* Fix bug where `OutputShapeFor(_MaxPoolNd)` did not respect `ceil_mode`.
* Fix bug where CPU implementation of non-max-suppression was different
* Add `__json__` method to `nh.XPU`
* Fix bug where snapshots are corrupted with an `EOFError`
* Fix bug where temporary directories were not cleaned up
* `harn._export` is now its own function


Version 0.1.0
==============
* Integrate the publicly released Pytorch exporter and deployer.
* Fix bug where train info was not written if you specified a custom train dpath.


Version 0.0.27
==============
* Add `DataFrameLight` to `nh.util`, which provides a subset of `pandas.DataFrame` functionality, but much much faster.


Version 0.0.26
==============
* Tentative Python 2.7 support


Version 0.0.25
==============
* Fix issue with per-instance FitHarn class loggers


Version 0.0.24
==============
* Fix tests and raised better errors if `tensorflow` does not exist


Version 0.0.23
==============
* Fix bug where `seed_global` did not set call `torch.cuda.manual_seed_all`


Version 0.0.22
==============
* Better support for torch.device with `nh.XPU`
* Minor reorganization of FitHarn, added more callbacks



Version 0.0.21
==============
* Fix issue with unseeded random states. Now defaults to the global `np.random` state.
* Fix bug in `load_arr`


Version 0.0.20
==============
* FitHarn now uses `StreamLogger` instead of print


Version 0.0.19
==============
* Fix torch 0.4.1 deprecation warnings in focal loss


Version 0.0.17
==============
* Fix tests
* Add `before_epochs` callback



Version 0.0.16
==============
* Add `nh.util.global_seed`
* Fix MNIST example
* Small improvements to outputs
* Better test images
* Better YOLO example
* Various minor bug fixes
* Other stuff I forgot to log, I'm doing this mostly in my spare time!


Version 0.0.15
==============
* Add `SlidingWindow` as simplified alternative to `SlidingSlices`


Version 0.0.14
==============
* Add zip-awareness to pre-trained loader 
* Expand COCO-API functionality
* Better detection metrics with alternative implementations
* Fix YOLO scheduler


Version 0.0.13
==============
* Fix issue with `autompl`. Now correctly detects if display is available. 

Version 0.0.12
==============
* Early and undocumented commits
