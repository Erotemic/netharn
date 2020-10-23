# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.5.10 - Unreleased


### Added
* `allow_unicode` option to `FitHarnPreferences`, which can be set to False to
  disable utf8 characters in output formatting.

* `IndexableWalker` in `netharn.util.util_json` (also exists in kwcoco)

* New helper methods in `data_containers.BatchContainer`

### Fixed
* Typo: directory `explit_checkpoints` renamed to `explicit_checkpoints`.

* Fixed bug where epoch 0 would write a snapshot if it failed.


### Changed

* Removed Python 3.5 support

* ProgIter information will now written to the log file pending release of ubelt 0.9.3.

* Progress information now includes warmup LR information in the first epoch.


### Deprecated
* Deprecate `colored` option in `FitHarnPreferences`. Use `NO_COLOR` environ to
  disable ANSI coloring instead.

* `netharn.export` has been deprecated for `torch_liberator` and `liberator`,
  and will be removed in the future. 


## Version 0.5.9 - Released 2020-08-26

### Changed
 
* `_dump_monitor_tensorboard` now additionally writes a bash script to quickly
  let the user re-visualize results in the case of mpl backend failure.

* `load_partial_state` now has an algorithm to better match model keys when the
  only difference is in key prefixes.
    - adds keyword arg association which defaults to prefix-hack, the old default was module-hack, and embedding is more theoretically correct but too slow.


### Fixes
* Optimizer.coerce now works correctly with any `torch.optim` or `torch_optimizer` optimizer.

### Added

* `BatchContainer.pack` for easier use of non-container aware models.
* `colored` option to `FitHarnPreferences`, which can be set to False to disable ANSI coloring


## Version 0.5.8 - Released

## Version 0.5.7 - Released

### Changed
* `harn.deploy_fpath` is now populated when the model is deployed.
* Improved docs on `netharn/data/toydata.py`
* Changed name of `torch_shapshots` directory name to `checkpoints`.

### Added
* Ported experimental `ChannelSpec` and `DataContainser` from bioharn to netharn.data.
* Added basic classification example that works on generic coco datasets
* Threshold curves to ConfusionVector metrics
* Initial weights are now saved in `initial_state` directory.
* New `plots` submodule.

### Fixed
* Fixed bug in XPU auto mode which caused it always to choose GPU 0.
* Bug in hyperparams where dict-based loader spec was not working.
* Display intervals were not working correctly with ProgIter, hacked in a temporary fix.


## Version 0.5.6 - Released 2020-04-16

### Changed
* Enhanced VOC ensure data 


### Fixed
* Version issues from last release


## Version 0.5.5

### Added
* Timeout to FitHarn.preferences
* Basic gradient logging
* Several new functions are now registered with OutputShapeFor to support efficientnet (F.pad, F.conv2d, torch.sigmoid)
* Balanced batch samplers 

### Changed
* Hyperparams "name" can now be specified instead of "nice". We will transition
  from "nice" to "name", for now both are supported but "nice" will eventually be deprecated.
* FitHarn.preferences now uses scriptconfig, which means the "help" sections are coupled with the object.
* Removed explicit support for Python 2.7
* Reverted default of `keyboard_debug` to True.
* Moved `analytic_for`, `output_shape_for`, and `receptive_field` for to the netharn.analytic subpackage. Original names are still available, but deprecated and will be removed in a future version. 
* Moved helpers from `netharn.hyperparams` to `netharn.util`
* Made pytorch-optimizer optional: https://github.com/jettify/pytorch-optimizer
* netharn now will timeout within an epoch

### Fixed
* Bug when a value in `harn.intervals` was zero.


## Version 0.5.4 - Released 2020-02-19 

### Added
* EfficientNet backbone and Swish activation

### Fixed 
* Handle "No running processes found" case in `XPU.coerce('auto')`
* Resize now works with newer `imgaug` versions
* Fixed incorrect use of the word "logit", what I was calling logits are
  actually log probabilities.

### Changed 
* Using new mode in `gpu_info`, this is more stable
* Examples are now in the netharn.examples directory, which means you can run
  them without having to git clone netharn.
* Moved data grabbers into netharn.data
* Moved unfinished examples to dev


## Version 0.5.3

### Added
* Add `tensorboard_groups` to config
* Add `min_lr` to Monitor
* Add `harn.iter_index`, new property which tracks the number iterations

### Fixed
* Reworked `remove_comments_and_docstrings`, so it always produces valid code.
* `nh.XPU` classmethods now work correctly for inheriting classes
* Iteration indexes are now correct in tensorboard.

### Changed
* `nh.XPU.cast` will now throw deprecation warnings use `nh.XPU.coerce` instead.
* `harn.config` is deprecated using `harn.preferences` instead.
* Progress display now counts epochs starting from 1, so the final epoch will
  read `({harn.epoch + 1})/({harn.monitor.max_epoch})`. The internal `harn.epoch` is still 0-based.
* Made `psutil` dependency optional.


## Version 0.5.2 - Release 2019-Nov-25

### Added
* Rectify nonlinearity now supports more torch activations

### Changed
* Smoothing no longer applied to lr (learning rate) and momentum monitor plots
* pandas and scipy are now optional (in this package)
* removed several old dependencies

### Fixed
* Small issues in CIFAR Example
* Small `imgaug` issue in `examples/sseg_camvid.py` and `examples/segmentation.py`
* FitHarn no longer fails when loaders are missing batch sizes
* Fixed windows issue in `util_zip.zopen`.
* Fixed runtime dependency on `strip_ansi` from xdoctest.


## Version 0.5.1

### Changed
* Second public release

## Version 0.4.1

### Added
* Add support for `main_device` in device `ModuleMixin`
* Add `coerce` to DeployedModel

### Changed
* Grad clipping dynamics now defaults to L2 norm. Can change the p-norm using `dynamic['grad_norm_type']`


## Version 0.4.0

### Added
* Add `AnalyticModule` 
* Add support for interpolate output-shape-for 
* Add PSPNet and DeepLab 
* Support for 'AdaptivePooling' in output-shape-for
* Added CamVid capable torch dataset 
* Add `nh.util.freeze_layers`
* Added `super_setup.py` to handle external utility dependencies.
* `FitHarn` now attempts to checkpoint the model if it encounters an error. 
* `DeployedModel.ensure_mounted_model` makes writing prediction scripts easier
* Add property `FitHarn.batch_index` that points to the current batch index.
* Add border mode and imgaug stochastic params to `Resize`.
* Add `InputNorm` layer, which couples input centering with a torch model.
* General segmentation example 
* General object detection example 
* Add `ForwardFor`
* Add `getitem`, `view`, and `shape` for  `ForwardFor` and `OutputShapeFor`
* Add `add`, `sub`, `mul`, and `div` for  `ForwardFor` and `OutputShapeFor` and  `ReceptiveFieldFor`
* `XPU.coerce('argv')` will now look for the `--xpu` CLI arg in addition to `cpu` and `gpu`.

### Changed

* `Hyperparams` now allows the user to specify pre-constructed instances of relevant classes (experimental).
* `Hyperparams` now tries to coerce `initkw` to json-compatible values.
* `train_hyper_id_brief` is no longer generated with `short=True`.
* `nh.Initializer.coerce` can now accept its argument as a string.
* `find_unused_gpu` now prioritizes the GPU with the fewest number of compute processes
* `nh.Pretrained` can now accept fuzzy paths, as long as they resolve to a single unique file.
* Netharn now creates symlink to a static "deploy.zip" version of the deployed models with robust name tags.
* Tensorboard mixins now dumps losses in both linear and symlog space by default. 
* Increased speed of dumping matplotlib outputs
* Breakdown requirements into runtime, tests, optional, etc...
* Defaults for `num_keep` and `keep_freq` have changed to 2 and 20 to reduce disk consumption.
* Reorganized focal loss code.
* The `step` scheduler can now specify all step points: e.g. step-90-140-250 
* The `stepXXX` scheduler code must now be given all step points: e.g. step-90-140-250 
* `run_tests.py` now returns the proper exit code

### Fixed
* Fixed issue extracting generic torch weights from zipfile with Pretrained initializer
* Fixed issue in closer, where attributes referenced in calls were not considered
* Bug fixes in focal loss
* Issues with new torchvision version
* Issue with large numbers in RunningStats
* Fixed issues with torch `1.1.0`
* Fixed export failure when `Hyperparams` is given a `MountedModel` instance

### Deprecated
* Deprecate SlidingSlices and SlidingSlicesDataset
* `util_fname` is now deprecated.

### Removed
* Old `_run_batch` internal function
* Removed the `initializer` argument of `nh.Pretrained` in favor of `leftover` (BREAKING).


### Future
* Support for classical classification (SVM / RF) on top of deep features. 


## Version 0.2.0

### Added
* Add `FitHarn.prepare_epoch` callback.

### Changed
* Refactored `netharn.utils` to depend on `kwarray`, `kwimage`, and `kwplot`, this removes a lot of the extra cruft added in `0.1.8`.
* Can now specify the package zip-file name when deploying.
* Add option `FitHarn.config['use_tensorboard'] = True` 
* `load_partial_state` now returns dict containing info on which keys were unused
* `nh.initializers.Pretrained` now returns info dict from `load_partial_state`
* `nll_focal_loss` now is as fast as `nll_loss` when `focus=0`


## Version 0.1.8

**Note: many of the changes in this version were immediately factored out into external modules**

### Added
* Backport `ndsampler` Coco-API 
* Add `arglexmax`, `argmaxima`, `argminima` 
* Add `util_distributions` 

### Changed
* Move `Boxes` and `DataFrameLight` from `netharn.util` to `netharn.util.structs` 
* Enhance `Boxes` and `DataFrameLight` functionality / docs 
* Add `netharn.util.structs.Detections` 
* Loss components are now automatically logged when loss is returned as a dict . 
* Add a small interactive debug interface on `KeyboardInterrupt` 
* Fix XPU.coerce / XPU.cast when input is multi-gpu
* Add `draw_clf_on_image` 
* Add `valign` to `draw_text_on_image` 
* Add `border` to `draw_text_on_image` 

* A handful of PF GGR-related commits stashed on my home machine meant for 0.1.7 
* Add `nh.data.batch_samplers.MatchingSamplerPK` 
* Add `shift_sat` and `shift_val` to HSV augmenter 
* Refactor and clean `api.py` 
* Refactor and clean `netharn.initializers` 
* Refactor `draw_boxes` and `draw_segments` into `mpl_draw` 
* Fixed issues with YOLO example 
* Add `torch_ravel_multi_index` to `nh.util` 


## Version 0.1.7

### Added
* Add `plot_surface3d` 
* Add `models.DescriptorNetwork` 
* `MLP` can now accept `dim=0` 

### Changed

* Modified batch outputs to all use `:g` format 
* Use `progiter` by default instead of `tqdm` 
* `nh.XPU.move` is now applied recursively to containers (e.g. dict list) 
* All `MovingAve` objects can now track variance  
* `CumMovingAve` can now track variance 
* `ExpMovingAve` can now track variance  
* `WindowedMovingAve` can now track variance  
* `imread` now attempts to return RGB or gray-scale by default. 
* `lr_range_test` now shows std-dev error bars 
* Improve API coerce methods (PF / IF)

### Removed
* `nh.XPU.variable` is deprecated and removed. 


## Version 0.1.6

### Fixed
* Fix Python2 compatibility issues. 
* Fixed bug in `IgnoreLayerContext` preventing it from being used with `DataParallel`. 

### Added
* Add `api.py` containing code to help reduce netharn boilerplate by parsing a config dictionary. 
* Add new `ListedScheduler` which is able to modify multiple optimizer attributes including learning rate and momentum. 
* Add variant of Leslie Smith's learning rate test 
* `nh.util.ExpMovingAve` now has a bias-correction option. 

### Removed
* Remove deprecated `_to_var`. 

### Changed
* FitHarn now logs momentum by default in addition to learning rate  


## Version 0.1.5

### Changed
* Switched to `skbuild` 

### Fixed
* Bug fixes 


## Version 0.1.4

### Added
* Ported `multi_plot` from `KWIL` 
* Add `devices` to `nh.layers.Module` 
* `FitHarn.config` can now specify `export_modules`, which will be modules to expand when running the pytorch exporter. 

### Changed
* Scheduler states are now saved by default 
* Netharn now dumps tensorboard plots every epoch by default 
* The default `prepare_batch` now returns a dictionary with keys `input` and `label`. 
* Ported modifications from KWIL to `imwrite`, `stack_imges`, etc... 
* Improve CIFAR example. 
* Improve MNIST example. 
* Rename internal variables of `nh.Monitor` 
* Improve doc-strings for `nh.Monitor`
* Move folder functionality into `hyperparams`. 

### Fixed
* Fix issue with relative imports in netharn exporter 
* Refactored the exporter closure-extractor into its own file. 

### Removed
* Deprecate and remove `HiddenShapesFor` 
* Move `HiddenShapesFor` functionality to `OutputShapeFor` 


## Version 0.1.3

### Added
* Add (hacked-in) better `imgaug` hyperparameter logging. 
* Add verbose kwarg to `Pretrained.forward`
* Add `IgnoreLayerContext`
* Add `nh.ReceptiveFieldFor`

### Changed
* `nh.util.DisableBatchNorm` renamed to  `nh.util.BatchNormContext`
* `train_info.json` now gets backed up if it would be overwritten


## Version 0.1.2

### Fixed
* Fix Python 2.7 bugs. 
* `nh.CocoAPI.show_image` now correctly clears the axis before drawing
* Fix bug in `FitHarn._check_divergence`

### Added
* Add `_demo_epoch` function to `FitHarn` which runs a single epoch for testing purposes.
* Add new layers: `GaussianBlurNd`, `L2Norm`, `Permute`, `Conv1d_pad`, `Conv2d_pad`
* `nh.XPU` now supports `__eq__`
* `one_hot_embedding` now supports the `dim` keyword argument.
* Add `nh.XPU.raw` to access the raw underlying model.
* Add `util_filesys` which has the function `get_file_info`.
* Add dependency on `astunparse` to fix bug where exporter could not handle complex assignments

### Changed
* Focal loss no longer produces warnings with newer versions of torch.
* The `nh.util.group_items` utility will now default to the `ubelt` implementation for object and string arrays.
* Improve efficiency of `DataFrameArray.groupby`
* `nh.Pretrained` can now discover weights inside deployment files.
* `nh.Pretrained` initializer now only requires the path to the deploy zip-file. It can figure out which files in the deployment are the weights.
* `nh.CocoAPI` can now look up images by filename
* `nh.CocoAPI` can now delete categories by category name


## Version 0.1.1

### Removed
* Deprecate and removed irrelevant parts of `CocoAPI`

### Added
* Add remove categories to `CocoAPI`
* Add experimental `_build_hashid` to `CocoAPI`
* Add compress to `ObjectList1D` in `CocoAPI`
* Add `hidden_shape_for`
* Add `__json__` method to `nh.XPU`

### Changed
* Remote annotations and categories now dynamically updates indexes `CocoAPI`
* `harn._export` is now its own function

### Fixed
* Fixed take in `ObjectList1D` in `CocoAPI`
* Fix bug where `OutputShapeFor(_MaxPoolNd)` did not respect `ceil_mode`.
* Fix bug where CPU implementation of non-max-suppression was different
* Fix bug where snapshots are corrupted with an `EOFError`
* Fix bug where temporary directories were not cleaned up


## Version 0.1.0
* Integrate the publicly released Pytorch exporter and deployer.
* Fix bug where train info was not written if you specified a custom train dpath.


## Version 0.0.27

### Added
* Add `DataFrameLight` to `nh.util`, which provides a subset of `pandas.DataFrame` functionality, but much much faster.


## Version 0.0.26

### Fixed
* Tentative Python 2.7 support


## Version 0.0.25

### Fixed
* Fix issue with per-instance FitHarn class loggers


## Version 0.0.24

### Fixed
* Fix tests and raised better errors if `tensorflow` does not exist


## Version 0.0.23

### Fixed
* Fix bug where `seed_global` did not set call `torch.cuda.manual_seed_all`


## Version 0.0.22

### Changed
* Better support for torch.device with `nh.XPU`
* Minor reorganization of FitHarn, added more callbacks


## Version 0.0.21

### Fixed
* Fix issue with unseeded random states. Now defaults to the global `np.random` state.
* Fix bug in `load_arr`


## Version 0.0.20

### Changed
* FitHarn now uses `StreamLogger` instead of print


## Version 0.0.19

### Fixed
* Fix torch 0.4.1 deprecation warnings in focal loss


## Version 0.0.17

### Added
* Add `before_epochs` callback

### Fixed
* Fix tests


## Version 0.0.16

### Added
* Add `nh.util.global_seed`

### Fixed
* Fix MNIST example
* Various minor bug fixes

### Changed
* Small improvements to outputs
* Better test images
* Better YOLO example
* Other stuff I forgot to log, I'm doing this mostly in my spare time!


## Version 0.0.15

### Added
* Add `SlidingWindow` as simplified alternative to `SlidingSlices`


## Version 0.0.14

### Added
* Add zip-awareness to pre-trained loader 
* Expand COCO-API functionality
* Better detection metrics with alternative implementations

### Fixed
* Fix YOLO scheduler


## Version 0.0.13

### Fixed
* Fix issue with `autompl`. Now correctly detects if display is available. 

## Version 0.0.12

### Added
* Early and undocumented commits
