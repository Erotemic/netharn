Version 0.2.0
==============
* Refactored `netharn.utils` to depend on `kwarray`, `kwimage`, and `kwplot`,
  this removes a lot of the extra cruft added in `0.1.8`.
* Can now specify the package zip-file name when deploying.
* Add option `FitHarn.config['use_tensorboard'] = True` 
* Add `FitHarn.prepare_epoch` callback.
* `load_partial_state` now returns dict containing info on which keys were unused
* `nh.initializers.Pretrained` now returns info dict from `load_partial_state`
* `nll_focal_loss` now is as fast as `nll_loss` when `focus=0`


Version 0.1.8
==============
* Backport `ndsampler` Coco-API (PF/IF)
* Move `Boxes` and `DataFrameLight` from `netharn.util` to `netharn.util.structs` (IF/PF)
* Enhance `Boxes` and `DataFrameLight` functionality / docs (IF/PF)
* Add `netharn.util.structs.Detections` (IF/PF)
# Note: above changes should be reworked to simply depend on ndsampler and kwimage
* Loss components are now automatically logged when loss is returned as a dict (IF). 
* Add a small interactive debug interface on `KeyboardInterrupt` (IF)
* Fix XPU.coerce / XPU.cast when input is multi-gpu
* Add `draw_clf_on_image` (PF)
* Add `valign` to `draw_text_on_image` (PF)
* Add `border` to `draw_text_on_image` (PF)
* A handful of PF GGR-related commits stashed on my home machine meant for 0.1.7 
* Add `nh.data.batch_samplers.MatchingSamplerPK` (PF)
* Add `shift_sat` and `shift_val` to HSV augmenter (PF)
* Refactor and clean `api.py` (PF)
* Refactor and clean `netharn.initializers` (PF)
* Refactor `draw_boxes` and `draw_segments` into `mpl_draw` (IF)
* Fixed issues with YOLO example (PF)
* Add `torch_ravel_multi_index` to `nh.util` (PF)
* Add `arglexmax`, `argmaxima`, `argminima` (IF)
* Add `util_distributions` (IF)


Version 0.1.7
==============
* Modified batch outputs to all use `:g` format (PF)
* Add `plot_surface3d` (PF)
* Use `progiter` by default instead of `tqdm` (PF)
* Add `models.DescriptorNetwork` (PF)
* `MLP` can now accept `dim=0` (PF)
* `nh.XPU.variable` is deprecated and removed. (IF)
* `nh.XPU.move` is now applied recursively to containers (e.g. dict list) (IF)
* All `MovingAve` objects can now track variance  (PF)
* `CumMovingAve` can now track variance (PF)
* `ExpMovingAve` can now track variance  (PF)
* `WindowedMovingAve` can now track variance  (PF)
* `lr_range_test` now shows std-dev error bars (IF)
* Improve API coerce methods (PF / IF)
* `imread` now attempts to return RGB or gray-scale by default. (IF)


Version 0.1.6
==============
* Fix Python2 compatibility issues. (IF)
* Fixed bug in `IgnoreLayerContext` preventing it from being used with `DataParallel`. (IF)
* Add `api.py` containing code to help reduce netharn boilerplate by parsing a config dictionary. (PF)
* Remove deprecated `_to_var`. (PF)
* Add new `ListedScheduler` which is able to modify multiple optimizer attributes including learning rate and momentum. (PF)
* FitHarn now logs momentum by default in addition to learning rate  (PF)
* Add variant of Leslie Smith's learning rate test (IF)
* `nh.util.ExpMovingAve` now has a bias-correction option. (IF)


Version 0.1.5
==============
* Switched to `skbuild` (PF)
* Bug fixes (IF)

Version 0.1.4
==============
* Ported `multi_plot` from `KWIL` (IF)
* Scheduler states are now saved by default (IF)
* Netharn now dumps tensorboard plots every epoch by default (IF)
* The default `prepare_batch` now returns a dictionary with keys `input` and `label`. (IF)
* `FitHarn.config` can now specify `export_modules`, which will be modules to
  expand when running the pytorch exporter. (IF)
* Ported modifications from KWIL to `imwrite`, `stack_imges`, etc... (IF)
* Fix issue with relative imports in netharn exporter (IF)
* Refactored the exporter closure-extractor into its own file. (IF)
* Add `devices` to `nh.layers.Module` (IF)
* Deprecate `HiddenShapesFor` (IF)
* Move `HiddenShapesFor` functionality to `OutputShapeFor` (IF)
* Improve CIFAR example. (PF)
* Improve MNIST example. (PF)
* Rename internal variables of `nh.Monitor` (IF)
* Improve doc-strings for `nh.Monitor`
* Move folder functionality into `hyperparams`. (IF)

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
* 


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


NOTE
====
PF = public funding (does not require public release)
IF = internal funding (requires public release)
