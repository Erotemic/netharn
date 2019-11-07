Version 0.1.0
==============
* Integrated the publicly released Pytorch exporter and deployer.
* Fixed bug where train info was not written if you specified a custom train dpath.


Version 0.0.27
==============
* Added DataFrameLight to util, which provides a subset of pandas DataFrame functionality, but much much faster.


Version 0.0.26
==============
* Tentative Python2.7 support


Version 0.0.25
==============
* Fixed issue with per-instance FitHarn class loggers


Version 0.0.24
==============
* Fixed tests and raised better errors if tensorflow does not exist


Version 0.0.23
==============
* Fixed bug where seed_global did not set call `torch.cuda.manual_seed_all`


Version 0.0.22
==============
* Better support for torch.device with nh.XPU
* Minor reorganization of FitHarn, added more callbacks



Version 0.0.21
==============
* Fixed issue with unseeded random states. Now defaults to the global np.random state.
* Fixed bug in `load_arr`


Version 0.0.20
==============
* FitHarn now uses StreamLogger instead of print


Version 0.0.19
==============
* Fixed torch 0.4.1 deprication warnings in focal loss


Version 0.0.17
==============
* Fixed tests
* Added before_epochs callback



Version 0.0.16
==============
* Added `nh.util.global_seed`
* Fixed MNIST example
* Small improvements to outputs
* Better test images
* Better YOLO example
* Various minor bugfixes
* Other stuff I forgot to log, I'm doing this mostly in my spare time!


Version 0.0.15
==============
* Added SlidingWindow as simplified alternative to SlidingSlices


Version 0.0.14
==============
* Added zip-awareness to pretrained loader 
* Expanded coco-api functionality
* Better detection metrics with alternative implementations
* Fixed YOLO scheduler


Version 0.0.13
==============
* Fixed issue with autompl. Now correctly detects if display is available. 
