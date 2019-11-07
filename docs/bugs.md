## Known Bugs

* The background matplotlib dumps of logged tensorboard scalars sometimes outputs garbage images. The fonts are in the wrong place, and parts of the image are cut off. This may be due to some race condition. Perhaps multiple dumps were spawned at once and were clobbering the encoding of one another?


* The per-batch iteration metrics seem to jump on the x-axis in the tensorboard logs. Not sure why this is. Perhaps there is a scheduler bug? 
