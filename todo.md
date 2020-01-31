This document contains a list of work that development items that need to be addressed.


- [ ] Plot variance around the loss and measure curves
- [ ] Removed logging dependency on `tensorboard` (use a simple in-house logging serialization format)
- [ ] Fine-tune default configuration for efficiency and intuitive appeal

- [ ] Extend closer functionality
    - [ ] Make sure we can close ourself. 
    - [ ] Allow `harn.config` to specify other classes to export (besides the model itself)
    - [ ] Need to make the closer generally robust. 

- [ ] Run one validation batch after every training batch in the first epoch to
  see how a single set of validation examples evolve.
