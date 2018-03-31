# -*- coding: utf-8 -*-
"""
"""
import torch  # NOQA
import ubelt as ub  # NOQA
import numpy as np  # NOQA
from nnharn import device  # NOQA
from nnharn import torch_utils  # NOQA
from nnharn.device import XPU  # NOQA
import logging  # NOQA


class FitHarn(object):
    """

    Args:
        hyper (nnharn.Hyperparameters): Parameters that determine the system.
            This serializable class encodes enough information to
            deterministically reproduce an experiment.

            Because it is serializable it also has an easy to use dict
            representation.


    """
    def __init__(self, hyper):
        self.hyper = hyper

    def run(harn):
        pass

    def setup_paths(harn):
        pass

    def initialize(harn):
        """
        Uses the hyper parameters to initialize the necessary resources and
        restart from previously
        """
        pass

    def run_epoch(harn):
        pass

    def prepare_batch(harn, batch):
        inputs, labels = batch
        inputs = harn.xpu.variables(*inputs)
        labels = harn.xpu.variables(*labels)
        return inputs, labels

    def run_batch(harn):
        pass

    # Internal
    def _current_lrs(harn):
        pass

    def _step_scheduler(harn):
        pass

    def _check_termination(harn):
        pass

    # Snapshoting
    def _clean_snapshots(harn):
        pass

    def load_snapshot(harn):
        pass

    def save_snapshot(harn):
        pass

    def prev_snapshots(harn):
        pass

    # Progress
    def _close_prog(harn):
        pass

    def _update_prog_description(harn):
        pass

    # Logging
    def log(harn, msg):
        pass

    def debug(harn, msg):
        pass

    def log_value(harn, msg):
        pass

    def log_image(harn, msg):
        pass

    # Basic callbacks
    def batch_metrics(harn):
        raise NotImplementedError('custom callback')

    def on_batch(harn):
        raise NotImplementedError('custom callback')

    def on_epoch(harn):
        raise NotImplementedError('custom callback')

    # Extra
    def _demo_batch(harn):
        pass
