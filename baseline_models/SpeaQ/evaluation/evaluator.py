import copy
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
from collections import Counter
import random
import cv2
import torch

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from detectron2.evaluation import COCOEvaluator

import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from .Image_Translation import image_translanting, get_co_occurence_matrix
from .Visualization import show_all


def scenegraph_inference_on_dataset(cfg, model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger('detectron2')
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    # evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)

    evaluator.reset(total * num_devices)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        "if we want to find the most likely object, we need to firstly calculate the co-occurence matrix"

      #  co_occurence_matrix = get_co_occurence_matrix(data_loader)
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            #        if len(inputs[0]['instances']) > 40:
            #           continue
            start_compute_time = time.perf_counter()
            'mode can be: related_object_in_image,similar_object_in_image,unlikely_onject_in_image,trained_object, untrained_object'
            'patch could be minimal_heuristic,maximal_heuristic,random_heuristic'
            'obi_in_rl is used to determine whether the object is used in reinforcement learning'
            #the two code can be used to test the model with different mode and patch under debug mode,  the origin image can be remained by runnning it
            input_copy = copy.deepcopy(inputs)
            inputs = copy.deepcopy(input_copy)

            mode = "related_object_in_image"
            patch = "random_heuristic"
            if mode != "origin":
                inputs[0] = image_translanting(inputs[0], matrix = None, mode = mode, patch = patch,  obi_in_rl = False)
            outputs = model(inputs)
            show_all(inputs, outputs, mode+'_'+patch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start

            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                # logger.info("Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                    name='detectron2'
                )

            if cfg.DEV_RUN and idx == 2:
                break

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


# _LOG_COUNTER = Counter()
# _LOG_TIMER = {}

# def log_every_n_seconds(lvl, msg, n=1, *, name=None):
#     """
#     Log no more than once per n seconds.

#     Args:
#         lvl (int): the logging level
#         msg (str):
#         n (int):
#         name (str): name of the logger to use. Will use the caller's module by default.
#     """
#     caller_module, key = _find_caller()
#     last_logged = _LOG_TIMER.get(key, None)
#     current_time = time.time()
#     if last_logged is None or current_time - last_logged >= n:
#         logging.getLogger('detectron2').log(lvl, msg)
#         _LOG_TIMER[key] = current_time

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)




