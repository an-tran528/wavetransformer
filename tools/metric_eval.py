#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap


__author__ = 'Samuel Lipping, Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['reformat_to_coco', 'evaluate_metrics_from_files', 'write_json']


def reformat_to_coco(predictions, ground_truths, ids=None):
    """Reformat annotation lists to the COCO format.

    :param predictions: List of predicted captions.
    :type predictions: list
    :param ground_truths: List of lists of reference captions.
    :type ground_truths: list
    :param ids: List of file(image) ids corresponding to predictions\
                and ground_truths.
    :type ids: list
    :return: Predicted and reference captions in the COCO format.
    :rtype: (list, list)
    """
    # Running number as ids for files if not given
    if ids is None:
        ids = range(len(predictions))

    # Captions need to be in format
    # [{
    #     "image_id": : int, (COCO code is for image captioning)
    #     "caption"  : str
    # ]},
    # as per the COCO results format.
    pred = []
    ref = {
        "info": {"description": "Clotho reference captions (2019)"},
        "images": [],
        "licenses": [
            {"id": 1},
            {"id": 2},
            {"id": 3}
        ],
        "type": "captions",
        "annotations": []
    }
    cap_id = 0
    for id, p, gt in zip(ids, predictions, ground_truths):
        p = p[0] if isinstance(p, list) else p
        pred.append({
            "image_id": id,
            "caption": p
        })

        ref["images"].append({
            "id": id
        })

        for cap in gt:
            ref["annotations"].append({
                "image_id": id,
                "id": cap_id,
                "caption": cap
            })
            cap_id += 1

    return pred, ref


def evaluate_metrics_from_files(pred_file, ref_file):
    """Evaluate the translation metrics from annotation files with the coco lib\
    Follows the example in the coco-caption repo.

    :param pred_file: File with predicted captions
    :type pred_file: pathlib.Path
    :param ref_file: File with reference captions
    :type ref_file: pathlib.Path
    :return: Metrics in dictionary
    """
    # Load annotations from files
    coco = COCO(str(ref_file))
    cocoRes = coco.loadRes(str(pred_file))

    # Create evaluation object and evaluate metrics
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # Make dict from metrics
    metrics = dict((m, s) for m, s in cocoEval.eval.items())
    return metrics


def write_json(data, path):
    """Write a json object into a file

    :param data: JSON object
    :type data: dict|list
    :param path: File to write in
    :type path: pathlib.Path
    """
    with path.open('w') as f:
        json.dump(data, f)

# EOF
