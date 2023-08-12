# -*- coding: utf-8 -*-

import os

import cv2
from tqdm import tqdm
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


class TestMetric(object):
    def __init__(self, root_preds_path, gt_path):
        """
        From https://github.com/lartpang/Py-SOD-VOS-EvalToolkit/blob/81ce89da6813fdd3e22e3f20e3a09fe1e4a1a87c/utils
        /recorders/metric_recorder.py
        """
        self.MAE = MAE()
        self.FM = Fmeasure()
        self.SM = Smeasure()
        self.EM = Emeasure()
        self.WFM = WeightedFmeasure()

        self.root_preds_path = root_preds_path
        self.gt_path = gt_path

    def back_results(self):
        data_root = "./test_data"
        mask_root = self.gt_path
        pred_root = self.root_preds_path

        mask_name_list = sorted(os.listdir(mask_root))
        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)
            pred_path = os.path.join(pred_root, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            self.FM.step(pred=pred, gt=mask)
            self.WFM.step(pred=pred, gt=mask)
            self.SM.step(pred=pred, gt=mask)
            self.EM.step(pred=pred, gt=mask)
            self.MAE.step(pred=pred, gt=mask)

        fm = self.FM.get_results()["fm"]
        wfm = self.WFM.get_results()["wfm"]
        sm = self.SM.get_results()["sm"]
        em = self.EM.get_results()["em"]
        mae = self.MAE.get_results()["mae"]

        results = {
            "maxFm": fm["curve"].max(),
            "wFmeasure": wfm,
            "MAE": mae,
            # "adpEm": em["adp"],
            "Smeasure": sm,
            "meanEm": em["curve"].mean(),
            # "maxEm": em["curve"].max(),
            # "adpFm": fm["adp"],
            # "meanFm": fm["curve"].mean(),
        }
        return results


if __name__ == "__main__":

    root_preds_path = r"/media/pjl307/data/experiment/ZZJ/DIS/DIS5K/DIS-VD/media/DIS-VD"
    gt_path = r"/media/pjl307/data/experiment/ZZJ/DIS/DIS5K/DIS-VD/gt"
    testMetric = TestMetric(root_preds_path, gt_path)
    results = testMetric.back_results()
    print(results)
