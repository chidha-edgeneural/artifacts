# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist

import mmcv
from mmcv.fileio.file_client import FileClient
from mmcv.utils import is_tuple_of, scandir
from mmcv.runner import HOOKS,Hook,LoggerHook
import requests
import json

"""
Modified from source of TextLoggerHook
"""

UPDATE_STATUS_URL = "https://enap.edgeneural.ai:3200/api/training/updateTrainingStatus/"
configs = json.loads(open(".tmp_vals",'r').read())
UPDATE_STATUS_URL = configs["UPDATE_STATUS_URL"]

@HOOKS.register_module()
class MetricsCSCVLoggerHook(LoggerHook):
    """Logger hook in text.

    In this logger hook, the information will be printed on terminal and
    saved in json file.

    Args:
        by_epoch (bool, optional): Whether EpochBasedRunner is used.
            Default: True.
        interval (int, optional): Logging interval (every k iterations).
            Default: 10.
        ignore_last (bool, optional): Ignore the log of last iterations in each
            epoch if less than :attr:`interval`. Default: True.
        reset_flag (bool, optional): Whether to clear the output buffer after
            logging. Default: False.
        interval_exp_name (int, optional): Logging interval for experiment
            name. This feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
        out_dir (str, optional): Logs are saved in ``runner.work_dir`` default.
            If ``out_dir`` is specified, logs will be copied to a new directory
            which is the concatenation of ``out_dir`` and the last level
            directory of ``runner.work_dir``. Default: None.
            `New in version 1.3.16.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be copied to ``out_dir``.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.3.16.`
        keep_local (bool, optional): Whether to keep local log when
            :attr:`out_dir` is specified. If False, the local log will be
            removed. Default: True.
            `New in version 1.3.16.`
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`
    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000,
                 out_dir=None,
                 out_suffix=('.log.json', '.log', '.py'),
                 keep_local=True,
                 file_client_args=None):
        interval = 1
        super(MetricsCSCVLoggerHook, self).__init__(interval, ignore_last, reset_flag,
                                             by_epoch)
        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name

        self.last_values = []

        if out_dir is None and file_client_args is not None:
            raise ValueError(
                'file_client_args should be "None" when `out_dir` is not'
                'specified.')
        self.out_dir = out_dir

        if not (out_dir is None or isinstance(out_dir, str)
                or is_tuple_of(out_dir, str)):
            raise TypeError('out_dir should be  "None" or string or tuple of '
                    'string, but got {out_dir}')
        self.out_suffix = out_suffix

        self.keep_local = keep_local
        self.file_client_args = file_client_args

        self.csv_log_file = None

    def before_run(self, runner):
        super(MetricsCSCVLoggerHook, self).before_run(runner)

        self.start_iter = runner.iter

        self.csv_log_file = open(osp.join(runner.work_dir,"metrics.csv"),'w')
        self.csv_log_file.write("Epoch,train_total_loss,val_loss,mAP,mAP@.5,mAP@.75,mAP_s,mAP_m,mAP_l\n")
        self.csv_log_file.flush()
        #TODO:add a line to enter the headers first

    def _log_info(self, log_dict, runner):

        if log_dict['mode'] == 'train':

            if not log_dict["iter"]==len(runner.data_loader) :
                return # to log only druing the end of an epoch

            self.last_values = [
                log_dict["epoch"],
                log_dict['loss']
            ]

        else: # in mAP evaluation epoch or validation epoch
            if "bbox_mAP" in log_dict.keys() : #Successful mAP evaluation epoch
                #A seperate evaluation epoch runs to evaluate bbox
                self.last_values += [
                    log_dict["bbox_mAP"],
                    log_dict["bbox_mAP_50"],
                    log_dict["bbox_mAP_75"],
                    log_dict["bbox_mAP_s"],
                    log_dict["bbox_mAP_m"],
                    log_dict["bbox_mAP_l"]
                ]
            elif "loss" not in log_dict.keys() : #Failed mAP epoch with Nans so just display 0s
                self.last_values += [
                        0,#bbox_mAP,
                        0,#bbox_mAP_50
                        0,#bbox_mAP_75
                        0,#bbox_mAP_s
                        0,#bbox_mAP_m
                        0,#bbox_mAP_l
                        ]
            else: #validation loss epoch
                self.last_values.insert(2,log_dict["loss"])#check the order of items in the header of file
                self.last_values = [str(x) for x in self.last_values]#converting the integers into strings for concatenation
                self.csv_log_file.write( ",".join(self.last_values) +"\n" )
                self.csv_log_file.flush()

                ##Sending recent status to 
                update_post_params = dict()
                update_post_params["status_code"] = 200
                update_post_params["error"] = ""
                update_post_params["status"] = "In Progress"
                update_post_params["epochs"] = self.last_values[0] #first value is the epoch number
                update_post_params["sessionid"] = configs["sessionid"]
                metrics_csv_file = [('metrics.csv', open('./artifacts/metrics.csv','rb'))]
                headers = {}
                print("Metrics CSV response====================================")
                response = requests.request("POST", UPDATE_STATUS_URL, headers=headers, data=update_post_params , files=metrics_csv_file)
                print(response.content)
                print("========================================================")

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    def _get_max_memory(self, runner):
        device = getattr(runner.model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=device)
        if runner.world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def log(self, runner):
        if 'eval_iter_num' in runner.log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = runner.log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(runner, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(runner),
            epoch=self.get_epoch(runner),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in runner.log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(runner)

        log_dict = dict(log_dict, **runner.log_buffer.output)

        self._log_info(log_dict, runner)
        return log_dict

