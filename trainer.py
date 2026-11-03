import sys
import logging
import copy
import os
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    gamma_list = copy.deepcopy(args.get("gamma", [0.0]))
    device = copy.deepcopy(args["device"])

    for gamma in gamma_list:
        args["gamma"] = gamma

        for seed in seed_list:
            args["seed"] = seed
            args["device"] = device
            _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    run_name = "{}_{}/{}_{}_init={}_incr={}/a={}_gamma={}_seed={}".format(
        args["model_name"],
        args.get("occe_criterion", "none"),
        args["prefix"],
        args["dataset"],
        init_cls,
        args["increment"],
        str(args.get("prior_a")).replace(".", "_"),
        str(args["gamma"]).replace(".", "_"),
        args["seed"],
    )
    logs_dir = os.path.join(args.get("log_dir", "logs"), run_name)
    ckpt_dir = os.path.join(args.get("checkpoint_dir", "checkpoints"), run_name)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=os.path.join(logs_dir, "logfile.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args.get("aug", 1),
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    for task in range(data_manager.nb_tasks):
        if load_model_from_task(model, data_manager, args, task, ckpt_dir):
            continue

        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if args.get("save_checkpoints", True):
            model.save_checkpoint(os.path.join(ckpt_dir, "model_on_task"))

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        cnn_matrix.append(_grouped_row(cnn_accy))
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        if nme_accy is not None:
            logging.info("NME: {}".format(nme_accy["grouped"]))
            nme_matrix.append(_grouped_row(nme_accy))
            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}".format(nme_curve["top5"]))
            logging.info("Average Accuracy (NME): {}\n".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))

    _log_forgetting("CNN", cnn_matrix, task)
    _log_forgetting("NME", nme_matrix, task)


def _grouped_row(accy):
    keys = sorted(key for key in accy["grouped"].keys() if "-" in key)
    return [accy["grouped"][key] for key in keys]


def _log_forgetting(name, matrix, task):
    if len(matrix) == 0 or task == 0:
        return
    np_acctable = np.zeros([task + 1, task + 1])
    for idxx, line in enumerate(matrix):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    np_acctable = np_acctable.T
    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    logging.info("Accuracy Matrix ({}):\n{}".format(name, np_acctable))
    logging.info("Forgetting ({}): {}".format(name, forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


def load_model_from_task(model, data_manager, args, current_task, ckpt_dir):
    """Resume a run: skip tasks before `load_from_task` and load its saved checkpoint."""
    if args.get("load_from_task", None) is None:
        return False

    model._total_classes = current_task*args["increment"] + args["init_cls"]

    if current_task < args["load_from_task"]:
        logging.info("Skipping task {} / {}".format(current_task, args["load_from_task"]))
        model._network.update_fc(model._total_classes)
        model._cur_task += 1
    else:
        logging.info("Load model from task {}".format(current_task))
        model._network.update_fc(model._total_classes)
        model._cur_task += 1
        path = os.path.join(ckpt_dir, "model_on_task_{}.pkl".format(current_task))
        if not os.path.exists(path):
            raise FileNotFoundError("Stored model for task {} not found in {}.".format(current_task, path))
        checkpoint = torch.load(path, map_location="cpu")
        model._network.load_state_dict(checkpoint["model_state_dict"])

        model._network.to(model._device)
        model._construct_exemplar_unified(data_manager, args["memory_size"] // model._total_classes)
        model.after_task()
        args["load_from_task"] = None

    return True
