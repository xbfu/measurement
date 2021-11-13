import argparse
import logging
import os
import threading
import time

import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import vgg, densenet, mobilenetv2, alexnet, googlenet, resnet3

import zeyu.net as znet

DORKER0_IP = "172.17.0.1"
INFO_PORT = 43256


class Logger(object):
    def __init__(self, job_name, file_dir, log_level=logging.INFO, mode="w"):
        self.logger = logging.getLogger(job_name)
        self.logger.setLevel(log_level)
        self.fh = logging.FileHandler(filename=file_dir, mode=mode)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)


class ParameterServer(object):
    def __init__(self, model, num_workers, lr, job_name):
        self.lock = threading.Lock()
        self.logger = Logger(job_name=job_name, file_dir=f"./measurement/logs/{job_name}_ps.log").logger
        self.cm_t1_start = np.zeros(num_workers)
        self.future_model = torch.futures.Future()
        self.batch_update_size = num_workers
        self.curr_update_size = 0
        self.stop_flag = False
        if model == 'resnet20':
            self.model = resnet3.resnet20()
        elif model == 'resnet56':
            self.model = resnet3.resnet56()
        elif model == 'vgg13':
            self.model = vgg.VGG13()
        elif model == 'vgg16':
            self.model = vgg.VGG16()
        elif model == 'densenet121':
            self.model = densenet.DenseNet121()
        elif model == 'alexnet':
            self.model = alexnet.AlexNet()
        elif model == 'googlenet':
            self.model = googlenet.GoogLeNet()
        elif model == 'mobilenet':
            self.model = mobilenetv2.MobileNetV2()
        self.lr = lr
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.info_socketm = znet.SocketMsger.tcp_connect(DORKER0_IP, INFO_PORT)
        self.info_socketm.send("PS")
        self.info_socketm.send(f"1.0\n/home/ubuntu/measurement/logs/{job_name}_info0.log\n{job_name}")
        self.ps_launched_lock = threading.Lock()
        self.ps_launched = False

    def get_model(self):
        return self.model

    def set_ps_launched_to_true(self):
        with self.ps_launched_lock:
            if self.ps_launched is False:
                self.ps_launched = True
                self.info_socketm.send("START")
                if self.info_socketm.recv() != "CONFIRM":
                    return

    def stop(self):
        self.stop_flag = True
        self.info_socketm.send("END")

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads, name, i, batch_idx, cm_t0_start, cm_t1_end):
        self = ps_rref.local_value()
        cm_t0_end = time.time()
        cm_t0 = 1000 * (cm_t0_end - cm_t0_start)
        worker_rank = int(name[-1])
        cm_t1 = 1000 * (cm_t1_end - self.cm_t1_start[worker_rank - 1])

        self.logger.info("Epoch: {:3d} | Batch: {:3d} | {:8s} | Communication O: {:7.2f} ms"
                         .format((i + 1), batch_idx - 1, name, cm_t1))
        self.logger.info("Epoch: {:3d} | Batch: {:3d} | {:8s} | Communication I: {:7.2f} ms"
                         .format((i + 1), batch_idx, name, cm_t0))

        with self.lock:
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grads):
                    p.grad = g

            t1 = time.time()
            self.optimizer.step()
            self.optimizer.zero_grad()

            t3 = time.time()
            self.logger.info("Update Time: {:.2f} ms".format(1000 * (t3 - t1)))

            fut = self.future_model

            self.cm_t1_start[worker_rank - 1] = time.time()

            self.logger.info(f"PS sending updated parameters to {name}")

            fut.set_result([self.model, self.stop_flag])
            self.future_model = torch.futures.Future()

        return fut


def run_worker(ps_rref, data_dir, batch_size, num_epochs, worker, job_name):
    worker_rank = int(worker[-1])
    info_socketm = znet.SocketMsger.tcp_connect(DORKER0_IP, INFO_PORT)
    info_socketm.send("WORKER")
    info_socketm.send(f"1.0\n/home/ubuntu/measurement/logs/{job_name}_info{worker_rank}.log\n{job_name}")

    logger = Logger(job_name=job_name, file_dir=f"./measurement/logs/{job_name}_{worker}.log").logger

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    name = rpc.get_worker_info().name

    ps_rref.rpc_sync().set_ps_launched_to_true()

    m = ps_rref.rpc_sync().get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    stop_flag = False

    info_socketm.send("START")
    if info_socketm.recv() != "CONFIRM":
        return

    cm_t1_end = time.time()
    tt0 = time.time()

    for i in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = m(data)
            loss = criterion(output, target)
            loss.backward()

            cm_t0_start = time.time()
            cp_t = 1000 * (cm_t0_start - cm_t1_end)

            logger.info("{:8s} | Epoch: {:3d} | Batch: {:3d} | Loss: {:6.2f} | Computation Time: {:7.2f} ms"
                        .format(name, (i + 1), (batch_idx + 1), loss.item(), cp_t))

            m, stop_flag = rpc.rpc_sync(
                to=ps_rref.owner(),
                func=ParameterServer.update_and_fetch_model,
                args=(ps_rref, [p.grad for p in m.cpu().parameters()], name, i, batch_idx, cm_t0_start, cm_t1_end))
            m.to(device)

            cm_t1_end = time.time()

            if stop_flag:
                break
        if stop_flag:
            break

    tt1 = time.time()

    info_socketm.send("END")

    logger.info("Time: {:.2f} seconds".format((tt1 - tt0)))


def get_accuracy(ps_rref, data_dir, test_batch_size, job_name, target_loss):
    logger = Logger(job_name=job_name, file_dir=f"./measurement/logs/{job_name}_tester.log").logger

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    dataset_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    t0 = time.time()
    logger.info("Start!")
    init = t0
    while True:
        t1 = time.time()
        if t1 - t0 > 40:
            t0 = t1
            m = ps_rref.rpc_sync().get_model().to(device)

            test_loss = 0
            correct = 0

            with torch.no_grad():
                for j, (data, target) in enumerate(test_loader):
                    data, target = data.to(device), target.to(device)
                    output = m(data)
                    loss = criterion(output, target)

                    test_loss += loss.item()
                    _, predicted = output.max(1)
                    correct += predicted.eq(target).sum().item()

            test_loss = test_loss * test_batch_size / len(test_loader.dataset)
            accuracy = 100. * correct / len(dataset_test)
            logger.info("Test Loss: {:6.3f} | Accuracy: {:5.2f} % | Time: {:7.2f} seconds"
                        .format(test_loss, accuracy, (t1 - init)))

            if test_loss < target_loss:
                ps_rref.rpc_sync().stop()
                break


def run(rank, num_workers, data_dir, model, batch_size, test_batch_size, lr, num_epochs, job_name, target_loss):
    logging.basicConfig(level=logging.INFO)
    world_size = num_workers + 2
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=0)

    if rank == 0:
        logging.info(f"PS{rank} initializing")
        rpc.init_rpc(f"PS{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"PS{rank} initialized")

        workers = [f"worker{r}" for r in range(1, world_size - 1)]
        ps_rref = rpc.RRef(ParameterServer(model, num_workers, lr, job_name))

        futs = []
        futs.append(rpc.rpc_async(to="tester",
                                  func=get_accuracy,
                                  args=(ps_rref, data_dir, test_batch_size, job_name, target_loss)))
        for worker in workers:
            futs.append(rpc.rpc_async(to=worker,
                                      func=run_worker,
                                      args=(ps_rref, data_dir, batch_size, num_epochs, worker, job_name)))

        torch.futures.wait_all(futs)
        logging.info(f"Finish training")

    elif rank == world_size - 1:
        logging.info(f"Tester initializing")
        rpc.init_rpc("tester", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"Tester initialized")

    else:
        logging.info(f"Worker{rank} initializing")
        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
        logging.info(f"Worker{rank} initialized")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on Imagenet under ASGD")
    parser.add_argument("--job_name", type=str, default="test", help="The job's name.")
    parser.add_argument("--model", type=str, default="resnet50", help="The model's name.")
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--num_workers", type=int, default=1, help="Total number of workers.")
    parser.add_argument("--data_dir", type=str, default="./measurement/data", help="The location of dataset.")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="29600", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of each worker during training.")
    parser.add_argument("--test_batch_size", type=int, default=64, help="Batch size during testing.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--target_loss", type=float, default=0.8, help="Targer accuracy.")

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    run(args.rank, args.num_workers, args.data_dir, args.model, args.batch_size, args.test_batch_size, args.lr,
        args.num_epochs, args.job_name, args.target_loss)
