#! /usr/bin/env python3


import argparse
import logging
import socket
import threading
import time

import gpustat
import psutil

import zeyu.net as znet

LSTN_PORT = 43256


class Logger(object):
    def __init__(self, job_name, file_dir, log_level=logging.INFO, mode="w"):
        self.logger = logging.getLogger(job_name)
        self.logger.setLevel(log_level)
        self.fh = logging.FileHandler(filename=file_dir, mode=mode)
        self.formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)


def get_cpu_memory_gpu():
    cpu_usage = psutil.cpu_percent(interval=1)

    virtual_memory = psutil.virtual_memory()
    memory_usage = virtual_memory.used / 1024 / 1024 / 1024

    try:
        query = gpustat.new_query()
    except Exception:
        gpu_memory = 0
        gpu_util = 0
    else:
        gpu = query.gpus[0]
        gpu_memory = 100.0 * gpu.memory_used / gpu.memory_total
        gpu_util = gpu.utilization

    return cpu_usage, memory_usage, gpu_memory, gpu_util


def run_logging(connm: znet.SocketMsger, args, logger):
    end_flag = [False]

    def ending_waiter(connm: znet.SocketMsger, end_flag):
        recv = connm.recv()
        if recv == "END":
            end_flag[0] = True
            connm.close()
        elif recv is None:
            end_flag[0] = True

    ending_waiter_t = threading.Thread(target=ending_waiter, args=(connm, end_flag))
    ending_waiter_t.start()

    memory_usage = psutil.virtual_memory().used / 1024 / 1024 / 1024

    bw = psutil.net_io_counters(pernic=True)

    # conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # conn.connect(('172.17.0.1', 24577))

    nic = socket.if_nameindex()[1][1]
    temp_recv = bw[nic].bytes_recv
    temp_sent = bw[nic].bytes_sent

    start_time = time.time()
    counter = 0
    total_cpu = 0.0
    total_mem = 0.0
    total_gpu_mem = 0.0
    total_recv = 0.0
    counter_recv = 0
    total_send = 0.0
    counter_send = 0
    total_gpu_util = 0.0
    counter_gpu_util = 0
    t0 = time.time()
    while True:
        if end_flag[0] is True:
            total_time = time.time() - start_time
            avg_cpu = total_cpu / counter
            avg_mem = total_mem / counter
            avg_gpu_mem = total_gpu_mem / counter
            avg_gpu_util = (
                0.0 if counter_gpu_util == 0 else total_gpu_util / counter_gpu_util
            )
            avg_recv = total_recv / counter_recv
            avg_send = total_send / counter_send
            logger.info(
                "======================================================================================================"
            )
            logger.info(f"JCT: {total_time}s")
            logger.info("Total:")
            logger.info(
                "cpu_usage: {:6.2f}% | memory_usage: {:5.2f} GB | gpu_memory: {:3.0f}% | gpu_util: {:3.0f}% "
                "| Bandwidth | recv: {:7.2f} MB | sent: {:7.2f} MB".format(
                    total_cpu,
                    total_mem,
                    total_gpu_mem,
                    total_gpu_util,
                    total_recv,
                    total_send,
                )
            )
            logger.info("Average:")
            logger.info(
                "cpu_usage: {:6.2f}% | memory_usage: {:5.2f} GB | gpu_memory: {:3.0f}% | gpu_util: {:3.0f}% "
                "| Bandwidth | recv: {:7.2f} MB | sent: {:7.2f} MB".format(
                    avg_cpu,
                    avg_mem,
                    avg_gpu_mem,
                    avg_gpu_util,
                    avg_recv,
                    avg_send,
                )
            )
            logger.info(
                "======================================================================================================"
            )
            return

        t1 = time.time()
        if t1 - t0 > args.interval:
            t0 = t1
            bw = psutil.net_io_counters(pernic=True)
            cpu_usage, memory_usage, gpu_memory, gpu_util = get_cpu_memory_gpu()

            recv_rate = (bw[nic].bytes_recv - temp_recv) / 1024.0 / 1024.0
            send_rate = (bw[nic].bytes_sent - temp_sent) / 1024.0 / 1024.0
            logger.info(
                "cpu_usage: {:6.2f}% | memory_usage: {:5.2f} GB | gpu_memory: {:3.0f}% | gpu_util: {:3d}% "
                "| Bandwidth | recv: {:7.2f} MB | sent: {:7.2f} MB".format(
                    cpu_usage,
                    memory_usage,
                    gpu_memory,
                    gpu_util,
                    recv_rate,
                    send_rate,
                )
            )
            counter += 1
            total_cpu += cpu_usage
            total_mem += memory_usage
            total_gpu_mem += gpu_memory
            if gpu_util > 3.0:
                total_gpu_util += gpu_util
                counter_gpu_util += 1
            if recv_rate > 3.0:
                total_recv += recv_rate
                counter_recv += 1
            if send_rate > 3.0:
                total_send += send_rate
                counter_send += 1
            # conn.send('cpu_usage: {:6.2f}%\n'.format(cpu_usage).encode('utf-8'))

            temp_recv = bw[nic].bytes_recv
            temp_sent = bw[nic].bytes_sent


def ps_logging(connm: znet.SocketMsger, args, logger):
    recv = connm.recv()
    recv = recv.split("\n")
    interval = float(recv[0])
    file_dir = recv[1]
    job_name = recv[2]
    fh = logging.FileHandler(file_dir, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    recv = connm.recv()
    if recv == "START":
        connm.send("CONFIRM")
        run_logging(connm, args, logger)


def worker_logging(connm: znet.SocketMsger, args, logger):
    recv = connm.recv()
    recv = recv.split("\n")
    interval = float(recv[0])
    file_dir = recv[1]
    job_name = recv[2]
    fh = logging.FileHandler(file_dir, mode="w")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    recv = connm.recv()
    if recv == "START":
        connm.send("CONFIRM")
        run_logging(connm, args, logger)


def conn_thread(connm: znet.SocketMsger, args, logger):
    recv = connm.recv()
    if recv == "PS":
        ps_logging(connm, args, logger)
    elif recv == "WORKER":
        worker_logging(connm, args, logger)
    else:
        return


def listener_thread(lstn_ip, lstn_port, args, logger):
    listener = znet.SocketMsger.tcp_listener(lstn_ip, lstn_port)
    while True:
        connm, _ = listener.accept()
        conn_t = threading.Thread(target=conn_thread, args=(connm, args, logger))
        conn_t.start()


def main_thread(args, logger):
    listener_t = threading.Thread(
        target=listener_thread, args=("0.0.0.0", LSTN_PORT, args, logger)
    )
    listener_t.start()
    listener_t.join()


def init_and_get_args_logger():
    parser = argparse.ArgumentParser(description="Worker information")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval")
    parser.add_argument(
        "--file_dir",
        type=str,
        default="./measurement/logs/info.log",
        help="The log location.",
    )
    parser.add_argument(
        "--job_name", type=str, default="jobname", help="The job's name."
    )

    args = parser.parse_args()

    logger = logging.getLogger("info")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # fh = logging.FileHandler(args.file_dir, mode="w")
    # fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
    ch.setFormatter(formatter)
    # fh.setFormatter(formatter)
    logger.addHandler(ch)
    # logger.addHandler(fh)

    return args, logger


if __name__ == "__main__":
    args, logger = init_and_get_args_logger()
    main_thread(args, logger)
