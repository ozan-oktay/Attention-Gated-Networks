#!/usr/bin/env python3
"""Script to check the state of GPU servers

This script is most useful in conjunction with an ssh-key, so a password does
not have to be entered for each SSH connection.
"""
import argparse
import os
import pwd
import subprocess
import sys
import xml.etree.ElementTree as ET

# Timeout in seconds after which SSH stops trying to connect
SSH_TIMEOUT = 3

SERVERS = [
    'monal02.doc.ic.ac.uk',
    'monal03.doc.ic.ac.uk',
    'monal04.doc.ic.ac.uk',
    'monal05.doc.ic.ac.uk',
    'monal06.doc.ic.ac.uk',
    'flamingo.doc.ic.ac.uk'
]

parser = argparse.ArgumentParser(description='Check state of GPU servers')
parser.add_argument('-l', '--list', action='store_true', help='Show used GPUs')
parser.add_argument('-f', '--finger', action='store_true',
                    help='Attempt to resolve user names to real names')
parser.add_argument('-m', '--me', action='store_true',
                    help='Show only GPUs used by current user')
parser.add_argument('-u', '--user', help='Shows only GPUs used by a user')
parser.add_argument('servers', nargs='*', default=SERVERS,
                    help='Servers to probe')


def run_nvidiasmi_local():
    cmd = 'nvidia-smi -q -x'
    try:
        res = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return None

    return ET.fromstring(res)


def run_nvidiasmi_remote(server):
    cmd = 'ssh -o "ConnectTimeout={}" {} nvidia-smi -q -x'.format(SSH_TIMEOUT,
                                                                  server)
    try:
        res = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return None

    return ET.fromstring(res)


def run_ps_remote(server, pids):
    pid_cmd = 'ps -o pid= -o ruser= -p {}'.format(','.join(pids))
    cmd = 'ssh -o "ConnectTimeout={}" {} {}'.format(SSH_TIMEOUT,
                                                    server,
                                                    pid_cmd)
    try:
        res = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError:
        return None

    return res.decode('ascii')


def get_users_by_pid(ps_output):
    users_by_pid = {}
    for line in ps_output.strip().split('\n'):
        pid, user = line.split()
        users_by_pid[pid] = user

    return users_by_pid


def get_gpu_infos(nvidiasmi_output):
    gpus = nvidiasmi_output.findall('gpu')

    gpu_infos = []
    for idx, gpu in enumerate(gpus):
        model = gpu.find('product_name').text
        processes = gpu.findall('processes')[0]
        pids = [process.find('pid').text for process in processes]
        gpu_infos.append({'idx': idx, 'model': model, 'pids': pids})

    return gpu_infos


def print_free_gpus(server, gpu_infos):
    free_gpus = [info for info in gpu_infos if len(info['pids']) == 0]

    if len(free_gpus) == 0:
        print('Server {}: No free GPUs :('.format(server))
    else:
        print('Server {}:'.format(server))
        for info in free_gpus:
            print('\tGPU {}, {}'.format(info['idx'], info['model']))


def print_gpu_infos(server, gpu_infos, filter_by_user=None,
                    translate_to_real_names=False):
    pids = [pid for info in gpu_infos for pid in info['pids']]
    ps = run_ps_remote(server, pids)
    if ps is None:
        print('Could not reach {} or error running ps'.format(server))

    users_by_pid = get_users_by_pid(ps)

    print('Server {}'.format(server))
    for info in gpu_infos:
        users = set((users_by_pid[pid] for pid in info['pids']))
        if filter_by_user is not None and filter_by_user not in users:
            continue

        if len(info['pids']) == 0:
            status = 'Free'
        else:
            if translate_to_real_names:
                real_names = []
                for user in users:
                    try:
                        real_names.append(pwd.getpwnam(user).pw_gecos)
                    except KeyError:
                        real_names.append('Unknown')
                users = ['{} ({})'.format(user, real_name)
                         for user, real_name in zip(users, real_names)]

            status = 'Used by {}'.format(', '.join(users))

        print('\tGPU {} ({}): {}'.format(info['idx'],
                                         info['model'],
                                         status))


def main(argv):
    args = parser.parse_args(argv)

    if args.me:
        args.user = pwd.getpwuid(os.getuid()).pw_name
    if args.user or args.finger:
        args.list = True

    for server in args.servers:
        nvidiasmi = run_nvidiasmi_remote(server)
        if nvidiasmi is None:
            print(('Could not reach {} or '
                   'error running nvidia-smi').format(server))
            continue

        gpu_infos = get_gpu_infos(nvidiasmi)

        if args.list:
            print_gpu_infos(server, gpu_infos,
                            filter_by_user=args.user,
                            translate_to_real_names=args.finger)
        else:
            print_free_gpus(server, gpu_infos)

if __name__ == '__main__':
    main(sys.argv[1:])
