import os
import sys
import pytz
import datetime
import functools
import subprocess
import contextlib


os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


def all_exist(fs):
    for f in fs:
        if not f[1]:
            return False
    return True


@contextlib.contextmanager
def tee_output(file, mode='w'):
    original_stdout = sys.stdout
    with open(file, mode) as f:
        class Tee:
            def write(self, obj):
                original_stdout.write(obj)
                f.write(obj)
            def flush(self):
                original_stdout.flush()
                f.flush()
        sys.stdout = Tee()
        try:
            yield
        finally:
            sys.stdout = original_stdout