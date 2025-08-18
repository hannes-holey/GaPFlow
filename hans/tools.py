#
# Copyright 2021, 2025 Hannes Holey
#
# ### MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#


import sys
from mpi4py import MPI
import numpy as np
import signal




def abort(errcode=1):
    if MPI.COMM_WORLD.Get_size() == 1:
        sys.exit(errcode)
    else:
        sys.stdout.flush()
        MPI.COMM_WORLD.Abort(errcode)


def progressbar(it, prefix="", size=40, out=sys.stdout):  # Python3.6+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(f"{prefix}|{u'█'*x}{('.'*(size-x))}| {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def bordered_text(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)


def power(base, exponent):

    complex_base = base.astype('complex128')
    result = np.float_power(complex_base, exponent)

    return result.real


def handle_signals(func):
    for s in [signal.SIGHUP, signal.SIGINT, signal.SIGHUP, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2]:
        signal.signal(s, func)
