#
# Copyright 2025 Hannes Holey
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
import numpy as np
import jax.numpy as jnp


def progressbar(it, prefix="", size=40, out=sys.stdout):  # Python3.6+
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(f"{prefix}|{u'█' * x}{('.' * (size - x))}| {j}/{count}", end='\r', file=out, flush=True)
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


def make_dumpable(obj):
    """
    Recursively convert numpy and other non-native types in dictionaries/lists
    to native Python types so they can be safely dumped (e.g., by ruamel.yaml or json).

    Args:
        obj: Any Python object (dict, list, tuple, numpy types, etc.)

    Returns:
        A copy of `obj` with only Python-native types.
    """
    if isinstance(obj, dict):
        return {make_dumpable(k): make_dumpable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        # preserve list/tuple type
        converted = [make_dumpable(x) for x in obj]
        return type(obj)(converted)
    elif isinstance(obj, (np.generic, jnp.generic)):  # all (jax)numpy scalar types
        return obj.item()
    elif hasattr(obj, "tolist") and not isinstance(obj, (str, bytes, bytearray)):
        # Convert numpy arrays
        try:
            return obj.tolist()
        except Exception:
            pass
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return obj
