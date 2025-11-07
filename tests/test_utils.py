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

import os
import numpy as np
import jax.numpy as jnp
from ruamel.yaml import YAML

from GaPFlow.utils import make_dumpable


yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4, sequence=4, offset=2)


def test_make_dumpable(tmp_path):

    Aj = jnp.array([1., 3.5, 5.])
    Bn = np.random.randint(10, size=10).astype(float)

    output_dict = {'A': Aj,
                   'B': Bn,
                   'b': np.mean(Bn),
                   's': 'test_value'}

    formatted_dict = make_dumpable(output_dict)

    fname = os.path.join(tmp_path, 'file.yaml')

    with open(fname, 'w') as outfile:
        yaml.dump(formatted_dict, outfile)

    with open(fname, 'r') as infile:
        input_dict = yaml.load(infile)

    assert np.all([np.isclose(i, Aji) for i, Aji in zip(input_dict['A'], Aj)])
    assert np.all([np.isclose(i, Bni) for i, Bni in zip(input_dict['B'], Bn)])
    assert np.isclose(input_dict['b'], np.mean(Bn))
    assert input_dict['s'] == 'test_value'
