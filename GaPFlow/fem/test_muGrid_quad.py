from muGrid import GlobalFieldCollection
import numpy as np

nb = 10
fc = GlobalFieldCollection((nb, ))
field1 = fc.real_field("sol")
quad1 = fc.real_field('quad', 2)

print(field1.shape)
field1.p = np.linspace(0, 1, nb)
print(field1.p)
