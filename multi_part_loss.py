import numpy

from chainer import function
from chainer.utils import type_check
from chainer  import cuda

class MultiPartLoss(function.Function):

    def __init__(self, coord, no_obj):
        super(MultiPartLoss, self).__init__()
        self.coord = coord
        self.no_obj = no_obj

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        #vec = [(class or back,x,y,w,h)*B,classes]
        x0, x1 = inputs
        self.diff = x0 - x1
        self.diff[:, 3:5] = numpy.sqrt(x0[:,3:5]) -numpy.sqrt(x1[:,3:5])
        self.diff[:, 5:7] = numpy.sqrt(x0[:,5:7]) -numpy.sqrt(x1[:,5:7])
        self.confidence = x1[:,0]
        diff1 = self.coord * (self.diff[:, 1:3]**2 + self.diff[:,6:8]**2)
        diff2 = self.coord * (self.diff[:, 3:5]**2 + self.diff[:,8:10]**2)
        diff3 = (self.diff[:,0]**2 + self.diff[:,5]**2) * self.confidence
        diff4 = self.no_obj * (self.diff[:,0]**2 + self.diff[:,5]**2) * (self.confidence == 0)
        diff5 = self.diff[:,10:]**2
        loss = sum([diff1.sum(),diff2.sum(),diff3.sum(),diff4.sum(),diff5.sum()])
        return numpy.array(loss, dtype=self.diff.dtype),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x0, x1 = inputs
        self.diff = x0 - x1
        self.diff[:, 3:5] = cupy.sqrt(x0[:,3:5]) - cupy.sqrt(x1[:,3:5])
        self.diff[:, 5:7] = cupy.sqrt(x0[:,5:7]) - cupy.sqrt(x1[:,5:7])
        self.confidence = x1[:,0]
        loss = (self.coord * (self.diff[:, 1:3]**2 + self.diff[:,6:8]**2)).sum()
        loss += (self.coord * (self.diff[:, 3:5]**2 + self.diff[:,8:10]**2)).sum()
        loss += ((self.diff[:,0]**2 + self.diff[:,5]**2) * self.confidence).sum()
        loss += (self.no_obj * (self.diff[:,0]**2 + self.diff[:,5]**2) * (self.confidence == 0)).sum()
        loss += (self.diff[:,10:]**2).sum()
        return (loss,)

    def backward_cpu(self, inputs, gy):
        x0, x1 = inputs
        coeff = self.diff.copy()
        coeff[:, 0] = self.diff[:, 0]*2 * x1[:, 0]
        coeff[:, 1] = 2 * self.coord * x1[:, 0] * self.diff[:, 1]
        coeff[:, 2] = 2 * self.coord * x1[:, 0] * self.diff[:, 2]
        coeff[:, 3] = 2 * self.coord * x1[:, 0] * self.diff[:, 3] / numpy.sqrt(x1[:, 3])**3
        coeff[:, 4] = 2 * self.coord * x1[:, 0] * self.diff[:, 4] / numpy.sqrt(x1[:, 4])**3
        coeff[:, 5] = self.diff[:, 5]*2 * x1[:, 5]
        coeff[:, 6] = 2 * self.coord * x1[:, 0] * self.diff[:, 6]
        coeff[:, 7] = 2 * self.coord * x1[:, 0] * self.diff[:, 7]
        coeff[:, 8] = 2 * self.coord * x1[:, 5] * self.diff[:, 8] / numpy.sqrt(x1[:, 8])**3
        coeff[:, 9] = 2 * self.coord * x1[:, 5] * self.diff[:, 9] / numpy.sqrt(x1[:, 9])**3
        coeff[:, 0] += self.no_obj * self.diff[:, 0] * 2 * (x1[:, 0]==0)
        coeff[:, 10:] = self.diff[:,10:] * 2
        gx0 = coeff * self.diff
        return gx0, -gx0

    def backward_gpu(self, inputs, gy):
        cupy = cuda.cupy
        x0, x1 = inputs
        coeff = self.diff.copy()
        coeff[:, 0] = self.diff[:, 0]*2 * x1[:, 0]
        coeff[:, 1] = 2 * self.coord * x1[:, 0] * self.diff[:, 1]
        coeff[:, 2] = 2 * self.coord * x1[:, 0] * self.diff[:, 2]
        coeff[:, 3] = 2 * self.coord * x1[:, 0] * self.diff[:, 3] / cupy.sqrt(x1[:, 3])**3
        coeff[:, 4] = 2 * self.coord * x1[:, 0] * self.diff[:, 4] / cupy.sqrt(x1[:, 4])**3
        coeff[:, 5] = self.diff[:, 5]*2 * x1[:, 5]
        coeff[:, 6] = 2 * self.coord * x1[:, 0] * self.diff[:, 6]
        coeff[:, 7] = 2 * self.coord * x1[:, 0] * self.diff[:, 7]
        coeff[:, 8] = 2 * self.coord * x1[:, 5] * self.diff[:, 8] / cupy.sqrt(x1[:, 8])**3
        coeff[:, 9] = 2 * self.coord * x1[:, 5] * self.diff[:, 9] / cupy.sqrt(x1[:, 9])**3
        coeff[:, 0] += self.no_obj * self.diff[:, 0] * 2 * (x1[:, 0]==0)
        coeff[:, 10:] = self.diff[:,10:] * 2
        gx0 = coeff * self.diff
        return gx0, -gx0

def multi_part_loss(x0, x1,coord=5, no_obj=0.5):
    return MultiPartLoss(coord, no_obj)(x0, x1)
