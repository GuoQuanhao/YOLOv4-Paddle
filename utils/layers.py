import paddle.nn.functional as F

from utils.general import *

import paddle
from paddle import nn


class Mish(nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (paddle.tanh(F.softplus(x)))
        return x


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Layer):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.reshape([x.shape[0], -1])


class Concat(nn.Layer):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return paddle.concat(x, self.d)


class FeatureConcat(nn.Layer):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return paddle.concat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class FeatureConcat2(nn.Layer):
    def __init__(self, layers):
        super(FeatureConcat2, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return paddle.concat([outputs[self.layers[0]], outputs[self.layers[1]].detach()], 1)


class FeatureConcat3(nn.Layer):
    def __init__(self, layers):
        super(FeatureConcat3, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return paddle.concat([outputs[self.layers[0]], outputs[self.layers[1]].detach(), outputs[self.layers[2]].detach()], 1)


class FeatureConcat_l(nn.Layer):
    def __init__(self, layers):
        super(FeatureConcat_l, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return paddle.concat([outputs[i][:,:outputs[i].shape[1]//2,:,:] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]][:,:outputs[self.layers[0]].shape[1]//2,:,:]


class WeightedFeatureFusion(nn.Layer):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        if weight:
            self.w = paddle.create_parameter(shape=[self.n], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = paddle.nn.functional.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class MixConv2d(nn.Layer):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = paddle.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.LayerList([nn.Conv2D(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return paddle.concat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(nn.Layer):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * paddle.nn.functional.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = paddle.nn.functional.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(nn.Layer):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.multiply(paddle.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = paddle.nn.functional.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Layer):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Layer):
    def forward(self, x):
        return MishImplementation.apply(x)


class Swish(nn.Layer):
    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x)


class HardSwish(nn.Layer):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


#class Mish(nn.Layer):  # https://github.com/digantamisra98/Mish
#    def forward(self, x):
#        return x * F.softplus(x).tanh()

class DeformConv2d(nn.Layer):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2D(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2D(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2D(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = paddle.nn.functional.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.shape[1] // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.transpose([0, 2, 3, 1])
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = paddle.concat([paddle.clip(q_lt[..., :N], 0, x.shape[2]-1), paddle.clip(q_lt[..., N:], 0, x.shape[3]-1)], axis=-1).astype('int64')
        q_rb = paddle.concat([paddle.clip(q_rb[..., :N], 0, x.shape[2]-1), paddle.clip(q_rb[..., N:], 0, x.shape[3]-1)], axis=-1).astype('int64')
        q_lb = paddle.concat([q_lt[..., :N], q_rb[..., N:]], axis=-1)
        q_rt = paddle.concat([q_rb[..., :N], q_lt[..., N:]], axis=-1)

        # clip p
        p = paddle.concat([paddle.clip(p[..., :N], 0, x.shape[2]-1), paddle.clip(p[..., N:], 0, x.shape[3]-1)], axis=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(axis=1) * x_q_lt + \
                   g_rb.unsqueeze(axis=1) * x_q_rb + \
                   g_lb.unsqueeze(axis=1) * x_q_lb + \
                   g_rt.unsqueeze(axis=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.permute(0, 2, 3, 1)
            m = m.unsqueeze(axis=1)
            m = paddle.concat([m for _ in range(x_offset.shape[1])], axis=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = paddle.meshgrid(
            paddle.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            paddle.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = paddle.concat([paddle.flatten(p_n_x), paddle.flatten(p_n_y)], 0)
        p_n = p_n.reshape([1, 2*N, 1, 1]).astype(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = paddle.meshgrid(
            paddle.arange(1, h*self.stride+1, self.stride),
            paddle.arange(1, w*self.stride+1, self.stride))
        p_0_x = paddle.flatten(p_0_x).reshape([1, 1, h, w]).repeat(1, N, 1, 1)
        p_0_y = paddle.flatten(p_0_y).reshape([1, 1, h, w]).repeat(1, N, 1, 1)
        p_0 = paddle.concat([p_0_x, p_0_y], 1).astype(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.shape[1]//2, offset.shape[2], offset.shape[3]

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.shape
        padded_w = x.shape[3]
        c = x.shape[1]
        # (b, c, h*w)
        x = x.reshape([b, c, -1])

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.unsqueeze(axis=1).expand(-1, c, -1, -1, -1).reshape([b, c, -1])

        x_offset = x.gather(axis=-1, index=index).reshape([b, c, h, w, N])

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.shape
        x_offset = paddle.concat([x_offset[..., s:s+ks].reshape([b, c, h, w*ks]) for s in range(0, N, ks)], axis=-1)
        x_offset = x_offset.reshape([b, c, h*ks, w*ks])

        return x_offset
