import numpy as np
import primitives as p
from numpy.lib.stride_tricks import sliding_window_view


class Conv2DLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channel   = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        self.kernels = [
            p.Tensor(np.random.randn(in_channels, kernel_size, kernel_size) * scale)
            for _ in range(out_channels)
        ]
        self.bias = p.Tensor(np.zeros(out_channels))

    def forward(self, x):
        is_unbatched = (x.value.ndim == 3)
        
        x_val = x.value[np.newaxis, ...] if is_unbatched else x.value
        K = self.kernel_size

        weights = np.stack([k.value for k in self.kernels], axis=0)
        windows = sliding_window_view(x_val, (K, K), axis=(2, 3))

        result = np.tensordot(windows, weights, axes=((1, 4, 5), (1, 2, 3)))
        result = np.moveaxis(result, -1, 1)
        result += self.bias.value[np.newaxis, :, np.newaxis, np.newaxis]

        if is_unbatched:
            result = result[0]

        def back_op(grad):
            grad_4d = grad[np.newaxis, ...] if is_unbatched else grad

            grad_bias = np.sum(grad_4d, axis=(0, 2, 3))
            self.bias.backward(grad_bias)

            grad_weights = np.tensordot(grad_4d, windows, axes=((0, 2, 3), (0, 2, 3)))

            for k_tensor, gw in zip(self.kernels, grad_weights):
                k_tensor.backward(gw)

            grad_padded = np.pad(
                grad_4d,
                ((0, 0), (0, 0), (K - 1, K - 1), (K - 1, K - 1)),
                mode='constant'
            )

            grad_windows    = sliding_window_view(grad_padded, (K, K), axis=(2, 3))
            weights_flipped = weights[:, :, ::-1, ::-1]

            grad_x = np.tensordot(grad_windows, weights_flipped, axes=((1, 4, 5), (0, 2, 3)))
            grad_x = np.moveaxis(grad_x, -1, 1)

            if is_unbatched:
                grad_x = grad_x[0]

            x.backward(grad_x)

        return p.Tensor(result, back_op=back_op)

    def parameters(self):
        return self.kernels + [self.bias]


class MaxPool2DLayer:
    def __init__(self, pool_size, stride):
        raise NotImplementedError("MaxPool2DLayer.__init__ is not implemented")

    def forward(self, x):
        # x: (channels, H, W) unbatched, nebo (batch_size, channels, H, W)
        raise NotImplementedError("MaxPool2DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("MaxPool2DLayer.parameters is not implemented")
