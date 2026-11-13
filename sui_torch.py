import numpy as np
import primitives as p


class Conv2DLayer:
    # Pocitejme s tim, ze stride = 1 a padding = 0.
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernels = None # Toto budou vahy konvolucni vrstvy
        raise NotImplementedError("Conv2DLayer.__init__ is not implemented")

    def forward(self, x):
        # x: (in_channels, H, W) unbatched, nebo (batch_size, in_channels, H, W)
        raise NotImplementedError("Conv2DLayer.forward is not implemented")

    def parameters(self):
        raise NotImplementedError("Conv2DLayer.parameters is not implemented")


class MaxPool2DLayer:
    def __init__(self, pool_size, stride):
        self._pool_size: int = pool_size
        self._stride: int = stride

    def forward(self, x):
        # x: (channels, H, W) unbatched, nebo (batch_size, channels, H, W)
        if x.value.ndim == 3:
            x_val = x.value[np.newaxis, ...]
        else:
            x_val = x.value

        batch, channels, height, width = x_val.shape

        windows = np.lib.stride_tricks.sliding_window_view(x_val, (self._pool_size, self._pool_size), axis=(2, 3))[: ,:, ::self._stride, ::self._stride, :, :]
        _, _, height_out, width_out, _, _ = windows.shape

        windows_flat = windows.reshape(batch, channels, height_out, width_out, self._pool_size * self._pool_size)

        out_val = np.max(windows_flat, axis=-1)

        max_indices = np.argmax(windows_flat, axis=-1)

        if x.value.ndim == 3:
            out_val = out_val[0]

        def back_op(grad):
            if x.value.ndim == 3:
                grad_4d = grad[np.newaxis, ...]
            else:
                grad_4d = grad

            grad_x = np.zeros_like(x_val)

            dh, dw = np.unravel_index(max_indices, (self._pool_size, self._pool_size))

            batch_idx, channels_idx, height_idx, width_idx = np.ogrid[:batch, :channels, :height_out, :width_out]

            height_in = height_idx * self._stride + dh
            width_in = width_idx * self._stride + dw

            np.add.at(grad_x, (batch_idx, channels_idx, height_in, width_in), grad_4d)

            if x.value.ndim == 3:
                grad_x = grad_x[0]

            x.backward(grad_x)

        return p.Tensor(out_val, back_op=back_op)

    def parameters(self):
        return []
