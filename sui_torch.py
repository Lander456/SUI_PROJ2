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
        #normalise input x to always have 4 dimensions (batch being a dummy dimension if it wasn't specified)
        if x.value.ndim == 3:
            x_val = x.value[np.newaxis, ...]
        else:
            x_val = x.value

        #extract individual input image arrays from passed x
        batch, channels, height, width = x_val.shape

        #create sliding windows over the image pixels across all channels and batches
        windows = np.lib.stride_tricks.sliding_window_view(
            x_val,
            (self._pool_size, self._pool_size),
            axis=(2, 3)
        )

        #apply stride to the windows, selecting only every stride-th window
        windows = windows[: ,:, ::self._stride, ::self._stride, :, :]
        _, _, height_out, width_out, _, _ = windows.shape

        #flatten windows to 1D to be able to apply numpy operations on them and avoid loops
        windows_flat = windows.reshape(batch, channels, height_out, width_out, self._pool_size * self._pool_size)

        #get pixels with max values for each window (most prominent ones, I guess I could call them that?)
        out_val = np.max(windows_flat, axis=-1)

        #get the positions of the max valued pixels inside each of the windows (most prominent, not sure what else to call them)
        max_indices = np.argmax(windows_flat, axis=-1)

        #if no batches, squeeze out the dummy dimension inserted for normalisation
        if x.value.ndim == 3:
            out_val = out_val.squeeze(axis=0)

        def back_op(grad):

            # check input dimension initialise grad to correspond with it (since backward propagation has to send a gradient to each input to tell it how the result depends on it)
            if x.value.ndim == 3:
                normalised_grad = grad[np.newaxis, ...]
            else:
                normalised_grad = grad

            #initialise gradient to zero (if the pixel was filtered out, it had no impact on the output)
            grad_x = np.zeros_like(x_val)

            #unravel to get the height and width offsets of the pixel that had the maximum value (passed on)
            height_offset, width_offset = np.unravel_index(
                max_indices,
                (self._pool_size, self._pool_size),
            )

            #get a vector of indices for each window
            batch_idx, channels_idx, height_idx, width_idx = np.ogrid[:batch, :channels, :height_out, :width_out]

            #multiply by stride and add the offsets to get the real indices of the pixels in the input
            height_in = height_idx * self._stride + height_offset
            width_in = width_idx * self._stride + width_offset

            #at each of these indices add the propagating gradient
            np.add.at(grad_x, (batch_idx, channels_idx, height_in, width_in), normalised_grad)

            #if input didn't have a batch dimension, squeeze it out
            if x.value.ndim == 3:
                grad_x = grad_x.squeeze(axis=0)

            #backward propagate
            x.backward(grad_x)

        #return the output tensor with the value and back_op
        return p.Tensor(out_val, back_op=back_op)

    def parameters(self):
        # Since MaxPool2D has no learnable parameters, return a blank list
        return []
