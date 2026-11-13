import numpy as np
import primitives as p
from numpy.lib.stride_tricks import sliding_window_view

# enums to make the tensor operation more readable
class WindowAxes:
    Batch   = 0
    Channel = 1
    H_Pos   = 2
    V_Pos   = 3
    Height  = 4
    Width   = 5


class FilterAxes:
    OutChannels = 0
    InChannels  = 1
    Height      = 2
    Width       = 3


class GradAxes:
    Batch      = 0
    OutChannel = 1
    H_Pos      = 2
    V_Pos      = 3


class Conv2DLayer:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channel   = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size

        # scale random weights using He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))

        # init the filters
        self.kernels = []
        for _ in range(out_channels):
            # use Z-score normalization to standardize the weights
            weights = np.random.randn(in_channels, kernel_size, kernel_size)
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)
            self.kernels.append(p.Tensor(weights * scale))

        self.bias = p.Tensor(np.zeros(out_channels))

    def forward(self, x):
        # x: (channels, H, W) unbatched, nebo (batch_size, channels, H, W)
        K, WA, FA, GA = self.kernel_size, WindowAxes, FilterAxes, GradAxes

        # check if the tensor is batched or unbatched
        batched = (x.value.ndim > 3)

        # in case of an unbatched tensor add the forth dimension of value 1
        x_val = x.value if batched else x.value.reshape(1, *x.value.shape)

        # stack filter weights and create a view of all sliding input windows
        filter  = np.stack([k.value for k in self.kernels], axis=0)
        windows = sliding_window_view(x_val, (K, K), axis=(WA.H_Pos, WA.V_Pos))

        # multiply all windows with the filter
        result = np.tensordot(windows, filter, axes=(
            (WA.Channel,    WA.Height, WA.Width),
            (FA.InChannels, FA.Height, FA.Width)
        ))
        # return OutChannels moved by tensordot back to position 1
        result = np.moveaxis(result, -1, 1)
        # add bias to result
        result += self.bias.value[np.newaxis, :, np.newaxis, np.newaxis]

        # unpack unbatched result
        if not batched:
            result = result[0]

        # function used for back-propagation
        def back_op(grad):
            # add forth dimension when not batched
            grad_4d = grad if batched else grad.reshape(1, *grad.shape)

            # compute the loss gradient to bias and pass it to the bias tensor
            grad_bias = np.sum(grad_4d, axis=(GA.Batch, GA.H_Pos, GA.V_Pos))
            self.bias.backward(grad_bias)

            # compute the loss gradient to filter weights
            grad_filter = np.tensordot(grad_4d, windows, axes=(
                (GA.Batch, GA.H_Pos, GA.V_Pos),
                (WA.Batch, WA.H_Pos, WA.V_Pos)
            ))

            # pass the gradient to each out channel of the filter
            for tensor, grad_weight in zip(self.kernels, grad_filter):
                tensor.backward(grad_weight)

            # zero-pad the output gradient
            grad_padded = np.pad(
                grad_4d,
                ((0, 0), (0, 0), (K - 1, K - 1), (K - 1, K - 1)),
                mode='constant'
            )

            # create sliding windows over the padded output gradient and rotate the filter by 180 degrees
            grad_windows   = sliding_window_view(grad_padded, (K, K), axis=(WA.H_Pos, WA.V_Pos))
            filter_flipped = filter[:, :, ::-1, ::-1]

            # calculate dotproduct of all the windows of the out gradient and the flipped filter
            grad_x = np.tensordot(grad_windows, filter_flipped, axes=(
                (WA.Channel,     WA.Height, WA.Width),
                (FA.OutChannels, FA.Height, FA.Width)
            ))
            # return OutChannels moved by tensordot back to position 1
            grad_x = np.moveaxis(grad_x, -1, 1)

            # unpack unbatched gradient
            if not batched:
                grad_x = grad_x[0]

            # pass the input gradient to the input tensor for further backpropagation
            x.backward(grad_x)

        # return the convolution output tensor with its backward function
        return p.Tensor(result, back_op=back_op)

    def parameters(self):
        return self.kernels + [self.bias]


class MaxPool2DLayer:
    def __init__(self, pool_size, stride):
        self._pool_size = pool_size
        self._stride = stride

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
