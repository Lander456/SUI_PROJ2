import numpy as np


class Tensor:
    def __init__(self, value, back_op=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.back_op = back_op

    def __str__(self):
        str_val = str(self.value)
        str_val = '\t' + '\n\t'.join(str_val.split('\n'))
        str_bwd = "lambda" if self.back_op and callable(self.back_op) else "None"
        return 'Tensor(\n' + str_val + '\n\tbwd: ' + str_bwd + '\n)'

    @property
    def shape(self):
        return self.value.shape

    def backward(self, deltas=None):
        # Neni prvnim uzlem grafu
        if deltas is not None:
            if deltas.shape != self.value.shape:
                raise ValueError(f'Expected gradient with shape {self.value.shape}, got {deltas.shape}')
            self.grad += deltas
            
            if self.back_op:
                # Neni listovy uzel - muzeme propagovat gradienty dal grafem.
                self.back_op(self.grad)
        # Je prvnim uzlem grafu - zaciname zpetny pruchod volanim
        # loss.backward()
        else:
            # Je ocekavano, ze loss (prvni uzel) je skalar.
            if self.shape != tuple() and np.prod(self.shape) != 1:
                raise ValueError(f'Can only backpropagate a scalar, got shape {self.shape}')

            # Pokud prvni uzel grafu nema definovany `back_op`, pak
            # neni co kam propagovat.
            if self.back_op is None:
                raise ValueError(f'Cannot start backpropagation from a leaf!')

            self.grad = np.ones_like(self.value)
            if self.back_op:
                self.back_op(self.grad)


def reduce_sum(tensor):
    # Operator redukujici tensor na jeden skalar - soucet vsech jeho hodnot
    sum_tensor = Tensor(np.array([[tensor.value.sum()]]), back_op=None)
    back_op = lambda grad: tensor.backward(np.full(tensor.value.shape, grad.item()))
    sum_tensor.back_op = back_op
    return sum_tensor


def add(a, b):
    # POZOR: Toto je zjednodusena verze "add", ktera pocita
    # s faktem, ze `a` a `b` maji stejny tvar. Pred pouzitim
    # tohoto operatoru je potreba zajistit spravny tvar.
    c = a.value + b.value
    back_op = lambda grad: (a.backward(grad), b.backward(grad))
    return Tensor(c, back_op=back_op)


def multiply(a, b):
    c = a.value * b.value
    back_op = lambda grad: (
        a.backward(grad * b.value),
        b.backward(grad * a.value)
    )
    return Tensor(c, back_op=back_op)


def relu(tensor):
    out = np.maximum(tensor.value, 0)
    back_op = lambda grad: tensor.backward(grad * np.where(tensor.value > 0, 1, 0))
    return Tensor(out, back_op=back_op)


def exp(tensor):
    out = np.exp(tensor.value)
    back_op = lambda grad: tensor.backward(grad * out)
    return Tensor(out, back_op=back_op)


def log(tensor):
    # Epsilon - mala konstanta pro numerickou stabilitu
    # (chceme se vyhnout log(0))
    out = np.log(tensor.value + 1e-8)
    back_op = lambda grad: tensor.backward(grad / (tensor.value + 1e-8))
    return Tensor(out, back_op=back_op)


def softmax(tensor):
    max_val = np.max(tensor.value, axis=-1, keepdims=True)
    exp_input = np.exp(tensor.value - max_val)
    sum_exp = np.sum(exp_input, axis=-1, keepdims=True)
    out = exp_input / sum_exp
    
    def softmax_back_op(grad):
        grad_sum = np.sum(grad * out, axis=-1, keepdims=True)
        grad_input = out * (grad - grad_sum)
        tensor.backward(grad_input)
    
    return Tensor(out, back_op=softmax_back_op)


def cross_entropy_loss(logits, lbl):
    is_unbatched = logits.value.ndim == 1
    batch_size = 1 if is_unbatched else logits.value.shape[0]

    probs = softmax(logits)
    log_probs = log(probs)

    # Vytvor one-hot encoding pro kazdou tridu (z pohledu derivace konstanta)
    one_hot = np.zeros(logits.value.shape)
    if is_unbatched:
        one_hot[lbl] = 1.0
    else:
        for i, label in enumerate(lbl):
            one_hot[i, label] = 1.0
    one_hot_tensor = Tensor(one_hot)

    selected = multiply(one_hot_tensor, log_probs)

    total_sum = reduce_sum(selected)
    # V pripade, ze mame vice vzorku (batch), potom
    # chceme prumernou loss skrze vsechny vzorky.
    # -1.0 je zde kvuli spravnemu znamenku -
    # vzorec pro Cross-Entropy zacina s minus.
    neg_one = Tensor(np.array([[-1.0 / batch_size]]))
    loss = multiply(total_sum, neg_one)
    
    return loss


def dot_product(a, b):
    unbatched = b.value.ndim == 1
    c = np.dot(b.value, a.value.T)

    def back_op(grad):
        # Jedna se o zjednoduseny vzorec -
        # derivace by byl vicedimenzionalni Jakobian
        if unbatched:
            a_grad = np.outer(grad, b.value)
            b_grad = np.dot(a.value.T, grad)
        else:
            a_grad = np.dot(grad.T, b.value)
            b_grad = np.dot(grad, a.value)

        a.backward(a_grad)
        b.backward(b_grad)

    return Tensor(c, back_op=back_op)


def reshape(tensor, new_shape):
    reshaped = tensor.value.reshape(new_shape)
    
    def reshape_back_op(grad):
        original_shape = tensor.value.shape
        grad_reshaped = grad.reshape(original_shape)
        tensor.backward(grad_reshaped)
    
    return Tensor(reshaped, back_op=reshape_back_op)


def flatten(tensor, start_dim=-3):
    shape = tensor.value.shape
    dim = start_dim if start_dim >= 0 else len(shape) + start_dim
    new_shape = shape[:dim] + (int(np.prod(shape[dim:])),)
    # Nepotrebujeme definovat back_op protoze k manipulaci
    # s tensorem pouzivame operator s jiz definovanym back_op.
    return reshape(tensor, new_shape)


def stack_tensors(tensors, axis=1):
    tensor_values = [t.value for t in tensors]
    stacked = np.stack(tensor_values, axis=axis)
    
    def stack_back_op(grad):
        splits = np.split(grad, len(tensors), axis=axis)
        for tensor, split_grad in zip(tensors, splits):
            split_grad = np.squeeze(split_grad, axis=axis)
            tensor.backward(split_grad)
    
    return Tensor(stacked, back_op=stack_back_op)

