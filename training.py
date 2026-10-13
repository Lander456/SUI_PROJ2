import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import primitives as p
import sui_torch as st
import data_generation as dg


class LinearLayer:
    def __init__(self, out_features, in_features):
        # Je potreba provest normalizaci vah, aby nedoslo k explozi gradientu.
        # Tenhle stav sice stale muze nastat, ale nemusime s nim rovnou zacinat... :)
        scale = 1.0 / np.sqrt(in_features)
        self.weights = p.Tensor(np.random.randn(out_features, in_features) * scale)
        self.bias = p.Tensor(np.zeros(out_features))

    def forward(self, x):
        # x je (in_features,) pro jeden obrazek, nebo (batch_size, in_features)
        # pro batch.
        out = p.dot_product(self.weights, x)

        unbatched = x.value.ndim == 1
        if unbatched:
            bias = self.bias
        else:
            batch_size = out.value.shape[0]
            # Bias je jeden vektor (out_features,) a ma byt pouzity na vsechny
            # obrazky v batchi. (batch_size, out_features)
            # Musime ho zde tedy zduplikovat - takoveto zmene tvaru pro kompatibilni
            # tensory se v podobnych knihovnach rika `broadcasting`. 
            bias = p.stack_tensors([self.bias] * batch_size, axis=0)

        return p.add(out, bias)

    def parameters(self):
        return [self.weights, self.bias]


class CIFAR10Classifier:
    def __init__(self, num_classes=10, in_channels=3):
        kernel_size = 3
        pool_size = 2
        pool_stride = 2
        channel_sizes = [in_channels, 32, 64, 64]
        hidden_size = 64

        self.blocks = []
        for c_in, c_out in zip(channel_sizes[:-1], channel_sizes[1:]):
            conv = st.Conv2DLayer(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size)
            pool = st.MaxPool2DLayer(pool_size=pool_size, stride=pool_stride)
            self.blocks.append((conv, pool))

        # Aktivace posledni vrstvy ma tvar (B, 64, 2, 2),
        # tzn. 256 float hodnot.
        flattened_size = 256
        self.fc1 = LinearLayer(hidden_size, flattened_size)
        self.fc2 = LinearLayer(num_classes, hidden_size)

    def forward(self, x):
        for conv, pool in self.blocks:
            x = conv.forward(x)
            x = p.relu(x)
            x = pool.forward(x)
            x = p.relu(x)

        x = p.flatten(x)
        x = p.relu(self.fc1.forward(x))
        return self.fc2.forward(x)

    def parameters(self):
        params = []
        for conv, _ in self.blocks:
            params += conv.parameters()
        return params + self.fc1.parameters() + self.fc2.parameters()


class StochasticGradientDescent:
    def __init__(self, tensors, lr):
        self.tensors = tensors
        self.lr = lr

    def step(self):
        for tensor in self.tensors:
            tensor.value -= self.lr * tensor.grad

    def zero_grad(self):
        for tensor in self.tensors:
            tensor.grad.fill(0)


def train_model(model, train_images, train_labels, optimizer, num_epochs, val_images, val_labels):
    losses = []
    val_accuracies = []
    num_samples = len(train_images)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0

        pbar = tqdm(zip(train_images, train_labels), total=num_samples, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (image, label) in enumerate(pbar, start=1):
            optimizer.zero_grad()

            image_tensor = p.Tensor(image)
            logits = model.forward(image_tensor)

            if np.any(np.isnan(logits.value)) or np.any(np.isinf(logits.value)):
                tqdm.write("NaN or Inf in model output, you are probably doing something wrong.")
                continue

            loss = p.cross_entropy_loss(logits, label)
            loss_scalar = float(loss.value.item() if hasattr(loss.value, "item") else loss.value)

            if np.isnan(loss_scalar) or np.isinf(loss_scalar):
                tqdm.write("NaN or Inf in loss, you are probably doing something wrong.")
                continue

            epoch_loss += loss_scalar
            if np.argmax(logits.value) == label:
                correct += 1

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{epoch_loss / i:.4f}", acc=f"{correct / i:.2%}")

        avg_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
        losses.append(avg_loss)

        _, val_acc = evaluate_model(model, val_images, val_labels, show_progress=False)
        val_accuracies.append(val_acc)

    return losses, val_accuracies


def evaluate_model(model, test_images, test_labels, show_progress=True):
    total_loss = 0.0
    correct = 0
    num_samples = len(test_images)

    pbar = tqdm(zip(test_images, test_labels), total=num_samples, desc="Evaluating", disable=not show_progress)
    for image, label in pbar:
        image_tensor = p.Tensor(image)
        logits = model.forward(image_tensor)
        loss = p.cross_entropy_loss(logits, label)
        total_loss += float(loss.value.item())
        if np.argmax(logits.value) == label:
            correct += 1

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train a small 3-layer CNN for CIFAR10 classification')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--num-train-samples', type=int, default=1000)
    parser.add_argument('--num-test-samples', type=int, default=300)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--do-plots', action='store_true')
    parser.add_argument('--save-plot', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    print("Loading data...")
    train_images, train_labels = dg.load_cifar10(root=args.data_root, train=True, num_samples=args.num_train_samples, seed=args.seed)
    test_images, test_labels = dg.load_cifar10(root=args.data_root, train=False, num_samples=args.num_test_samples, seed=args.seed)

    # Validacni sada jde z trenovaci sady -- testovaci sada zustava
    # nedotcena az do finalniho vyhodnoceni.
    train_images, train_labels, val_images, val_labels = dg.split_train_val(
        train_images, train_labels, val_fraction=args.val_fraction, seed=args.seed
    )

    print(f"Train samples: {len(train_images)}")
    print(f"Val samples: {len(val_images)}")
    print(f"Test samples: {len(test_images)}")

    model = CIFAR10Classifier(num_classes=len(dg.EASY_CLASSES))
    optimizer = StochasticGradientDescent(model.parameters(), lr=args.lr)

    print("\nTraining...")
    losses, val_accuracies = train_model(model, train_images, train_labels, optimizer, args.num_epochs, val_images, val_labels)

    print("\nEvaluating...")
    test_loss, test_acc = evaluate_model(model, test_images, test_labels)

    print(f"\nResults:")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")

    if args.do_plots or args.save_plot:
        num_examples = min(3, len(test_images))
        class_names = [dg.CIFAR10_CLASSES[c] for c in dg.EASY_CLASSES]

        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(2, max(num_examples, 1), height_ratios=[4, 1])

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(losses, color='tab:blue', label='Train Loss')
        ax_main.set_xlabel('Epoch')
        ax_main.set_ylabel('Loss', color='tab:blue')
        ax_main.tick_params(axis='y', labelcolor='tab:blue')
        ax_main.set_title('Training Loss & Validation Accuracy')
        ax_main.grid(True)

        ax_acc = ax_main.twinx()
        ax_acc.plot(val_accuracies, color='tab:orange', label='Val Accuracy')
        ax_acc.set_ylabel('Val Accuracy', color='tab:orange')
        ax_acc.tick_params(axis='y', labelcolor='tab:orange')
        ax_acc.set_ylim(0, 1)

        lines_main, labels_main = ax_main.get_legend_handles_labels()
        lines_acc, labels_acc = ax_acc.get_legend_handles_labels()
        ax_main.legend(lines_main + lines_acc, labels_main + labels_acc, loc='lower left')

        for col in range(num_examples):
            i = np.random.randint(0, len(test_images))
            image, label = test_images[i], test_labels[i]
            logits = model.forward(p.Tensor(image))
            predicted = int(np.argmax(logits.value))

            ax_img = fig.add_subplot(gs[1, col])
            ax_img.imshow(np.transpose(image, (1, 2, 0)))
            ax_img.set_title(f'GT: {class_names[label]}; Pred: {class_names[predicted]}', fontsize=7)
            ax_img.axis('off')

        plt.tight_layout()

        if args.save_plot:
            plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {args.save_plot}")

        if args.do_plots:
            plt.show()


if __name__ == '__main__':
    main()
