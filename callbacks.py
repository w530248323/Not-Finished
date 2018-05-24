import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt


class MonitorLRDecay(object):
    """
    Decay learning rate with some patience
    """

    def __init__(self, decay_factor, patience):
        self.best_loss = 999999
        self.decay_factor = decay_factor
        self.patience = patience
        self.count = 0

    def __call__(self, current_loss, current_lr):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.count = 0
        elif self.count > self.patience:
            current_lr = current_lr * self.decay_factor
            print(" > New learning rate -- {0:}".format(current_lr))
            self.count = 0
        else:
            self.count += 1
        return current_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_images_for_debug(dir_img, imgs):
    """
    2x3x12x224x224 --> [BS, C, seq_len, H, W]
    """
    print("Saving images to {}".format(dir_img))

    imgs = imgs.permute(0, 2, 3, 4, 1)  # [BS, seq_len, H, W, C]
    imgs = imgs.mul(255).numpy()
    if not os.path.exists(dir_img):
        os.makedirs(dir_img)
    print(imgs.shape)
    for batch_id, batch in enumerate(imgs):
        batch_dir = os.path.join(dir_img, "batch{}".format(batch_id + 1))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        for j, img in enumerate(batch):
            plt.imsave(os.path.join(batch_dir, "frame{%04d}.png" % (j + 1)),
                       img.astype("uint8"))
