import custom_losses as cls
from tensorflow.keras.models import load_model


def load_custom_model(model_path, loss_type, size=None):
    if loss_type == 'weighted_ce':
        model = load_model(model_path, custom_objects={'weighted_loss': cls.weighted_crossentropy(size)})
    elif loss_type == 'dice':
        model = load_model(model_path, custom_objects={'compute_dice_loss': cls.dice_loss()})
    else:
        print('unrecognized loss, assuming regular binary cross entropy')
        model = load_model(model_path)
    return model

