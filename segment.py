import matplotlib.pyplot as plt
import tensorflow as tf

def segment_image(im, image_path, model):

    pred_mask = model.predict(im)

    im = create_mask(pred_mask)
    im = tf.keras.preprocessing.image.array_to_img(im)
    fig, ax = plt.subplots()
    ax.imshow(im)
    fig.savefig(image_path)    

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]