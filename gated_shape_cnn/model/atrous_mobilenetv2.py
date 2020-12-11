import tensorflow as tf


def build_mobilenetv2():
    """
    Create tf.keras.applications.MobileNetV2
    which uses the pretrained image net weights
    """
    model = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=[None, None, 3],)

    atrous_mobilenetv2 = tf.keras.models.model_from_json(model.to_json())
    atrous_mobilenetv2.set_weights(model.get_weights())
        
    for layer in atrous_mobilenetv2.layers:
        layer.trainable = False
    
    return atrous_mobilenetv2




class AtrousMobilenetv2(tf.keras.models.Model):
    def __init__(self, **kwargs):
        mobilenet = build_mobilenetv2()
        super(AtrousMobilenetv2, self).__init__(inputs=mobilenet.inputs, outputs=mobilenet.outputs, **kwargs)


if __name__ == '__main__':
    build_mobilenetv2()

