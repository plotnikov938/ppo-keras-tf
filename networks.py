from tensorflow.python.keras.layers import Dense, Dropout


act = 'tanh'


def actor(input_state, drop_rate=0.0, trainable=True):
    out = Dense(64, act, trainable=trainable)(input_state)
    out = Dropout(rate=drop_rate)(out, training=True)
    features = Dense(64, act, trainable=trainable)(out)

    return features


def critic(input_state, drop_rate=0.0, trainable=True):
    out = Dense(64, act, trainable=trainable)(input_state)
    out = Dropout(rate=drop_rate)(out, training=True)
    features = Dense(64, act, trainable=trainable)(out)

    return features
