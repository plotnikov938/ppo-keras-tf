from tensorflow.python.keras.layers import Dense, Dropout


act, drop_rate, dc = 'tanh', 0.5, False


def actor(input_state, trainable=True):
    out = Dense(64, act, trainable=trainable)(input_state)
    out = Dropout(rate=drop_rate)(out) if dc else out
    features = Dense(64, act, trainable=trainable)(out)

    return features


def critic(input_state, trainable=True):
    out = Dense(64, act, trainable=trainable)(input_state)
    out = Dropout(rate=drop_rate)(out) if dc else out
    features = Dense(64, act, trainable=trainable)(out)

    return features
