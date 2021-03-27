import sys
from utils import *

DATA_PARAMS = dict(
    target_size=(128, 128),
    batch_size=64,
    directory="dataset/simpsons_train/",
    seed=1337,
    image_data_generator=ImageDataGenerator(validation_split=0.2),
    class_mode="categorical",
    interpolation="bicubic",
)

TRAIN_PARAMS = dict(learning_rate=1e-4, dropout=0.3, epochs=1)


def make_xception(input_shape=(128, 128), dropout=0.3):
    inputs = keras.Input(shape=input_shape + (3,))

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = layers.experimental.preprocessing.RandomWidth(0.3)(x)
    x = layers.experimental.preprocessing.RandomHeight(0.3)(x)
    x = layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(x)
    x = layers.experimental.preprocessing.RandomRotation(0.3)(x)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(units=20, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def train(
    train_ds,
    val_ds,
    learning_rate=1e-4,
    epochs=1,
    input_shape=(299, 299),
    dropout=0.3,
    optimizer=tf.keras.optimizers.Adam,
    callbacks=None,
):
    model = make_xception(input_shape=input_shape, dropout=dropout)

    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    history = model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds
    )

    return model, history


def main():
    init_gpu()
    wandb.init(project="deakin-simpsons", entity="cask", mode="offline")
    if len(sys.argv) > 1:
        args = vars(parse_args())
        args = {k: v for k, v in args.items() if v is not None}
        TRAIN_PARAMS.update(args)

    train_ds, val_ds, class_names, num_classes = create_datasets(DATA_PARAMS)
    config = {
        "input_shape": DATA_PARAMS["target_size"],
        "learning_rate": TRAIN_PARAMS["learning_rate"],
        "dropout": TRAIN_PARAMS["dropout"],
        "epochs": TRAIN_PARAMS["epochs"],
    }
    wandb.config.update(config)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
        WandbCallback(
            monitor="val_accuracy",
            mode="max",
            training_data=train_ds,
            validation_data=val_ds,
            data_type="image",
            labels=class_names,
            log_gradients=True,
        ),
    ]

    model, history = train(train_ds, val_ds, callbacks=callbacks, **config)

    val_ds.reset()
    val_ds.shuffle = False
    val_ds.next()
    y_prob = model.predict(val_ds)
    y_pred = y_prob.argmax(axis=-1)
    y_true = val_ds.labels

    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, class_names)

    wandb.log(
        {
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        }
    )


if __name__ == "__main__":
    main()