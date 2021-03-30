import sys
import argparse
from utils import *
from tensorflow.keras.applications import ResNet50


GLOBAL_PARAMS = dict(
    target_size=64,
    batch_size=64,
    directory="dataset/simpsons_train/",
    seed=1337,
    image_data_generator=ImageDataGenerator(validation_split=0.2),
    class_mode="categorical",
    interpolation="bicubic",
    train_learning_rate=1e-3,
    fine_tune_learning_rate=1e-5,
    dropout=0.2,
    train_epochs=20,
    fine_tune_epochs=10,
)

image_augmentation = models.Sequential(
    [
        layers.experimental.preprocessing.Rescaling(1.0 / 255),
        layers.experimental.preprocessing.RandomContrast(0.3),
        layers.experimental.preprocessing.RandomWidth(0.3),
        layers.experimental.preprocessing.RandomHeight(0.3),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.3),
    ],
    name="image_augmentation",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_learning_rate", type=float)
    parser.add_argument("--fine_tune_learning_rate", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--train_epochs", type=int)
    parser.add_argument("--fine_tune_epochs", type=int)
    parser.add_argument("--target_size", type=int)
    parser.add_argument("--batch_size", type=int)
    return parser.parse_args()


def resnet(target_size=64, dropout=0.2, **kwargs):
    inputs = layers.Input(shape=(target_size, target_size, 3))
    x = image_augmentation(inputs)
    x = keras.applications.resnet50.preprocess_input(x)
    model = ResNet50(include_top=False, input_tensor=x, weights="imagenet")

    # freeze pretrained weights
    model.trainable = False

    # rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(dropout, name="top_dropout")(x)
    outputs = layers.Dense(20, activation="softmax", name="pred")(x)

    # compile
    model = keras.Model(inputs, outputs, name="ResNet")
    return model


def train(train_ds, val_ds, callbacks, train_epochs, train_learning_rate, **kwargs):
    model = resnet(**kwargs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=train_learning_rate),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=train_epochs, callbacks=callbacks
    )

    return model, history


def fine_tune(
    model: keras.Model,
    train_ds,
    val_ds,
    fine_tune_learning_rate,
    fine_tune_epochs,
    callbacks,
):
    model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks,
    )

    return model, history


def main():
    init_gpu()
    wandb.init(project="deakin-simpsons", entity="cask", mode="online")
    if len(sys.argv) > 1:
        args = vars(parse_args())
        args = {k: v for k, v in args.items() if v is not None}
        GLOBAL_PARAMS.update(args)

    wandb.config.update(GLOBAL_PARAMS)

    data_params = [
        "target_size",
        "batch_size",
        "directory",
        "seed",
        "image_data_generator",
        "class_mode",
        "interpolation",
    ]
    data_params = {param: GLOBAL_PARAMS[param] for param in data_params}

    train_ds, val_ds, test_ds, class_names, num_classes = create_datasets(data_params)

    hp = [
        "target_size",
        "train_learning_rate",
        "fine_tune_learning_rate",
        "dropout",
        "train_epochs",
        "fine_tune_epochs",
    ]
    hp = {param: GLOBAL_PARAMS[param] for param in hp}

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3),
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

    model, history = train(
        train_ds,
        val_ds,
        callbacks=callbacks,
        **hp,
    )

    test_ds.reset()
    test_ds.shuffle = False
    test_ds.next()
    y_prob = model.predict(test_ds)
    y_pred = y_prob.argmax(axis=-1)
    y_true = test_ds.labels

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