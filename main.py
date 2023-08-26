import model as md
import tensorflow as tf
from train import train_generator_fn, val_generator_fn
import datetime

B_size=16

def compile_model():
    tf.config.run_functions_eagerly(True)
    model = md.SegNet(out_channels=32)
    #sample_input_shape = (300, 300, 3)
    #model.build(input_shape=(None,) + sample_input_shape)


    # Print the model summary
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('Camvid-Weights1.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-6)
    Earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, mode="auto", restore_best_weights=True)
    CSVLogger = tf.keras.callbacks.CSVLogger(filename='training.csv', append=True)

    epochs = 50
    #steps = int(369 / B_size)
    #valid_steps = int(100 / B_size)

    Hisotry = model.fit(train_generator_fn(),
                        validation_data=val_generator_fn(),
                        epochs=epochs,
                        verbose=2,
                        validation_freq=5,
                        workers=-1,
                        callbacks=[CSVLogger, reduce_lr, Earlystop, model_checkpoint])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compile_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
