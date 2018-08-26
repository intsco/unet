from unet.model import *
from unet.data import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
myGen = train_generator(batch_size=2,
                        train_path='data/membrane/train',
                        image_folder='image', mask_folder='label',
                        aug_dict=data_gen_args,
                        save_to_dir=None)

model = unet(input_size=(256, 256, 1))
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGen, steps_per_epoch=300, epochs=5, callbacks=[model_checkpoint])

test_gen = test_generator("data/membrane/test")
# results = model.predict_generator((img for img in images), steps=30, verbose=1)
results = [model.predict(batch, batch_size=1, verbose=1)[0,:,:,0]
           for batch in test_gen]
save_result("data/membrane/test_pred", results)
