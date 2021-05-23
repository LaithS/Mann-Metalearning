import numpy as np
import os
from PIL import Image
from numpy.random import default_rng
import io
import tensorflow as tf
import json
rng = default_rng()

ch = {'RGB': 3, 'L': 1, '1': 1}


def get_images_labels(all_images,
                      nb_classes,
                      nb_samples_per_class,
                      image_size,
                      colorspace,
                      augment,
                      sample_stategy="random"):
    sample_classes = rng.choice(range(len(all_images)),
                                replace=True,
                                size=nb_classes)
    if sample_stategy == "random":
        labels = rng.integers(0,
                              nb_classes,
                              nb_classes * nb_samples_per_class)
    elif sample_stategy == "uniform":
        labels = np.concatenate([[i] * nb_samples_per_class
                                for i in range(nb_classes)])
        rng.shuffle(labels)
    angles = rng.integers(0, 4, nb_classes) * 90

    images = []
    if augment:
        images = [
            image_transform(all_images[sample_classes[i]]
                            [rng.integers(0,
                                          len(all_images[sample_classes[i]])
                                          )
                             ],
                            angle=angles[i]+(rng.random()-0.5)*22.5,
                            trans=rng.integers(-10,
                                               11,
                                               size=2
                                               ).tolist(),
                            size=image_size,
                            colorspace=colorspace
                            ) for i in labels
                  ]
    else:
        images = [
            image_notransform(all_images[sample_classes[i]]
                              [rng.integers(0,
                                            len(all_images[sample_classes[i]])
                                            )
                               ],
                              size=image_size,
                              colorspace=colorspace
                              ) for i in labels
                  ]

    return images, labels


def image_notransform(image,
                      size=(20, 20),
                      colorspace='L'):
    image = image.resize(size)
    np_image = np.reshape(np.array(image,
                                   dtype=np.float32),
                          newshape=(np.prod(size)*ch[colorspace]))
    max_value = np.max(np_image)
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image


def image_transform(image,
                    angle=0.,
                    trans=(0., 0.),
                    size=(20, 20),
                    colorspace='L'):
    image = image.rotate(angle, translate=trans)\
                    .resize(size)
    np_image = np.reshape(np.array(image,
                                   dtype=np.float32),
                          newshape=(np.prod(size)*ch[colorspace]))
    max_value = np.max(np_image)
    if max_value > 0.:
        np_image = np_image / max_value
    return np_image


image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)


class DatasetGenerator(object):
    def __init__(self,
                 data_folder,
                 splits,  # train test eval
                 nb_classes=5,
                 nb_samples_per_class=10,
                 img_size=(20, 20),
                 colorspace='L',
                 pre_scale=(126, 126),
                 augment=True
                 ):
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.img_size = img_size
        self.colorspace = colorspace
        self.augment = augment
        with open(os.path.join(data_folder, 'dataset_spec.json')) as f:
            dspec = json.load(f)
        self.images = [[] for _ in dspec['class_names'].keys()]
        ds = tf.data.TFRecordDataset([os.path.join(data_folder,
                                                   f'{cls}.tfrecords')
                                      for cls in dspec['class_names'].keys()])

        parsed_image_dataset_class = ds.map(_parse_image_function)

        for img in parsed_image_dataset_class:
            img_data = Image.open(io.BytesIO(img['image'].numpy()))\
                            .convert(colorspace).resize(pre_scale)
            self.images[img['label'].numpy()].append(img_data)
        print()
        self.splits = {}
        self.splits['train'] = self.images[:splits[0]]
        self.splits['test'] = self.images[splits[0]:sum(splits[0:2])]
        self.splits['eval'] = self.images[sum(splits[0:2]):sum(splits[0:3])]

    def generate_batch(self, batch_type, batch_size, sample_strategy="random"):

        data = self.splits[batch_type]
        chs = ch[self.colorspace]
        sampled_inputs = np.zeros((batch_size,
                                   self.nb_classes * self.nb_samples_per_class,
                                   np.prod(self.img_size)*chs),
                                  dtype=np.float32)
        sampled_outputs = np.zeros((batch_size,
                                    self.nb_classes
                                    * self.nb_samples_per_class),
                                   dtype=np.int32)

        for i in range(batch_size):
            images, labels = get_images_labels(data,
                                               self.nb_classes,
                                               self.nb_samples_per_class,
                                               self.img_size,
                                               self.colorspace,
                                               self.augment,
                                               sample_strategy)
            sampled_inputs[i] = np.asarray(images, dtype=np.float32)
            sampled_outputs[i] = np.asarray(labels, dtype=np.int32)
        return sampled_inputs, sampled_outputs
