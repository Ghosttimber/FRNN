
# generate_tfrecord.py
 
 
import os
import io
import pandas as pd
import tensorflow as tf
 
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
 
# os.chdir('./images/test')
 
flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string("img_path", "", "Path to images")
FLAGS = flags.FLAGS
 
 
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'head':     # 需改动为自己的分类
        return 1
    if row_label == 'helmet':     # 需改动为自己的分类
        return 2
    if row_label == 'person':     # 需改动为自己的分类
        return 3

    else:
        None
 
 
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
 
 
def create_tf_example(group, path):
    encoded_jpg = tf.gfile.FastGFile(os.path.join(path, '{}'.format(group.filename)), 'rb').read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
 
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
for index, row in group.object.iterrows():
        
        ########### ADDITIONAL CHECKS START HERE ###################

        xmn = row['xmin'] / width
        if xmn < 0.0:
            xmn = 0.0
        elif xmn > 1.0:
            xmn = 1.0
        xmins.append(xmn)

        xmx = row['xmax'] / width
        if xmx < 0.0:
            xmx = 0.0
        elif xmx > 1.0:
            xmx = 1.0
        xmaxs.append(xmx)

        ymn = row['ymin'] / height
        if ymn < 0.0:
            ymn = 0.0
        elif ymn > 1.0:
            ymn = 1.0
        ymins.append(ymn)

        ymx = row['ymax'] / height
        if ymx < 0.0:
            ymx = 0.0
        elif ymx > 1.0:
            ymx = 1.0
        ymaxs.append(ymx)

        ############ ADDITIONAL CHECKS END HERE ####################

        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
 
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example
 
 
def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # path = os.path.join(os.getcwd(), 'test')         #  有问题，此处用绝对地址出错，用相对 
      #地址正确，网友可以测试下，有其他答案可留言
    path = os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
 
    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
 
 
if __name__ == '__main__':
    tf.app.run()