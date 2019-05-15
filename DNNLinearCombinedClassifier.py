#-*-coding:utf-8-*-
import argparse
import json
import os
import multiprocessing
import tensorflow as tf
#import constants

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("LABEL_COLUMN","income_bracket",'output label')
tf.app.flags.DEFINE_string("TRAIN_DATA","./wjm_adult.data.csv",'input data')
tf.app.flags.DEFINE_string("EVAL_DATA","./wjm_adult.test.csv",'test data')
tf.app.flags.DEFINE_string("EXPORT_NAME","census",'export name')
tf.app.flags.DEFINE_string("EVAL_NAME","census-eval",'eval name')
tf.app.flags.DEFINE_string("MODEL_DIR","output",'output dir folder')

tf.app.flags.DEFINE_integer("MAX_STEPS",1000,'train max steps')
tf.app.flags.DEFINE_integer("EVAL_STEPS",100,'eval steps')
tf.app.flags.DEFINE_integer("TRAIN_BATCH_SIZE",40,'tain batch size')
tf.app.flags.DEFINE_integer("EVAL_BATCH_SIZE",40,'eval batch size')

INPUT_COLUMNS = [
    tf.feature_column.categorical_column_with_vocabulary_list('gender', [' Female', ' Male']),

    tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=100, dtype=tf.string),

    tf.feature_column.numeric_column('age'),
]

def train_input():

    return input_fn(FLAGS.TRAIN_DATA,batch_size=FLAGS.TRAIN_BATCH_SIZE)

def eval_input():

    return input_fn(FLAGS.EVAL_DATA,batch_size=FLAGS.EVAL_BATCH_SIZE,shuffle=False)


def _decode_csv(line):
    """Takes the string input tensor and returns a dict of rank-2 tensors."""

    # Takes a rank-1 tensor and converts it into rank-2 tensor
    # Example if the data is ['csv,line,1', 'csv,line,2', ..] to
    # [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
    # tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
    CSV_COLUMN_DEFAULTS = [[0], [''], [''], ['']]
    CSV_COLUMNS = [
	'age', 'occupation', 'gender', 'income_bracket'
    ]

    row_columns = tf.expand_dims(line, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    return features

def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=0, batch_size=200, num_parallel_calls=None, prefetch_buffer_size=None):

    # -----contribute features
    if num_parallel_calls is None:
        num_parallel_calls = multiprocessing.cpu_count()
    if prefetch_buffer_size is None:
        prefetch_buffer_size = 1024
    dataset = tf.data.TextLineDataset(filenames).skip(skip_header_lines).map(_decode_csv, num_parallel_calls).prefetch(prefetch_buffer_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
    features = iterator.get_next()

    # -----contribute lable
    LABELS = [' <=50K', ' >50K']
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(LABELS))

    # Use the hash table to convert string labels to ints and one-hot encode
    label = table.lookup(features.pop(FLAGS.LABEL_COLUMN))

    
    return features, label


def build_estimator(config, embedding_size=8, hidden_units=None):
    (gender, occupation, age )= INPUT_COLUMNS

    # Reused Transformations.
    # Continuous columns can be converted to categorical via bucketization
    age_buckets = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Wide columns and deep columns.
    wide_columns = [
        gender,
        occupation,
        age_buckets,
    ]
    deep_columns = [
        # Use indicator columns for low dimensional vocabularies
        tf.feature_column.indicator_column(gender),

        # Use embedding columns for high dimensional vocabularies
        tf.feature_column.embedding_column(
            occupation, dimension=embedding_size),
        age,
    ]

    return tf.estimator.DNNLinearCombinedClassifier(
        config=config,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units or [100, 70, 50, 25])

# [START serving-function]
def json_serving_input_fn():
    """Build the serving inputs."""
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)

    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
}

def _get_session_config_from_env_var():
    """Returns a tf.ConfigProto instance that has appropriate device_filters
    set."""


    tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

    if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
            'index' in tf_config['task']):
        # Master should only communicate with itself and ps
        if tf_config['task']['type'] == 'master':
            return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
        # Worker should only communicate with itself and ps
        elif tf_config['task']['type'] == 'worker':
            return tf.ConfigProto(device_filters=[
                '/job:ps',
                '/job:worker/task:%d' % tf_config['task']['index']
            ])
    return None


def training_job():
    """Run the training and evaluate using the high level API."""

    train_spec = tf.estimator.TrainSpec(train_input, max_steps=FLAGS.MAX_STEPS)
    
    exporter = tf.estimator.FinalExporter(FLAGS.EXPORT_NAME, SERVING_FUNCTIONS['JSON'])
    
    eval_spec = tf.estimator.EvalSpec(eval_input, steps=FLAGS.EVAL_STEPS, exporters=[exporter], name=FLAGS.EVAL_NAME)

    run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())

    run_config = run_config.replace(model_dir=FLAGS.MODEL_DIR)
    print('Model dir %s' % run_config.model_dir)

    estimator = build_estimator( embedding_size=8,
 	# Construct layers sizes with exponential decay
        hidden_units=[
            max(2, int(100 * 0.7**i))
            for i in range(4)
        ],
        config=run_config)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)
    training_job()
