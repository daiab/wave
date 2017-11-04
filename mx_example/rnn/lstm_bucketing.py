# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
import mxnet as mx
import argparse
import os


# parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
#                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--num-layers', type=int, default=2,
#                     help='number of stacked RNN layers')
# parser.add_argument('--num-hidden', type=int, default=200,
#                     help='hidden layer size')
# parser.add_argument('--num-embed', type=int, default=200,
#                     help='embedding layer size')
# parser.add_argument('--gpus', type=str,
#                     help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
#                          'Increase batch size when using multiple gpus for best performance.')
# parser.add_argument('--kv-store', type=str, default='device',
#                     help='key-value store type')
# parser.add_argument('--num-epochs', type=int, default=25,
#                     help='max num of epochs')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='initial learning rate')
# parser.add_argument('--optimizer', type=str, default='sgd',
#                     help='the optimizer type')
# parser.add_argument('--mom', type=float, default=0.0,
#                     help='momentum for sgd')
# parser.add_argument('--wd', type=float, default=0.00001,
#                     help='weight decay for sgd')
# parser.add_argument('--batch-size', type=int, default=32,
#                     help='the batch size.')
# parser.add_argument('--disp-batches', type=int, default=50,
#                     help='show progress for every n batches')

class Arg:
    def __init__(self):
        self.num_layers = 2
        self.num_hidden = 111
        self.num_embed = 234
        self.gpus = '0'
        self.kv_store = 'device'
        self.num_epochs = 25
        self.lr = 0.01
        self.optimizer = 'sgd'
        self.mom = 0.0
        self.wd = 0.00001
        self.batch_size = 32
        self.disp_batches = 20


def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    if not os.path.isfile(fname):
        raise IOError("Please use get_ptb_data.sh to download requied file (data/ptb.train.txt)")
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label,
                                               start_label=start_label)
    # \n: 0
    return sentences, vocab


def desc_net(net, data_shape, label_shape):
    arg = net.list_arguments()
    arg_shapes = net.infer_shape(data=data_shape, softmax_label=label_shape)
    print('=============== arguments shapes ================')
    for kv in zip(arg, arg_shapes[0]):
        print('key: %s ## shape: %s' % kv)
    layer_output = net.get_internals().list_outputs()
    print('================ output shapes ===================')
    internals = net.get_internals()
    for name in layer_output:
        if not name.endswith('_output'):
            continue
        layer = internals[name]
        try:
            shape = layer.infer_shape(data=data_shape)
            print('key: %s ## shape: %s' % (name, shape[1][0]))
        except:
            continue


if __name__ == '__main__':
    import logging

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    # args = parser.parse_args()
    args = Arg()

    # buckets = []
    buckets = [10, 20, 30, 40, 50, 60]

    start_label = 1
    invalid_label = 0

    train_sent, vocab = tokenize_text("./data/ptb.train.txt", start_label=start_label,
                                      invalid_label=invalid_label)
    val_sent, _ = tokenize_text("./data/ptb.test.txt", vocab=vocab, start_label=start_label,
                                invalid_label=invalid_label)

    print('vocab len == %d' % len(vocab))
    data_train = mx.rnn.BucketSentenceIter(train_sent, args.batch_size, buckets=buckets,
                                           invalid_label=invalid_label)
    data_val = mx.rnn.BucketSentenceIter(val_sent, args.batch_size, buckets=buckets,
                                         invalid_label=invalid_label)

    stack = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        stack.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_' % i))


    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=args.num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs, shape=(-1, args.num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    contexts = [mx.gpu(int(i)) for i in [0]]
    model = mx.mod.BucketingModule(
        sym_gen=sym_gen,
        default_bucket_key=data_train.default_bucket_key,
        context=contexts)

    model.bind(data_shapes=data_train.provide_data, label_shapes=data_train.provide_label,
              for_training=True)
    model.init_params()

    data_train.reset()
    # for _ in range(1):
    #     data = data_train.next()
        # print(data)
        # print(data.data[0].asnumpy())
        # print(data.label[0].asnumpy())
    # data_train.reset()
    # print(data_train.provide_data)
    # net, _, _ = sym_gen(30)
    # desc_net(net, (32, 30), (32, 30))
    # print(model.get_params())
    # print(model.get_outputs())
    # print(model.get_states(merge_multi_context=False))
    # exit()

    model.fit(
        train_data=data_train,
        eval_data=data_val,
        eval_metric=mx.metric.Perplexity(invalid_label),
        kvstore=args.kv_store,
        optimizer=args.optimizer,
        optimizer_params={'learning_rate': args.lr,
                          'momentum': args.mom,
                          'wd': args.wd},
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch=args.num_epochs,
        batch_end_callback=mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False))


