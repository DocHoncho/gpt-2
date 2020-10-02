#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    init_context="Chapter 1",
    generations=2,
    outfile='output.txt',
    temperature=1,
    top_k=0,
    top_p=0.0
):
    """
    Run the model iteratively, feeding part of each result back into the model as context

    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :init_context=Chapter 1 : Initial context value to use
    :generations=2 : Number of generations to run
    :outfile=output.txt : File to write output to
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """

    batch_size = 1

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    length = 1024

    # if length is None:
    #     length = hparams.n_ctx // 2
    # elif length > hparams.n_ctx:
    #     raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        context = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        if init_context:
            context_text = init_context
        else:
            context_text = "Chapter 1"

        outf = open(outfile, 'w', encoding="utf8")
        outf.write(init_context)
        outf.flush()

        for g in range(generations):
            print('Calculating generation {} of {}'.format(g+1, generations))
            ctx_tokens = enc.encode(context_text)
            out = sess.run(output, feed_dict={
                context: [ctx_tokens]
            })[:, len(ctx_tokens):]
            text = enc.decode(out[0])
            outf.write(text)
            outf.flush()
            context_text = text.split('\n')[-1]

        # while True:
        #     raw_text = input("Model prompt >>> ")
        #     while not raw_text:
        #         print('Prompt should not be empty!')
        #         raw_text = input("Model prompt >>> ")
        #     context_tokens = enc.encode(raw_text)
        #     generated = 0
        #     for _ in range(nsamples // batch_size):
        #         out = sess.run(output, feed_dict={
        #             context: [context_tokens for _ in range(batch_size)]
        #         })[:, len(context_tokens):]
        #         for i in range(batch_size):
        #             generated += 1
        #             text = enc.decode(out[i])
        #             print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
        #             print(text)
        #     print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)
