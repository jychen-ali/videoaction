import tensorflow as tf
import numpy as np
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net


class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.sample_id, self.d, self.ch, self.y1, self.y2, self.vf, self.v = batch.get_next()

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable(
            "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.d_mask = tf.cast(self.d, tf.bool)
        self.v_mask = tf.cast(self.v, tf.bool)

        self.d_len = tf.reduce_sum(tf.cast(self.d_mask, tf.int32), axis=1)
        self.v_len = tf.reduce_sum(tf.cast(self.v_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.d_maxlen = tf.reduce_max(self.d_len)
            self.v_maxlen = tf.reduce_max(self.v_len)

            self.d = tf.slice(self.d, [0, 0], [N, self.d_maxlen])
            self.d_mask = tf.slice(self.d_mask, [0, 0], [N, self.d_maxlen])

            self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.d_maxlen, CL])
           
            self.y1 = tf.slice(self.y1, [0, 0], [N, self.v_maxlen])
            self.y2 = tf.slice(self.y2, [0, 0], [N, self.v_maxlen])

        else:
            self.d_maxlen = config.word_limit

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])

        self.ready()        
         
        if trainable:

#            total_paramters = 0
#            for variable in tf.trainable_variables():
#                shape = variable.get_shape()
#                if shape is None:
#                    continue
#                print(shape)
#                variable_parameters = 1
#                for dim in shape:

#                    variable_parameters *= dim.value
#                total_paramters += variable_parameters
#            print('Total paras: ' + str(total_parameters))

            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.lr, epsilon=1e-08)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        
        config = self.config
        N, DL, VL, CL, dh, dc, dg = config.batch_size, self.d_maxlen, self.v_maxlen, config.char_limit, config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            v_emb = self.vf  
            v_emb = tf.reshape(v_emb, [N, config.frame_limit, config.frame_vec_size])

#            with tf.variable_scope("char"):
#                ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.ch), [N * DL, CL, dc]) 
#                ch_emb = dropout(
#                    ch_emb, keep_prob=config.keep_prob, is_train=self.is_train)
#                cell_fw = tf.contrib.rnn.GRUCell(dg)
#                cell_bw = tf.contrib.rnn.GRUCell(dg)
#                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
#                    cell_fw, cell_bw, ch_emb, self.ch_len, dtype=tf.float32)
#                ch_emb = tf.concat([state_fw, state_bw], axis=1)
#                ch_emb = tf.reshape(ch_emb, [N, DL, 2 * dg])


            with tf.name_scope("word"):
                d_emb = tf.nn.embedding_lookup(self.word_mat, self.d)

#            d_emb = tf.concat([d_emb, ch_emb], axis=2)

        with tf.variable_scope("encoding"):

            rnn = gru(num_layers=3, num_units=dh, batch_size=N, input_size=d_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            d = rnn(d_emb, seq_len=self.d_len)
            rnn_v = gru(num_layers=3, num_units=dh, batch_size=N, input_size=v_emb.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
          
            v = rnn_v(v_emb, seq_len=self.v_len)	

        with tf.variable_scope("attention"):
            dv_att, self.dv_logits = dot_attention(v, d, mask=self.d_mask, hidden=dh,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=dh, batch_size=N, input_size=dv_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(dv_att, seq_len=self.v_len)
        
        with tf.variable_scope("match"):
            self_att, self.self_logits = dot_attention(
                att, att, mask=self.v_mask, hidden=dh, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=dh, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.v_len)

        #with tf.variable_scope("lattention"):
            #l_att = direct_attention(
            #    att, att, mask=self.lmask, hidden=dh, keep_prob=config.keep_prob, is_train=self.is_train)
            #rnn = gru(num_layers=1, num_units=dh, batch_size=N, input_size=l_att.get_shape(
            #).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            #l_match = rnn(l_att, seq_len=self.v_len)

        #with tf.variable_scope("rattention"):
            #r_att = direct_attention(
            #    att, att, mask=self.rmask, hidden=dh, keep_prob=config.keep_prob, is_train=self.is_train)
            #rnn = gru(num_layers=1, num_units=dh, batch_size=N, input_size=r_att.get_shape(
            #).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            #r_match = rnn(r_att, seq_len=self.v_len)


#        with tf.variable_scope("match"):
#            self_att = dot_attention(
#                att, att, mask=self.v_mask, hidden=dh, keep_prob=config.keep_prob, is_train=self.is_train)
#            rnn = gru(num_layers=1, num_units=dh, batch_size=N, input_size=self_att.get_shape(
#            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
#            match = rnn(self_att, seq_len=self.v_len)

        with tf.variable_scope("pointer"):
            init = summ(d[:, :, -2 * dh:], dh, mask=self.d_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, dh, self.v_mask)

#        with tf.variable_scope("pointer"):
#            init = summ(d[:, :, -2 * dh:], dh, mask=self.d_mask,
#                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
#            lpointer = direct_ptr_net(batch=N, hidden=init.get_shape().as_list(
#            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train, scope="left_ptr_net")
#            logits1 = lpointer(init, l_match, dh, self.v_mask)
#            rpointer = direct_ptr_net(batch=N, hidden=init.get_shape().as_list(
#            )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train, scope="right_ptr_net")
#            logits2 = rpointer(init, r_match, dh, self.v_mask)

        with tf.variable_scope("predict"):
            #logits1 = tf.nn.softmax(logits1)
            #logits2 = tf.nn.softmax(logtis2)
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            self.outer = tf.matrix_band_part(outer, 0, 5)
            self.yp1 = tf.argmax(tf.reduce_max(self.outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(self.outer, axis=1), axis=1)
            #print(self.yp1, self.y1, self.yp2, self.y2)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
