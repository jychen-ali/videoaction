import tensorflow as tf
import ujson as json
import numpy as np
import os

from model import Model
from util import get_record_parser, get_record_parser_test, convert_tokens, evaluate, get_batch_dataset, get_dataset

def iou(pred, gt):
    assert pred[0]<=pred[1]
    intersection = max(0, min(pred[1], gt[1]) - max(pred[0],gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    iou = float(intersection) / union 
    return iou

def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    meta = {int(k):v for k,v in meta.items()}
    dev_total = len(meta)

    print("Building model...")
    parser = get_record_parser(config)
    parser_test = get_record_parser_test(config)    
    train_dataset = get_batch_dataset(config.train_record_file, parser, config)
    dev_dataset = get_dataset(config.dev_record_file, parser_test, config)
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    model = Model(config, iterator, word_mat, char_mat)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    loss_save = 100.0
    patience = 0
    lr = config.init_lr

    with tf.Session(config=sess_config) as sess:
        writer = tf.summary.FileWriter(config.log_dir)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train_handle = sess.run(train_iterator.string_handle())
        dev_handle = sess.run(dev_iterator.string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
        
        for step in range(1, config.num_steps + 1):
            global_step = sess.run(model.global_step) + 1
            loss, train_op = sess.run([model.loss, model.train_op],
                                                                feed_dict={handle: train_handle})

            print("step: {}  global_step: {}  loss: {:.3f}".format(step, global_step, loss))

            if global_step % config.period == 0:
                loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss), ])
                writer.add_summary(loss_sum, global_step)

            if global_step % config.checkpoint == 0:
                sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
                #_, summ = evaluate_batch(
                #    model, config.val_num_batches, sess, "train", handle, train_handle)
                #for s in summ:
                #    writer.add_summary(s, global_step)

                count1,count5,miou,loss_m,summ = evaluate_batch(model, meta, dev_total // config.batch_size + 1, sess, "dev", handle, dev_handle)
                print(count1,count5,miou)
                #print("mean iou metric: {}".format(metrics['miou']))
                #print("rank1 metric: {}".format(metrics['rank1']))
                sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))

                dev_loss = loss_m
                if dev_loss < loss_save:
                    loss_save = dev_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= config.patience:
                    print("learning rate: {}".format(lr))
                    lr = lr*0.8
                    loss_save = dev_loss
                    patience = 0
                sess.run(tf.assign(model.lr, tf.constant(lr, dtype=tf.float32)))
                for s in summ:
                    writer.add_summary(s, global_step)
                writer.flush()
                filename = os.path.join(config.save_dir, "model_{}.ckpt".format(global_step))
                saver.save(sess, filename)


def evaluate_batch(model, meta, num_batches, sess, data_type, handle, str_handle):
    answer_dict = {}
    losses = []
    for _ in range(1, num_batches + 1):
        loss, sample_ids, outer_matrix, y1s, y2s = sess.run([model.loss, model.sample_id, model.outer, model.y1, model.y2], feed_dict={handle: str_handle})
        #loss, sample_ids, y1s, y2s, yp1s, yp2s, = sess.run([model.loss, model.sample_id, model.y1, model.y2, model.yp1, model.yp2], feed_dict={handle: str_handle})
        #for sample_id, y1, y2, yp1, yp2 in zip(sample_ids, y1s, y2s, yp1s, yp2s):
        #    answer_dict[sample_id] = [yp1, yp2]
        for ii in range(outer_matrix.shape[0]):
            y1 = np.where(y1s[ii]==1.)[0][0]
            y2 = np.where(y2s[ii]==1.)[0][0]
            cur = outer_matrix[ii]
            idxes1 = np.unravel_index(np.argsort(cur.ravel())[-1:], cur.shape)
            idxes2 = np.unravel_index(np.argsort(cur.ravel())[-5:], cur.shape)
            print(idxes2)
            print(idxes1)
            answer_dict[sample_ids[ii]]=[idxes1, idxes2, y1, y2]
        return
        losses.append(loss)
    loss = np.mean(losses)
    count1, count5, miou = evaluate(answer_dict, meta)
    miou_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/miou".format(data_type), simple_value=loss), ])
    loss_sum      = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=loss), ])
    count1_10_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_10".format(data_type), simple_value=count1[0]), ])
    count1_09_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_09".format(data_type), simple_value=count1[1]), ])
    count1_08_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_08".format(data_type), simple_value=count1[2]), ])
    count1_07_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_07".format(data_type), simple_value=count1[3]), ])
    count1_06_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_06".format(data_type), simple_value=count1[4]), ])
    count1_05_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_05".format(data_type), simple_value=count1[5]), ])
    count1_04_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_04".format(data_type), simple_value=count1[6]), ])
    count1_03_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_03".format(data_type), simple_value=count1[7]), ])
    count1_02_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_02".format(data_type), simple_value=count1[8]), ])
    count1_01_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count1_01".format(data_type), simple_value=count1[9]), ])

    count5_10_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_10".format(data_type), simple_value=count5[0]), ])
    count5_09_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_09".format(data_type), simple_value=count5[1]), ])
    count5_08_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_08".format(data_type), simple_value=count5[2]), ])
    count5_07_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_07".format(data_type), simple_value=count5[3]), ])
    count5_06_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_06".format(data_type), simple_value=count5[4]), ])
    count5_05_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_05".format(data_type), simple_value=count5[5]), ])
    count5_04_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_04".format(data_type), simple_value=count5[6]), ])
    count5_03_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_03".format(data_type), simple_value=count5[7]), ])
    count5_02_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_02".format(data_type), simple_value=count5[8]), ])
    count5_01_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/count5_01".format(data_type), simple_value=count5[9]), ])

    return count1, count5, miou, loss, [miou_sum, loss_sum, count1_10_sum, count1_09_sum, count1_08_sum, count1_07_sum, count1_06_sum,
                                            count1_05_sum, count1_04_sum, count1_03_sum, count1_02_sum, count1_01_sum,
                                            count5_10_sum, count5_09_sum, count5_08_sum, count5_07_sum, count5_06_sum,
                                            count5_05_sum, count5_04_sum, count5_03_sum, count5_02_sum, count5_01_sum]

    #metrics["loss"] = loss
    #loss_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    #miou_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/miou".format(data_type), simple_value=metrics["miou"]), ])
    #rank1_sum = tf.Summary(value=[tf.Summary.Value(tag="{}/rank1".format(data_type), simple_value=metrics["rank1"]), ])
    #return metrics, [loss_sum, miou_sum, rank1_sum]

def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.char_emb_file, "r") as fh:
        char_mat = np.array(json.load(fh), dtype=np.float32)

    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)
  
    with open(config.idx2word_file, "r") as fh:
        idx2word = json.load(fh)
    
    meta = {int(k):v for k,v in meta.items()}
    total = len(meta)
    
    #total = meta['total']

    print("Loading model...")
    test_batch = get_dataset(config.test_record_file, get_record_parser_test(config, is_test=True), config).make_one_shot_iterator()

    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    answer_dict = []
    answers = []
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
        losses = []
        for step in range(total // config.batch_size + 1):
            vids, des, y1s, y2s, yp1, yp2, dv_logits, self_logits = sess.run([model.sample_id, model.d, model.y1, model.y2, model.yp1, model.yp2, model.dv_logits, model.self_logits])
            for ii in range(len(vids)):
                y1 = np.where(y1s[ii]==1.)[0][0]
                y2 = np.where(y2s[ii]==1.)[0][0]
                iouscore = iou([yp1[ii],yp2[ii]], [y1,y2]) 
                if iouscore == 1. and y1 < y2 and vids[ii]==930:
                    print('vid:'+str(vids[ii])+'\t'+str(yp1[ii])+'\t'+str(yp2[ii])+'\t')
                    for idxs in range(len(des[ii])):
                        if des[ii][idxs] == 0 or des[ii][idxs]==1:
                            break
                        print(idx2word[str(des[ii][idxs])]+' ', end='')  #+'des:'+str(des[ii]))
                    print(str(dv_logits[ii]))
                    print('-----------------------------------')
                    print(str(self_logits[ii]))
                    print('\n')
                    #print('vid:'+str(vids[ii])+'\t'+str(yp1[ii])+'\t'+str(yp2[ii])+'\t'+str(dv_logits[ii])+'\t'+str(self_logits[ii])+'\t'+'des:'+str(des[ii]))

#def test(config):
#    with open(config.word_emb_file, "r") as fh:
#        word_mat = np.array(json.load(fh), dtype=np.float32)
#    with open(config.char_emb_file, "r") as fh:
#        char_mat = np.array(json.load(fh), dtype=np.float32)
#    with open(config.test_eval_file, "r") as fh:
#        eval_file = json.load(fh)
#    with open(config.test_meta, "r") as fh:
#        meta = json.load(fh)
#
#    total = meta["total"]
#
#    print("Loading model...")
#    test_batch = get_dataset(config.test_record_file, get_record_parser(config, is_test=True), config).make_one_shot_iterator()

#    model = Model(config, test_batch, word_mat, char_mat, trainable=False)

#    sess_config = tf.ConfigProto(allow_soft_placement=True)
#    sess_config.gpu_options.allow_growth = True

#    with tf.Session(config=sess_config) as sess:
#        sess.run(tf.global_variables_initializer())
#        saver = tf.train.Saver()
#        saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
#        sess.run(tf.assign(model.is_train, tf.constant(False, dtype=tf.bool)))
#        losses = []
#        answer_dict = {}
#        remapped_dict = {}
#        for step in range(total // config.batch_size + 1):
#            qa_id, loss, yp1, yp2 = sess.run([model.qa_id, model.loss, model.yp1, model.yp2])
#            answer_dict_, remapped_dict_ = convert_tokens(eval_file, qa_id.tolist(), yp1.tolist(), yp2.tolist())
#            answer_dict.update(answer_dict_)
#            remapped_dict.update(remapped_dict_)
#            losses.append(loss)
#        loss = np.mean(losses)
#        metrics = evaluate(eval_file, answer_dict)
#        with open(config.answer_file, "w") as fh:
#            json.dump(remapped_dict, fh)
#        print("Exact Match: {}, F1: {}".format(metrics['exact_match'], metrics['f1']))
