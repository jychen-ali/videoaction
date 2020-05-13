import tensorflow as tf
import re
from collections import Counter
import string
import numpy as np

def get_record_parser_test(config, is_test=True):
    def parse_test(example):
        word_limit = config.word_limit
        char_limit = config.char_limit
        frame_limit = config.frame_limit
        frame_vec_size = config.frame_vec_size
        features = tf.parse_single_example(example,
                   features={
                   "id": tf.FixedLenFeature([], tf.int64),
                   "des_idxs": tf.FixedLenFeature([], tf.string),
                   "des_char_idxs": tf.FixedLenFeature([], tf.string), 
                   "y1": tf.FixedLenFeature([], tf.string),
                   "y2": tf.FixedLenFeature([], tf.string),
                   "vfeat": tf.FixedLenFeature([], tf.string),
                   "v": tf.FixedLenFeature([], tf.string),
                   "lmask": tf.FixedLenFeature([], tf.string),
                   "rmask": tf.FixedLenFeature([], tf.string)
                   })
        sample_id = tf.cast(features["id"], tf.int32)
        des_idxs = tf.reshape(tf.decode_raw(features["des_idxs"], tf.int32), [word_limit])
        des_char_idxs = tf.reshape(tf.decode_raw(features["des_char_idxs"], tf.int32), [word_limit, char_limit])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [frame_limit])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [frame_limit])
        vfeat = tf.reshape(tf.decode_raw(features["vfeat"], tf.float32), [frame_limit, frame_vec_size])
        v = tf.reshape(tf.decode_raw(features["v"], tf.int32), [frame_limit])
        lmask = tf.reshape(tf.decode_raw(features["lmask"], tf.int32), [frame_limit, frame_limit])
        rmask = tf.reshape(tf.decode_raw(features["rmask"], tf.int32), [frame_limit, frame_limit])
        
        return sample_id, des_idxs, des_char_idxs, y1, y2, vfeat, v#, lmask, rmask

    return parse_test
                   

def get_record_parser(config, is_test=False):
    def parse(example):

        word_limit = config.test_word_limit if is_test else config.word_limit
        char_limit = config.char_limit
        frame_limit = config.frame_limit
        frame_vec_size = config.frame_vec_size
        features = tf.parse_single_example(example,
                                           features={
                                               "des_idxs": tf.FixedLenFeature([], tf.string),  
                                               "des_char_idxs": tf.FixedLenFeature([], tf.string),                                      
                                               "y1": tf.FixedLenFeature([], tf.string),
                                               "y2": tf.FixedLenFeature([], tf.string),
                                               "vfeat": tf.FixedLenFeature([], tf.string),
                                               "v": tf.FixedLenFeature([], tf.string),
                                               "lmask": tf.FixedLenFeature([], tf.string),
                                               "rmask": tf.FixedLenFeature([], tf.string)
                                           })
       
        des_idxs = tf.reshape(tf.decode_raw(features["des_idxs"], tf.int32), [word_limit])
        #print("des_idxs: {}".format(des_idxs))

        des_char_idxs = tf.reshape(tf.decode_raw(features["des_char_idxs"], tf.int32), [word_limit, char_limit])
        #print("des_char_idxs: {}".format(des_char_idxs))

        #print("feature y1: {}".format(features["y1"]))
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.float32), [frame_limit])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.float32), [frame_limit])
        #print("feature vfeat: {}".format(features["vfeat"]))
        vfeat = tf.reshape(tf.decode_raw(features["vfeat"], tf.float32), [frame_limit, frame_vec_size])
        v = tf.reshape(tf.decode_raw(features["v"], tf.int32), [frame_limit])
        lmask = tf.reshape(tf.decode_raw(features["lmask"], tf.int32), [frame_limit, frame_limit])
        rmask = tf.reshape(tf.decode_raw(features["rmask"], tf.int32), [frame_limit, frame_limit])
        return 0, des_idxs, des_char_idxs, y1, y2, vfeat, v#, lmask, rmask
        #print("y1: {}".format(y1))
        #print("y2: {}".format(y2))
        #print("vfeat: {}".format(vfeat))
 
        #return 0, des_idxs, des_char_idxs, y1, y2, vfeat, v
    
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        start_idx = spans[p1][0]
        end_idx = spans[p2][1]
        answer_dict[str(qid)] = context[start_idx: end_idx]
        remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict

def iou(pred, gt):
   
    assert pred[0] <= pred[1]
    intersection = max(0, min(pred[1], gt[1])+1 - max(pred[0],gt[0]))
    union = max(pred[1], gt[1])+1 - min(pred[0], gt[0])
    iou = float(intersection) / union

    return iou

def rank(pred, gt):
    gt = [gt[0],gt[1]]
    if pred == gt:
        return 1
    else:
        return 2

#def evaluate(answer_dict, meta):
#    ious = []
#    average_iou = []
#    average_rank = []
#    total = 0
#    count1 = [0,0,0,0,0,0,0,0,0,0]
#    count5 = [0,0,0,0,0,0,0,0,0,0]
#    miou = 0.0
   
#    for sample_id in answer_dict.keys():
#        pred = answer_dict[sample_id]
#        ious = [iou(pred, temp) for temp in meta[sample_id]]
#        average_iou.append(np.mean(np.sort(ious)[-3:]))
#        ranks = [rank(pred, temp) for temp in meta[sample_id]]
#        average_rank.append(np.mean(np.sort(ranks)[:3]))
    
#    rank1 = np.sum(np.array(average_rank) <= 1)/float(len(average_rank))
#    miou = np.mean(average_iou)
#    return {'miou': miou, 'rank1': rank1}

def evaluate(answer_dict, meta):
    average_ious_top1 = []
    average_ious_top5 = []
    for sample_id in answer_dict.keys():
        [idxes1, idxes2, y1, y2] = answer_dict[sample_id]
        yp1 = idxes1[0][0]
        yp2 = idxes1[1][0]
        iou0 = [iou([yp1,yp2],temp) for temp in meta[sample_id]]

        pred_0 = [idxes2[0][0], idxes2[1][0]]
        ious_0 = [iou(pred_0, temp) for temp in meta[sample_id]]
        pred_1 = [idxes2[0][1], idxes2[1][1]]
        ious_1 = [iou(pred_1, temp) for temp in meta[sample_id]]
        pred_2 = [idxes2[0][2], idxes2[1][2]]
        ious_2 = [iou(pred_2, temp) for temp in meta[sample_id]]
        pred_3 = [idxes2[0][3], idxes2[1][3]]
        ious_3 = [iou(pred_3, temp) for temp in meta[sample_id]]
        pred_4 = [idxes2[0][4], idxes2[1][4]]
        ious_4 = [iou(pred_4, temp) for temp in meta[sample_id]]

        average_ious_top1.append(np.mean(np.sort(iou0)[-3:]))
        average_ious_top5.append(max(np.mean(np.sort(ious_0)[-3:]), 
                                     np.mean(np.sort(ious_1)[-3:]),
                                     np.mean(np.sort(ious_2)[-3:]),
                                     np.mean(np.sort(ious_3)[-3:]),
                                     np.mean(np.sort(ious_4)[-3:])))
    miou = np.mean(average_ious_top1)

    # R@1, IoU from 0.1 to 1.0, step 0.1
    R_at_1_iou1 = np.sum(np.array(average_ious_top1) >= 0.1)/float(len(average_ious_top1))
    R_at_1_iou2 = np.sum(np.array(average_ious_top1) >= 0.2)/float(len(average_ious_top1))
    R_at_1_iou3 = np.sum(np.array(average_ious_top1) >= 0.3)/float(len(average_ious_top1))
    R_at_1_iou4 = np.sum(np.array(average_ious_top1) >= 0.4)/float(len(average_ious_top1))
    R_at_1_iou5 = np.sum(np.array(average_ious_top1) >= 0.5)/float(len(average_ious_top1))
    R_at_1_iou6 = np.sum(np.array(average_ious_top1) >= 0.6)/float(len(average_ious_top1))
    R_at_1_iou7 = np.sum(np.array(average_ious_top1) >= 0.7)/float(len(average_ious_top1))
    R_at_1_iou8 = np.sum(np.array(average_ious_top1) >= 0.8)/float(len(average_ious_top1))
    R_at_1_iou9 = np.sum(np.array(average_ious_top1) >= 0.9)/float(len(average_ious_top1))
    R_at_1_iou10 = np.sum(np.array(average_ious_top1) >= 1.0)/float(len(average_ious_top1))

    #print "R@1, iou >= 0.1: %f" % R_at_1_iou1
    #print "R@1, iou >= 0.2: %f" % R_at_1_iou2
    #print "R@1, iou >= 0.3: %f" % R_at_1_iou3
    #print "R@1, iou >= 0.4: %f" % R_at_1_iou4
    #print "R@1, iou >= 0.5: %f" % R_at_1_iou5
    #print "R@1, iou >= 0.6: %f" % R_at_1_iou6
    #print "R@1, iou >= 0.7: %f" % R_at_1_iou7
    #print "R@1, iou >= 0.8: %f" % R_at_1_iou8
    #print "R@1, iou >= 0.9: %f" % R_at_1_iou9
    #print "R@1, iou >= 1.0: %f" % R_at_1_iou10

    # R@5, IoU from 0.1 to 1.0, step 0.1
    R_at_5_iou1 = np.sum(np.array(average_ious_top5) >= 0.1)/float(len(average_ious_top5))
    R_at_5_iou2 = np.sum(np.array(average_ious_top5) >= 0.2)/float(len(average_ious_top5))
    R_at_5_iou3 = np.sum(np.array(average_ious_top5) >= 0.3)/float(len(average_ious_top5))
    R_at_5_iou4 = np.sum(np.array(average_ious_top5) >= 0.4)/float(len(average_ious_top5))
    R_at_5_iou5 = np.sum(np.array(average_ious_top5) >= 0.5)/float(len(average_ious_top5))
    R_at_5_iou6 = np.sum(np.array(average_ious_top5) >= 0.6)/float(len(average_ious_top5))
    R_at_5_iou7 = np.sum(np.array(average_ious_top5) >= 0.7)/float(len(average_ious_top5))
    R_at_5_iou8 = np.sum(np.array(average_ious_top5) >= 0.8)/float(len(average_ious_top5))
    R_at_5_iou9 = np.sum(np.array(average_ious_top5) >= 0.9)/float(len(average_ious_top5))
    R_at_5_iou10 = np.sum(np.array(average_ious_top5) >= 1.0)/float(len(average_ious_top5))

    #print "R@5, iou >= 0.1: %f" % R_at_5_iou1
    #print "R@5, iou >= 0.2: %f" % R_at_5_iou2
    #print "R@5, iou >= 0.3: %f" % R_at_5_iou3
    #print "R@5, iou >= 0.4: %f" % R_at_5_iou4
    #print "R@5, iou >= 0.5: %f" % R_at_5_iou5
    #print "R@5, iou >= 0.6: %f" % R_at_5_iou6
    #print "R@5, iou >= 0.7: %f" % R_at_5_iou7
    #print "R@5, iou >= 0.8: %f" % R_at_5_iou8
    #print "R@5, iou >= 0.9: %f" % R_at_5_iou9
    #print "R@5, iou >= 1.0: %f" % R_at_5_iou10
 
    #print "Average iou: %f" % miou

    return [R_at_1_iou10,R_at_1_iou9,R_at_1_iou8,R_at_1_iou7,R_at_1_iou6,R_at_1_iou5,R_at_1_iou4,R_at_1_iou3,R_at_1_iou2,R_at_1_iou1],[R_at_5_iou10,R_at_5_iou9,R_at_5_iou8,R_at_5_iou7,R_at_5_iou6,R_at_5_iou5,R_at_5_iou4,R_at_5_iou3,R_at_5_iou2,R_at_5_iou1], miou

    
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
