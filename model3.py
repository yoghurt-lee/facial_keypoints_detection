import tensorflow as tf
import numpy as np
from Facial_Keypoints_Detection.kfkd import load
from sklearn.utils import shuffle

save_path = "./model3/model"
# 前向传播算法
W = 96
H = 96
D = 1
OUTPUT_NODE = 30

def weight_variable(shape,regularizer):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if (regularizer != None):
        # 将权重参数的正则化项加入损失集合
        tf.add_to_collection("losses", regularizer(tf.Variable(initial)))
    return tf.Variable(initial)
# 根据shape初始化bias变量
def bias_variable(shape,value=0.1):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def inference(input_tensor,regularizer=None,keepprob=1.0):
    with tf.variable_scope("layer1"): #第一层 卷积+池化 96x96x1 -> 94x94x32 -> 47x47x32
        W_conv1 = weight_variable([3,3,1,32],regularizer)
        b_conv1 = bias_variable([32])
        conv1 = conv2d(input_tensor,W_conv1)
        relu1 = tf.nn.relu(conv1+b_conv1)
        pool1 = max_pool_2x2(relu1)

    with tf.variable_scope("layer2"): #第二层 卷积+池化 47x47x32 -> 46x46x64 -> 23x23x64
        W_conv2 = weight_variable([3,3,32,64],regularizer)
        b_conv2 = bias_variable([64])
        conv2 = conv2d(pool1,W_conv2)
        relu2 = tf.nn.relu(conv2+b_conv2)
        pool2 = max_pool_2x2(relu2)

    with tf.variable_scope("layer3"): #第三层 卷积+池化 23x23x64 -> 22x22x128 -> 11x11x128
        W_conv3 = weight_variable([3,3,64,128],regularizer)
        b_conv3 = bias_variable([128])
        conv3 = conv2d(pool2,W_conv3)
        relu3 = tf.nn.relu(conv3+b_conv3)
        pool3 = max_pool_2x2(relu3)
        node_size = 11 * 11 * 128
        reshaped = tf.reshape(pool3, [-1, node_size])

    with tf.variable_scope("layer4"): #第4层 全连接层
        W_fc1 = weight_variable([node_size,512],regularizer)
        b_fc1 = bias_variable([512])
        fc1 = tf.matmul(reshaped,W_fc1)+b_fc1
        relu4 = tf.nn.relu(fc1)

    with tf.variable_scope("layer5"):  # 第5层 全连接层
        W_fc2 = weight_variable([512, 512],regularizer)
        b_fc2 = bias_variable([512])
        fc2 = tf.matmul(relu4, W_fc2) + b_fc2
        relu5 = tf.nn.relu(fc2)
        relu5_drop = tf.nn.dropout(relu5,keep_prob=keepprob)

    with tf.variable_scope("layer6"):
        W_fc3 = weight_variable([512,30],regularizer)
        b_fc3 = bias_variable([30])
        output = tf.matmul(relu5_drop,W_fc3)+b_fc3

    return output


# 训练算法
REGULARIZATION_RATE = 0.0001
LEARNING_RATE = 0.001
TRAIN_SIZE = 1000 #验证集大小
N_EPOCH = 500 #迭代次数
EARLY_STOP_PATIENCE = 20 #若往后在训练这么多轮测试集的结果不改变，则证明过拟合，应该早停止
BATCH_SIZE = 32
INF = 100000000000.0
def minibatch(inputs,labels,batch_size,shuffle=False): #每次在训练集取出一批数据,最后一个参数代表打乱(训练集需要打乱)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], labels[excerpt]
def train_data():
    x = tf.placeholder(tf.float32,[None,W,H,D],name = "x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = "y-input")
    keep_prob = tf.placeholder(tf.float32,name="keep_prob")
    X, y = load()
    X = X.reshape((-1, 96, 96, 1))
    TRAIN_SIZE = int(X.shape[0] * 0.8)  # 百分之80的数据做训练集
    x_train, y_train = X[:TRAIN_SIZE], y[:TRAIN_SIZE]
    x_valid, y_valid = X[TRAIN_SIZE:], y[TRAIN_SIZE:]

    current_epoch = 0
    # 定义正则化器及前向传播过程
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    res = inference(x, regularizer=regularizer,keepprob=keep_prob)
    #res = model(x,keep_prob)
    #保存模型
    b = tf.constant(value=1, dtype=tf.float32)
    res_save = tf.multiply(res, b, name='y_save')
    # 定义误差 (均方根误差)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(res-y_))) + tf.add_n(tf.get_collection("losses"))

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                     TRAIN_SIZE/BATCH_SIZE*4, 0.98,staircase=True)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # 训练器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(rmse,global_step=global_step)
    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())

    best_validation_error = INF

    saver = tf.train.Saver()

    for epoch in range(N_EPOCH):
        x_train, y_train = shuffle(x_train, y_train, random_state=42)  # 随机打乱 X,y
        train_loss,n_batch = 0, 0
        for x_train_a, y_train_a in minibatch(x_train, y_train, BATCH_SIZE, shuffle=True):
            _, err = sess.run([train_step,rmse], feed_dict={x: x_train_a, y_: y_train_a,keep_prob : 0.5})
            train_loss += err;
            n_batch += 1
        print("learning rate : ",sess.run(learning_rate))
        validate_loss = 0
        for x_valid_a, y_valid_a in minibatch(x_valid, y_valid, BATCH_SIZE, shuffle=True):
            err = sess.run(rmse, feed_dict={x: x_valid_a, y_: y_valid_a,keep_prob : 1.0})
            validate_loss += err;
            n_batch += 1
        print("validate loss : ",validate_loss*48+48)


        if validate_loss<best_validation_error: #保存测试集误差最小的模型
            best_validation_error = validate_loss
            current_epoch = epoch
            saver.save(sess,save_path)
        elif epoch-current_epoch>EARLY_STOP_PATIENCE:
            print("early stop!")
            break
    print(sess.run(res, feed_dict={x: X[0:1]}))
    sess.close()


# 测试集
def showpic(X,y):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)  # 设置子区域的边框间隔
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        img = X[i].reshape((96,96))
        ax.imshow(img, cmap='gray')
        ax.scatter(y[i][0::2] * 48 + 48, y[i][1::2] * 48 + 48, s=10, c='r', marker='x')
    plt.show()

keypoint_dict = {
        'left_eye_center_x':0,
        'left_eye_center_y':1,
        'right_eye_center_x':2,
        'right_eye_center_y':3,
        'left_eye_inner_corner_x':4,
        'left_eye_inner_corner_y':5,
        'left_eye_outer_corner_x':6,
        'left_eye_outer_corner_y':7,
        'right_eye_inner_corner_x':8,
        'right_eye_inner_corner_y':9,
        'right_eye_outer_corner_x':10,
        'right_eye_outer_corner_y':11,
        'left_eyebrow_inner_end_x':12,
        'left_eyebrow_inner_end_y':13,
        'left_eyebrow_outer_end_x':14,
        'left_eyebrow_outer_end_y':15,
        'right_eyebrow_inner_end_x':16,
        'right_eyebrow_inner_end_y':17,
        'right_eyebrow_outer_end_x':18,
        'right_eyebrow_outer_end_y':19,
        'nose_tip_x':20,
        'nose_tip_y':21,
        'mouth_left_corner_x':22,
        'mouth_left_corner_y':23,
        'mouth_right_corner_x':24,
        'mouth_right_corner_y':25,
        'mouth_center_top_lip_x':26,
        'mouth_center_top_lip_y':27,
        'mouth_center_bottom_lip_x':28,
        'mouth_center_bottom_lip_y':29
}
def mytest():
    X,y = load(test=True)
    x_test = X.reshape((-1, 96, 96, 1))
    y_pred = []

    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph("./model3/model.meta")
    saver.restore(sess,save_path)
    x = sess.graph.get_tensor_by_name("x-input:0")
    net = sess.graph.get_tensor_by_name("y_save:0")
    keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
    TEST_SIZE = x_test.shape[0]


    for idx in range(0,TEST_SIZE,BATCH_SIZE):
        y_batch = sess.run(net, feed_dict={x: x_test[idx:idx+BATCH_SIZE],keep_prob:1.0})
        y_pred.extend(y_batch)

    showpic(X[32:48],y_pred[32:48])
    print("test image predict has done!")

    output_file = open("SampleSubmission.csv","w")
    output_file.write("RowId,Location\n")

    IdLookupTable = open("IdLookupTable.csv")
    IdLookupTable.readline()

    for line in IdLookupTable:
        RowId, ImageId, FeatureName = line.rstrip().split(',')
        imageid = int(ImageId)-1
        featureid = keypoint_dict[FeatureName]
        feature_location = y_pred[imageid][featureid]*48+48
        if feature_location<0: feature_location = 0
        elif feature_location>96 :feature_location = 96
        output_file.write("{0},{1}\n".format(RowId,feature_location))

    output_file.close()
    IdLookupTable.close()
    sess.close()


def main():
    #train_data()
    mytest()

if __name__ == "__main__":
    main()


