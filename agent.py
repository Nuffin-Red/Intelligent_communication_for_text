import numpy as np
import tensorflow as tf
import reset_environment
import os
from memory import replay_memory
import sys
import math


my_config = tf.ConfigProto()
my_config.gpu_options.allow_growth=True

class DQN_Agent(object):
    def __init__(self, state_size,discrete_actions_size,continuou_action_size):
        self.discount = 0.99
        self.double_q = True
        self.state_size = state_size
        self.discrete_action_size = discrete_actions_size
        self.continuou_action_size = continuou_action_size
        self.memory = replay_memory(self.state_size,discrete_actions_size,continuou_action_size)

class DDPG_Agent(object):
    def __init__(self, state_size, discrete_actions_size, continuou_action_size):
        self.discount = 0.99
        self.double_q = True
        self.state_size = state_size
        self.discrete_action_size = discrete_actions_size
        self.continuou_action_size = continuou_action_size
        self.memory = replay_memory(self.state_size, discrete_actions_size, continuou_action_size)

fadding_factor_hd=3.75
fadding_factor_BR=fadding_factor_Ru=2.2
[M,N,C,K,K1,K2,M1,M2,N1,N2,B,PT]=[6,16,7,3,10,10,3,2,4,4,10e5,3]
snr_min=(1e-4)*np.asarray([10, 8.5, 9])         #10,10,10
h_d=reset_environment.hd_channels(fadding_factor_hd,M,N,K,C).Large_scale_fadding()
h_BR=reset_environment.Reflection_path(fadding_factor_BR,fadding_factor_Ru
                                       ,M,N,K,C,K1,K2,N1,N2,M1,M2).BS_IRS_fadding()
h_Ru=reset_environment.Reflection_path(fadding_factor_BR,fadding_factor_Ru
                                       ,M,N,K,C,K1,K2,N1,N2,M1,M2).IRS_user_fadding()

n_episode =200
epsi_final = 0.01           #最终的探索率0.02
n_step_per_episode=120
epsi_anneal_length = int(0.35*n_episode)         #探索率变化的前80%集0.8,0.65
mini_batch_step = 120            #每轮步数
target_update_step =120*4

def get_state(C,K,M,N,PT,h_d,h_Ru,h_BR,N1,N2,B,snr_min):

    state=reset_environment.reset_init_state(C,K,M,N,PT,h_d,h_Ru,h_BR,N1,N2,B,snr_min).get_init_state()
    return state

#==============================DQN网络================================
dqn_hiddle_1=300
dqn_hiddle_2=200
dqn_hiddle_3=100
dqn_input=ddpg_a_input=C+N+C*M+2*C*M+1+2*K
dqn_output=K

#===========================DDPG======================================
ddpg_a_hiddle_1=300
ddpg_a_hiddle_2=100
ddpg_a_output=N+M*C
ddpg_c_input=ddpg_a_input+ddpg_a_output
ddpg_c_hiddle_1=300
ddpg_c_hiddle_2=100
ddpg_c_output=1

g = tf.Graph()
with g.as_default():
    #===============================DQN_eval_network=============================
    x_dqn = tf.placeholder(tf.float32, [None, dqn_input])

    w_1 = tf.Variable(tf.truncated_normal([dqn_input, dqn_hiddle_1], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([dqn_hiddle_1, dqn_hiddle_2], stddev=0.1))
    w_3 = tf.Variable(tf.truncated_normal([dqn_hiddle_2, dqn_hiddle_3], stddev=0.1))
    w_4 = tf.Variable(tf.truncated_normal([dqn_hiddle_3, dqn_output], stddev=0.1))

    b_1 = tf.Variable(tf.truncated_normal([dqn_hiddle_1], stddev=0.1))
    b_2 = tf.Variable(tf.truncated_normal([dqn_hiddle_2], stddev=0.1))
    b_3 = tf.Variable(tf.truncated_normal([dqn_hiddle_3], stddev=0.1))
    b_4 = tf.Variable(tf.truncated_normal([dqn_output], stddev=0.1))

    layer_1 =tf.sigmoid(tf.add(tf.matmul(x_dqn, w_1), b_1))
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, w_2), b_2))
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2, w_3), b_3))
    y = tf.sigmoid(tf.add(tf.matmul(layer_3, w_4), b_4))
    g_q_action = tf.argmax(y, axis=1)

    # compute loss
    g_target_q_t = tf.placeholder(tf.float32, None, name="target_value")
    g_action = tf.placeholder(tf.int32, None, name='g_action')
    action_one_hot = tf.one_hot(g_action, dqn_output, 1.0, 0.0, name='action_one_hot')
    q_acted = tf.reduce_sum(y*action_one_hot, reduction_indices=1, name='q_acted')
    q_acted_= tf.reduce_sum(q_acted)
    g_loss = tf.reduce_mean(tf.square(g_target_q_t - q_acted_), name='g_loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_loss)



    g_target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
    target_q_with_idx = tf.gather_nd(y, g_target_q_idx)

    #===========================DDPG_eval_actor_network================================
    x_ddpg_a = tf.placeholder(tf.float32, [None, ddpg_a_input])
    with tf.variable_scope('Actor'):
        with tf.variable_scope('eval_net'):
            w_ddpg_a_1 = tf.Variable(tf.truncated_normal([ddpg_a_input, ddpg_a_hiddle_1], stddev=0.1))
            w_ddpg_a_2 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_1, ddpg_a_hiddle_2], stddev=0.1))
            w_ddpg_a_3 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_2, ddpg_a_output], stddev=0.1))

            b_ddpg_a_1 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_1], stddev=0.1))
            b_ddpg_a_2 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_2], stddev=0.1))
            b_ddpg_a_3 = tf.Variable(tf.truncated_normal([ddpg_a_output], stddev=0.1))

            layer_ddpg_a_1 = tf.sigmoid(tf.add(tf.matmul(x_ddpg_a, w_ddpg_a_1), b_ddpg_a_1))
            layer_ddpg_a_2 = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_a_1, w_ddpg_a_2), b_ddpg_a_2))
            ddpg_actor_out = tf.nn.tanh(tf.add(tf.matmul(layer_ddpg_a_2, w_ddpg_a_3), b_ddpg_a_3))

    #============================DDPG_target_actor_network================================
    x_ddpg_a_t = tf.placeholder(tf.float32, [None, ddpg_a_input])
    with tf.variable_scope('Actor'):
        with tf.variable_scope('target_net'):

            w_ddpg_a_t_1 = tf.Variable(tf.truncated_normal([ddpg_a_input, ddpg_a_hiddle_1], stddev=0.1))
            w_ddpg_a_t_2 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_1, ddpg_a_hiddle_2], stddev=0.1))
            w_ddpg_a_t_3 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_2, ddpg_a_output], stddev=0.1))

            b_ddpg_a_t_1 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_1], stddev=0.1))
            b_ddpg_a_t_2 = tf.Variable(tf.truncated_normal([ddpg_a_hiddle_2], stddev=0.1))
            b_ddpg_a_t_3 = tf.Variable(tf.truncated_normal([ddpg_a_output], stddev=0.1))

            layer_ddpg_a_t_1 = tf.sigmoid(tf.add(tf.matmul(x_ddpg_a_t, w_ddpg_a_t_1), b_ddpg_a_t_1))
            layer_ddpg_a_t_2 = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_a_t_1, w_ddpg_a_t_2), b_ddpg_a_t_2))
            ddpg_actor_t_out = tf.nn.tanh(tf.add(tf.matmul(layer_ddpg_a_t_2, w_ddpg_a_t_3), b_ddpg_a_t_3))
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
    # ================================DDPG_eval_critic_network==============================
    x_ddpg_s_c = tf.placeholder(tf.float32, [None, ddpg_a_input])
    x_ddpg_a_c = tf.placeholder(tf.float32, [None, ddpg_a_output])

    w_ddpg_c_s_1 = tf.Variable(tf.truncated_normal([ddpg_a_input, ddpg_c_hiddle_1], stddev=0.1))
    w_ddpg_c_a_1 = tf.Variable(tf.truncated_normal([ddpg_a_output, ddpg_c_hiddle_1], stddev=0.1))
    w_ddpg_c_2 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_1, ddpg_c_hiddle_2], stddev=0.1))
    w_ddpg_c_3 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_2, ddpg_c_output], stddev=0.1))

    b_ddpg_c_1 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_1], stddev=0.1))
    b_ddpg_c_2 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_2], stddev=0.1))
    b_ddpg_c_3 = tf.Variable(tf.truncated_normal([ddpg_c_output], stddev=0.1))

    layer_ddpg_c_1 = tf.sigmoid(
        tf.add(tf.matmul(x_ddpg_s_c, w_ddpg_c_s_1) + tf.matmul(x_ddpg_a_c, w_ddpg_c_a_1), b_ddpg_c_1))
    layer_ddpg_c_2 = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_c_1, w_ddpg_c_2), b_ddpg_c_2))
    y_ddpg_c = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_c_2, w_ddpg_c_3), b_ddpg_c_3))

    #critic_loss
    g_target_ddpg_c_t = tf.placeholder(tf.float32, None, name="ddpg_target_value")      #
    g_ddpg_c_loss = tf.reduce_mean(tf.square(g_target_ddpg_c_t - y_ddpg_c), name='ddpg_g_loss')
    optim_c = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01).minimize(g_ddpg_c_loss)
    #actor_update
    a_grads = tf.gradients(y_ddpg_c, ddpg_actor_out)
    policy_grads = tf.gradients(ys=ddpg_actor_out, xs=e_params, grad_ys=a_grads)
    opt = tf.train.AdamOptimizer(-0.001)
    train_op = opt.apply_gradients(zip(policy_grads, e_params))

    # ================================DDPG_target_critic_network==============================
    x_ddpg_c_s_t = tf.placeholder(tf.float32, [None, ddpg_a_input])
    x_ddpg_a_a_c = tf.placeholder(tf.float32, [None, ddpg_a_output])

    w_ddpg_c_t_s_1 = tf.Variable(tf.truncated_normal([ddpg_a_input, ddpg_c_hiddle_1], stddev=0.1))
    w_ddpg_c_t_a_1 = tf.Variable(tf.truncated_normal([ddpg_a_output, ddpg_c_hiddle_1], stddev=0.1))
    w_ddpg_c_t_2 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_1, ddpg_c_hiddle_2], stddev=0.1))
    w_ddpg_c_t_3 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_2, ddpg_c_output], stddev=0.1))

    b_ddpg_c_t_1 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_1], stddev=0.1))
    b_ddpg_c_t_2 = tf.Variable(tf.truncated_normal([ddpg_c_hiddle_2], stddev=0.1))
    b_ddpg_c_t_3 = tf.Variable(tf.truncated_normal([ddpg_c_output], stddev=0.1))

    layer_ddpg_c_t_1 = tf.sigmoid(tf.add(tf.matmul(x_ddpg_c_s_t, w_ddpg_c_t_s_1)+tf.matmul(x_ddpg_a_a_c, w_ddpg_c_t_a_1), b_ddpg_c_t_1))
    layer_ddpg_c_t_2 = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_c_t_1, w_ddpg_c_t_2), b_ddpg_c_t_2))
    y_ddpg_c_target = tf.sigmoid(tf.add(tf.matmul(layer_ddpg_c_t_2, w_ddpg_c_t_3), b_ddpg_c_t_3))

    init = tf.global_variables_initializer()

def update_target_DDPG_network(sess):

    sess.run(w_ddpg_a_t_1.assign(sess.run(w_ddpg_a_1)))
    sess.run(w_ddpg_a_t_2.assign(sess.run(w_ddpg_a_2)))
    sess.run(w_ddpg_a_t_3.assign(sess.run(w_ddpg_a_3)))


    sess.run(b_ddpg_a_t_1.assign(sess.run(b_ddpg_a_1)))
    sess.run(b_ddpg_a_t_2.assign(sess.run(b_ddpg_a_2)))
    sess.run(b_ddpg_a_t_3.assign(sess.run(b_ddpg_a_3)))

    sess.run(w_ddpg_c_t_s_1.assign(sess.run(w_ddpg_c_s_1)))
    sess.run(w_ddpg_c_t_a_1.assign(sess.run(w_ddpg_c_a_1)))
    sess.run(w_ddpg_c_t_2.assign(sess.run(w_ddpg_c_2)))
    sess.run(w_ddpg_c_t_3.assign(sess.run(w_ddpg_c_3)))

    sess.run(b_ddpg_c_t_1.assign(sess.run(b_ddpg_c_1)))
    sess.run(b_ddpg_c_t_2.assign(sess.run(b_ddpg_c_2)))
    sess.run(b_ddpg_c_t_3.assign(sess.run(b_ddpg_c_3)))

def dqn_learning_mini_batch(current_agent, current_sess):
    batch_s_t, batch_s_t_add_1, batch_d_action, batch_c_action, batch_reward = current_agent.memory.sample()
    pred_action = current_sess.run(g_q_action, feed_dict={x_dqn: batch_s_t_add_1})
    yy = current_sess.run(y, feed_dict={x_dqn: batch_s_t_add_1})
    q_t_add_1 = current_sess.run(target_q_with_idx, {x_dqn: batch_s_t_add_1,
                                                      g_target_q_idx: [[idx, pred_a] for idx, pred_a in
                                                                       enumerate(pred_action)]})
    batch_target_q_t = current_agent.discount * q_t_add_1 + batch_reward
    _, loss_val_dqn = current_sess.run([optim, g_loss],{g_target_q_t: batch_target_q_t, g_action: batch_d_action, x_dqn: batch_s_t})
    return loss_val_dqn

def ddpg_critic_learning_mini_batch(current_agent, current_sess):
    batch_s_t, batch_s_t_add_1, batch_d_action, batch_c_action, batch_reward = current_agent.memory.sample()
    a=batch_c_action
    #a = tf.stop_gradient(batch_c_action)
    batch_c_action_t_add_1=current_sess.run(ddpg_actor_t_out, feed_dict={x_ddpg_a_t: batch_s_t_add_1})          #actor_target
    y_ddpg_c_target_t_add_1=current_sess.run(y_ddpg_c_target, feed_dict={x_ddpg_c_s_t: batch_s_t_add_1,x_ddpg_a_a_c:batch_c_action_t_add_1})        #得到critic_target网络St+1时刻Q值
    batch_target_q_ddpg_c_t = current_agent.discount * y_ddpg_c_target_t_add_1 + batch_reward           #目标值
    _, loss_val_critic = current_sess.run([optim_c, g_ddpg_c_loss],
                                   {g_target_ddpg_c_t: batch_target_q_ddpg_c_t, x_ddpg_s_c: batch_s_t,x_ddpg_a_c:a})
    return loss_val_critic

def ddpg_actor_learning_mini_batch(current_agent, current_sess):
    batch_s_t, batch_s_t_add_1, batch_d_action, batch_c_action, batch_reward = current_agent.memory.sample()
    #batch_critic_eval_input_t = np.concatenate((batch_s_t, batch_c_action),axis=1)
    policy_grad,_=current_sess.run([policy_grads, train_op],
                                   {x_ddpg_a: batch_s_t, x_ddpg_s_c: batch_s_t,x_ddpg_a_c:batch_c_action})
    return policy_grad

def predict_dqn(sess, s_t, ep):
    if np.random.rand() < ep:
        pred_action = np.random.randint(K)
    else:
        pred_action = sess.run(g_q_action, feed_dict={x_dqn: [s_t]})[0]
    return pred_action

def predict_ddpg(sess, s_t, var):
    pred_action = sess.run(ddpg_actor_out, feed_dict={x_ddpg_a:[s_t]})[0]
    pred_action = np.clip(np.random.normal(pred_action, var), -1, 1)
    return pred_action


#初始化智能体
agents = []
sesses = []
for ind_agent in range(C+1):  # initialize agents
    print("Initializing agent", ind_agent)
    if ind_agent==C:
        agent = DDPG_Agent(dqn_input, 1, ddpg_a_output)        #42+16,把连续动作存到离散里面了
    else:
        agent = DQN_Agent(dqn_input,1,ddpg_a_output)
    agents.append(agent)

    sess = tf.Session(graph=g,config=my_config)
    sess.run(init)
    sesses.append(sess)

record_reward = np.zeros([n_episode*n_step_per_episode, 1])
reward_plot_data=[]
record_loss = []



var=2
for i_episode in range(n_episode):
    print("-------------------------")
    print('Episode:', i_episode)
    var *= 0.985
    rr=0
    if i_episode < epsi_anneal_length:
        epsi = 0.12 - i_episode * (0.12- epsi_final) / (epsi_anneal_length - 1)
    else:
        epsi = epsi_final
    d_action_all_training = np.zeros((1, C), dtype='int32')
    c_action_all_training = np.zeros((1, ddpg_a_output), dtype='float64')
    state = get_state(C, K, M, N, PT, h_d, h_Ru, h_BR, N1, N2,
                      B, snr_min)
    for i_step in range(n_step_per_episode):

        time_step = i_episode * n_step_per_episode + i_step
        state_old_all = []
        d_action_all = []
        c_action_all = []
        state_old_all.append(state)
        for k in range(C+1):
            if k<C:
                d_action = predict_dqn(sesses[k], state, epsi)
                d_action_all.append(d_action)
                d_action_all_training[0,k]=d_action
            else:

                c_action = predict_ddpg(sesses[k], state,var)
                d_action_all.append(9)
                c_action_all.append(c_action)
                c_action_all_training=c_action_all
        c_action_temp = c_action_all_training.copy()
        d_action_temp = d_action_all_training.copy()
        train_reward,_,sn,sum_rate,next_state=reset_environment.calculation_SNR_and_get_next_state\
            (snr_min,h_d, h_Ru, h_BR,B,M,K,N,C,PT).SNR_and_get_next_state(d_action_temp,c_action_temp)

        record_reward[time_step] = train_reward
        rr+=sum_rate
        reward_plot_data.append(train_reward)
        state=next_state

        if i_step == n_step_per_episode-1:
            print('Episode_sum_rate:',rr)
        for c in range(C + 1):
            state_old = state_old_all[0]
            d__action = d_action_all[c]         #可以改，将没有信道用户的随机挑选
            c__action = c_action_all[0]
            state_new = next_state
            if c < C:
                agents[c].memory.add(state_old, state_new,train_reward,c__action,d__action)     #对应连续变量是个数组，只能填进一个位置，虽然有58个数
            else:
                agents[c].memory.add(state_old, state_new, train_reward, c__action, d__action)

            if time_step % mini_batch_step == mini_batch_step - 1:
                if c<C:
                    loss_val_batch = dqn_learning_mini_batch(agents[c], sesses[c])

                else:
                    loss_val_batch = ddpg_actor_learning_mini_batch(agents[c], sesses[c])
                    loss_critic_batch = ddpg_critic_learning_mini_batch(agents[c], sesses[c])
                if c==0:
                    print('step:', time_step, 'agent', 'DQN_1', 'loss', loss_val_batch)
                elif c==C:
                    print('step:', time_step, 'agent', 'DDPG', 'loss', loss_critic_batch)

            if time_step % target_update_step == target_update_step - 1:
                if c==C:
                    update_target_DDPG_network(sesses[c])
                    print('Update target DDPG network...')

for sess in sesses:
    sess.close()

