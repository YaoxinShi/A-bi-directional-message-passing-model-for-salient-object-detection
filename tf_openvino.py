#####################################################
# (1) create TF model
#####################################################

print(">>> create TF model")
import MainModel as MM
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

model = MM.Model()
model.build_model()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state('model')
saver = tf.train.Saver()
saver.restore(sess, ckpt.model_checkpoint_path)

# refer to:
# https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html
# https://stackoverflow.com/questions/47267636/tensorflow-how-do-i-find-my-output-node-in-my-tensorflow-trained-model
# https://blog.csdn.net/Murdock_C/article/details/87281575
# https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/

# from the code,
#       sal_map,result = sess.run([model.Score,model.Prob], feed_dict={model.input_holder: img})
#       self.Score = tf.reshape(prev1, [-1, 2])
#       self.Prob = tf.nn.softmax(self.Score)
# we know the output is reshape and softmax

# by print all nodes, we know their name are "Reshape" and "Softmax"
print(">>> show all nodes")
for op in sess.graph.get_operations(): 
    print(op.name, op.outputs)

# Or we can get from pbtxt
#   node {
#     name: "Reshape"
#     op: "Reshape"
#     input: "fusion/add_11"
#     input: "Reshape/shape"
#   }
#   node {
#     name: "Softmax"
#     op: "Softmax"
#     input: "Reshape"
#   }
#print(">>> save pbtxt")
#tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir='.', name='out.pbtxt', as_text=True)
#print(">>> save pb")
#from tensorflow.python.tools import freeze_graph
#freeze_graph.freeze_graph('out.pbtxt', input_saver='', input_binary=False, input_checkpoint='./model/model.ckpt', output_node_names='Reshape, Softmax', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph='out.pb', clear_devices=True, initializer_nodes='')

print(">>> save pb")
from tensorflow.python.framework import graph_io
frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Reshape", "Softmax"])
graph_io.write_graph(frozen, './', 'out.pb', as_text=False)        
    
#####################################################
# (2) convert onnx to OpenVINO
#####################################################

print(">>> convert TF to OpenVINO")
import os
os.system('python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model out.pb --data_type FP16 --model_name BDMP_FP16')

# from "https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html"
# we can use MetaGraph directly.
#   python3 mo_tf.py --input_meta_graph <INPUT_META_GRAPH>.meta
# but I haven't test it.
