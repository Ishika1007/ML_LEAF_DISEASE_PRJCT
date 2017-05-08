# coding: utf-8

# # Annotations for the Sirajology Python NN Example
# 
# This code comes from a demo NN program from the YouTube video https://youtu.be/h3l4qz76JhQ. The program creates an neural network that simulates the exclusive OR function with two inputs and one output. 
# 
# 

# In[23]:

import numpy as np  # Note: there is a typo on this line in the video

# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate. 
# 
# The implementation of this function does double duty. If the deriv=True flag is passed in, the function instead calculates the derivative of the function, which is used in the error backpropogation step. 

# In[24]:

#import openpyxl


def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))
    
    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video


# The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the bias term and is not part of the input. 

# In[25]:

#b = openpyxl.load_workbook('Train.xlsx')
#type(wb)
X = np.array([
[0.30028667,	0.863010651,	0.203012789,	0.86881893,	102.8537599,	46.24576532,	7.52211964,	15.91789225,	918.5834672,	2.998034312,	0.291624192],
[0.279788087,	0.901439966,	0.19760853,	0.917894589,	119.8105267,	43.31752976,	7.25510284,	15.96770953,	1202.130528,	4.031290556,	0.53784907],
#[0.273595216,	0.862206545,	0.218812762,	0.908355893,	124.6806634,	42.44161013,	7.274046258,	15.96830244,	1009.211755,	3.807920191,	0.597221939],
#[0.232727273,	0.907807285,	0.193553454,	0.892669856,	100.8666095,	38.16742415,	7.243519935,	15.92042393,	855.3164189,	2.673260133,	-0.02796292],
#[0.225544275,	0.967831408,	0.134072798,	0.902087207,	94.72867397,	56.84669046,	7.442911614,	15.79086218,	1820.172199,	1.698241576,	-0.042240627],
#[0.124086873,	0.970673248,	0.170648135,	0.952410646,	108.3088974,	48.4906068,	7.475355584,	15.94046338,	1587.762756,	2.282837742,	0.343183869],
#[0.122807205,	0.978012586,	0.152806646,	0.954501531,	119.0383872,	53.99513185,	7.645874337,	15.57625587,	1865.414721,	2.673926791,	-0.413806931],
[0.532661711,	0.954498815,	0.396477104,	0.886489046,	186.1208171,	93.07716221,	4.1274721,	15.69124254,	5910.948845,	1.848799909,	-0.773672764],
[0.327366076,	0.797399871,	0.210078758,	0.868728729,	110.672677,	40.09606884,	7.314909828,	15.9287476,	705.8810616,	2.57114264,	-0.261829914]
]
)



#input data
#X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
 #           [0,1,1],
  #          [1,0,1],
   #         [1,1,1]])


# The output of the exclusive OR function follows. 

# In[26]:

#output data
y = np.array([[0],
             [0],
#             [0],
#             [0],
#              [0],
#              [1],
#             [1],
             [1],
             [1]])


# The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

# In[27]:

np.random.seed(1)


# Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example. 

# In[28]:

#synapses
syn0 = 2*np.random.random((3,4)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((4,1)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.


# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases. 

# In[29]:

#training step
# Python2 Note: In the follow command, you may improve 
#   performance by replacing 'range' with 'xrange'. 
for j in range(60000):  
    
    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    
    # Back propagation of errors using the chain rule. 
    l2_error = y - l2
    if(j % 10000) == 0:   # Only print the error every 10000 steps, to save time and limit the amount of output. 
        print("Error: " + str(np.mean(np.abs(l2_error))))
        
    l2_delta = l2_error*nonlin(l2, deriv=True)
    
    l1_error = l2_delta.dot(syn1.T)
    
    l1_delta = l1_error * nonlin(l1,deriv=True)
    
    #update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
print("Output after training")
print(l2)
    
    


# See how the final output closely approximates the true output [0, 1, 1, 0]. If you increase the number of interations in the training loop (currently 60000), the final output will be even closer. 

# In[30]:

#get_ipython().run_cell_magic(u'HTML', u'', u'#The following line is for embedding the YouTube video \n#   in this Jupyter Notebook. You may remove it without peril. \n<iframe width="560" height="315" src="https://www.youtube.com/embed/h3l4qz76JhQ" frameborder="0" allowfullscreen></iframe>')


# In[ ]:
