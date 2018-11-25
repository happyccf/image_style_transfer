[TOC]

# 1 风格化概念

​        在计算机视觉领域，纹理合成一直是常见的问题。风格可以作为纹理的一种扩展，所以风格迁移实际上也是纹理合成的一种扩展。风格迁移的目标是将风格图中的风格提取出来，并应用到另一张内容图中。所以风格迁移涉及到风格提取与风格重建。参考知乎上的一张图，可以将风格迁移总结一下。

![style_transfer](/figures/style_transfer.jpg)




# 2 慢速和快速风格化综述

## 2.1 慢速参数化的风格迁移

### 2.1.1 Image Style Transfer Using Convolutional Neural Network

**（*CVPR 2016，* 基于神经网络的风格迁移的开山大作）**

#### 2.1.1.1 简介

​        **摘要**：使用CNN(Vgg)网络对输入的风格图片和内容图片分别进行低维+高维的风格提取重建、高维内容抽取重建，然后控制loss重建新的风格图片。

​        **注意点**：每次训练迭代更新的不是VGG网络本身参数，而是随机初始化的输入x；由于输入x是随机初始化的，最终得到的“合成画”会有差异；每次生成一幅“合成画”都需重新训练，速度慢。

#### 2.1.1.2 损失函数

​        **内容loss**：CNN网络的较高层表达是较为抽象的图像内容，所以根据比较两幅图高层特征的相似性来比较图像内容相似性；这里使用欧式距离得到内容loss：

   ![ist_using_cnn_content_loss](/figures/ist_using_cnn_content_loss.png)

​        **风格loss**：比较两幅图的风格纹理特征可以比较他们在CNN网络中的低层特征；但是低层特征不仅包含纹理特征还包含高很多图像局部内容特征，会影响比较结果，这里使用Gram矩阵计算不同响应层的联系，同时去除图像内容的影响。基于此即可定义风格loss：

   ![ist_using_cnn_gram_matrix](/figures/ist_using_cnn_gram_matrix.png)

   ![ist_using_cnn_style_loss1](/figures/ist_using_cnn_style_loss1.png)

   ![ist_using_cnn_style_loss2](/figures/ist_using_cnn_style_loss2.png)

​        **总loss**：权衡内容loss和风格loss即可得到最终的loss：

![ist_using_cnn_total_loss](/figures/ist_using_cnn_total_loss.png)

####  2.1.1.3 训练过程

![ist_using_cnn_train1](/figures/ist_using_cnn_train1.png)

​                                                               图1 CNN网络中低维纹理特征和高维内容特征的表达



![ist_using_cnn_train2](/figures/ist_using_cnn_train2.png)

​                                                                    图2 Image Style Transfer算法流程



​        在实际实验中，内容层提取提取的特征来自**conv4_2**，风格层使用的特征来自**conv1_1，conv2_1，conv3_1，conv4_1，conv5_1**。训练过程中，将白噪声图像x输入到网络，根据以上层中提取的特征计算loss，使用梯度下降法更新x，最终得到结果。

#### 2.1.1.4 实验结果说明   

​        内容和风格的权衡，$\alpha$ 和 $\beta$ 系数比不同会有不同的效果；不同层次的特征捕捉到的特征也不相同；不同初始化的方式也对应着不同的结果；也可以迁移一些照片的风格等等。

#### 2.1.1.5 算法复现

基于keras和tensorflow的实现：/home/caichengfei/image_style_transfer/slow_online_parametric/Neural-Style-Transfer-keras/INetwork.py

### 2.1.2 Laplacian-Steered Neural Style Transfer

**（*ACM MM 2017，*针对Gatys的风格迁移中，内容图中的边缘等细节难以保留，增加了LaplacianLoss）**

#### 2.1.2.1 简介

​        **摘要**：基于Gatys的Neural Style Transfer在提取内容图片的高维特征时，会舍弃一些低维的边缘信息等，此时风格图中的一些低维的边缘信息将会在的合成的风格图中占主导。本文通过增加新的关于内容Laplacian loss约束，修正合成的图。

#### 2.1.2.2 损失函数

​         除了**内容loss**和**风格loss**，增加了一项**拉普拉斯Loss**：另D为拉普拉斯滤波器，那么图像x的拉普拉斯操作为D(x)，基于此定义拉普拉斯Loss如下。

![Laplacian_ist_Dfilter](/figures/Laplacian_ist_Dfilter.png)

![Laplacian_ist_lap_loss](/figures/Laplacian_ist_lap_loss.png)

​         **总loss**：

![Laplacian_ist_total_loss](/figures/Laplacian_ist_total_loss.png)

#### 2.1.2.3 训练过程

![Laplacian_ist_train](/figures/Laplacian_ist_train.png)

​                                                         图3  Lapstyle风格网络结构

​        额外需要考虑的是，在RGB图像的3个channels都需要进行拉普拉斯操作，然后取3个通道值的和作为最终的D(x)；另一点需要注意的是，为了防止小的扰动带来的影响，对输入的图片首先进行了PxP的均值平滑操作。

#### 2.1.2.4 实验结果说明   

​       不同的P均值平滑和$\gama$会带来不同的结果，以及使用不同的Laplacian算子也会有不同的效果。

#### 2.1.2.5 算法复现

​        基于theano，https://github.com/askerlee/lapstyle

### 2.1.3 Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses

**（*arXiv 2017，*针对Gatys的风格迁移中，风格loss的Gram矩阵不稳定导致风格合成图纹理模糊需要大量人工参数调整等，提出利用直方图来修正）**

#### 2.1.3.1 简介 

​        **摘要：**论文解释了为什么Gram矩阵会导致合成图的不稳定情况发生，然后针对此，提出了使用直方图loss取解决这个文图，然后利用多尺度的生成优化了一些参数和生成图的质量。 

#### 2.1.3.2 损失函数 

​          **Gram矩阵的不稳定性**：下图4是基于Gatys纹理生成结果的不稳定性展示，这种不稳定性可以通过调参去解决，但是经常需要大量调参。产生这种问题的理论解释如（5）和（6）所示，对应的样例如图5所示。

![instabilities_formulation](/figures/Histogram_ist_instabilities_formulation.png)

 

![instabilities](/figures/Histogram_ist_instabilities.png)

​                                                        图4 Gatys提出的方法中的不稳定因素需要人为调参解决



​                                                     ![instabilities_reson](/figures/Histogram_ist_instabilities_reason.png)

​                                          图5 当两幅图标准差和均值平方之和相等，Gram矩阵的结果就是一样的 

​        **解决方法，直方图loss: ** 同时参考前人的经验也增加了total variation loss作为风格纹理loss中的一部分。定义直方图loss和风格纹理loss如下：

![histogram_loss](/figures/Histogram_ist_histogram_loss.png)

![texture_loss](/figures/Histogram_ist_texture_loss.png)

​        **总loss:**

![total_loss](/figures/Histogram_ist_total_loss.png)

## 2.2 慢速无参MRF风格迁移

### 2.2.1 Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis

**（*CVPR 2016，*针对Gatys的风格迁移中，风格loss的Gram矩阵考虑的是纹理像素级的特征，这里使用包含纹理局部区域特征的MRF regularizer代替，提升合成图的质量）**           

#### 2.2.1.1 简介 

​        **摘要：**Gatys提出的风格迁移方法中，Gram矩阵考虑的是像素级的特征关系，空间布局限制不能被很好补货，所以严格的局部真实感很难保留。本文将MRF和dCNN结合起来，将Gram矩阵用MRF loss进行了代替，利用CNN提取内容特征的能力和MRF的空间布局限制，提高图像生成的质量。 

​        **注意点：**MRF虽然解决了局部区域的视觉可信度，但并没有考虑到整体全局布局，这里使用多分辨率金字塔来解决这个问题。

#### 2.2.1.2 损失函数 

​          **MRFs loss：**这里我们使用relu3_1和 relu4_1层的feature map记为Φ(x)计算MRFloss，Ψ(Φ(x))为从Φ(x)抽取来的patch，尺寸为k * k * C，k为patch宽和高，C为channel数。所以基于MRF的style loss如下，NN(i)是使用归一的互相关性来找到最匹配的patch。

![MRF_ist_style_loss](/figures/MRF_ist_style_loss.png)

![MRF_ist_style_loss2](/figures/MRF_ist_style_loss2.png)

​        **内容loss：**依然使用relu4_2层的欧式距离计算loss。

![MRF_ist_content_loss](/figures/MRF_ist_content_loss.png)

​        **正则项：**在网络的训练过程中，有图像部分的细节损失，会引起噪声和非自然现象的产生，使用平方梯度正则进行平滑。

![MRF_ist_regularizer_loss](/figures/MRF_ist_regularizer_loss.png)

​        **总loss：**最小化以上loss的加权和，使用                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             BFGS，。

![MRF_ist_total_loss](/figures/MRF_ist_total_loss.png)

#### 2.2.1.3 实验细节

​        **MRF实现：**将MRFloss用卷积反向传播解决，从relu3_1和relu4_1抽取3x3，步长为1的稠密patchs。匹配过程用一个卷积层实现，把patch作为filter，那么对query patch有最大响应的就是最好的匹配patch。

​        **多分辨率金字塔：**构建了一个图像金字塔，不停的以倍数2缩放，直到最长边的长度小于64为止。然后将上一步的生成图像放大2倍，作为下一个分辨率的生成图像的初始值。为了避免内容图和风格图的视野和尺度的影响，对style Image做了多尺寸和多旋转变化。

​        **结果：**

![MRF_ist_result](/figures/MRF_ist_result.png)

​                                                                        图6 实验结果与Gatys比较

#### 2.2.1.4 技术限制

​        只在内容图像中的patch和风格图像中的patch可以吻合的比较好的时候才会有效。否则，那就会倾向于把整幅图都变化了，如下图所示。

![MRF_ist_result_limit](/figures/MRF_ist_result_limit.png)



#### 2.2.1.5 源码

https://github.com/chuanli11/CNNMRF

### 2.2.2 Arbitrary Style Transfer with Deep Feature Reshuffle

**（*CVPR 2018，*有参风格生成模型能很好学习style整体特征，但是很能保证局部纹理结构；无参风格生成模型可以很好保证loc  ，但无法保证对整体的模仿，基于此使用reshuffle方式解决问题）**           

#### 2.2.2.1 简介 

​        **摘要：**目前常用的风格化迁移中，有参模型更多考虑的是特征的全局    统计信息， 无参模型则更多匹配样本的局部patch，双方各有优缺点，本文则是提出使用deep feature reshuffle集合两种模型的优点。   

#### 2.2.2.2 损失函数

​        **Reshuffle Loss：**定义了新的style的损失Reshuffle Loss，同时控制源风格图中的每个patch只能被使用一次，具体如下：

![Reshuffle_ist_reshuffle_loss](/figures/Reshuffle_ist_reshuffle_loss.png)

![Reshuffle_ist_reshuffle_loss2](/figures/Reshuffle_ist_reshuffle_loss2.png)

​        **总loss：**所以最终需要优化的目标就是：

![Reshuffle_ist_total_loss](/figures/Reshuffle_ist_total_loss.png)

#### 2.2.2.3 训练过程

​        实际训练中，使用了多层渐进优化。

![Reshuffle_ist_train](/figures/Reshuffle_ist_train.png)

#### 2.2.2.4 源码

​    https://github.com/msracver/Style-Feature-Reshuffle

## 2.3 快速单一网络单一风格迁移

### 2.3.1 Texture Networks: Feed-forward Synthesis of Textures and Stylized Images

**（*ICML 2016*，Gatys风格图生产耗时~10s左右，远远不能在工业中落地，本文针对一个风格训练一个生成网络，这样只需前馈就可以完成该风格生成图的制作，耗时~20ms）**     

#### 2.3.1.1 简介 

​        **摘要：**Gatys风格图效果虽然好，但是耗时太久。针对此，本文训练了CNN的生成网络，训练时将噪声和内容图输入，生成纹理图，使用类似于Gatys的loss函数反馈回来，更新参数；测试时，只需要将噪声和内容图输入到训练好的网络，即可生成某种风格的生成图。

​        **注意点：**训练完成一组参数对应着一种风格；输入是随机噪声和内容图在channal-wise的concatenate；输入进行了多尺度得到下采样。

#### 2.3.1.2 损失函数

​         这里使用的loss参考Gatys提出的texture loss和content loss。

![Texture_network_loss](/figures/Texture_network_loss.png)

#### 2.3.1.3 训练过程

​        训练需要更新的参数是一个CNN的生成网络g，用来计算loss的Descriptor网络是固定的vgg19；只训练纹理合成时，只使用texture loss，训练内容+纹理合成时，使用加权的texture 和 content loss。

![Texture_network_train](/figures/Texture_network_train.png)

#### 2.3.1.4 实验细节与结果

​         实验发现输入为多尺度结构时，效果更好，同时输入噪声和内容图是channel-wise的concat；使用了circular convolution消除边界效应；插入了BN层在生成网络也会有好的效果提升；最终的训练完成之后，生成一张图大约~20ms，相比Gatys的~10s提升了500倍。实验结果如下：

![Texture_network_result](/figures/Texture_network_result.png)

#### 2.3.1.5 算法复现

源码：

TensorFlow https://github.com/tgyg-jegli/tf_texture_net 

Torch https://github.com/DmitryUlyanov/texture_nets

Tensorflow版本的有前馈网络但没有训练网络，lua版本的有训练网络。

### 2.3.2 Perceptual Losses for Real-Time Style Transfer and Super-Resolution

**（*ECCV 2016*，同样为了解决Gatys中生成图耗时太久的问题，类似于上一篇使用了训练好的前馈网络生成）**     

#### 2.3.2.1 简介 

​        **摘要：**使用像素级的loss监督训练的CNN网络为图像转换提供了一种思路，同时使用perceptual loss 函数能够生成高质量的图像。本文结合两者，提出了一种快速的风格迁移的方法，同时能够提升输入图像的像素。

​        **注意点：**网络实现了两个任务风格迁移和超像素；训练的只是Transform网络，计算loss的网络为已经训练好的VGG16。

#### 2.3.2.2 损失函数

​         这里使用的loss参考Gatys提出的texture loss和content loss，另外增加了一些简单的loss。

![Perceptual_total_loss](/figures/Perceptual_total_loss.png)

#### 2.3.2.3 训练过程

​        网络结构如下图所示。

![Perceptual_train](/figures/Perceptual_train.png)

#### 2.3.2.4 实验细节与结果

​        在图像转换的网络上，没有使用不使用pooling层，而是使用strided和fractionally strided卷积来做downsampling和upsampling；使用了五个残差模块等等，具体可以参考原文。

​        两个任务中，对于风格转换来说，输入和输出的大小都是256×256×3；对于图片清晰化来说，输出是288×288×3，而输入是288/f×288/f×3，f是压缩比。

​        对于风格转化的视觉效果和花费时间如下(与Gatys方法的对比)，时间大约提升3个量级

![Perceptual_result1](/figures/Perceptual_result1.png)

![Perceptual_result1_1](/figures/Perceptual_result1_1.png)

   对于超像素，实验结果如下：

![Perceptual_result2](/figures/Perceptual_result2.png)

  #### 2.3.2.5 代码复现

源码：

TensorFlow https://github.com/lengstrom/fast-style-transfer

Torch https://github.com/jcjohnson/fast-neural-style

TensorFlow有训练过程，可以较好的参考。

### 2.3.3 Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks

**（*ECCV 2016*，提出了MGANs来合成纹理图片，也是训练一个前向网络加速合成图的生成，利用纹理的无参统计量）**     

#### 2.3.3.1 简介 

​        **摘要：**深度Markovian模型能够产生比传统MRF更好的效果，但是却很耗时，为了改善这一点提出了 马尔科夫生成对抗网络MGANs。论文利用strided卷积网络捕获Markovian patches的统计特征，训练了一个前馈网络。

​        **注意点：**网络合成的图片可是任意尺寸的；但是限制和上面的模型一样，都是一个网络只用合成一种风格的图像。

#### 2.3.3.2 损失函数

​        论文使用的loss，依旧根以上一致，使用了比较内容的content loss，比较纹理的texture loss， 和平滑项r(x)loss。需要注意的是，这里使用GAN的结构，使得挑选texture patch变成了一个判别网络，即判断生成图的patch是为和example的patch相似，通过控制判别器的输出score学习纹理特征。

![MGANs_texture_loss](/figures/MGANs_texture_loss.png)

![MGANs_total_loss](/figures/MGANs_total_loss.png)

 #### 2.3.3.3 训练过程

​        网络结构如下图所示。

![MGANs_train](/figures/MGANs_train.png)

​        灰色的部分是参考vgg19网络，固定参数的。网络通过学习生成网络和判别网络的参数，最小化上述的loss。判别网络D使用最大边缘准则Hinge loss来代替Sigmoid function 和二值交叉熵能有效地避免梯度消失；MGANs跟GANs的区别在于并没有对一整张图片进行操作，而是在同一张图片的patch上进行操作。这样做有利于利用patch之间的上下文联系，跟学习整个对象类的分布或学习上下文无关的数据映射相比这样能让learning更容易也更有效。

​        具体的一些其他细节可以参考论文。

#### 2.3.3.4 实验结果

​        作者分别控制G和D不动，调整D和G，查看G和D的中间结果，选择合适的特征层计算loss。最终的结果如下。

![MGANs_result](/figures/MGANs_result.png)

#### 2.3.3.5 源码

基于torch https://github.com/chuanli11/MGANs

#### 2.3.3.6 想法

​        快速的有参和无参的生成模型各有优缺点，在18年cvpr中融合两者的慢速网络可以用来借鉴，发表一篇**融合两者的快速的生成模型**。

## 2.4 单一模型多种风格的快速生成方法

### 2.4.1 A Learned Representation for Artistic Style

**（*ICLR 2017*，基于李飞飞和Johnson提出的2.3.1和2.3.2的结构网络，增加了条件实例归一化层CIN，CIN存着关于多种风格化的scaled和transform参数。这样的话只需要更新CIN的参数，其他内容的参数保持固定即可实现多种风格图的生成）**     

#### 2.4.1.1 简介 

​        **摘要：**单一风格快速的风格迁移相比于Gatys提出的方法，速度增加了百倍甚至千倍，但是由于移动设备内存与计算能力的限制，如果每种风格都用一个网络，肯定是无法实用的。本文针对这个问题，通过单个scalable的深度网络来描述多种风格。

​        **注意点：**通过增加了条件实例归一化层conditional instance normalization，来学习多种风格。每一个CIN层的参数是2NC，N为style数量，C为feature map数量。模型总计1.6M参数，只有3K(~总数的0.2%)是关于风格的CIN参数。

#### 2.4.1.2 损失函数

​        网络的损失函数和以前的网络一样，定义内容损失和风格损失，然后通过mirror-padding和最近邻上采样消除了total variation的影响。

![Learning_representation_loss](/figures/Learning_representation_loss.png)

![Learning_representation_loss2](/figures/Learning_representation_loss2.png)

#### 2.4.1.3 网络结构

​        网络中需要注意的是CIN层的定义，CIN本质上把一个feature通过一些平移和缩放，变到另一个分布特征上，公式如下。

![Learning_representation_CIN](/figures/Learning_representation_CIN.png)

![Learning_representation_CIN2](/figures/Learning_representation_CIN2.png)

​        总的网络结构如表所示：

![Learning_representation_network](/figures/Learning_representation_network.png)

#### 2.4.1.4 训练细节和实验结果

​        训练N个风格和单个风格相比，收敛速度慢一些，但是能够捕获到不同的风格，而且能通过参数调整产生新的风格。

![Learning_representation_result](/figures/Learning_representation_result.png)

#### 2.4.1.5 源码

TensorFlow https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization

### 2.4.2 StyleBank: An Explicit Representation for Neural Image Style Transfer

**（*CVPR 2017*，类似于上一篇的内容，这里使用了Style Bank，而不是CIN，提出auto encode和decode的结构，进行多风格的迁移）**     

#### 2.4.2.1 简介 

​        **摘要：**本文提出了Style Bank（多个卷积滤波器banks组成），每一组bank对应着一种风格。然后提供了一个风格的显示表达，将Style Bank和auto-encoder联合训练，利用前向网络进行风格图的生成。

​        **注意点：**可以进行区域风格的融合；增量学习新的风格。

#### 2.4.2.2 损失函数

​        在auto-encoder分支中，利用MSE计算input与output之间的loss。

![Stylebank_autoencoder_loss](/figures/Stylebank_autoencoder_loss.png)

​        在stylize分支中，使用的loss和前人用的loss一致，contentloss用的是VGG16的relu4_2，styleloss用的是relu1_2 relu2_2 relu3_2 relu4_2。

![Stylebank_stylebranch_loss](/figures/Stylebank_stylebranch_loss.png)

#### 2.4.2.3 网络结构

​        网络主要分为3个模块，2个分支。其中auto-encoder分支将内容图分解到多个featuremap中，然后和style分支中的Stylebank结合，由stylize分支生成最终的风格图。

![Stylebank_network](/figures/Stylebank_network.png)

#### 2.4.2.4 训练细节与实验结果

​        训练中使用了 T+1 step迭代训练策略，来平衡两个分支的训练过程。具体训练算法如下，这里T=2，lambda=1：

![Stylebank_train](/figures/Stylebank_train.png)

​       增加新的风格时，只需要重新训练style分支，更新bank参数即可。另外训练完成的Stylebank可以进行线性融合或者区域融合。

![Stylebank_result_linearfusion](/figures/Stylebank_result_linearfusion.png)

![Stylebank_result_regionfusion](/figures/Stylebank_result_regionfusion.png)

![Stylebank_result_compare](/figures/Stylebank_result_compare.png)



### 2.4.3 Diversified Texture Synthesis With Feed-Forward Networks

**（*CVPR 2017*，与以上两篇的出发点相似，用前馈网络加速生成风格图，但是完全只需要一个网络，和以上两篇不同，网络不会跟着风格增多而变大）**     

#### 2.4.3.1 简介 

​        **摘要：**以往的基于feed-forward的纹理生成算法缺少泛华能力，而且缺少差异性，不是最优解。本文针对这些，提出了使用一个固定网络产生多风格纹理的算法。

​        **注意点：**更新了Gram矩阵loss；提出了差异性loss；增量学习训练策略。

#### 2.4.3.2 损失函数

​        **纹理loss**：类似于以往的纹理loss，本文也是用Gram矩阵，但是Gram矩阵无法很好的区分风格1与风格2之间的差异，这里使用了减均值的Gram矩阵。传统的Gram矩阵与减均值的Gram矩阵的效果如下图所示。

![Diversified_network_texture_loss](/figures/Diversified_network_texture_loss.png)

![Diversified_network_texture_loss2](/figures/Diversified_network_texture_loss2.png)

![Diversified_network_gram_result](/figures/Diversified_network_gram_result.png)

​                                         图 中间一行传统Gram生成结果，下面一行减均值的Gram生成结果

​        **差异性loss**：对于纹理生成，以往不增加差异性loss时，导致了输出视觉上差异性不大，而且于输入的噪声没有关系。这里增加了差异性loss。效果如下。

![Diversified_network_diversity_loss](/figures/Diversified_network_diversity_loss.png)

![Diversified_network_diversity_result](/figures/Diversified_network_diversity_result.png)

​                               图 中间列是没加差异性loss时生成纹理差异性不大；右侧是加入了差异性loss

​         **总loss**：

![Diversified_network_total_loss](/figures/Diversified_network_total_loss.png)

#### 2.4.3.3 网络结构

​        多纹理生成网络主要两部分组成，生成网络和选择网络。生成网络用来生成纹理图，选择器网络则输入one-hot形式的向量来表示各种风格，指导生成网络，网络结构如下。

![Diversified_network_texture_network](/figures/Diversified_network_texture_network.png)

​        当用来生成风格图片时，需要对网络做一些调整，网络如下所示。

![Diversified_network_style_network](/figures/Diversified_network_style_network.png)

#### 2.4.3.4 训练细节与实验结果

​        如果将所有的风格一起训练，会导致各种风格之间的参数相互影响，导致参数更新效率变低。这里训练中使用了增量学习训练策略，首先训练一种风格的参数，然后逐渐增加风格，更新风格参数，训练策略如下所示。

![Diversified_network_style_train](/figures/Diversified_network_style_train.png)

​        另外，各种风格之间可以进行插值处理，生成新得风格结果。最终的风格结果如图所示。

![Diversified_network_texture_result](/figures/Diversified_network_texture_result.png)

![Diversified_network_style_result](/figures/Diversified_network_style_result.png)

![Diversified_network_style_result2](/figures/Diversified_network_style_result2.png)

#### 2.4.3.5 代码实现

Torch实现 https://github.com/Yijunmaverick/MultiTextureSynthesis

## 2.5 单一模型任意风格的快速生成方法

### 2.5.1 Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

**（*ICCV 2017*，第一次提出了一种实时任意风格转化技术，核心是使用了AdaIN自适应实例归一化层，来将内容图的均值方差迁移到风格图上的均值方差）**     

#### 2.5.1.1 简介 

​        **摘要：**以往基于优化过程的风格化虽然能处理任意风格，但速度极慢；基于前向网络的风格化虽然速度快，但是风格数量受限；受到instance normalization启发，本文提出的风格迁移网络，基于AdaIN层将内容图特征和风格图特征对齐，第一次使得任意风格生成算法达到实时。

​        **注意点：**Fast patch-based提出的任意风格转化大部分时间花在了style swap上，导致了算法无法实时使用。

#### 2.5.1.2 损失函数

​        **内容loss：**本文使用AdaIN的输出t与最终生成的风格图经过f编码得到后的f(g(t))的欧式距离作为内容loss，可以加速收敛。

![AdaIN_network_content_loss.png](/figures/AdaIN_network_content_loss.png)

​        **风格loss:** 通过比较生成图与风格图在vgg网络中某些层的feature map对应的均值和方差的差异。

![AdaIN_network_style_loss.png](/figures/AdaIN_network_style_loss.png)

​        **总loss: **

![AdaIN_network_total_loss.png](/figures/AdaIN_network_total_loss.png)

#### 2.5.1.3 网络结构

​        网络主要包括encoder、AdaIN和Decoder几块，如下图所示。

![AdaIN_network_train.png](/figures/AdaIN_network_train.png)

​        其中，edcoder是VGG19网络中的一些固定层，decoder通过训练学习到风格图，AdaIN层则是用来将内容和风格对准的层，公式如下。与以前的BN、IN和CIN不同，AdaIN没有需要学习的放射变化系数，用风格y的方差来缩放尺度用均值来平移。

![AdaIN_network_adain_layer](/figures/AdaIN_network_adain_layer.png)

​        

#### 2.5.1.4 训练细节与实验结果

​        训练中使用MSCOCO作为内容图像，WikiArt作为风格；每组80000张数据，使用adam，batchsize为8，resize成N * 512后，使用256 * 256crop。结果和速度如下：

![AdaIN_network_result1.png](/figures/AdaIN_network_result1.png)

![AdaIN_network_result2.png](/figures/AdaIN_network_result2.png)

​        另外，用户可以自己控制例如内容与风格得到均衡、风格与风格间的插值、空间与颜色的控制等等。

#### 2.5.1.5 源码

作者torch版：https://github.com/xunhuang1995/AdaIN-style

其他TensorFlow版本：https://github.com/eridgd/AdaIN-TF

### 2.5.2 Exploring the Structure of a Real-time, Arbitrary Neural Artistic Stylization Network

**（*BMVC 2017*，另一篇数据驱动的ASPM，通过一个专门的风格预测网络，来预测CIN层风格的仿射变换参数）**     

#### 2.5.2.1 简介 

​        **摘要：**通过大量数据，训练一个style predict network，学习每个风格仿射变换参数得到向量S，然后输入到风格迁移网络T，得到最终的风格合成图。

​        **注意点：**能够生成任意风格的图片，对未在训练集中的风格泛华能力强；但是确定在于需要大量训练集去训练，不然效果会降低。

#### 2.5.2.2 损失函数

​        **风格与内容loss：**本文使用loss函数类似于Gatys使用的loss，基于Gram的风格loss和欧式距离的内容loss。

![Exploring_stylenetwork_loss1](/figures/Exploring_stylenetwork_loss1.png)

​        **总loss:** 最终优化的问题，则是Ls与Lc之间的权衡。

![Exploring_stylenetwork_loss2](/figures/Exploring_stylenetwork_loss2.png)

#### 2.5.2.3 网络结构

![Exploring_stylenetwork_train](/figures/Exploring_stylenetwork_train.png)

​        网络主要包含风格预测网络P和风格迁移网络T。P的作用是从style中学习得到CIN正则参数，传入到迁移网络T中。其中预测网络借鉴于Inception-V3，风格迁移网络借鉴于2.4.1Dumoulin的 A learned representation for artistic style，计算loss的网络依然参考VGG网络。     

#### 2.5.2.4 训练细节与实验结果

​        文章关于训练的细节可以参考文章附录A6。文章在大约8w张风格图和6k张纹理图训练，得到的结果泛化能力强，可以在任意风格或者纹理，包括未见过的风格。

![Exploring_stylenetwork_result](/figures/Exploring_stylenetwork_result.png)

​        文章发现，增加训练风格数量，可以有效的提升泛华能力。训练的风格预测网络可以很好的捕获到风格结构。

#### 2.5.2.5 源码

 基于TensorFlow的源码，很详细https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization

### 2.5.3 Universal Style Transfer via Feature Transforms

**（*NIPS 2017*，不需要通过大量学习训练提取放射变换系数，单纯使用了白化和颜色迁移操作WCT，将内容图的风格信息去掉，用颜色迁移将风格中的色彩迁移上去，得到风格图）**     

#### 2.5.3.1 简介 

​        **摘要：**以往的大多数基于feed-forward的模型，大多数不能泛华到所有风格。本文使用了一个简单有效的方法，whitening & coloring transform（WCT）操作，将内容图中的特征协方差映射到风格中。

​        **注意点：**不需要大量训练学习，就能生成任意风格的图片；用户可控，包括风格权重和局部迁移。

#### 2.5.3.2 损失函数

​        **重建网络的loss：**本文需要训练是重建网络中的decoder部分参数，WCT操作不需要学习训练，重建网络中的encoder部分取自于VGG-19；decoder部分与VGG-19相似，使用了最近邻的上采样，loss为输入输出的图像像素loss和feature map间的loss。

![Universal_WCT_decoder_loss](/figures/Universal_WCT_decoder_loss.png)

#### 2.5.3.3 网络结构

![Universal_WCT_train](/figures/Universal_WCT_train.png)

​        网络首先训练了五个重建网络中的decoder，X=1,2...5；然后固定该网络，在每个 encoder和decoder之间加入一个WCT，在5个层匹配内容和风格的特征统计值。

​        WCT操作主要包含了Whitening转换和Color转换，其中White转换主要将输入的图像剥去风格保留内容，然后通过颜色迁移，将风格的颜色等信息匹配到内容图上。白化操作的过程，先将feature map减均值，然后取方差特征，结果如下。

![Universal_WCT_white_transfer](/figures/Universal_WCT_white_transfer.png)

![Universal_WCT_white_result](/figures/Universal_WCT_white_result.png)

​        然后进行颜色迁移，主要步骤如下，也可以控制alpha来调整风格权重。

![Universal_WCT_color_transfer](/figures/Universal_WCT_color_transfer.png)

![Universal_WCT_color_transfer2](/figures/Universal_WCT_color_transfer2.png)

![Universal_WCT_transfer](/figures/Universal_WCT_transfer.png)

#### 2.5.3.4 训练细节与实验结果

​        文章训练decoder使用Microsoft Cocodataset，得到的风格图效果如下。另外用户可以控制，生成过程方便高效等。

![Universal_WCT_result1](/figures/Universal_WCT_result1.png)

![Universal_WCT_result2](/figures/Universal_WCT_result2.png)

#### 2.5.3.5 源码

官方torch源码  https://github.com/Yijunmaverick/UniversalStyleTransfer

 基于TensorFlow的源码，很详细  https://github.com/eridgd/WCT-TF

### 2.5.4 Avatar-Net: Multi-Scale Zero-Shot Style Transfer by Feature Decoration

**（*CVPR 2018*，任意风格转化中，生成效果和效率之间的权衡一直是备受关注的一点。本文提出了一种Avatar-Net，其中使用了风格修饰器style decorator来修饰内容，使内容和风格语义一致。）**     

#### 2.5.4.1 简介 

​        **摘要：**本文提出一种新的风格迁移模型Avatar-Net，使用基于patch的修饰器使得内容特征和风格模式特征在语义上一致。主要包括以下三步，1）通过encoder模块抽取内容与风格特征；2）基于patch的修饰器，修饰内容特征；3）多尺度逐级的风格特征的decode。

​        **注意点：**基于patch的风格修饰使得生成图的内容和风格结合的更近；多尺度网络使得生成图效果更真实；提出的方法效率和质量更高。

#### 2.5.4.2 损失函数

​        **目标loss：**loss主要为像素的欧式距离，层特征map的欧式距离以及总变差组成。

![Avaternet_loss](/figures/Avaternet_loss.png)

#### 2.5.4.3 网络结构

![Avaternet_train](/figures/Avaternet_train.png)

​        网络encode采用了VGG19的一些结构，decode则是类似于encode结构随机初始化参数。Style transfer模块使用了一个Style Decorator，彻底将使得生成图完美匹配风格的特征统计如下图。

![Avaternet_style_decorater.png](/figures/Avaternet_style_decorater.png)

​        风格修饰主要包括三个步骤：1）归一化映射；2）匹配和重新组装；3）重建。对应着下面的3个公式。

![Avaternet_decorater_step1.png](/figures/Avaternet_decorater_step1.png)

![Avaternet_decorater_step2.png](/figures/Avaternet_decorater_step2.png)

![Avaternet_decorater_step3.png](/figures/Avaternet_decorater_step3.png)

#### 2.5.4.4 训练细节与实验结果 

​        网络训练在MSCOCO datatset的80000张训练样本数，ADAM优化，cropped 256*256。

![Avaternet_result](/figures/Avaternet_result.png)

![Avaternet_result2](/figures/Avaternet_result2.png)

#### 2.5.4.5 源码

tensorflow详细：https://github.com/LucasSheng/avatar-net                

​      

# 3 TO-DO：

#### 1）2.5.2.5 源码

 基于TensorFlow的源码，很详细https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization                                                                                                                                                                                                                                                                                          

#### 2）2.5.1.5 源码 

其他TensorFlow版本：https://github.com/eridgd/AdaIN-TF

#### 3）2.5.3.5 源码

 基于TensorFlow的源码，很详细  https://github.com/eridgd/WCT-TF

#### 4）2.5.4.5 源码

TensorFlow详细 https://github.com/LucasSheng/avatar-net          

#### 5) 老照片上色源码

other 老照片上色，https://github.com/jantic/DeOldify



​                                                                                                                                                                                                                         # image_style_transfer
