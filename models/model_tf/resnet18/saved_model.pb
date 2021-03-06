??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
E
AssignSubVariableOp
resource
value"dtype"
dtypetype?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b5??

initNoOp
w
dataPlaceholder*$
shape:?????????@@*
dtype0*/
_output_shapes
:?????????@@
f
batch_normalization/ConstConst*
_output_shapes
:*
valueB*  ??*
dtype0
?
*batch_normalization/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *+
_class!
loc:@batch_normalization/beta
?
batch_normalization/betaVarHandleOp*
dtype0*+
_class!
loc:@batch_normalization/beta*
_output_shapes
: *
	container *)
shared_namebatch_normalization/beta*
shape:
?
9batch_normalization/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
?
batch_normalization/beta/AssignAssignVariableOpbatch_normalization/beta*batch_normalization/beta/Initializer/zeros*+
_class!
loc:@batch_normalization/beta*
dtype0
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*+
_class!
loc:@batch_normalization/beta*
dtype0
?
1batch_normalization/moving_mean/Initializer/zerosConst*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0*
valueB*    *
_output_shapes
:
?
batch_normalization/moving_meanVarHandleOp*
dtype0*
shape:*
_output_shapes
: *
	container *0
shared_name!batch_normalization/moving_mean*2
_class(
&$loc:@batch_normalization/moving_mean
?
@batch_normalization/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
?
&batch_normalization/moving_mean/AssignAssignVariableOpbatch_normalization/moving_mean1batch_normalization/moving_mean/Initializer/zeros*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
dtype0*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean
?
4batch_normalization/moving_variance/Initializer/onesConst*
valueB*  ??*
_output_shapes
:*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
shape:*6
_class,
*(loc:@batch_normalization/moving_variance*4
shared_name%#batch_normalization/moving_variance*
	container *
dtype0
?
Dbatch_normalization/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
?
*batch_normalization/moving_variance/AssignAssignVariableOp#batch_normalization/moving_variance4batch_normalization/moving_variance/Initializer/ones*6
_class,
*(loc:@batch_normalization/moving_variance*
dtype0
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
\
keras_learning_phase/inputConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
|
keras_learning_phasePlaceholderWithDefaultkeras_learning_phase/input*
_output_shapes
: *
shape: *
dtype0

x
batch_normalization/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

q
!batch_normalization/cond/switch_tIdentity!batch_normalization/cond/Switch:1*
_output_shapes
: *
T0

o
!batch_normalization/cond/switch_fIdentitybatch_normalization/cond/Switch*
_output_shapes
: *
T0

c
 batch_normalization/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
'batch_normalization/cond/ReadVariableOpReadVariableOp0batch_normalization/cond/ReadVariableOp/Switch:1*
_output_shapes
:*
dtype0
?
.batch_normalization/cond/ReadVariableOp/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
_output_shapes
: : *+
_class!
loc:@batch_normalization/beta*
T0
?
batch_normalization/cond/ConstConst"^batch_normalization/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
 batch_normalization/cond/Const_1Const"^batch_normalization/cond/switch_t*
valueB *
_output_shapes
: *
dtype0
?
'batch_normalization/cond/FusedBatchNormFusedBatchNorm0batch_normalization/cond/FusedBatchNorm/Switch:12batch_normalization/cond/FusedBatchNorm/Switch_1:1'batch_normalization/cond/ReadVariableOpbatch_normalization/cond/Const batch_normalization/cond/Const_1*
T0*G
_output_shapes5
3:?????????@@::::*
data_formatNHWC*
epsilon%?ŧ7*
is_training(
?
.batch_normalization/cond/FusedBatchNorm/SwitchSwitchdata batch_normalization/cond/pred_id*J
_output_shapes8
6:?????????@@:?????????@@*
T0*
_class
	loc:@data
?
0batch_normalization/cond/FusedBatchNorm/Switch_1Switchbatch_normalization/Const batch_normalization/cond/pred_id* 
_output_shapes
::*
T0*,
_class"
 loc:@batch_normalization/Const
?
)batch_normalization/cond/ReadVariableOp_1ReadVariableOp0batch_normalization/cond/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
?
0batch_normalization/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization/beta batch_normalization/cond/pred_id*
T0*
_output_shapes
: : *+
_class!
loc:@batch_normalization/beta
?
8batch_normalization/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOp?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes
:*
dtype0
?
?batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitchbatch_normalization/moving_mean batch_normalization/cond/pred_id*
_output_shapes
: : *
T0*2
_class(
&$loc:@batch_normalization/moving_mean
?
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpAbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:
?
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch#batch_normalization/moving_variance batch_normalization/cond/pred_id*
_output_shapes
: : *6
_class,
*(loc:@batch_normalization/moving_variance*
T0
?
)batch_normalization/cond/FusedBatchNorm_1FusedBatchNorm0batch_normalization/cond/FusedBatchNorm_1/Switch2batch_normalization/cond/FusedBatchNorm_1/Switch_1)batch_normalization/cond/ReadVariableOp_18batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1*
is_training( *
epsilon%?ŧ7*
data_formatNHWC*
T0*G
_output_shapes5
3:?????????@@::::
?
0batch_normalization/cond/FusedBatchNorm_1/SwitchSwitchdata batch_normalization/cond/pred_id*J
_output_shapes8
6:?????????@@:?????????@@*
T0*
_class
	loc:@data
?
2batch_normalization/cond/FusedBatchNorm_1/Switch_1Switchbatch_normalization/Const batch_normalization/cond/pred_id*
T0* 
_output_shapes
::*,
_class"
 loc:@batch_normalization/Const
?
batch_normalization/cond/MergeMerge)batch_normalization/cond/FusedBatchNorm_1'batch_normalization/cond/FusedBatchNorm*1
_output_shapes
:?????????@@: *
N*
T0
?
 batch_normalization/cond/Merge_1Merge+batch_normalization/cond/FusedBatchNorm_1:1)batch_normalization/cond/FusedBatchNorm:1*
_output_shapes

:: *
N*
T0
?
 batch_normalization/cond/Merge_2Merge+batch_normalization/cond/FusedBatchNorm_1:2)batch_normalization/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes

:: 
z
!batch_normalization/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization/cond_1/switch_tIdentity#batch_normalization/cond_1/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization/cond_1/switch_fIdentity!batch_normalization/cond_1/Switch*
_output_shapes
: *
T0

e
"batch_normalization/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
 batch_normalization/cond_1/ConstConst$^batch_normalization/cond_1/switch_t*
_output_shapes
: *
valueB
 *?p}?*
dtype0
?
"batch_normalization/cond_1/Const_1Const$^batch_normalization/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
 batch_normalization/cond_1/MergeMerge"batch_normalization/cond_1/Const_1 batch_normalization/cond_1/Const*
T0*
N*
_output_shapes
: : 
?
)batch_normalization/AssignMovingAvg/sub/xConst*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
'batch_normalization/AssignMovingAvg/subSub)batch_normalization/AssignMovingAvg/sub/x batch_normalization/cond_1/Merge*
T0*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
: 
?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
)batch_normalization/AssignMovingAvg/sub_1Sub2batch_normalization/AssignMovingAvg/ReadVariableOp batch_normalization/cond/Merge_1*
T0*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean
?
'batch_normalization/AssignMovingAvg/mulMul)batch_normalization/AssignMovingAvg/sub_1'batch_normalization/AssignMovingAvg/sub*2
_class(
&$loc:@batch_normalization/moving_mean*
_output_shapes
:*
T0
?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpbatch_normalization/moving_mean'batch_normalization/AssignMovingAvg/mul*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
?
4batch_normalization/AssignMovingAvg/ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean8^batch_normalization/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:*2
_class(
&$loc:@batch_normalization/moving_mean*
dtype0
?
+batch_normalization/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*6
_class,
*(loc:@batch_normalization/moving_variance*
_output_shapes
: *
dtype0
?
)batch_normalization/AssignMovingAvg_1/subSub+batch_normalization/AssignMovingAvg_1/sub/x batch_normalization/cond_1/Merge*
_output_shapes
: *6
_class,
*(loc:@batch_normalization/moving_variance*
T0
?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
dtype0*
_output_shapes
:
?
+batch_normalization/AssignMovingAvg_1/sub_1Sub4batch_normalization/AssignMovingAvg_1/ReadVariableOp batch_normalization/cond/Merge_2*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance*
T0
?
)batch_normalization/AssignMovingAvg_1/mulMul+batch_normalization/AssignMovingAvg_1/sub_1)batch_normalization/AssignMovingAvg_1/sub*
T0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance
?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp#batch_normalization/moving_variance)batch_normalization/AssignMovingAvg_1/mul*
dtype0*6
_class,
*(loc:@batch_normalization/moving_variance
?
6batch_normalization/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp#batch_normalization/moving_variance:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*
_output_shapes
:*6
_class,
*(loc:@batch_normalization/moving_variance
?
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
?
zero_padding2d/PadPadbatch_normalization/cond/Mergezero_padding2d/Pad/paddings*
T0*
	Tpaddings0*/
_output_shapes
:?????????FF
?
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@conv2d/kernel*%
valueB"         @   *
_output_shapes
:
?
,conv2d/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *?N?*
dtype0* 
_class
loc:@conv2d/kernel
?
,conv2d/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?N>*
dtype0* 
_class
loc:@conv2d/kernel
?
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
T0*

seed * 
_class
loc:@conv2d/kernel*&
_output_shapes
:@*
seed2 *
dtype0
?
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
T0*
_output_shapes
: 
?
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel*
T0
?
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
:@* 
_class
loc:@conv2d/kernel
?
conv2d/kernelVarHandleOp*
	container *
shared_nameconv2d/kernel* 
_class
loc:@conv2d/kernel*
_output_shapes
: *
dtype0*
shape:@
k
.conv2d/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d/kernel*
_output_shapes
: 
?
conv2d/kernel/AssignAssignVariableOpconv2d/kernel(conv2d/kernel/Initializer/random_uniform* 
_class
loc:@conv2d/kernel*
dtype0
?
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0* 
_class
loc:@conv2d/kernel*&
_output_shapes
:@
e
conv2d/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
r
conv2d/Conv2D/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:@*
dtype0
?
conv2d/Conv2DConv2Dzero_padding2d/Padconv2d/Conv2D/ReadVariableOp*
data_formatNHWC*
strides
*/
_output_shapes
:?????????  @*
	dilations
*
T0*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID
?
,batch_normalization_1/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0*
_output_shapes
:@*
valueB@*  ??
?
batch_normalization_1/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_1/gamma*,
shared_namebatch_normalization_1/gamma*
shape:@*
_output_shapes
: *
dtype0*
	container 
?
<batch_normalization_1/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
?
"batch_normalization_1/gamma/AssignAssignVariableOpbatch_normalization_1/gamma,batch_normalization_1/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_1/gamma*
dtype0
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
dtype0*
_output_shapes
:@*.
_class$
" loc:@batch_normalization_1/gamma
?
,batch_normalization_1/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*-
_class#
!loc:@batch_normalization_1/beta*
valueB@*    
?
batch_normalization_1/betaVarHandleOp*
dtype0*
shape:@*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: *
	container *+
shared_namebatch_normalization_1/beta
?
;batch_normalization_1/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
?
!batch_normalization_1/beta/AssignAssignVariableOpbatch_normalization_1/beta,batch_normalization_1/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_1/beta*
dtype0
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
dtype0*
_output_shapes
:@*-
_class#
!loc:@batch_normalization_1/beta
?
3batch_normalization_1/moving_mean/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
!batch_normalization_1/moving_meanVarHandleOp*2
shared_name#!batch_normalization_1/moving_mean*
shape:@*
	container *
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_1/moving_mean
?
Bbatch_normalization_1/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
?
(batch_normalization_1/moving_mean/AssignAssignVariableOp!batch_normalization_1/moving_mean3batch_normalization_1/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
6batch_normalization_1/moving_variance/Initializer/onesConst*
valueB@*  ??*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
	container *
_output_shapes
: *6
shared_name'%batch_normalization_1/moving_variance*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance*
shape:@
?
Fbatch_normalization_1/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
?
,batch_normalization_1/moving_variance/AssignAssignVariableOp%batch_normalization_1/moving_variance6batch_normalization_1/moving_variance/Initializer/ones*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
dtype0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance
z
!batch_normalization_1/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_1/cond/switch_tIdentity#batch_normalization_1/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_1/cond/switch_fIdentity!batch_normalization_1/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_1/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_1/cond/ReadVariableOpReadVariableOp2batch_normalization_1/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:@
?
0batch_normalization_1/cond/ReadVariableOp/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*.
_class$
" loc:@batch_normalization_1/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_1/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_1/cond/ReadVariableOp_1/Switch:1*
_output_shapes
:@*
dtype0
?
2batch_normalization_1/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_1/beta*
_output_shapes
: : 
?
 batch_normalization_1/cond/ConstConst$^batch_normalization_1/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
"batch_normalization_1/cond/Const_1Const$^batch_normalization_1/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
)batch_normalization_1/cond/FusedBatchNormFusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm/Switch:1)batch_normalization_1/cond/ReadVariableOp+batch_normalization_1/cond/ReadVariableOp_1 batch_normalization_1/cond/Const"batch_normalization_1/cond/Const_1*
T0*
data_formatNHWC*G
_output_shapes5
3:?????????  @:@:@:@:@*
is_training(*
epsilon%?ŧ7
?
0batch_normalization_1/cond/FusedBatchNorm/SwitchSwitchconv2d/Conv2D"batch_normalization_1/cond/pred_id*J
_output_shapes8
6:?????????  @:?????????  @*
T0* 
_class
loc:@conv2d/Conv2D
?
+batch_normalization_1/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_2/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_1/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_1/gamma"batch_normalization_1/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_1/gamma*
_output_shapes
: : 
?
+batch_normalization_1/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_1/cond/ReadVariableOp_3/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_1/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_1/beta"batch_normalization_1/cond/pred_id*
T0*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_1/beta
?
:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:@
?
Abatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_1/moving_mean"batch_normalization_1/cond/pred_id*4
_class*
(&loc:@batch_normalization_1/moving_mean*
T0*
_output_shapes
: : 
?
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:@
?
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_1/moving_variance"batch_normalization_1/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
+batch_normalization_1/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_1/cond/FusedBatchNorm_1/Switch+batch_normalization_1/cond/ReadVariableOp_2+batch_normalization_1/cond/ReadVariableOp_3:batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
data_formatNHWC*
is_training( *G
_output_shapes5
3:?????????  @:@:@:@:@*
epsilon%?ŧ7
?
2batch_normalization_1/cond/FusedBatchNorm_1/SwitchSwitchconv2d/Conv2D"batch_normalization_1/cond/pred_id*
T0*J
_output_shapes8
6:?????????  @:?????????  @* 
_class
loc:@conv2d/Conv2D
?
 batch_normalization_1/cond/MergeMerge+batch_normalization_1/cond/FusedBatchNorm_1)batch_normalization_1/cond/FusedBatchNorm*
N*1
_output_shapes
:?????????  @: *
T0
?
"batch_normalization_1/cond/Merge_1Merge-batch_normalization_1/cond/FusedBatchNorm_1:1+batch_normalization_1/cond/FusedBatchNorm:1*
_output_shapes

:@: *
N*
T0
?
"batch_normalization_1/cond/Merge_2Merge-batch_normalization_1/cond/FusedBatchNorm_1:2+batch_normalization_1/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes

:@: 
|
#batch_normalization_1/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_1/cond_1/switch_tIdentity%batch_normalization_1/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_1/cond_1/switch_fIdentity#batch_normalization_1/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_1/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
"batch_normalization_1/cond_1/ConstConst&^batch_normalization_1/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
$batch_normalization_1/cond_1/Const_1Const&^batch_normalization_1/cond_1/switch_f*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
"batch_normalization_1/cond_1/MergeMerge$batch_normalization_1/cond_1/Const_1"batch_normalization_1/cond_1/Const*
N*
_output_shapes
: : *
T0
?
+batch_normalization_1/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
)batch_normalization_1/AssignMovingAvg/subSub+batch_normalization_1/AssignMovingAvg/sub/x"batch_normalization_1/cond_1/Merge*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
: *
T0
?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
+batch_normalization_1/AssignMovingAvg/sub_1Sub4batch_normalization_1/AssignMovingAvg/ReadVariableOp"batch_normalization_1/cond/Merge_1*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean
?
)batch_normalization_1/AssignMovingAvg/mulMul+batch_normalization_1/AssignMovingAvg/sub_1)batch_normalization_1/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:@
?
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_1/moving_mean)batch_normalization_1/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_1/moving_mean*
dtype0
?
6batch_normalization_1/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_1/moving_mean:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_1/moving_variance*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
+batch_normalization_1/AssignMovingAvg_1/subSub-batch_normalization_1/AssignMovingAvg_1/sub/x"batch_normalization_1/cond_1/Merge*8
_class.
,*loc:@batch_normalization_1/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp"batch_normalization_1/cond/Merge_2*
T0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
+batch_normalization_1/AssignMovingAvg_1/mulMul-batch_normalization_1/AssignMovingAvg_1/sub_1+batch_normalization_1/AssignMovingAvg_1/sub*
T0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_1/moving_variance+batch_normalization_1/AssignMovingAvg_1/mul*
dtype0*8
_class.
,*loc:@batch_normalization_1/moving_variance
?
8batch_normalization_1/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_1/moving_variance<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
s
activation/ReluRelu batch_normalization_1/cond/Merge*/
_output_shapes
:?????????  @*
T0
?
zero_padding2d_1/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
zero_padding2d_1/PadPadactivation/Reluzero_padding2d_1/Pad/paddings*
	Tpaddings0*/
_output_shapes
:?????????""@*
T0
?
max_pooling2d/MaxPoolMaxPoolzero_padding2d_1/Pad*
data_formatNHWC*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
*
T0
?
,batch_normalization_2/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0*
valueB@*  ??*
_output_shapes
:@
?
batch_normalization_2/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_2/gamma*
_output_shapes
: *
dtype0*
	container *,
shared_namebatch_normalization_2/gamma*
shape:@
?
<batch_normalization_2/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
?
"batch_normalization_2/gamma/AssignAssignVariableOpbatch_normalization_2/gamma,batch_normalization_2/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*.
_class$
" loc:@batch_normalization_2/gamma*
dtype0
?
,batch_normalization_2/beta/Initializer/zerosConst*
_output_shapes
:@*-
_class#
!loc:@batch_normalization_2/beta*
dtype0*
valueB@*    
?
batch_normalization_2/betaVarHandleOp*
dtype0*-
_class#
!loc:@batch_normalization_2/beta*
	container *+
shared_namebatch_normalization_2/beta*
shape:@*
_output_shapes
: 
?
;batch_normalization_2/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
?
!batch_normalization_2/beta/AssignAssignVariableOpbatch_normalization_2/beta,batch_normalization_2/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_2/beta*
dtype0
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0*-
_class#
!loc:@batch_normalization_2/beta
?
3batch_normalization_2/moving_mean/Initializer/zerosConst*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:@*
valueB@*    
?
!batch_normalization_2/moving_meanVarHandleOp*
dtype0*
shape:@*
_output_shapes
: *2
shared_name#!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
	container 
?
Bbatch_normalization_2/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
?
(batch_normalization_2/moving_mean/AssignAssignVariableOp!batch_normalization_2/moving_mean3batch_normalization_2/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:@
?
6batch_normalization_2/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0*
valueB@*  ??
?
%batch_normalization_2/moving_varianceVarHandleOp*
shape:@*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0*
	container *6
shared_name'%batch_normalization_2/moving_variance
?
Fbatch_normalization_2/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
?
,batch_normalization_2/moving_variance/AssignAssignVariableOp%batch_normalization_2/moving_variance6batch_normalization_2/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
z
!batch_normalization_2/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_2/cond/switch_tIdentity#batch_normalization_2/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_2/cond/switch_fIdentity!batch_normalization_2/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_2/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_2/cond/ReadVariableOpReadVariableOp2batch_normalization_2/cond/ReadVariableOp/Switch:1*
_output_shapes
:@*
dtype0
?
0batch_normalization_2/cond/ReadVariableOp/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_2/gamma
?
+batch_normalization_2/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_2/cond/ReadVariableOp_1/Switch:1*
_output_shapes
:@*
dtype0
?
2batch_normalization_2/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*-
_class#
!loc:@batch_normalization_2/beta*
_output_shapes
: : *
T0
?
 batch_normalization_2/cond/ConstConst$^batch_normalization_2/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
"batch_normalization_2/cond/Const_1Const$^batch_normalization_2/cond/switch_t*
valueB *
_output_shapes
: *
dtype0
?
)batch_normalization_2/cond/FusedBatchNormFusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm/Switch:1)batch_normalization_2/cond/ReadVariableOp+batch_normalization_2/cond/ReadVariableOp_1 batch_normalization_2/cond/Const"batch_normalization_2/cond/Const_1*
is_training(*
epsilon%?ŧ7*G
_output_shapes5
3:?????????@:@:@:@:@*
data_formatNHWC*
T0
?
0batch_normalization_2/cond/FusedBatchNorm/SwitchSwitchmax_pooling2d/MaxPool"batch_normalization_2/cond/pred_id*
T0*J
_output_shapes8
6:?????????@:?????????@*(
_class
loc:@max_pooling2d/MaxPool
?
+batch_normalization_2/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:@
?
2batch_normalization_2/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_2/gamma"batch_normalization_2/cond/pred_id*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_2/gamma*
T0
?
+batch_normalization_2/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_2/cond/ReadVariableOp_3/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_2/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_2/beta"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_2/beta
?
:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes
:@*
dtype0
?
Abatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_2/moving_mean"batch_normalization_2/cond/pred_id*
_output_shapes
: : *
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes
:@*
dtype0
?
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_2/moving_variance"batch_normalization_2/cond/pred_id*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: : 
?
+batch_normalization_2/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_2/cond/FusedBatchNorm_1/Switch+batch_normalization_2/cond/ReadVariableOp_2+batch_normalization_2/cond/ReadVariableOp_3:batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1*
is_training( *
data_formatNHWC*G
_output_shapes5
3:?????????@:@:@:@:@*
epsilon%?ŧ7*
T0
?
2batch_normalization_2/cond/FusedBatchNorm_1/SwitchSwitchmax_pooling2d/MaxPool"batch_normalization_2/cond/pred_id*
T0*(
_class
loc:@max_pooling2d/MaxPool*J
_output_shapes8
6:?????????@:?????????@
?
 batch_normalization_2/cond/MergeMerge+batch_normalization_2/cond/FusedBatchNorm_1)batch_normalization_2/cond/FusedBatchNorm*1
_output_shapes
:?????????@: *
T0*
N
?
"batch_normalization_2/cond/Merge_1Merge-batch_normalization_2/cond/FusedBatchNorm_1:1+batch_normalization_2/cond/FusedBatchNorm:1*
_output_shapes

:@: *
N*
T0
?
"batch_normalization_2/cond/Merge_2Merge-batch_normalization_2/cond/FusedBatchNorm_1:2+batch_normalization_2/cond/FusedBatchNorm:2*
T0*
_output_shapes

:@: *
N
|
#batch_normalization_2/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

y
%batch_normalization_2/cond_1/switch_tIdentity%batch_normalization_2/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_2/cond_1/switch_fIdentity#batch_normalization_2/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_2/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
"batch_normalization_2/cond_1/ConstConst&^batch_normalization_2/cond_1/switch_t*
valueB
 *?p}?*
_output_shapes
: *
dtype0
?
$batch_normalization_2/cond_1/Const_1Const&^batch_normalization_2/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"batch_normalization_2/cond_1/MergeMerge$batch_normalization_2/cond_1/Const_1"batch_normalization_2/cond_1/Const*
T0*
_output_shapes
: : *
N
?
+batch_normalization_2/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
)batch_normalization_2/AssignMovingAvg/subSub+batch_normalization_2/AssignMovingAvg/sub/x"batch_normalization_2/cond_1/Merge*4
_class*
(&loc:@batch_normalization_2/moving_mean*
T0*
_output_shapes
: 
?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
dtype0*
_output_shapes
:@
?
+batch_normalization_2/AssignMovingAvg/sub_1Sub4batch_normalization_2/AssignMovingAvg/ReadVariableOp"batch_normalization_2/cond/Merge_1*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
)batch_normalization_2/AssignMovingAvg/mulMul+batch_normalization_2/AssignMovingAvg/sub_1)batch_normalization_2/AssignMovingAvg/sub*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_2/moving_mean
?
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_2/moving_mean)batch_normalization_2/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
6batch_normalization_2/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_2/moving_mean:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_2/moving_mean*
dtype0
?
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_2/moving_variance
?
+batch_normalization_2/AssignMovingAvg_1/subSub-batch_normalization_2/AssignMovingAvg_1/sub/x"batch_normalization_2/cond_1/Merge*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
: *
T0
?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
dtype0*
_output_shapes
:@
?
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp"batch_normalization_2/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:@
?
+batch_normalization_2/AssignMovingAvg_1/mulMul-batch_normalization_2/AssignMovingAvg_1/sub_1+batch_normalization_2/AssignMovingAvg_1/sub*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_2/moving_variance*
T0
?
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_2/moving_variance+batch_normalization_2/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_2/moving_variance*
dtype0
?
8batch_normalization_2/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_2/moving_variance<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_2/moving_variance*
_output_shapes
:@
u
activation_1/ReluRelu batch_normalization_2/cond/Merge*
T0*/
_output_shapes
:?????????@
?
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
:
?
.conv2d_1/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_1/kernel*
valueB
 *qĜ?*
dtype0*
_output_shapes
: 
?
.conv2d_1/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_1/kernel*
dtype0*
_output_shapes
: *
valueB
 *qĜ>
?
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@*

seed *
seed2 *
dtype0
?
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: *
T0
?
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*&
_output_shapes
:@@*
T0
?
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
:@@
?
conv2d_1/kernelVarHandleOp*"
_class
loc:@conv2d_1/kernel*
_output_shapes
: * 
shared_nameconv2d_1/kernel*
shape:@@*
dtype0*
	container 
o
0conv2d_1/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
?
conv2d_1/kernel/AssignAssignVariableOpconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_1/kernel*
dtype0
?
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
dtype0*&
_output_shapes
:@@
g
conv2d_1/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
v
conv2d_1/Conv2D/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_1/Conv2DConv2Dactivation_1/Reluconv2d_1/Conv2D/ReadVariableOp*/
_output_shapes
:?????????@*
strides
*
T0*
data_formatNHWC*
	dilations
*
paddingVALID*
explicit_paddings
 *
use_cudnn_on_gpu(
?
zero_padding2d_2/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_2/PadPadactivation_1/Reluzero_padding2d_2/Pad/paddings*
	Tpaddings0*
T0*/
_output_shapes
:?????????@
?
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *"
_class
loc:@conv2d_2/kernel*
dtype0*
_output_shapes
:
?
.conv2d_2/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: *
valueB
 *?ѽ*
dtype0
?
.conv2d_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *??=*
dtype0*"
_class
loc:@conv2d_2/kernel*
_output_shapes
: 
?
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_2/kernel
?
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*
T0*
_output_shapes
: 
?
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_2/kernel*&
_output_shapes
:@@*
T0
?
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
:@@
?
conv2d_2/kernelVarHandleOp*
	container *
_output_shapes
: * 
shared_nameconv2d_2/kernel*
shape:@@*
dtype0*"
_class
loc:@conv2d_2/kernel
o
0conv2d_2/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
?
conv2d_2/kernel/AssignAssignVariableOpconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_2/kernel*
dtype0
?
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0*"
_class
loc:@conv2d_2/kernel
g
conv2d_2/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
v
conv2d_2/Conv2D/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@
?
conv2d_2/Conv2DConv2Dzero_padding2d_2/Padconv2d_2/Conv2D/ReadVariableOp*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingVALID*
T0*
data_formatNHWC*
	dilations
*/
_output_shapes
:?????????@*
strides

?
,batch_normalization_3/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@*
valueB@*  ??*
dtype0
?
batch_normalization_3/gammaVarHandleOp*
shape:@*
_output_shapes
: *
	container *
dtype0*,
shared_namebatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma
?
<batch_normalization_3/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
?
"batch_normalization_3/gamma/AssignAssignVariableOpbatch_normalization_3/gamma,batch_normalization_3/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_3/gamma*
dtype0
?
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*.
_class$
" loc:@batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
?
,batch_normalization_3/beta/Initializer/zerosConst*
_output_shapes
:@*
dtype0*
valueB@*    *-
_class#
!loc:@batch_normalization_3/beta
?
batch_normalization_3/betaVarHandleOp*-
_class#
!loc:@batch_normalization_3/beta*
shape:@*
	container *
_output_shapes
: *+
shared_namebatch_normalization_3/beta*
dtype0
?
;batch_normalization_3/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
?
!batch_normalization_3/beta/AssignAssignVariableOpbatch_normalization_3/beta,batch_normalization_3/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_3/beta*
dtype0
?
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*-
_class#
!loc:@batch_normalization_3/beta*
_output_shapes
:@*
dtype0
?
3batch_normalization_3/moving_mean/Initializer/zerosConst*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_3/moving_mean*
valueB@*    *
dtype0
?
!batch_normalization_3/moving_meanVarHandleOp*
	container *
shape:@*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0*2
shared_name#!batch_normalization_3/moving_mean
?
Bbatch_normalization_3/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
?
(batch_normalization_3/moving_mean/AssignAssignVariableOp!batch_normalization_3/moving_mean3batch_normalization_3/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
:@
?
6batch_normalization_3/moving_variance/Initializer/onesConst*
valueB@*  ??*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
?
%batch_normalization_3/moving_varianceVarHandleOp*
	container *
_output_shapes
: *8
_class.
,*loc:@batch_normalization_3/moving_variance*
shape:@*
dtype0*6
shared_name'%batch_normalization_3/moving_variance
?
Fbatch_normalization_3/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
?
,batch_normalization_3/moving_variance/AssignAssignVariableOp%batch_normalization_3/moving_variance6batch_normalization_3/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
?
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
dtype0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_3/moving_variance
z
!batch_normalization_3/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_3/cond/switch_tIdentity#batch_normalization_3/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_3/cond/switch_fIdentity!batch_normalization_3/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_3/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_3/cond/ReadVariableOpReadVariableOp2batch_normalization_3/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:@
?
0batch_normalization_3/cond/ReadVariableOp/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_3/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_3/cond/ReadVariableOp_1/Switch:1*
_output_shapes
:@*
dtype0
?
2batch_normalization_3/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
T0*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_3/beta
?
 batch_normalization_3/cond/ConstConst$^batch_normalization_3/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
"batch_normalization_3/cond/Const_1Const$^batch_normalization_3/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
)batch_normalization_3/cond/FusedBatchNormFusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm/Switch:1)batch_normalization_3/cond/ReadVariableOp+batch_normalization_3/cond/ReadVariableOp_1 batch_normalization_3/cond/Const"batch_normalization_3/cond/Const_1*
epsilon%?ŧ7*
data_formatNHWC*
T0*G
_output_shapes5
3:?????????@:@:@:@:@*
is_training(
?
0batch_normalization_3/cond/FusedBatchNorm/SwitchSwitchconv2d_2/Conv2D"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*
T0*"
_class
loc:@conv2d_2/Conv2D
?
+batch_normalization_3/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:@
?
2batch_normalization_3/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_3/gamma"batch_normalization_3/cond/pred_id*.
_class$
" loc:@batch_normalization_3/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_3/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_3/cond/ReadVariableOp_3/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_3/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_3/beta"batch_normalization_3/cond/pred_id*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_3/beta*
T0
?
:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes
:@
?
Abatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_3/moving_mean"batch_normalization_3/cond/pred_id*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: : *
T0
?
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:@
?
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_3/moving_variance"batch_normalization_3/cond/pred_id*
_output_shapes
: : *8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
?
+batch_normalization_3/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_3/cond/FusedBatchNorm_1/Switch+batch_normalization_3/cond/ReadVariableOp_2+batch_normalization_3/cond/ReadVariableOp_3:batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
epsilon%?ŧ7*
is_training( *G
_output_shapes5
3:?????????@:@:@:@:@*
data_formatNHWC
?
2batch_normalization_3/cond/FusedBatchNorm_1/SwitchSwitchconv2d_2/Conv2D"batch_normalization_3/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*"
_class
loc:@conv2d_2/Conv2D*
T0
?
 batch_normalization_3/cond/MergeMerge+batch_normalization_3/cond/FusedBatchNorm_1)batch_normalization_3/cond/FusedBatchNorm*
N*1
_output_shapes
:?????????@: *
T0
?
"batch_normalization_3/cond/Merge_1Merge-batch_normalization_3/cond/FusedBatchNorm_1:1+batch_normalization_3/cond/FusedBatchNorm:1*
_output_shapes

:@: *
N*
T0
?
"batch_normalization_3/cond/Merge_2Merge-batch_normalization_3/cond/FusedBatchNorm_1:2+batch_normalization_3/cond/FusedBatchNorm:2*
T0*
_output_shapes

:@: *
N
|
#batch_normalization_3/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

y
%batch_normalization_3/cond_1/switch_tIdentity%batch_normalization_3/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_3/cond_1/switch_fIdentity#batch_normalization_3/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_3/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
"batch_normalization_3/cond_1/ConstConst&^batch_normalization_3/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
$batch_normalization_3/cond_1/Const_1Const&^batch_normalization_3/cond_1/switch_f*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
"batch_normalization_3/cond_1/MergeMerge$batch_normalization_3/cond_1/Const_1"batch_normalization_3/cond_1/Const*
T0*
_output_shapes
: : *
N
?
+batch_normalization_3/AssignMovingAvg/sub/xConst*
dtype0*
valueB
 *  ??*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_3/moving_mean
?
)batch_normalization_3/AssignMovingAvg/subSub+batch_normalization_3/AssignMovingAvg/sub/x"batch_normalization_3/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_3/moving_mean*
_output_shapes
: 
?
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
dtype0*
_output_shapes
:@
?
+batch_normalization_3/AssignMovingAvg/sub_1Sub4batch_normalization_3/AssignMovingAvg/ReadVariableOp"batch_normalization_3/cond/Merge_1*
T0*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
)batch_normalization_3/AssignMovingAvg/mulMul+batch_normalization_3/AssignMovingAvg/sub_1)batch_normalization_3/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_3/moving_mean*
T0*
_output_shapes
:@
?
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_3/moving_mean)batch_normalization_3/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_3/moving_mean*
dtype0
?
6batch_normalization_3/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_3/moving_mean:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:@*
dtype0*4
_class*
(&loc:@batch_normalization_3/moving_mean
?
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*8
_class.
,*loc:@batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
+batch_normalization_3/AssignMovingAvg_1/subSub-batch_normalization_3/AssignMovingAvg_1/sub/x"batch_normalization_3/cond_1/Merge*
T0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_3/moving_variance
?
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
?
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp"batch_normalization_3/cond/Merge_2*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0
?
+batch_normalization_3/AssignMovingAvg_1/mulMul-batch_normalization_3/AssignMovingAvg_1/sub_1+batch_normalization_3/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_3/moving_variance*
T0*
_output_shapes
:@
?
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_3/moving_variance+batch_normalization_3/AssignMovingAvg_1/mul*
dtype0*8
_class.
,*loc:@batch_normalization_3/moving_variance
?
8batch_normalization_3/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_3/moving_variance<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_3/moving_variance*
dtype0
u
activation_2/ReluRelu batch_normalization_3/cond/Merge*/
_output_shapes
:?????????@*
T0
?
zero_padding2d_3/Pad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
zero_padding2d_3/PadPadactivation_2/Reluzero_padding2d_3/Pad/paddings*
T0*/
_output_shapes
:?????????@*
	Tpaddings0
?
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
:*%
valueB"      @   @   
?
.conv2d_3/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: *
dtype0*
valueB
 *?ѽ
?
.conv2d_3/kernel/Initializer/random_uniform/maxConst*
valueB
 *??=*
dtype0*"
_class
loc:@conv2d_3/kernel*
_output_shapes
: 
?
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*&
_output_shapes
:@@*
T0*

seed *"
_class
loc:@conv2d_3/kernel*
dtype0*
seed2 
?
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*
T0*
_output_shapes
: 
?
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
:@@*"
_class
loc:@conv2d_3/kernel
?
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@@*
T0
?
conv2d_3/kernelVarHandleOp* 
shared_nameconv2d_3/kernel*
shape:@@*
_output_shapes
: *"
_class
loc:@conv2d_3/kernel*
	container *
dtype0
o
0conv2d_3/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
?
conv2d_3/kernel/AssignAssignVariableOpconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_3/kernel*
dtype0
?
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
g
conv2d_3/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
v
conv2d_3/Conv2D/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_3/Conv2DConv2Dzero_padding2d_3/Padconv2d_3/Conv2D/ReadVariableOp*/
_output_shapes
:?????????@*
T0*
paddingVALID*
use_cudnn_on_gpu(*
strides
*
explicit_paddings
 *
data_formatNHWC*
	dilations

j
add/addAddconv2d_3/Conv2Dconv2d_1/Conv2D*/
_output_shapes
:?????????@*
T0
?
,batch_normalization_4/gamma/Initializer/onesConst*
dtype0*
valueB@*  ??*
_output_shapes
:@*.
_class$
" loc:@batch_normalization_4/gamma
?
batch_normalization_4/gammaVarHandleOp*
	container *
shape:@*.
_class$
" loc:@batch_normalization_4/gamma*
_output_shapes
: *,
shared_namebatch_normalization_4/gamma*
dtype0
?
<batch_normalization_4/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
?
"batch_normalization_4/gamma/AssignAssignVariableOpbatch_normalization_4/gamma,batch_normalization_4/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_4/gamma*
dtype0
?
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
:@*.
_class$
" loc:@batch_normalization_4/gamma
?
,batch_normalization_4/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:@*
dtype0*
valueB@*    
?
batch_normalization_4/betaVarHandleOp*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
: *
	container *+
shared_namebatch_normalization_4/beta*
dtype0*
shape:@
?
;batch_normalization_4/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
?
!batch_normalization_4/beta/AssignAssignVariableOpbatch_normalization_4/beta,batch_normalization_4/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_4/beta*
dtype0
?
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
dtype0*-
_class#
!loc:@batch_normalization_4/beta*
_output_shapes
:@
?
3batch_normalization_4/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *4
_class*
(&loc:@batch_normalization_4/moving_mean
?
!batch_normalization_4/moving_meanVarHandleOp*
shape:@*2
shared_name#!batch_normalization_4/moving_mean*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean*
	container *
dtype0
?
Bbatch_normalization_4/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
?
(batch_normalization_4/moving_mean/AssignAssignVariableOp!batch_normalization_4/moving_mean3batch_normalization_4/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
?
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
6batch_normalization_4/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0*
valueB@*  ??
?
%batch_normalization_4/moving_varianceVarHandleOp*
shape:@*
	container *6
shared_name'%batch_normalization_4/moving_variance*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
?
Fbatch_normalization_4/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
?
,batch_normalization_4/moving_variance/AssignAssignVariableOp%batch_normalization_4/moving_variance6batch_normalization_4/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
?
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
z
!batch_normalization_4/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_4/cond/switch_tIdentity#batch_normalization_4/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_4/cond/switch_fIdentity!batch_normalization_4/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_4/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
)batch_normalization_4/cond/ReadVariableOpReadVariableOp2batch_normalization_4/cond/ReadVariableOp/Switch:1*
_output_shapes
:@*
dtype0
?
0batch_normalization_4/cond/ReadVariableOp/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_4/gamma
?
+batch_normalization_4/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_4/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:@
?
2batch_normalization_4/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*-
_class#
!loc:@batch_normalization_4/beta*
T0*
_output_shapes
: : 
?
 batch_normalization_4/cond/ConstConst$^batch_normalization_4/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
"batch_normalization_4/cond/Const_1Const$^batch_normalization_4/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
)batch_normalization_4/cond/FusedBatchNormFusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm/Switch:1)batch_normalization_4/cond/ReadVariableOp+batch_normalization_4/cond/ReadVariableOp_1 batch_normalization_4/cond/Const"batch_normalization_4/cond/Const_1*
data_formatNHWC*
epsilon%?ŧ7*
is_training(*
T0*G
_output_shapes5
3:?????????@:@:@:@:@
?
0batch_normalization_4/cond/FusedBatchNorm/SwitchSwitchadd/add"batch_normalization_4/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*
T0*
_class
loc:@add/add
?
+batch_normalization_4/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes
:@
?
2batch_normalization_4/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_4/gamma"batch_normalization_4/cond/pred_id*.
_class$
" loc:@batch_normalization_4/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_4/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_4/cond/ReadVariableOp_3/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_4/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_4/beta"batch_normalization_4/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_4/beta
?
:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes
:@*
dtype0
?
Abatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_4/moving_mean"batch_normalization_4/cond/pred_id*4
_class*
(&loc:@batch_normalization_4/moving_mean*
T0*
_output_shapes
: : 
?
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:@
?
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_4/moving_variance"batch_normalization_4/cond/pred_id*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: : *
T0
?
+batch_normalization_4/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_4/cond/FusedBatchNorm_1/Switch+batch_normalization_4/cond/ReadVariableOp_2+batch_normalization_4/cond/ReadVariableOp_3:batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*
is_training( *
epsilon%?ŧ7*
data_formatNHWC*G
_output_shapes5
3:?????????@:@:@:@:@
?
2batch_normalization_4/cond/FusedBatchNorm_1/SwitchSwitchadd/add"batch_normalization_4/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*
T0*
_class
loc:@add/add
?
 batch_normalization_4/cond/MergeMerge+batch_normalization_4/cond/FusedBatchNorm_1)batch_normalization_4/cond/FusedBatchNorm*1
_output_shapes
:?????????@: *
T0*
N
?
"batch_normalization_4/cond/Merge_1Merge-batch_normalization_4/cond/FusedBatchNorm_1:1+batch_normalization_4/cond/FusedBatchNorm:1*
N*
_output_shapes

:@: *
T0
?
"batch_normalization_4/cond/Merge_2Merge-batch_normalization_4/cond/FusedBatchNorm_1:2+batch_normalization_4/cond/FusedBatchNorm:2*
T0*
_output_shapes

:@: *
N
|
#batch_normalization_4/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_4/cond_1/switch_tIdentity%batch_normalization_4/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_4/cond_1/switch_fIdentity#batch_normalization_4/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_4/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
"batch_normalization_4/cond_1/ConstConst&^batch_normalization_4/cond_1/switch_t*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
$batch_normalization_4/cond_1/Const_1Const&^batch_normalization_4/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
"batch_normalization_4/cond_1/MergeMerge$batch_normalization_4/cond_1/Const_1"batch_normalization_4/cond_1/Const*
N*
_output_shapes
: : *
T0
?
+batch_normalization_4/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean
?
)batch_normalization_4/AssignMovingAvg/subSub+batch_normalization_4/AssignMovingAvg/sub/x"batch_normalization_4/cond_1/Merge*
T0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_4/moving_mean
?
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
:@
?
+batch_normalization_4/AssignMovingAvg/sub_1Sub4batch_normalization_4/AssignMovingAvg/ReadVariableOp"batch_normalization_4/cond/Merge_1*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
)batch_normalization_4/AssignMovingAvg/mulMul+batch_normalization_4/AssignMovingAvg/sub_1)batch_normalization_4/AssignMovingAvg/sub*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_4/moving_mean
?
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_4/moving_mean)batch_normalization_4/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_4/moving_mean*
dtype0
?
6batch_normalization_4/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_4/moving_mean:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp*4
_class*
(&loc:@batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
?
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*
dtype0*
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_4/moving_variance*
_output_shapes
: 
?
+batch_normalization_4/AssignMovingAvg_1/subSub-batch_normalization_4/AssignMovingAvg_1/sub/x"batch_normalization_4/cond_1/Merge*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
: 
?
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
:@
?
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp"batch_normalization_4/cond/Merge_2*8
_class.
,*loc:@batch_normalization_4/moving_variance*
T0*
_output_shapes
:@
?
+batch_normalization_4/AssignMovingAvg_1/mulMul-batch_normalization_4/AssignMovingAvg_1/sub_1+batch_normalization_4/AssignMovingAvg_1/sub*
_output_shapes
:@*
T0*8
_class.
,*loc:@batch_normalization_4/moving_variance
?
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_4/moving_variance+batch_normalization_4/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_4/moving_variance*
dtype0
?
8batch_normalization_4/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_4/moving_variance<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes
:@*
dtype0*8
_class.
,*loc:@batch_normalization_4/moving_variance
u
activation_3/ReluRelu batch_normalization_4/cond/Merge*
T0*/
_output_shapes
:?????????@
?
zero_padding2d_4/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
?
zero_padding2d_4/PadPadactivation_3/Reluzero_padding2d_4/Pad/paddings*/
_output_shapes
:?????????@*
T0*
	Tpaddings0
?
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*"
_class
loc:@conv2d_4/kernel*%
valueB"      @   @   
?
.conv2d_4/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_4/kernel*
valueB
 *?ѽ*
_output_shapes
: 
?
.conv2d_4/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_4/kernel*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*
T0*&
_output_shapes
:@@*
seed2 *
dtype0*

seed *"
_class
loc:@conv2d_4/kernel
?
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
T0
?
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@
?
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*&
_output_shapes
:@@*"
_class
loc:@conv2d_4/kernel*
T0
?
conv2d_4/kernelVarHandleOp*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_4/kernel*
	container *
shape:@@* 
shared_nameconv2d_4/kernel
o
0conv2d_4/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
?
conv2d_4/kernel/AssignAssignVariableOpconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_4/kernel
?
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:@@*
dtype0
g
conv2d_4/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
v
conv2d_4/Conv2D/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_4/Conv2DConv2Dzero_padding2d_4/Padconv2d_4/Conv2D/ReadVariableOp*
	dilations
*
use_cudnn_on_gpu(*
strides
*
paddingVALID*
T0*
explicit_paddings
 */
_output_shapes
:?????????@*
data_formatNHWC
?
,batch_normalization_5/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0*
_output_shapes
:@*
valueB@*  ??
?
batch_normalization_5/gammaVarHandleOp*.
_class$
" loc:@batch_normalization_5/gamma*,
shared_namebatch_normalization_5/gamma*
	container *
_output_shapes
: *
dtype0*
shape:@
?
<batch_normalization_5/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_5/gamma*
_output_shapes
: 
?
"batch_normalization_5/gamma/AssignAssignVariableOpbatch_normalization_5/gamma,batch_normalization_5/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_5/gamma*
dtype0
?
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
dtype0*
_output_shapes
:@*.
_class$
" loc:@batch_normalization_5/gamma
?
,batch_normalization_5/beta/Initializer/zerosConst*-
_class#
!loc:@batch_normalization_5/beta*
dtype0*
valueB@*    *
_output_shapes
:@
?
batch_normalization_5/betaVarHandleOp*
dtype0*-
_class#
!loc:@batch_normalization_5/beta*
_output_shapes
: *
shape:@*+
shared_namebatch_normalization_5/beta*
	container 
?
;batch_normalization_5/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_5/beta*
_output_shapes
: 
?
!batch_normalization_5/beta/AssignAssignVariableOpbatch_normalization_5/beta,batch_normalization_5/beta/Initializer/zeros*
dtype0*-
_class#
!loc:@batch_normalization_5/beta
?
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*-
_class#
!loc:@batch_normalization_5/beta*
dtype0
?
3batch_normalization_5/moving_mean/Initializer/zerosConst*
valueB@*    *
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0
?
!batch_normalization_5/moving_meanVarHandleOp*2
shared_name#!batch_normalization_5/moving_mean*4
_class*
(&loc:@batch_normalization_5/moving_mean*
shape:@*
dtype0*
	container *
_output_shapes
: 
?
Bbatch_normalization_5/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_5/moving_mean*
_output_shapes
: 
?
(batch_normalization_5/moving_mean/AssignAssignVariableOp!batch_normalization_5/moving_mean3batch_normalization_5/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_5/moving_mean*
dtype0
?
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
?
6batch_normalization_5/moving_variance/Initializer/onesConst*
valueB@*  ??*
_output_shapes
:@*
dtype0*8
_class.
,*loc:@batch_normalization_5/moving_variance
?
%batch_normalization_5/moving_varianceVarHandleOp*8
_class.
,*loc:@batch_normalization_5/moving_variance*
shape:@*
_output_shapes
: *
dtype0*
	container *6
shared_name'%batch_normalization_5/moving_variance
?
Fbatch_normalization_5/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_5/moving_variance*
_output_shapes
: 
?
,batch_normalization_5/moving_variance/AssignAssignVariableOp%batch_normalization_5/moving_variance6batch_normalization_5/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0
?
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
dtype0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:@
z
!batch_normalization_5/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_5/cond/switch_tIdentity#batch_normalization_5/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_5/cond/switch_fIdentity!batch_normalization_5/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_5/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
)batch_normalization_5/cond/ReadVariableOpReadVariableOp2batch_normalization_5/cond/ReadVariableOp/Switch:1*
_output_shapes
:@*
dtype0
?
0batch_normalization_5/cond/ReadVariableOp/SwitchSwitchbatch_normalization_5/gamma"batch_normalization_5/cond/pred_id*
T0*.
_class$
" loc:@batch_normalization_5/gamma*
_output_shapes
: : 
?
+batch_normalization_5/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_5/cond/ReadVariableOp_1/Switch:1*
_output_shapes
:@*
dtype0
?
2batch_normalization_5/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_5/beta"batch_normalization_5/cond/pred_id*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_5/beta*
T0
?
 batch_normalization_5/cond/ConstConst$^batch_normalization_5/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
"batch_normalization_5/cond/Const_1Const$^batch_normalization_5/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
)batch_normalization_5/cond/FusedBatchNormFusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm/Switch:1)batch_normalization_5/cond/ReadVariableOp+batch_normalization_5/cond/ReadVariableOp_1 batch_normalization_5/cond/Const"batch_normalization_5/cond/Const_1*
is_training(*
epsilon%?ŧ7*G
_output_shapes5
3:?????????@:@:@:@:@*
data_formatNHWC*
T0
?
0batch_normalization_5/cond/FusedBatchNorm/SwitchSwitchconv2d_4/Conv2D"batch_normalization_5/cond/pred_id*"
_class
loc:@conv2d_4/Conv2D*
T0*J
_output_shapes8
6:?????????@:?????????@
?
+batch_normalization_5/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_5/cond/ReadVariableOp_2/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_5/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_5/gamma"batch_normalization_5/cond/pred_id*.
_class$
" loc:@batch_normalization_5/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_5/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_5/cond/ReadVariableOp_3/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_5/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_5/beta"batch_normalization_5/cond/pred_id*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_5/beta*
T0
?
:batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes
:@*
dtype0
?
Abatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_5/moving_mean"batch_normalization_5/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean*
_output_shapes
: : 
?
<batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes
:@*
dtype0
?
Cbatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_5/moving_variance"batch_normalization_5/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
?
+batch_normalization_5/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_5/cond/FusedBatchNorm_1/Switch+batch_normalization_5/cond/ReadVariableOp_2+batch_normalization_5/cond/ReadVariableOp_3:batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*G
_output_shapes5
3:?????????@:@:@:@:@*
is_training( *
epsilon%?ŧ7*
T0
?
2batch_normalization_5/cond/FusedBatchNorm_1/SwitchSwitchconv2d_4/Conv2D"batch_normalization_5/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*"
_class
loc:@conv2d_4/Conv2D*
T0
?
 batch_normalization_5/cond/MergeMerge+batch_normalization_5/cond/FusedBatchNorm_1)batch_normalization_5/cond/FusedBatchNorm*
T0*1
_output_shapes
:?????????@: *
N
?
"batch_normalization_5/cond/Merge_1Merge-batch_normalization_5/cond/FusedBatchNorm_1:1+batch_normalization_5/cond/FusedBatchNorm:1*
_output_shapes

:@: *
T0*
N
?
"batch_normalization_5/cond/Merge_2Merge-batch_normalization_5/cond/FusedBatchNorm_1:2+batch_normalization_5/cond/FusedBatchNorm:2*
_output_shapes

:@: *
T0*
N
|
#batch_normalization_5/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_5/cond_1/switch_tIdentity%batch_normalization_5/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_5/cond_1/switch_fIdentity#batch_normalization_5/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_5/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
"batch_normalization_5/cond_1/ConstConst&^batch_normalization_5/cond_1/switch_t*
_output_shapes
: *
valueB
 *?p}?*
dtype0
?
$batch_normalization_5/cond_1/Const_1Const&^batch_normalization_5/cond_1/switch_f*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
"batch_normalization_5/cond_1/MergeMerge$batch_normalization_5/cond_1/Const_1"batch_normalization_5/cond_1/Const*
_output_shapes
: : *
N*
T0
?
+batch_normalization_5/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_5/moving_mean
?
)batch_normalization_5/AssignMovingAvg/subSub+batch_normalization_5/AssignMovingAvg/sub/x"batch_normalization_5/cond_1/Merge*
T0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_5/moving_mean
?
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
dtype0*
_output_shapes
:@
?
+batch_normalization_5/AssignMovingAvg/sub_1Sub4batch_normalization_5/AssignMovingAvg/ReadVariableOp"batch_normalization_5/cond/Merge_1*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_5/moving_mean
?
)batch_normalization_5/AssignMovingAvg/mulMul+batch_normalization_5/AssignMovingAvg/sub_1)batch_normalization_5/AssignMovingAvg/sub*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_5/moving_mean*
T0
?
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_5/moving_mean)batch_normalization_5/AssignMovingAvg/mul*
dtype0*4
_class*
(&loc:@batch_normalization_5/moving_mean
?
6batch_normalization_5/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_5/moving_mean:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp*
_output_shapes
:@*
dtype0*4
_class*
(&loc:@batch_normalization_5/moving_mean
?
-batch_normalization_5/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
: *
dtype0
?
+batch_normalization_5/AssignMovingAvg_1/subSub-batch_normalization_5/AssignMovingAvg_1/sub/x"batch_normalization_5/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance
?
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
?
-batch_normalization_5/AssignMovingAvg_1/sub_1Sub6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp"batch_normalization_5/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:@
?
+batch_normalization_5/AssignMovingAvg_1/mulMul-batch_normalization_5/AssignMovingAvg_1/sub_1+batch_normalization_5/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:@*
T0
?
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_5/moving_variance+batch_normalization_5/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_5/moving_variance*
dtype0
?
8batch_normalization_5/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_5/moving_variance<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*8
_class.
,*loc:@batch_normalization_5/moving_variance*
_output_shapes
:@
u
activation_4/ReluRelu batch_normalization_5/cond/Merge*
T0*/
_output_shapes
:?????????@
?
zero_padding2d_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
zero_padding2d_5/PadPadactivation_4/Reluzero_padding2d_5/Pad/paddings*
	Tpaddings0*/
_output_shapes
:?????????@*
T0
?
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
:*%
valueB"      @   @   
?
.conv2d_5/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?ѽ*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel
?
.conv2d_5/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@conv2d_5/kernel*
valueB
 *??=*
dtype0*
_output_shapes
: 
?
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_5/kernel*
T0*&
_output_shapes
:@@*
dtype0*

seed 
?
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_5/kernel*
_output_shapes
: 
?
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_5/kernel*
T0*&
_output_shapes
:@@
?
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@*
T0
?
conv2d_5/kernelVarHandleOp*
dtype0*
	container * 
shared_nameconv2d_5/kernel*
_output_shapes
: *"
_class
loc:@conv2d_5/kernel*
shape:@@
o
0conv2d_5/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
?
conv2d_5/kernel/AssignAssignVariableOpconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
dtype0*"
_class
loc:@conv2d_5/kernel
?
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:@@*
dtype0
g
conv2d_5/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
v
conv2d_5/Conv2D/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:@@*
dtype0
?
conv2d_5/Conv2DConv2Dzero_padding2d_5/Padconv2d_5/Conv2D/ReadVariableOp*
data_formatNHWC*
	dilations
*
T0*
strides
*
use_cudnn_on_gpu(*/
_output_shapes
:?????????@*
explicit_paddings
 *
paddingVALID
d
	add_1/addAddconv2d_5/Conv2Dadd/add*/
_output_shapes
:?????????@*
T0
?
,batch_normalization_6/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*
valueB@*  ??*.
_class$
" loc:@batch_normalization_6/gamma
?
batch_normalization_6/gammaVarHandleOp*
dtype0*
shape:@*
_output_shapes
: *,
shared_namebatch_normalization_6/gamma*.
_class$
" loc:@batch_normalization_6/gamma*
	container 
?
<batch_normalization_6/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_6/gamma*
_output_shapes
: 
?
"batch_normalization_6/gamma/AssignAssignVariableOpbatch_normalization_6/gamma,batch_normalization_6/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_6/gamma*
dtype0
?
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*.
_class$
" loc:@batch_normalization_6/gamma*
_output_shapes
:@*
dtype0
?
,batch_normalization_6/beta/Initializer/zerosConst*
valueB@*    *-
_class#
!loc:@batch_normalization_6/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_6/betaVarHandleOp*
shape:@*
dtype0*
_output_shapes
: *
	container *+
shared_namebatch_normalization_6/beta*-
_class#
!loc:@batch_normalization_6/beta
?
;batch_normalization_6/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_6/beta*
_output_shapes
: 
?
!batch_normalization_6/beta/AssignAssignVariableOpbatch_normalization_6/beta,batch_normalization_6/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_6/beta*
dtype0
?
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*-
_class#
!loc:@batch_normalization_6/beta*
dtype0
?
3batch_normalization_6/moving_mean/Initializer/zerosConst*
valueB@*    *
dtype0*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_6/moving_mean
?
!batch_normalization_6/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*4
_class*
(&loc:@batch_normalization_6/moving_mean*
	container *2
shared_name#!batch_normalization_6/moving_mean
?
Bbatch_normalization_6/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_6/moving_mean*
_output_shapes
: 
?
(batch_normalization_6/moving_mean/AssignAssignVariableOp!batch_normalization_6/moving_mean3batch_normalization_6/moving_mean/Initializer/zeros*
dtype0*4
_class*
(&loc:@batch_normalization_6/moving_mean
?
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0
?
6batch_normalization_6/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0*
_output_shapes
:@*
valueB@*  ??
?
%batch_normalization_6/moving_varianceVarHandleOp*
	container *8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes
: *6
shared_name'%batch_normalization_6/moving_variance*
shape:@*
dtype0
?
Fbatch_normalization_6/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_6/moving_variance*
_output_shapes
: 
?
,batch_normalization_6/moving_variance/AssignAssignVariableOp%batch_normalization_6/moving_variance6batch_normalization_6/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0
?
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0*8
_class.
,*loc:@batch_normalization_6/moving_variance
z
!batch_normalization_6/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_6/cond/switch_tIdentity#batch_normalization_6/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_6/cond/switch_fIdentity!batch_normalization_6/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_6/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_6/cond/ReadVariableOpReadVariableOp2batch_normalization_6/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes
:@
?
0batch_normalization_6/cond/ReadVariableOp/SwitchSwitchbatch_normalization_6/gamma"batch_normalization_6/cond/pred_id*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_6/gamma*
T0
?
+batch_normalization_6/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_6/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes
:@
?
2batch_normalization_6/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_6/beta"batch_normalization_6/cond/pred_id*-
_class#
!loc:@batch_normalization_6/beta*
T0*
_output_shapes
: : 
?
 batch_normalization_6/cond/ConstConst$^batch_normalization_6/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
"batch_normalization_6/cond/Const_1Const$^batch_normalization_6/cond/switch_t*
_output_shapes
: *
dtype0*
valueB 
?
)batch_normalization_6/cond/FusedBatchNormFusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm/Switch:1)batch_normalization_6/cond/ReadVariableOp+batch_normalization_6/cond/ReadVariableOp_1 batch_normalization_6/cond/Const"batch_normalization_6/cond/Const_1*
epsilon%?ŧ7*G
_output_shapes5
3:?????????@:@:@:@:@*
is_training(*
T0*
data_formatNHWC
?
0batch_normalization_6/cond/FusedBatchNorm/SwitchSwitch	add_1/add"batch_normalization_6/cond/pred_id*
_class
loc:@add_1/add*
T0*J
_output_shapes8
6:?????????@:?????????@
?
+batch_normalization_6/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_6/cond/ReadVariableOp_2/Switch*
_output_shapes
:@*
dtype0
?
2batch_normalization_6/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_6/gamma"batch_normalization_6/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_6/gamma
?
+batch_normalization_6/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_6/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes
:@
?
2batch_normalization_6/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_6/beta"batch_normalization_6/cond/pred_id*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_6/beta*
T0
?
:batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes
:@*
dtype0
?
Abatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_6/moving_mean"batch_normalization_6/cond/pred_id*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes
: : *
T0
?
<batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes
:@
?
Cbatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_6/moving_variance"batch_normalization_6/cond/pred_id*
_output_shapes
: : *
T0*8
_class.
,*loc:@batch_normalization_6/moving_variance
?
+batch_normalization_6/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_6/cond/FusedBatchNorm_1/Switch+batch_normalization_6/cond/ReadVariableOp_2+batch_normalization_6/cond/ReadVariableOp_3:batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1*
is_training( *
T0*G
_output_shapes5
3:?????????@:@:@:@:@*
epsilon%?ŧ7*
data_formatNHWC
?
2batch_normalization_6/cond/FusedBatchNorm_1/SwitchSwitch	add_1/add"batch_normalization_6/cond/pred_id*J
_output_shapes8
6:?????????@:?????????@*
T0*
_class
loc:@add_1/add
?
 batch_normalization_6/cond/MergeMerge+batch_normalization_6/cond/FusedBatchNorm_1)batch_normalization_6/cond/FusedBatchNorm*
T0*1
_output_shapes
:?????????@: *
N
?
"batch_normalization_6/cond/Merge_1Merge-batch_normalization_6/cond/FusedBatchNorm_1:1+batch_normalization_6/cond/FusedBatchNorm:1*
N*
_output_shapes

:@: *
T0
?
"batch_normalization_6/cond/Merge_2Merge-batch_normalization_6/cond/FusedBatchNorm_1:2+batch_normalization_6/cond/FusedBatchNorm:2*
_output_shapes

:@: *
N*
T0
|
#batch_normalization_6/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_6/cond_1/switch_tIdentity%batch_normalization_6/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_6/cond_1/switch_fIdentity#batch_normalization_6/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_6/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
"batch_normalization_6/cond_1/ConstConst&^batch_normalization_6/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
$batch_normalization_6/cond_1/Const_1Const&^batch_normalization_6/cond_1/switch_f*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
"batch_normalization_6/cond_1/MergeMerge$batch_normalization_6/cond_1/Const_1"batch_normalization_6/cond_1/Const*
T0*
N*
_output_shapes
: : 
?
+batch_normalization_6/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0*4
_class*
(&loc:@batch_normalization_6/moving_mean
?
)batch_normalization_6/AssignMovingAvg/subSub+batch_normalization_6/AssignMovingAvg/sub/x"batch_normalization_6/cond_1/Merge*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes
: 
?
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
?
+batch_normalization_6/AssignMovingAvg/sub_1Sub4batch_normalization_6/AssignMovingAvg/ReadVariableOp"batch_normalization_6/cond/Merge_1*
_output_shapes
:@*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean
?
)batch_normalization_6/AssignMovingAvg/mulMul+batch_normalization_6/AssignMovingAvg/sub_1)batch_normalization_6/AssignMovingAvg/sub*
T0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes
:@
?
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_6/moving_mean)batch_normalization_6/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_6/moving_mean*
dtype0
?
6batch_normalization_6/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_6/moving_mean:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp*
dtype0*4
_class*
(&loc:@batch_normalization_6/moving_mean*
_output_shapes
:@
?
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0*
valueB
 *  ??
?
+batch_normalization_6/AssignMovingAvg_1/subSub-batch_normalization_6/AssignMovingAvg_1/sub/x"batch_normalization_6/cond_1/Merge*8
_class.
,*loc:@batch_normalization_6/moving_variance*
_output_shapes
: *
T0
?
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
dtype0*
_output_shapes
:@
?
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp"batch_normalization_6/cond/Merge_2*
T0*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_6/moving_variance
?
+batch_normalization_6/AssignMovingAvg_1/mulMul-batch_normalization_6/AssignMovingAvg_1/sub_1+batch_normalization_6/AssignMovingAvg_1/sub*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_6/moving_variance*
T0
?
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_6/moving_variance+batch_normalization_6/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0
?
8batch_normalization_6/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_6/moving_variance<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes
:@*8
_class.
,*loc:@batch_normalization_6/moving_variance*
dtype0
u
activation_5/ReluRelu batch_normalization_6/cond/Merge*
T0*/
_output_shapes
:?????????@
?
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_6/kernel*%
valueB"      @   ?   *
_output_shapes
:*
dtype0
?
.conv2d_6/kernel/Initializer/random_uniform/minConst*
valueB
 *qĜ?*
dtype0*"
_class
loc:@conv2d_6/kernel*
_output_shapes
: 
?
.conv2d_6/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_6/kernel*
valueB
 *qĜ>*
_output_shapes
: 
?
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*'
_output_shapes
:@?*
dtype0*

seed *
seed2 *
T0*"
_class
loc:@conv2d_6/kernel
?
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*
T0*
_output_shapes
: 
?
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@conv2d_6/kernel*'
_output_shapes
:@?
?
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*
T0*'
_output_shapes
:@?*"
_class
loc:@conv2d_6/kernel
?
conv2d_6/kernelVarHandleOp* 
shared_nameconv2d_6/kernel*
	container *
shape:@?*"
_class
loc:@conv2d_6/kernel*
_output_shapes
: *
dtype0
o
0conv2d_6/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_6/kernel*
_output_shapes
: 
?
conv2d_6/kernel/AssignAssignVariableOpconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_6/kernel*
dtype0
?
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*
dtype0*'
_output_shapes
:@?*"
_class
loc:@conv2d_6/kernel
g
conv2d_6/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
w
conv2d_6/Conv2D/ReadVariableOpReadVariableOpconv2d_6/kernel*'
_output_shapes
:@?*
dtype0
?
conv2d_6/Conv2DConv2Dactivation_5/Reluconv2d_6/Conv2D/ReadVariableOp*
use_cudnn_on_gpu(*
data_formatNHWC*
explicit_paddings
 *0
_output_shapes
:??????????*
strides
*
paddingVALID*
T0*
	dilations

?
zero_padding2d_6/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_6/PadPadactivation_5/Reluzero_padding2d_6/Pad/paddings*/
_output_shapes
:?????????@*
T0*
	Tpaddings0
?
0conv2d_7/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_7/kernel*
_output_shapes
:*
dtype0*%
valueB"      @   ?   
?
.conv2d_7/kernel/Initializer/random_uniform/minConst*
valueB
 *?ѽ*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_7/kernel
?
.conv2d_7/kernel/Initializer/random_uniform/maxConst*
valueB
 *??=*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_7/kernel
?
8conv2d_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_7/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@conv2d_7/kernel*'
_output_shapes
:@?*
dtype0*
seed2 *

seed 
?
.conv2d_7/kernel/Initializer/random_uniform/subSub.conv2d_7/kernel/Initializer/random_uniform/max.conv2d_7/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_7/kernel
?
.conv2d_7/kernel/Initializer/random_uniform/mulMul8conv2d_7/kernel/Initializer/random_uniform/RandomUniform.conv2d_7/kernel/Initializer/random_uniform/sub*
T0*'
_output_shapes
:@?*"
_class
loc:@conv2d_7/kernel
?
*conv2d_7/kernel/Initializer/random_uniformAdd.conv2d_7/kernel/Initializer/random_uniform/mul.conv2d_7/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_7/kernel*
T0*'
_output_shapes
:@?
?
conv2d_7/kernelVarHandleOp*"
_class
loc:@conv2d_7/kernel*
dtype0*
	container *
shape:@?*
_output_shapes
: * 
shared_nameconv2d_7/kernel
o
0conv2d_7/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_7/kernel*
_output_shapes
: 
?
conv2d_7/kernel/AssignAssignVariableOpconv2d_7/kernel*conv2d_7/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_7/kernel*
dtype0
?
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*
dtype0*'
_output_shapes
:@?*"
_class
loc:@conv2d_7/kernel
g
conv2d_7/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
w
conv2d_7/Conv2D/ReadVariableOpReadVariableOpconv2d_7/kernel*
dtype0*'
_output_shapes
:@?
?
conv2d_7/Conv2DConv2Dzero_padding2d_6/Padconv2d_7/Conv2D/ReadVariableOp*
explicit_paddings
 *0
_output_shapes
:??????????*
data_formatNHWC*
use_cudnn_on_gpu(*
T0*
	dilations
*
paddingVALID*
strides

?
,batch_normalization_7/gamma/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*.
_class$
" loc:@batch_normalization_7/gamma
?
batch_normalization_7/gammaVarHandleOp*
	container *
shape:?*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes
: *
dtype0*,
shared_namebatch_normalization_7/gamma
?
<batch_normalization_7/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_7/gamma*
_output_shapes
: 
?
"batch_normalization_7/gamma/AssignAssignVariableOpbatch_normalization_7/gamma,batch_normalization_7/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0
?
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_7/gamma*
dtype0
?
,batch_normalization_7/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*-
_class#
!loc:@batch_normalization_7/beta*
valueB?*    
?
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
	container *+
shared_namebatch_normalization_7/beta*
shape:?*
dtype0*-
_class#
!loc:@batch_normalization_7/beta
?
;batch_normalization_7/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_7/beta*
_output_shapes
: 
?
!batch_normalization_7/beta/AssignAssignVariableOpbatch_normalization_7/beta,batch_normalization_7/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_7/beta*
dtype0
?
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*-
_class#
!loc:@batch_normalization_7/beta*
dtype0*
_output_shapes	
:?
?
3batch_normalization_7/moving_mean/Initializer/zerosConst*
dtype0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
valueB?*    *
_output_shapes	
:?
?
!batch_normalization_7/moving_meanVarHandleOp*
dtype0*
	container *2
shared_name#!batch_normalization_7/moving_mean*4
_class*
(&loc:@batch_normalization_7/moving_mean*
shape:?*
_output_shapes
: 
?
Bbatch_normalization_7/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_7/moving_mean*
_output_shapes
: 
?
(batch_normalization_7/moving_mean/AssignAssignVariableOp!batch_normalization_7/moving_mean3batch_normalization_7/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0
?
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:?*
dtype0*4
_class*
(&loc:@batch_normalization_7/moving_mean
?
6batch_normalization_7/moving_variance/Initializer/onesConst*8
_class.
,*loc:@batch_normalization_7/moving_variance*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
%batch_normalization_7/moving_varianceVarHandleOp*
	container *
shape:?*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0*6
shared_name'%batch_normalization_7/moving_variance
?
Fbatch_normalization_7/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_7/moving_variance*
_output_shapes
: 
?
,batch_normalization_7/moving_variance/AssignAssignVariableOp%batch_normalization_7/moving_variance6batch_normalization_7/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0
?
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
dtype0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:?
z
!batch_normalization_7/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_7/cond/switch_tIdentity#batch_normalization_7/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_7/cond/switch_fIdentity!batch_normalization_7/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_7/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
)batch_normalization_7/cond/ReadVariableOpReadVariableOp2batch_normalization_7/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
0batch_normalization_7/cond/ReadVariableOp/SwitchSwitchbatch_normalization_7/gamma"batch_normalization_7/cond/pred_id*.
_class$
" loc:@batch_normalization_7/gamma*
_output_shapes
: : *
T0
?
+batch_normalization_7/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_7/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
2batch_normalization_7/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_7/beta"batch_normalization_7/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_7/beta*
_output_shapes
: : 
?
 batch_normalization_7/cond/ConstConst$^batch_normalization_7/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
"batch_normalization_7/cond/Const_1Const$^batch_normalization_7/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
)batch_normalization_7/cond/FusedBatchNormFusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm/Switch:1)batch_normalization_7/cond/ReadVariableOp+batch_normalization_7/cond/ReadVariableOp_1 batch_normalization_7/cond/Const"batch_normalization_7/cond/Const_1*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training(*
epsilon%?ŧ7*
T0
?
0batch_normalization_7/cond/FusedBatchNorm/SwitchSwitchconv2d_7/Conv2D"batch_normalization_7/cond/pred_id*"
_class
loc:@conv2d_7/Conv2D*
T0*L
_output_shapes:
8:??????????:??????????
?
+batch_normalization_7/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_7/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
2batch_normalization_7/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_7/gamma"batch_normalization_7/cond/pred_id*.
_class$
" loc:@batch_normalization_7/gamma*
T0*
_output_shapes
: : 
?
+batch_normalization_7/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_7/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
2batch_normalization_7/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_7/beta"batch_normalization_7/cond/pred_id*
T0*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_7/beta
?
:batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Abatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_7/moving_mean"batch_normalization_7/cond/pred_id*
_output_shapes
: : *4
_class*
(&loc:@batch_normalization_7/moving_mean*
T0
?
<batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Cbatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_7/moving_variance"batch_normalization_7/cond/pred_id*
_output_shapes
: : *8
_class.
,*loc:@batch_normalization_7/moving_variance*
T0
?
+batch_normalization_7/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_7/cond/FusedBatchNorm_1/Switch+batch_normalization_7/cond/ReadVariableOp_2+batch_normalization_7/cond/ReadVariableOp_3:batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1*
is_training( *
data_formatNHWC*
T0*
epsilon%?ŧ7*L
_output_shapes:
8:??????????:?:?:?:?
?
2batch_normalization_7/cond/FusedBatchNorm_1/SwitchSwitchconv2d_7/Conv2D"batch_normalization_7/cond/pred_id*
T0*"
_class
loc:@conv2d_7/Conv2D*L
_output_shapes:
8:??????????:??????????
?
 batch_normalization_7/cond/MergeMerge+batch_normalization_7/cond/FusedBatchNorm_1)batch_normalization_7/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:??????????: 
?
"batch_normalization_7/cond/Merge_1Merge-batch_normalization_7/cond/FusedBatchNorm_1:1+batch_normalization_7/cond/FusedBatchNorm:1*
_output_shapes
	:?: *
T0*
N
?
"batch_normalization_7/cond/Merge_2Merge-batch_normalization_7/cond/FusedBatchNorm_1:2+batch_normalization_7/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:?: 
|
#batch_normalization_7/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_7/cond_1/switch_tIdentity%batch_normalization_7/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_7/cond_1/switch_fIdentity#batch_normalization_7/cond_1/Switch*
T0
*
_output_shapes
: 
g
$batch_normalization_7/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
"batch_normalization_7/cond_1/ConstConst&^batch_normalization_7/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
$batch_normalization_7/cond_1/Const_1Const&^batch_normalization_7/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
"batch_normalization_7/cond_1/MergeMerge$batch_normalization_7/cond_1/Const_1"batch_normalization_7/cond_1/Const*
N*
T0*
_output_shapes
: : 
?
+batch_normalization_7/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0
?
)batch_normalization_7/AssignMovingAvg/subSub+batch_normalization_7/AssignMovingAvg/sub/x"batch_normalization_7/cond_1/Merge*
T0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_7/moving_mean
?
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:?*
dtype0
?
+batch_normalization_7/AssignMovingAvg/sub_1Sub4batch_normalization_7/AssignMovingAvg/ReadVariableOp"batch_normalization_7/cond/Merge_1*
_output_shapes	
:?*4
_class*
(&loc:@batch_normalization_7/moving_mean*
T0
?
)batch_normalization_7/AssignMovingAvg/mulMul+batch_normalization_7/AssignMovingAvg/sub_1)batch_normalization_7/AssignMovingAvg/sub*
_output_shapes	
:?*4
_class*
(&loc:@batch_normalization_7/moving_mean*
T0
?
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_7/moving_mean)batch_normalization_7/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_7/moving_mean*
dtype0
?
6batch_normalization_7/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_7/moving_mean:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp*
dtype0*4
_class*
(&loc:@batch_normalization_7/moving_mean*
_output_shapes	
:?
?
-batch_normalization_7/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes
: *
dtype0
?
+batch_normalization_7/AssignMovingAvg_1/subSub-batch_normalization_7/AssignMovingAvg_1/sub/x"batch_normalization_7/cond_1/Merge*
T0*
_output_shapes
: *8
_class.
,*loc:@batch_normalization_7/moving_variance
?
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
dtype0*
_output_shapes	
:?
?
-batch_normalization_7/AssignMovingAvg_1/sub_1Sub6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp"batch_normalization_7/cond/Merge_2*
T0*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:?
?
+batch_normalization_7/AssignMovingAvg_1/mulMul-batch_normalization_7/AssignMovingAvg_1/sub_1+batch_normalization_7/AssignMovingAvg_1/sub*
T0*
_output_shapes	
:?*8
_class.
,*loc:@batch_normalization_7/moving_variance
?
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_7/moving_variance+batch_normalization_7/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_7/moving_variance*
dtype0
?
8batch_normalization_7/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_7/moving_variance<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_7/moving_variance*
_output_shapes	
:?*
dtype0
v
activation_6/ReluRelu batch_normalization_7/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_7/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_7/PadPadactivation_6/Reluzero_padding2d_7/Pad/paddings*
T0*0
_output_shapes
:?????????

?*
	Tpaddings0
?
0conv2d_8/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?   ?   *"
_class
loc:@conv2d_8/kernel*
dtype0*
_output_shapes
:
?
.conv2d_8/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *:͓?*"
_class
loc:@conv2d_8/kernel*
dtype0
?
.conv2d_8/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_8/kernel*
_output_shapes
: *
valueB
 *:͓=
?
8conv2d_8/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_8/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*"
_class
loc:@conv2d_8/kernel*
T0*

seed *(
_output_shapes
:??
?
.conv2d_8/kernel/Initializer/random_uniform/subSub.conv2d_8/kernel/Initializer/random_uniform/max.conv2d_8/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_8/kernel*
_output_shapes
: *
T0
?
.conv2d_8/kernel/Initializer/random_uniform/mulMul8conv2d_8/kernel/Initializer/random_uniform/RandomUniform.conv2d_8/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_8/kernel*
T0*(
_output_shapes
:??
?
*conv2d_8/kernel/Initializer/random_uniformAdd.conv2d_8/kernel/Initializer/random_uniform/mul.conv2d_8/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_8/kernel*(
_output_shapes
:??*
T0
?
conv2d_8/kernelVarHandleOp*
	container *
shape:??*
dtype0*
_output_shapes
: * 
shared_nameconv2d_8/kernel*"
_class
loc:@conv2d_8/kernel
o
0conv2d_8/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_8/kernel*
_output_shapes
: 
?
conv2d_8/kernel/AssignAssignVariableOpconv2d_8/kernel*conv2d_8/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_8/kernel*
dtype0
?
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*"
_class
loc:@conv2d_8/kernel*(
_output_shapes
:??*
dtype0
g
conv2d_8/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
x
conv2d_8/Conv2D/ReadVariableOpReadVariableOpconv2d_8/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_8/Conv2DConv2Dzero_padding2d_7/Padconv2d_8/Conv2D/ReadVariableOp*0
_output_shapes
:??????????*
explicit_paddings
 *
	dilations
*
data_formatNHWC*
T0*
use_cudnn_on_gpu(*
strides
*
paddingVALID
m
	add_2/addAddconv2d_8/Conv2Dconv2d_6/Conv2D*0
_output_shapes
:??????????*
T0
?
,batch_normalization_8/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0*
_output_shapes	
:?*
valueB?*  ??
?
batch_normalization_8/gammaVarHandleOp*
	container *
_output_shapes
: *
shape:?*
dtype0*.
_class$
" loc:@batch_normalization_8/gamma*,
shared_namebatch_normalization_8/gamma
?
<batch_normalization_8/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_8/gamma*
_output_shapes
: 
?
"batch_normalization_8/gamma/AssignAssignVariableOpbatch_normalization_8/gamma,batch_normalization_8/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_8/gamma*
dtype0
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*.
_class$
" loc:@batch_normalization_8/gamma*
_output_shapes	
:?*
dtype0
?
,batch_normalization_8/beta/Initializer/zerosConst*
dtype0*-
_class#
!loc:@batch_normalization_8/beta*
valueB?*    *
_output_shapes	
:?
?
batch_normalization_8/betaVarHandleOp*+
shared_namebatch_normalization_8/beta*-
_class#
!loc:@batch_normalization_8/beta*
dtype0*
shape:?*
_output_shapes
: *
	container 
?
;batch_normalization_8/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_8/beta*
_output_shapes
: 
?
!batch_normalization_8/beta/AssignAssignVariableOpbatch_normalization_8/beta,batch_normalization_8/beta/Initializer/zeros*
dtype0*-
_class#
!loc:@batch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:?*
dtype0*-
_class#
!loc:@batch_normalization_8/beta
?
3batch_normalization_8/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes	
:?
?
!batch_normalization_8/moving_meanVarHandleOp*4
_class*
(&loc:@batch_normalization_8/moving_mean*
shape:?*2
shared_name#!batch_normalization_8/moving_mean*
	container *
dtype0*
_output_shapes
: 
?
Bbatch_normalization_8/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_8/moving_mean*
_output_shapes
: 
?
(batch_normalization_8/moving_mean/AssignAssignVariableOp!batch_normalization_8/moving_mean3batch_normalization_8/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:?*
dtype0*4
_class*
(&loc:@batch_normalization_8/moving_mean
?
6batch_normalization_8/moving_variance/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0*8
_class.
,*loc:@batch_normalization_8/moving_variance
?
%batch_normalization_8/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_8/moving_variance*
shape:?*
	container *
dtype0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
: 
?
Fbatch_normalization_8/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_8/moving_variance*
_output_shapes
: 
?
,batch_normalization_8/moving_variance/AssignAssignVariableOp%batch_normalization_8/moving_variance6batch_normalization_8/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_8/moving_variance*
dtype0
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
z
!batch_normalization_8/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

u
#batch_normalization_8/cond/switch_tIdentity#batch_normalization_8/cond/Switch:1*
_output_shapes
: *
T0

s
#batch_normalization_8/cond/switch_fIdentity!batch_normalization_8/cond/Switch*
_output_shapes
: *
T0

e
"batch_normalization_8/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_8/cond/ReadVariableOpReadVariableOp2batch_normalization_8/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
0batch_normalization_8/cond/ReadVariableOp/SwitchSwitchbatch_normalization_8/gamma"batch_normalization_8/cond/pred_id*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_8/gamma*
T0
?
+batch_normalization_8/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_8/cond/ReadVariableOp_1/Switch:1*
_output_shapes	
:?*
dtype0
?
2batch_normalization_8/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_8/beta"batch_normalization_8/cond/pred_id*
_output_shapes
: : *
T0*-
_class#
!loc:@batch_normalization_8/beta
?
 batch_normalization_8/cond/ConstConst$^batch_normalization_8/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
"batch_normalization_8/cond/Const_1Const$^batch_normalization_8/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
)batch_normalization_8/cond/FusedBatchNormFusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm/Switch:1)batch_normalization_8/cond/ReadVariableOp+batch_normalization_8/cond/ReadVariableOp_1 batch_normalization_8/cond/Const"batch_normalization_8/cond/Const_1*L
_output_shapes:
8:??????????:?:?:?:?*
is_training(*
epsilon%?ŧ7*
data_formatNHWC*
T0
?
0batch_normalization_8/cond/FusedBatchNorm/SwitchSwitch	add_2/add"batch_normalization_8/cond/pred_id*
T0*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_2/add
?
+batch_normalization_8/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_8/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
2batch_normalization_8/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_8/gamma"batch_normalization_8/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_8/gamma
?
+batch_normalization_8/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_8/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
2batch_normalization_8/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_8/beta"batch_normalization_8/cond/pred_id*-
_class#
!loc:@batch_normalization_8/beta*
T0*
_output_shapes
: : 
?
:batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Abatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_8/moving_mean"batch_normalization_8/cond/pred_id*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes
: : 
?
<batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Cbatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_8/moving_variance"batch_normalization_8/cond/pred_id*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
: : *
T0
?
+batch_normalization_8/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_8/cond/FusedBatchNorm_1/Switch+batch_normalization_8/cond/ReadVariableOp_2+batch_normalization_8/cond/ReadVariableOp_3:batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%?ŧ7*
is_training( *
data_formatNHWC*
T0*L
_output_shapes:
8:??????????:?:?:?:?
?
2batch_normalization_8/cond/FusedBatchNorm_1/SwitchSwitch	add_2/add"batch_normalization_8/cond/pred_id*
_class
loc:@add_2/add*
T0*L
_output_shapes:
8:??????????:??????????
?
 batch_normalization_8/cond/MergeMerge+batch_normalization_8/cond/FusedBatchNorm_1)batch_normalization_8/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:??????????: 
?
"batch_normalization_8/cond/Merge_1Merge-batch_normalization_8/cond/FusedBatchNorm_1:1+batch_normalization_8/cond/FusedBatchNorm:1*
_output_shapes
	:?: *
T0*
N
?
"batch_normalization_8/cond/Merge_2Merge-batch_normalization_8/cond/FusedBatchNorm_1:2+batch_normalization_8/cond/FusedBatchNorm:2*
T0*
_output_shapes
	:?: *
N
|
#batch_normalization_8/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_8/cond_1/switch_tIdentity%batch_normalization_8/cond_1/Switch:1*
_output_shapes
: *
T0

w
%batch_normalization_8/cond_1/switch_fIdentity#batch_normalization_8/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_8/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
"batch_normalization_8/cond_1/ConstConst&^batch_normalization_8/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
$batch_normalization_8/cond_1/Const_1Const&^batch_normalization_8/cond_1/switch_f*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
"batch_normalization_8/cond_1/MergeMerge$batch_normalization_8/cond_1/Const_1"batch_normalization_8/cond_1/Const*
N*
T0*
_output_shapes
: : 
?
+batch_normalization_8/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ??*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0
?
)batch_normalization_8/AssignMovingAvg/subSub+batch_normalization_8/AssignMovingAvg/sub/x"batch_normalization_8/cond_1/Merge*
T0*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_8/moving_mean
?
4batch_normalization_8/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
dtype0*
_output_shapes	
:?
?
+batch_normalization_8/AssignMovingAvg/sub_1Sub4batch_normalization_8/AssignMovingAvg/ReadVariableOp"batch_normalization_8/cond/Merge_1*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean*
_output_shapes	
:?
?
)batch_normalization_8/AssignMovingAvg/mulMul+batch_normalization_8/AssignMovingAvg/sub_1)batch_normalization_8/AssignMovingAvg/sub*
_output_shapes	
:?*
T0*4
_class*
(&loc:@batch_normalization_8/moving_mean
?
9batch_normalization_8/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_8/moving_mean)batch_normalization_8/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0
?
6batch_normalization_8/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_8/moving_mean:^batch_normalization_8/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:?*4
_class*
(&loc:@batch_normalization_8/moving_mean*
dtype0
?
-batch_normalization_8/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
dtype0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
valueB
 *  ??
?
+batch_normalization_8/AssignMovingAvg_1/subSub-batch_normalization_8/AssignMovingAvg_1/sub/x"batch_normalization_8/cond_1/Merge*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes
: 
?
6batch_normalization_8/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
?
-batch_normalization_8/AssignMovingAvg_1/sub_1Sub6batch_normalization_8/AssignMovingAvg_1/ReadVariableOp"batch_normalization_8/cond/Merge_2*
_output_shapes	
:?*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance
?
+batch_normalization_8/AssignMovingAvg_1/mulMul-batch_normalization_8/AssignMovingAvg_1/sub_1+batch_normalization_8/AssignMovingAvg_1/sub*
_output_shapes	
:?*
T0*8
_class.
,*loc:@batch_normalization_8/moving_variance
?
;batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_8/moving_variance+batch_normalization_8/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_8/moving_variance*
dtype0
?
8batch_normalization_8/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_8/moving_variance<^batch_normalization_8/AssignMovingAvg_1/AssignSubVariableOp*8
_class.
,*loc:@batch_normalization_8/moving_variance*
_output_shapes	
:?*
dtype0
v
activation_7/ReluRelu batch_normalization_8/cond/Merge*
T0*0
_output_shapes
:??????????
?
zero_padding2d_8/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
?
zero_padding2d_8/PadPadactivation_7/Reluzero_padding2d_8/Pad/paddings*
	Tpaddings0*
T0*0
_output_shapes
:?????????

?
?
0conv2d_9/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@conv2d_9/kernel*
dtype0*
_output_shapes
:*%
valueB"      ?   ?   
?
.conv2d_9/kernel/Initializer/random_uniform/minConst*"
_class
loc:@conv2d_9/kernel*
_output_shapes
: *
dtype0*
valueB
 *:͓?
?
.conv2d_9/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *
dtype0*"
_class
loc:@conv2d_9/kernel
?
8conv2d_9/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_9/kernel/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*"
_class
loc:@conv2d_9/kernel*(
_output_shapes
:??*
T0
?
.conv2d_9/kernel/Initializer/random_uniform/subSub.conv2d_9/kernel/Initializer/random_uniform/max.conv2d_9/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_9/kernel*
T0*
_output_shapes
: 
?
.conv2d_9/kernel/Initializer/random_uniform/mulMul8conv2d_9/kernel/Initializer/random_uniform/RandomUniform.conv2d_9/kernel/Initializer/random_uniform/sub*(
_output_shapes
:??*"
_class
loc:@conv2d_9/kernel*
T0
?
*conv2d_9/kernel/Initializer/random_uniformAdd.conv2d_9/kernel/Initializer/random_uniform/mul.conv2d_9/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@conv2d_9/kernel*(
_output_shapes
:??
?
conv2d_9/kernelVarHandleOp*"
_class
loc:@conv2d_9/kernel*
_output_shapes
: *
dtype0*
shape:??*
	container * 
shared_nameconv2d_9/kernel
o
0conv2d_9/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_9/kernel*
_output_shapes
: 
?
conv2d_9/kernel/AssignAssignVariableOpconv2d_9/kernel*conv2d_9/kernel/Initializer/random_uniform*"
_class
loc:@conv2d_9/kernel*
dtype0
?
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*"
_class
loc:@conv2d_9/kernel*
dtype0*(
_output_shapes
:??
g
conv2d_9/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
x
conv2d_9/Conv2D/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_9/Conv2DConv2Dzero_padding2d_8/Padconv2d_9/Conv2D/ReadVariableOp*
data_formatNHWC*
explicit_paddings
 *
strides
*
	dilations
*
paddingVALID*0
_output_shapes
:??????????*
T0*
use_cudnn_on_gpu(
?
,batch_normalization_9/gamma/Initializer/onesConst*.
_class$
" loc:@batch_normalization_9/gamma*
valueB?*  ??*
_output_shapes	
:?*
dtype0
?
batch_normalization_9/gammaVarHandleOp*
shape:?*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_9/gamma*
	container *.
_class$
" loc:@batch_normalization_9/gamma
?
<batch_normalization_9/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_9/gamma*
_output_shapes
: 
?
"batch_normalization_9/gamma/AssignAssignVariableOpbatch_normalization_9/gamma,batch_normalization_9/gamma/Initializer/ones*.
_class$
" loc:@batch_normalization_9/gamma*
dtype0
?
/batch_normalization_9/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_9/gamma*
dtype0*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes	
:?
?
,batch_normalization_9/beta/Initializer/zerosConst*
valueB?*    *
dtype0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes	
:?
?
batch_normalization_9/betaVarHandleOp*
dtype0*+
shared_namebatch_normalization_9/beta*
shape:?*
	container *-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: 
?
;batch_normalization_9/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_9/beta*
_output_shapes
: 
?
!batch_normalization_9/beta/AssignAssignVariableOpbatch_normalization_9/beta,batch_normalization_9/beta/Initializer/zeros*-
_class#
!loc:@batch_normalization_9/beta*
dtype0
?
.batch_normalization_9/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_9/beta*
dtype0*
_output_shapes	
:?*-
_class#
!loc:@batch_normalization_9/beta
?
3batch_normalization_9/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*4
_class*
(&loc:@batch_normalization_9/moving_mean
?
!batch_normalization_9/moving_meanVarHandleOp*2
shared_name#!batch_normalization_9/moving_mean*
_output_shapes
: *
shape:?*4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0*
	container 
?
Bbatch_normalization_9/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp!batch_normalization_9/moving_mean*
_output_shapes
: 
?
(batch_normalization_9/moving_mean/AssignAssignVariableOp!batch_normalization_9/moving_mean3batch_normalization_9/moving_mean/Initializer/zeros*4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0
?
5batch_normalization_9/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes	
:?*
dtype0
?
6batch_normalization_9/moving_variance/Initializer/onesConst*
dtype0*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes	
:?*
valueB?*  ??
?
%batch_normalization_9/moving_varianceVarHandleOp*
	container *
shape:?*8
_class.
,*loc:@batch_normalization_9/moving_variance*6
shared_name'%batch_normalization_9/moving_variance*
dtype0*
_output_shapes
: 
?
Fbatch_normalization_9/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp%batch_normalization_9/moving_variance*
_output_shapes
: 
?
,batch_normalization_9/moving_variance/AssignAssignVariableOp%batch_normalization_9/moving_variance6batch_normalization_9/moving_variance/Initializer/ones*8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0
?
9batch_normalization_9/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes	
:?*
dtype0
z
!batch_normalization_9/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
u
#batch_normalization_9/cond/switch_tIdentity#batch_normalization_9/cond/Switch:1*
T0
*
_output_shapes
: 
s
#batch_normalization_9/cond/switch_fIdentity!batch_normalization_9/cond/Switch*
T0
*
_output_shapes
: 
e
"batch_normalization_9/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
)batch_normalization_9/cond/ReadVariableOpReadVariableOp2batch_normalization_9/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
0batch_normalization_9/cond/ReadVariableOp/SwitchSwitchbatch_normalization_9/gamma"batch_normalization_9/cond/pred_id*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: : *
T0
?
+batch_normalization_9/cond/ReadVariableOp_1ReadVariableOp4batch_normalization_9/cond/ReadVariableOp_1/Switch:1*
_output_shapes	
:?*
dtype0
?
2batch_normalization_9/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_9/beta"batch_normalization_9/cond/pred_id*
T0*
_output_shapes
: : *-
_class#
!loc:@batch_normalization_9/beta
?
 batch_normalization_9/cond/ConstConst$^batch_normalization_9/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
"batch_normalization_9/cond/Const_1Const$^batch_normalization_9/cond/switch_t*
valueB *
_output_shapes
: *
dtype0
?
)batch_normalization_9/cond/FusedBatchNormFusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm/Switch:1)batch_normalization_9/cond/ReadVariableOp+batch_normalization_9/cond/ReadVariableOp_1 batch_normalization_9/cond/Const"batch_normalization_9/cond/Const_1*L
_output_shapes:
8:??????????:?:?:?:?*
T0*
data_formatNHWC*
is_training(*
epsilon%?ŧ7
?
0batch_normalization_9/cond/FusedBatchNorm/SwitchSwitchconv2d_9/Conv2D"batch_normalization_9/cond/pred_id*"
_class
loc:@conv2d_9/Conv2D*L
_output_shapes:
8:??????????:??????????*
T0
?
+batch_normalization_9/cond/ReadVariableOp_2ReadVariableOp2batch_normalization_9/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
2batch_normalization_9/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_9/gamma"batch_normalization_9/cond/pred_id*.
_class$
" loc:@batch_normalization_9/gamma*
_output_shapes
: : *
T0
?
+batch_normalization_9/cond/ReadVariableOp_3ReadVariableOp2batch_normalization_9/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
2batch_normalization_9/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_9/beta"batch_normalization_9/cond/pred_id*
T0*-
_class#
!loc:@batch_normalization_9/beta*
_output_shapes
: : 
?
:batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpAbatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:?*
dtype0
?
Abatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch!batch_normalization_9/moving_mean"batch_normalization_9/cond/pred_id*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: : *
T0
?
<batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpCbatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Cbatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch%batch_normalization_9/moving_variance"batch_normalization_9/cond/pred_id*8
_class.
,*loc:@batch_normalization_9/moving_variance*
T0*
_output_shapes
: : 
?
+batch_normalization_9/cond/FusedBatchNorm_1FusedBatchNorm2batch_normalization_9/cond/FusedBatchNorm_1/Switch+batch_normalization_9/cond/ReadVariableOp_2+batch_normalization_9/cond/ReadVariableOp_3:batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp<batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
data_formatNHWC*
epsilon%?ŧ7*
T0
?
2batch_normalization_9/cond/FusedBatchNorm_1/SwitchSwitchconv2d_9/Conv2D"batch_normalization_9/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
T0*"
_class
loc:@conv2d_9/Conv2D
?
 batch_normalization_9/cond/MergeMerge+batch_normalization_9/cond/FusedBatchNorm_1)batch_normalization_9/cond/FusedBatchNorm*
N*2
_output_shapes 
:??????????: *
T0
?
"batch_normalization_9/cond/Merge_1Merge-batch_normalization_9/cond/FusedBatchNorm_1:1+batch_normalization_9/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:?: 
?
"batch_normalization_9/cond/Merge_2Merge-batch_normalization_9/cond/FusedBatchNorm_1:2+batch_normalization_9/cond/FusedBatchNorm:2*
_output_shapes
	:?: *
N*
T0
|
#batch_normalization_9/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
y
%batch_normalization_9/cond_1/switch_tIdentity%batch_normalization_9/cond_1/Switch:1*
T0
*
_output_shapes
: 
w
%batch_normalization_9/cond_1/switch_fIdentity#batch_normalization_9/cond_1/Switch*
_output_shapes
: *
T0

g
$batch_normalization_9/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
"batch_normalization_9/cond_1/ConstConst&^batch_normalization_9/cond_1/switch_t*
_output_shapes
: *
valueB
 *?p}?*
dtype0
?
$batch_normalization_9/cond_1/Const_1Const&^batch_normalization_9/cond_1/switch_f*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
"batch_normalization_9/cond_1/MergeMerge$batch_normalization_9/cond_1/Const_1"batch_normalization_9/cond_1/Const*
T0*
_output_shapes
: : *
N
?
+batch_normalization_9/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
dtype0*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes
: 
?
)batch_normalization_9/AssignMovingAvg/subSub+batch_normalization_9/AssignMovingAvg/sub/x"batch_normalization_9/cond_1/Merge*
_output_shapes
: *4
_class*
(&loc:@batch_normalization_9/moving_mean*
T0
?
4batch_normalization_9/AssignMovingAvg/ReadVariableOpReadVariableOp!batch_normalization_9/moving_mean*
dtype0*
_output_shapes	
:?
?
+batch_normalization_9/AssignMovingAvg/sub_1Sub4batch_normalization_9/AssignMovingAvg/ReadVariableOp"batch_normalization_9/cond/Merge_1*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes	
:?*
T0
?
)batch_normalization_9/AssignMovingAvg/mulMul+batch_normalization_9/AssignMovingAvg/sub_1)batch_normalization_9/AssignMovingAvg/sub*4
_class*
(&loc:@batch_normalization_9/moving_mean*
_output_shapes	
:?*
T0
?
9batch_normalization_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp!batch_normalization_9/moving_mean)batch_normalization_9/AssignMovingAvg/mul*4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0
?
6batch_normalization_9/AssignMovingAvg/ReadVariableOp_1ReadVariableOp!batch_normalization_9/moving_mean:^batch_normalization_9/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:?*4
_class*
(&loc:@batch_normalization_9/moving_mean*
dtype0
?
-batch_normalization_9/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *
valueB
 *  ??*8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0
?
+batch_normalization_9/AssignMovingAvg_1/subSub-batch_normalization_9/AssignMovingAvg_1/sub/x"batch_normalization_9/cond_1/Merge*
_output_shapes
: *
T0*8
_class.
,*loc:@batch_normalization_9/moving_variance
?
6batch_normalization_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp%batch_normalization_9/moving_variance*
_output_shapes	
:?*
dtype0
?
-batch_normalization_9/AssignMovingAvg_1/sub_1Sub6batch_normalization_9/AssignMovingAvg_1/ReadVariableOp"batch_normalization_9/cond/Merge_2*
_output_shapes	
:?*8
_class.
,*loc:@batch_normalization_9/moving_variance*
T0
?
+batch_normalization_9/AssignMovingAvg_1/mulMul-batch_normalization_9/AssignMovingAvg_1/sub_1+batch_normalization_9/AssignMovingAvg_1/sub*8
_class.
,*loc:@batch_normalization_9/moving_variance*
_output_shapes	
:?*
T0
?
;batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp%batch_normalization_9/moving_variance+batch_normalization_9/AssignMovingAvg_1/mul*8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0
?
8batch_normalization_9/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp%batch_normalization_9/moving_variance<^batch_normalization_9/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*8
_class.
,*loc:@batch_normalization_9/moving_variance*
dtype0
v
activation_8/ReluRelu batch_normalization_9/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_9/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_9/PadPadactivation_8/Reluzero_padding2d_9/Pad/paddings*0
_output_shapes
:?????????

?*
	Tpaddings0*
T0
?
1conv2d_10/kernel/Initializer/random_uniform/shapeConst*
dtype0*#
_class
loc:@conv2d_10/kernel*
_output_shapes
:*%
valueB"      ?   ?   
?
/conv2d_10/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*#
_class
loc:@conv2d_10/kernel*
_output_shapes
: *
dtype0
?
/conv2d_10/kernel/Initializer/random_uniform/maxConst*
valueB
 *:͓=*
_output_shapes
: *#
_class
loc:@conv2d_10/kernel*
dtype0
?
9conv2d_10/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_10/kernel/Initializer/random_uniform/shape*(
_output_shapes
:??*
T0*#
_class
loc:@conv2d_10/kernel*
seed2 *
dtype0*

seed 
?
/conv2d_10/kernel/Initializer/random_uniform/subSub/conv2d_10/kernel/Initializer/random_uniform/max/conv2d_10/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@conv2d_10/kernel*
_output_shapes
: 
?
/conv2d_10/kernel/Initializer/random_uniform/mulMul9conv2d_10/kernel/Initializer/random_uniform/RandomUniform/conv2d_10/kernel/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_10/kernel
?
+conv2d_10/kernel/Initializer/random_uniformAdd/conv2d_10/kernel/Initializer/random_uniform/mul/conv2d_10/kernel/Initializer/random_uniform/min*(
_output_shapes
:??*
T0*#
_class
loc:@conv2d_10/kernel
?
conv2d_10/kernelVarHandleOp*
	container *!
shared_nameconv2d_10/kernel*
_output_shapes
: *#
_class
loc:@conv2d_10/kernel*
dtype0*
shape:??
q
1conv2d_10/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_10/kernel*
_output_shapes
: 
?
conv2d_10/kernel/AssignAssignVariableOpconv2d_10/kernel+conv2d_10/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_10/kernel*
dtype0
?
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*
dtype0*#
_class
loc:@conv2d_10/kernel*(
_output_shapes
:??
h
conv2d_10/dilation_rateConst*
valueB"      *
_output_shapes
:*
dtype0
z
conv2d_10/Conv2D/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_10/Conv2DConv2Dzero_padding2d_9/Padconv2d_10/Conv2D/ReadVariableOp*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID*
data_formatNHWC*
strides
*0
_output_shapes
:??????????*
T0*
	dilations

h
	add_3/addAddconv2d_10/Conv2D	add_2/add*0
_output_shapes
:??????????*
T0
?
-batch_normalization_10/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*
valueB?*  ??*/
_class%
#!loc:@batch_normalization_10/gamma
?
batch_normalization_10/gammaVarHandleOp*-
shared_namebatch_normalization_10/gamma*/
_class%
#!loc:@batch_normalization_10/gamma*
	container *
dtype0*
shape:?*
_output_shapes
: 
?
=batch_normalization_10/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_10/gamma*
_output_shapes
: 
?
#batch_normalization_10/gamma/AssignAssignVariableOpbatch_normalization_10/gamma-batch_normalization_10/gamma/Initializer/ones*
dtype0*/
_class%
#!loc:@batch_normalization_10/gamma
?
0batch_normalization_10/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_10/gamma*/
_class%
#!loc:@batch_normalization_10/gamma*
_output_shapes	
:?*
dtype0
?
-batch_normalization_10/beta/Initializer/zerosConst*
valueB?*    *.
_class$
" loc:@batch_normalization_10/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_10/betaVarHandleOp*
	container *
_output_shapes
: *,
shared_namebatch_normalization_10/beta*.
_class$
" loc:@batch_normalization_10/beta*
dtype0*
shape:?
?
<batch_normalization_10/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_10/beta*
_output_shapes
: 
?
"batch_normalization_10/beta/AssignAssignVariableOpbatch_normalization_10/beta-batch_normalization_10/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_10/beta*
dtype0
?
/batch_normalization_10/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_10/beta*.
_class$
" loc:@batch_normalization_10/beta*
dtype0*
_output_shapes	
:?
?
4batch_normalization_10/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_10/moving_mean*
valueB?*    
?
"batch_normalization_10/moving_meanVarHandleOp*3
shared_name$"batch_normalization_10/moving_mean*
_output_shapes
: *
	container *
shape:?*5
_class+
)'loc:@batch_normalization_10/moving_mean*
dtype0
?
Cbatch_normalization_10/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_10/moving_mean*
_output_shapes
: 
?
)batch_normalization_10/moving_mean/AssignAssignVariableOp"batch_normalization_10/moving_mean4batch_normalization_10/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_10/moving_mean*
dtype0
?
6batch_normalization_10/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*5
_class+
)'loc:@batch_normalization_10/moving_mean*
dtype0*
_output_shapes	
:?
?
7batch_normalization_10/moving_variance/Initializer/onesConst*9
_class/
-+loc:@batch_normalization_10/moving_variance*
dtype0*
valueB?*  ??*
_output_shapes	
:?
?
&batch_normalization_10/moving_varianceVarHandleOp*7
shared_name(&batch_normalization_10/moving_variance*9
_class/
-+loc:@batch_normalization_10/moving_variance*
dtype0*
_output_shapes
: *
	container *
shape:?
?
Gbatch_normalization_10/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_10/moving_variance*
_output_shapes
: 
?
-batch_normalization_10/moving_variance/AssignAssignVariableOp&batch_normalization_10/moving_variance7batch_normalization_10/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_10/moving_variance*
dtype0
?
:batch_normalization_10/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_10/moving_variance
{
"batch_normalization_10/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

w
$batch_normalization_10/cond/switch_tIdentity$batch_normalization_10/cond/Switch:1*
T0
*
_output_shapes
: 
u
$batch_normalization_10/cond/switch_fIdentity"batch_normalization_10/cond/Switch*
_output_shapes
: *
T0

f
#batch_normalization_10/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
*batch_normalization_10/cond/ReadVariableOpReadVariableOp3batch_normalization_10/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
1batch_normalization_10/cond/ReadVariableOp/SwitchSwitchbatch_normalization_10/gamma#batch_normalization_10/cond/pred_id*
T0*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_10/gamma
?
,batch_normalization_10/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_10/cond/ReadVariableOp_1/Switch:1*
_output_shapes	
:?*
dtype0
?
3batch_normalization_10/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_10/beta#batch_normalization_10/cond/pred_id*.
_class$
" loc:@batch_normalization_10/beta*
T0*
_output_shapes
: : 
?
!batch_normalization_10/cond/ConstConst%^batch_normalization_10/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
#batch_normalization_10/cond/Const_1Const%^batch_normalization_10/cond/switch_t*
valueB *
_output_shapes
: *
dtype0
?
*batch_normalization_10/cond/FusedBatchNormFusedBatchNorm3batch_normalization_10/cond/FusedBatchNorm/Switch:1*batch_normalization_10/cond/ReadVariableOp,batch_normalization_10/cond/ReadVariableOp_1!batch_normalization_10/cond/Const#batch_normalization_10/cond/Const_1*
is_training(*
data_formatNHWC*
T0*
epsilon%?ŧ7*L
_output_shapes:
8:??????????:?:?:?:?
?
1batch_normalization_10/cond/FusedBatchNorm/SwitchSwitch	add_3/add#batch_normalization_10/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
T0*
_class
loc:@add_3/add
?
,batch_normalization_10/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_10/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_10/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_10/gamma#batch_normalization_10/cond/pred_id*
T0*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_10/gamma
?
,batch_normalization_10/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_10/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_10/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_10/beta#batch_normalization_10/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_10/beta
?
;batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Bbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_10/moving_mean#batch_normalization_10/cond/pred_id*
T0*
_output_shapes
: : *5
_class+
)'loc:@batch_normalization_10/moving_mean
?
=batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Dbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_10/moving_variance#batch_normalization_10/cond/pred_id*9
_class/
-+loc:@batch_normalization_10/moving_variance*
_output_shapes
: : *
T0
?
,batch_normalization_10/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_10/cond/FusedBatchNorm_1/Switch,batch_normalization_10/cond/ReadVariableOp_2,batch_normalization_10/cond/ReadVariableOp_3;batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
T0*
is_training( *
epsilon%?ŧ7
?
3batch_normalization_10/cond/FusedBatchNorm_1/SwitchSwitch	add_3/add#batch_normalization_10/cond/pred_id*
T0*
_class
loc:@add_3/add*L
_output_shapes:
8:??????????:??????????
?
!batch_normalization_10/cond/MergeMerge,batch_normalization_10/cond/FusedBatchNorm_1*batch_normalization_10/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:??????????: 
?
#batch_normalization_10/cond/Merge_1Merge.batch_normalization_10/cond/FusedBatchNorm_1:1,batch_normalization_10/cond/FusedBatchNorm:1*
N*
_output_shapes
	:?: *
T0
?
#batch_normalization_10/cond/Merge_2Merge.batch_normalization_10/cond/FusedBatchNorm_1:2,batch_normalization_10/cond/FusedBatchNorm:2*
T0*
N*
_output_shapes
	:?: 
}
$batch_normalization_10/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

{
&batch_normalization_10/cond_1/switch_tIdentity&batch_normalization_10/cond_1/Switch:1*
_output_shapes
: *
T0

y
&batch_normalization_10/cond_1/switch_fIdentity$batch_normalization_10/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_10/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_10/cond_1/ConstConst'^batch_normalization_10/cond_1/switch_t*
dtype0*
valueB
 *?p}?*
_output_shapes
: 
?
%batch_normalization_10/cond_1/Const_1Const'^batch_normalization_10/cond_1/switch_f*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
#batch_normalization_10/cond_1/MergeMerge%batch_normalization_10/cond_1/Const_1#batch_normalization_10/cond_1/Const*
T0*
N*
_output_shapes
: : 
?
,batch_normalization_10/AssignMovingAvg/sub/xConst*5
_class+
)'loc:@batch_normalization_10/moving_mean*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
*batch_normalization_10/AssignMovingAvg/subSub,batch_normalization_10/AssignMovingAvg/sub/x#batch_normalization_10/cond_1/Merge*5
_class+
)'loc:@batch_normalization_10/moving_mean*
_output_shapes
: *
T0
?
5batch_normalization_10/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_10/moving_mean*
_output_shapes	
:?*
dtype0
?
,batch_normalization_10/AssignMovingAvg/sub_1Sub5batch_normalization_10/AssignMovingAvg/ReadVariableOp#batch_normalization_10/cond/Merge_1*5
_class+
)'loc:@batch_normalization_10/moving_mean*
_output_shapes	
:?*
T0
?
*batch_normalization_10/AssignMovingAvg/mulMul,batch_normalization_10/AssignMovingAvg/sub_1*batch_normalization_10/AssignMovingAvg/sub*
T0*5
_class+
)'loc:@batch_normalization_10/moving_mean*
_output_shapes	
:?
?
:batch_normalization_10/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_10/moving_mean*batch_normalization_10/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_10/moving_mean*
dtype0
?
7batch_normalization_10/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_10/moving_mean;^batch_normalization_10/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:?*
dtype0*5
_class+
)'loc:@batch_normalization_10/moving_mean
?
.batch_normalization_10/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_10/moving_variance*
dtype0*
valueB
 *  ??
?
,batch_normalization_10/AssignMovingAvg_1/subSub.batch_normalization_10/AssignMovingAvg_1/sub/x#batch_normalization_10/cond_1/Merge*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_10/moving_variance*
T0
?
7batch_normalization_10/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_10/moving_variance*
_output_shapes	
:?*
dtype0
?
.batch_normalization_10/AssignMovingAvg_1/sub_1Sub7batch_normalization_10/AssignMovingAvg_1/ReadVariableOp#batch_normalization_10/cond/Merge_2*9
_class/
-+loc:@batch_normalization_10/moving_variance*
T0*
_output_shapes	
:?
?
,batch_normalization_10/AssignMovingAvg_1/mulMul.batch_normalization_10/AssignMovingAvg_1/sub_1,batch_normalization_10/AssignMovingAvg_1/sub*9
_class/
-+loc:@batch_normalization_10/moving_variance*
_output_shapes	
:?*
T0
?
<batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_10/moving_variance,batch_normalization_10/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_10/moving_variance*
dtype0
?
9batch_normalization_10/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_10/moving_variance=^batch_normalization_10/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_10/moving_variance
w
activation_9/ReluRelu!batch_normalization_10/cond/Merge*
T0*0
_output_shapes
:??????????
?
1conv2d_11/kernel/Initializer/random_uniform/shapeConst*%
valueB"      ?      *#
_class
loc:@conv2d_11/kernel*
dtype0*
_output_shapes
:
?
/conv2d_11/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *׳]?*
_output_shapes
: *#
_class
loc:@conv2d_11/kernel
?
/conv2d_11/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*
dtype0*#
_class
loc:@conv2d_11/kernel*
_output_shapes
: 
?
9conv2d_11/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_11/kernel/Initializer/random_uniform/shape*(
_output_shapes
:??*

seed *#
_class
loc:@conv2d_11/kernel*
seed2 *
T0*
dtype0
?
/conv2d_11/kernel/Initializer/random_uniform/subSub/conv2d_11/kernel/Initializer/random_uniform/max/conv2d_11/kernel/Initializer/random_uniform/min*#
_class
loc:@conv2d_11/kernel*
_output_shapes
: *
T0
?
/conv2d_11/kernel/Initializer/random_uniform/mulMul9conv2d_11/kernel/Initializer/random_uniform/RandomUniform/conv2d_11/kernel/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_11/kernel
?
+conv2d_11/kernel/Initializer/random_uniformAdd/conv2d_11/kernel/Initializer/random_uniform/mul/conv2d_11/kernel/Initializer/random_uniform/min*#
_class
loc:@conv2d_11/kernel*(
_output_shapes
:??*
T0
?
conv2d_11/kernelVarHandleOp*!
shared_nameconv2d_11/kernel*
shape:??*
	container *
_output_shapes
: *#
_class
loc:@conv2d_11/kernel*
dtype0
q
1conv2d_11/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_11/kernel*
_output_shapes
: 
?
conv2d_11/kernel/AssignAssignVariableOpconv2d_11/kernel+conv2d_11/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_11/kernel*
dtype0
?
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:??*
dtype0*#
_class
loc:@conv2d_11/kernel
h
conv2d_11/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
z
conv2d_11/Conv2D/ReadVariableOpReadVariableOpconv2d_11/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_11/Conv2DConv2Dactivation_9/Reluconv2d_11/Conv2D/ReadVariableOp*0
_output_shapes
:??????????*
	dilations
*
data_formatNHWC*
explicit_paddings
 *
T0*
use_cudnn_on_gpu(*
strides
*
paddingVALID
?
zero_padding2d_10/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
zero_padding2d_10/PadPadactivation_9/Reluzero_padding2d_10/Pad/paddings*
T0*
	Tpaddings0*0
_output_shapes
:?????????

?
?
1conv2d_12/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"      ?      *#
_class
loc:@conv2d_12/kernel*
dtype0
?
/conv2d_12/kernel/Initializer/random_uniform/minConst*
valueB
 *:͓?*
dtype0*#
_class
loc:@conv2d_12/kernel*
_output_shapes
: 
?
/conv2d_12/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:͓=*#
_class
loc:@conv2d_12/kernel*
dtype0
?
9conv2d_12/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_12/kernel/Initializer/random_uniform/shape*
dtype0*
seed2 *#
_class
loc:@conv2d_12/kernel*
T0*(
_output_shapes
:??*

seed 
?
/conv2d_12/kernel/Initializer/random_uniform/subSub/conv2d_12/kernel/Initializer/random_uniform/max/conv2d_12/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *#
_class
loc:@conv2d_12/kernel
?
/conv2d_12/kernel/Initializer/random_uniform/mulMul9conv2d_12/kernel/Initializer/random_uniform/RandomUniform/conv2d_12/kernel/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_12/kernel
?
+conv2d_12/kernel/Initializer/random_uniformAdd/conv2d_12/kernel/Initializer/random_uniform/mul/conv2d_12/kernel/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_12/kernel
?
conv2d_12/kernelVarHandleOp*
dtype0*
_output_shapes
: *
	container *!
shared_nameconv2d_12/kernel*
shape:??*#
_class
loc:@conv2d_12/kernel
q
1conv2d_12/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_12/kernel*
_output_shapes
: 
?
conv2d_12/kernel/AssignAssignVariableOpconv2d_12/kernel+conv2d_12/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_12/kernel*
dtype0
?
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:??*
dtype0*#
_class
loc:@conv2d_12/kernel
h
conv2d_12/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
conv2d_12/Conv2D/ReadVariableOpReadVariableOpconv2d_12/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_12/Conv2DConv2Dzero_padding2d_10/Padconv2d_12/Conv2D/ReadVariableOp*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
	dilations
*0
_output_shapes
:??????????*
explicit_paddings
 *
data_formatNHWC
?
-batch_normalization_11/gamma/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*/
_class%
#!loc:@batch_normalization_11/gamma
?
batch_normalization_11/gammaVarHandleOp*
dtype0*
_output_shapes
: *
	container */
_class%
#!loc:@batch_normalization_11/gamma*-
shared_namebatch_normalization_11/gamma*
shape:?
?
=batch_normalization_11/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_11/gamma*
_output_shapes
: 
?
#batch_normalization_11/gamma/AssignAssignVariableOpbatch_normalization_11/gamma-batch_normalization_11/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_11/gamma*
dtype0
?
0batch_normalization_11/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_11/gamma*/
_class%
#!loc:@batch_normalization_11/gamma*
_output_shapes	
:?*
dtype0
?
-batch_normalization_11/beta/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_11/beta
?
batch_normalization_11/betaVarHandleOp*
dtype0*,
shared_namebatch_normalization_11/beta*.
_class$
" loc:@batch_normalization_11/beta*
	container *
shape:?*
_output_shapes
: 
?
<batch_normalization_11/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_11/beta*
_output_shapes
: 
?
"batch_normalization_11/beta/AssignAssignVariableOpbatch_normalization_11/beta-batch_normalization_11/beta/Initializer/zeros*
dtype0*.
_class$
" loc:@batch_normalization_11/beta
?
/batch_normalization_11/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_11/beta*
dtype0*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_11/beta
?
4batch_normalization_11/moving_mean/Initializer/zerosConst*5
_class+
)'loc:@batch_normalization_11/moving_mean*
valueB?*    *
_output_shapes	
:?*
dtype0
?
"batch_normalization_11/moving_meanVarHandleOp*
dtype0*
shape:?*
	container *
_output_shapes
: *3
shared_name$"batch_normalization_11/moving_mean*5
_class+
)'loc:@batch_normalization_11/moving_mean
?
Cbatch_normalization_11/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_11/moving_mean*
_output_shapes
: 
?
)batch_normalization_11/moving_mean/AssignAssignVariableOp"batch_normalization_11/moving_mean4batch_normalization_11/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_11/moving_mean*
dtype0
?
6batch_normalization_11/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_11/moving_mean*
dtype0
?
7batch_normalization_11/moving_variance/Initializer/onesConst*9
_class/
-+loc:@batch_normalization_11/moving_variance*
_output_shapes	
:?*
valueB?*  ??*
dtype0
?
&batch_normalization_11/moving_varianceVarHandleOp*
	container *
shape:?*
dtype0*9
_class/
-+loc:@batch_normalization_11/moving_variance*7
shared_name(&batch_normalization_11/moving_variance*
_output_shapes
: 
?
Gbatch_normalization_11/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_11/moving_variance*
_output_shapes
: 
?
-batch_normalization_11/moving_variance/AssignAssignVariableOp&batch_normalization_11/moving_variance7batch_normalization_11/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_11/moving_variance*
dtype0
?
:batch_normalization_11/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_11/moving_variance*
dtype0
{
"batch_normalization_11/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

w
$batch_normalization_11/cond/switch_tIdentity$batch_normalization_11/cond/Switch:1*
_output_shapes
: *
T0

u
$batch_normalization_11/cond/switch_fIdentity"batch_normalization_11/cond/Switch*
T0
*
_output_shapes
: 
f
#batch_normalization_11/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
*batch_normalization_11/cond/ReadVariableOpReadVariableOp3batch_normalization_11/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
1batch_normalization_11/cond/ReadVariableOp/SwitchSwitchbatch_normalization_11/gamma#batch_normalization_11/cond/pred_id*/
_class%
#!loc:@batch_normalization_11/gamma*
T0*
_output_shapes
: : 
?
,batch_normalization_11/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_11/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_11/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_11/beta#batch_normalization_11/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_11/beta
?
!batch_normalization_11/cond/ConstConst%^batch_normalization_11/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
#batch_normalization_11/cond/Const_1Const%^batch_normalization_11/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
*batch_normalization_11/cond/FusedBatchNormFusedBatchNorm3batch_normalization_11/cond/FusedBatchNorm/Switch:1*batch_normalization_11/cond/ReadVariableOp,batch_normalization_11/cond/ReadVariableOp_1!batch_normalization_11/cond/Const#batch_normalization_11/cond/Const_1*L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
epsilon%?ŧ7*
T0*
is_training(
?
1batch_normalization_11/cond/FusedBatchNorm/SwitchSwitchconv2d_12/Conv2D#batch_normalization_11/cond/pred_id*#
_class
loc:@conv2d_12/Conv2D*L
_output_shapes:
8:??????????:??????????*
T0
?
,batch_normalization_11/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_11/cond/ReadVariableOp_2/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_11/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_11/gamma#batch_normalization_11/cond/pred_id*/
_class%
#!loc:@batch_normalization_11/gamma*
T0*
_output_shapes
: : 
?
,batch_normalization_11/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_11/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_11/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_11/beta#batch_normalization_11/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_11/beta
?
;batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Bbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_11/moving_mean#batch_normalization_11/cond/pred_id*
T0*
_output_shapes
: : *5
_class+
)'loc:@batch_normalization_11/moving_mean
?
=batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Dbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_11/moving_variance#batch_normalization_11/cond/pred_id*
_output_shapes
: : *
T0*9
_class/
-+loc:@batch_normalization_11/moving_variance
?
,batch_normalization_11/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_11/cond/FusedBatchNorm_1/Switch,batch_normalization_11/cond/ReadVariableOp_2,batch_normalization_11/cond/ReadVariableOp_3;batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
T0*
epsilon%?ŧ7*
is_training( 
?
3batch_normalization_11/cond/FusedBatchNorm_1/SwitchSwitchconv2d_12/Conv2D#batch_normalization_11/cond/pred_id*L
_output_shapes:
8:??????????:??????????*#
_class
loc:@conv2d_12/Conv2D*
T0
?
!batch_normalization_11/cond/MergeMerge,batch_normalization_11/cond/FusedBatchNorm_1*batch_normalization_11/cond/FusedBatchNorm*
N*
T0*2
_output_shapes 
:??????????: 
?
#batch_normalization_11/cond/Merge_1Merge.batch_normalization_11/cond/FusedBatchNorm_1:1,batch_normalization_11/cond/FusedBatchNorm:1*
T0*
_output_shapes
	:?: *
N
?
#batch_normalization_11/cond/Merge_2Merge.batch_normalization_11/cond/FusedBatchNorm_1:2,batch_normalization_11/cond/FusedBatchNorm:2*
N*
T0*
_output_shapes
	:?: 
}
$batch_normalization_11/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

{
&batch_normalization_11/cond_1/switch_tIdentity&batch_normalization_11/cond_1/Switch:1*
_output_shapes
: *
T0

y
&batch_normalization_11/cond_1/switch_fIdentity$batch_normalization_11/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_11/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_11/cond_1/ConstConst'^batch_normalization_11/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *?p}?
?
%batch_normalization_11/cond_1/Const_1Const'^batch_normalization_11/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#batch_normalization_11/cond_1/MergeMerge%batch_normalization_11/cond_1/Const_1#batch_normalization_11/cond_1/Const*
T0*
N*
_output_shapes
: : 
?
,batch_normalization_11/AssignMovingAvg/sub/xConst*
valueB
 *  ??*5
_class+
)'loc:@batch_normalization_11/moving_mean*
dtype0*
_output_shapes
: 
?
*batch_normalization_11/AssignMovingAvg/subSub,batch_normalization_11/AssignMovingAvg/sub/x#batch_normalization_11/cond_1/Merge*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_11/moving_mean*
T0
?
5batch_normalization_11/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_11/moving_mean*
_output_shapes	
:?*
dtype0
?
,batch_normalization_11/AssignMovingAvg/sub_1Sub5batch_normalization_11/AssignMovingAvg/ReadVariableOp#batch_normalization_11/cond/Merge_1*
T0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_11/moving_mean
?
*batch_normalization_11/AssignMovingAvg/mulMul,batch_normalization_11/AssignMovingAvg/sub_1*batch_normalization_11/AssignMovingAvg/sub*
T0*5
_class+
)'loc:@batch_normalization_11/moving_mean*
_output_shapes	
:?
?
:batch_normalization_11/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_11/moving_mean*batch_normalization_11/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_11/moving_mean*
dtype0
?
7batch_normalization_11/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_11/moving_mean;^batch_normalization_11/AssignMovingAvg/AssignSubVariableOp*
dtype0*5
_class+
)'loc:@batch_normalization_11/moving_mean*
_output_shapes	
:?
?
.batch_normalization_11/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0*9
_class/
-+loc:@batch_normalization_11/moving_variance
?
,batch_normalization_11/AssignMovingAvg_1/subSub.batch_normalization_11/AssignMovingAvg_1/sub/x#batch_normalization_11/cond_1/Merge*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_11/moving_variance
?
7batch_normalization_11/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_11/moving_variance*
dtype0*
_output_shapes	
:?
?
.batch_normalization_11/AssignMovingAvg_1/sub_1Sub7batch_normalization_11/AssignMovingAvg_1/ReadVariableOp#batch_normalization_11/cond/Merge_2*9
_class/
-+loc:@batch_normalization_11/moving_variance*
_output_shapes	
:?*
T0
?
,batch_normalization_11/AssignMovingAvg_1/mulMul.batch_normalization_11/AssignMovingAvg_1/sub_1,batch_normalization_11/AssignMovingAvg_1/sub*9
_class/
-+loc:@batch_normalization_11/moving_variance*
T0*
_output_shapes	
:?
?
<batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_11/moving_variance,batch_normalization_11/AssignMovingAvg_1/mul*
dtype0*9
_class/
-+loc:@batch_normalization_11/moving_variance
?
9batch_normalization_11/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_11/moving_variance=^batch_normalization_11/AssignMovingAvg_1/AssignSubVariableOp*9
_class/
-+loc:@batch_normalization_11/moving_variance*
_output_shapes	
:?*
dtype0
x
activation_10/ReluRelu!batch_normalization_11/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_11/Pad/paddingsConst*9
value0B."                             *
_output_shapes

:*
dtype0
?
zero_padding2d_11/PadPadactivation_10/Reluzero_padding2d_11/Pad/paddings*
	Tpaddings0*
T0*0
_output_shapes
:??????????
?
1conv2d_13/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*#
_class
loc:@conv2d_13/kernel
?
/conv2d_13/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?Q?*#
_class
loc:@conv2d_13/kernel
?
/conv2d_13/kernel/Initializer/random_uniform/maxConst*
dtype0*
valueB
 *?Q=*#
_class
loc:@conv2d_13/kernel*
_output_shapes
: 
?
9conv2d_13/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_13/kernel/Initializer/random_uniform/shape*

seed *(
_output_shapes
:??*
T0*#
_class
loc:@conv2d_13/kernel*
seed2 *
dtype0
?
/conv2d_13/kernel/Initializer/random_uniform/subSub/conv2d_13/kernel/Initializer/random_uniform/max/conv2d_13/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@conv2d_13/kernel*
T0
?
/conv2d_13/kernel/Initializer/random_uniform/mulMul9conv2d_13/kernel/Initializer/random_uniform/RandomUniform/conv2d_13/kernel/Initializer/random_uniform/sub*#
_class
loc:@conv2d_13/kernel*(
_output_shapes
:??*
T0
?
+conv2d_13/kernel/Initializer/random_uniformAdd/conv2d_13/kernel/Initializer/random_uniform/mul/conv2d_13/kernel/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_13/kernel
?
conv2d_13/kernelVarHandleOp*
dtype0*
_output_shapes
: *#
_class
loc:@conv2d_13/kernel*
	container *
shape:??*!
shared_nameconv2d_13/kernel
q
1conv2d_13/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_13/kernel*
_output_shapes
: 
?
conv2d_13/kernel/AssignAssignVariableOpconv2d_13/kernel+conv2d_13/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_13/kernel*
dtype0
?
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*#
_class
loc:@conv2d_13/kernel*(
_output_shapes
:??*
dtype0
h
conv2d_13/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
z
conv2d_13/Conv2D/ReadVariableOpReadVariableOpconv2d_13/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_13/Conv2DConv2Dzero_padding2d_11/Padconv2d_13/Conv2D/ReadVariableOp*
explicit_paddings
 *
	dilations
*
strides
*
data_formatNHWC*
T0*0
_output_shapes
:??????????*
use_cudnn_on_gpu(*
paddingVALID
o
	add_4/addAddconv2d_13/Conv2Dconv2d_11/Conv2D*
T0*0
_output_shapes
:??????????
?
-batch_normalization_12/gamma/Initializer/onesConst*
valueB?*  ??*
dtype0*/
_class%
#!loc:@batch_normalization_12/gamma*
_output_shapes	
:?
?
batch_normalization_12/gammaVarHandleOp*-
shared_namebatch_normalization_12/gamma*
dtype0*/
_class%
#!loc:@batch_normalization_12/gamma*
	container *
shape:?*
_output_shapes
: 
?
=batch_normalization_12/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_12/gamma*
_output_shapes
: 
?
#batch_normalization_12/gamma/AssignAssignVariableOpbatch_normalization_12/gamma-batch_normalization_12/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_12/gamma*
dtype0
?
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:?*/
_class%
#!loc:@batch_normalization_12/gamma*
dtype0
?
-batch_normalization_12/beta/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_12/beta
?
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *,
shared_namebatch_normalization_12/beta*
shape:?*
	container *
dtype0*.
_class$
" loc:@batch_normalization_12/beta
?
<batch_normalization_12/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_12/beta*
_output_shapes
: 
?
"batch_normalization_12/beta/AssignAssignVariableOpbatch_normalization_12/beta-batch_normalization_12/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_12/beta*
dtype0
?
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
dtype0*.
_class$
" loc:@batch_normalization_12/beta*
_output_shapes	
:?
?
4batch_normalization_12/moving_mean/Initializer/zerosConst*
dtype0*
valueB?*    *
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_12/moving_mean
?
"batch_normalization_12/moving_meanVarHandleOp*5
_class+
)'loc:@batch_normalization_12/moving_mean*3
shared_name$"batch_normalization_12/moving_mean*
_output_shapes
: *
shape:?*
dtype0*
	container 
?
Cbatch_normalization_12/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_12/moving_mean*
_output_shapes
: 
?
)batch_normalization_12/moving_mean/AssignAssignVariableOp"batch_normalization_12/moving_mean4batch_normalization_12/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_12/moving_mean*
dtype0
?
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
dtype0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_12/moving_mean
?
7batch_normalization_12/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_12/moving_variance*
dtype0
?
&batch_normalization_12/moving_varianceVarHandleOp*9
_class/
-+loc:@batch_normalization_12/moving_variance*
shape:?*7
shared_name(&batch_normalization_12/moving_variance*
dtype0*
_output_shapes
: *
	container 
?
Gbatch_normalization_12/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_12/moving_variance*
_output_shapes
: 
?
-batch_normalization_12/moving_variance/AssignAssignVariableOp&batch_normalization_12/moving_variance7batch_normalization_12/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_12/moving_variance*
dtype0
?
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_12/moving_variance*
dtype0
{
"batch_normalization_12/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
w
$batch_normalization_12/cond/switch_tIdentity$batch_normalization_12/cond/Switch:1*
_output_shapes
: *
T0

u
$batch_normalization_12/cond/switch_fIdentity"batch_normalization_12/cond/Switch*
T0
*
_output_shapes
: 
f
#batch_normalization_12/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
*batch_normalization_12/cond/ReadVariableOpReadVariableOp3batch_normalization_12/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
1batch_normalization_12/cond/ReadVariableOp/SwitchSwitchbatch_normalization_12/gamma#batch_normalization_12/cond/pred_id*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_12/gamma*
T0
?
,batch_normalization_12/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_12/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_12/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_12/beta#batch_normalization_12/cond/pred_id*.
_class$
" loc:@batch_normalization_12/beta*
_output_shapes
: : *
T0
?
!batch_normalization_12/cond/ConstConst%^batch_normalization_12/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
#batch_normalization_12/cond/Const_1Const%^batch_normalization_12/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
*batch_normalization_12/cond/FusedBatchNormFusedBatchNorm3batch_normalization_12/cond/FusedBatchNorm/Switch:1*batch_normalization_12/cond/ReadVariableOp,batch_normalization_12/cond/ReadVariableOp_1!batch_normalization_12/cond/Const#batch_normalization_12/cond/Const_1*
data_formatNHWC*
T0*
epsilon%?ŧ7*
is_training(*L
_output_shapes:
8:??????????:?:?:?:?
?
1batch_normalization_12/cond/FusedBatchNorm/SwitchSwitch	add_4/add#batch_normalization_12/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_4/add*
T0
?
,batch_normalization_12/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_12/cond/ReadVariableOp_2/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_12/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_12/gamma#batch_normalization_12/cond/pred_id*
T0*/
_class%
#!loc:@batch_normalization_12/gamma*
_output_shapes
: : 
?
,batch_normalization_12/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_12/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_12/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_12/beta#batch_normalization_12/cond/pred_id*.
_class$
" loc:@batch_normalization_12/beta*
T0*
_output_shapes
: : 
?
;batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:?*
dtype0
?
Bbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_12/moving_mean#batch_normalization_12/cond/pred_id*
_output_shapes
: : *
T0*5
_class+
)'loc:@batch_normalization_12/moving_mean
?
=batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Dbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_12/moving_variance#batch_normalization_12/cond/pred_id*
_output_shapes
: : *
T0*9
_class/
-+loc:@batch_normalization_12/moving_variance
?
,batch_normalization_12/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_12/cond/FusedBatchNorm_1/Switch,batch_normalization_12/cond/ReadVariableOp_2,batch_normalization_12/cond/ReadVariableOp_3;batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1*
data_formatNHWC*
is_training( *L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%?ŧ7*
T0
?
3batch_normalization_12/cond/FusedBatchNorm_1/SwitchSwitch	add_4/add#batch_normalization_12/cond/pred_id*
T0*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_4/add
?
!batch_normalization_12/cond/MergeMerge,batch_normalization_12/cond/FusedBatchNorm_1*batch_normalization_12/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
N*
T0
?
#batch_normalization_12/cond/Merge_1Merge.batch_normalization_12/cond/FusedBatchNorm_1:1,batch_normalization_12/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:?: 
?
#batch_normalization_12/cond/Merge_2Merge.batch_normalization_12/cond/FusedBatchNorm_1:2,batch_normalization_12/cond/FusedBatchNorm:2*
_output_shapes
	:?: *
N*
T0
}
$batch_normalization_12/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
{
&batch_normalization_12/cond_1/switch_tIdentity&batch_normalization_12/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_12/cond_1/switch_fIdentity$batch_normalization_12/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_12/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_12/cond_1/ConstConst'^batch_normalization_12/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *?p}?
?
%batch_normalization_12/cond_1/Const_1Const'^batch_normalization_12/cond_1/switch_f*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
#batch_normalization_12/cond_1/MergeMerge%batch_normalization_12/cond_1/Const_1#batch_normalization_12/cond_1/Const*
_output_shapes
: : *
N*
T0
?
,batch_normalization_12/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_12/moving_mean*
dtype0
?
*batch_normalization_12/AssignMovingAvg/subSub,batch_normalization_12/AssignMovingAvg/sub/x#batch_normalization_12/cond_1/Merge*
T0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_12/moving_mean
?
5batch_normalization_12/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_12/AssignMovingAvg/sub_1Sub5batch_normalization_12/AssignMovingAvg/ReadVariableOp#batch_normalization_12/cond/Merge_1*
T0*5
_class+
)'loc:@batch_normalization_12/moving_mean*
_output_shapes	
:?
?
*batch_normalization_12/AssignMovingAvg/mulMul,batch_normalization_12/AssignMovingAvg/sub_1*batch_normalization_12/AssignMovingAvg/sub*
_output_shapes	
:?*
T0*5
_class+
)'loc:@batch_normalization_12/moving_mean
?
:batch_normalization_12/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_12/moving_mean*batch_normalization_12/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_12/moving_mean*
dtype0
?
7batch_normalization_12/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_12/moving_mean;^batch_normalization_12/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_12/moving_mean*
dtype0
?
.batch_normalization_12/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
_output_shapes
: *
dtype0*9
_class/
-+loc:@batch_normalization_12/moving_variance
?
,batch_normalization_12/AssignMovingAvg_1/subSub.batch_normalization_12/AssignMovingAvg_1/sub/x#batch_normalization_12/cond_1/Merge*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_12/moving_variance*
T0
?
7batch_normalization_12/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
dtype0*
_output_shapes	
:?
?
.batch_normalization_12/AssignMovingAvg_1/sub_1Sub7batch_normalization_12/AssignMovingAvg_1/ReadVariableOp#batch_normalization_12/cond/Merge_2*9
_class/
-+loc:@batch_normalization_12/moving_variance*
T0*
_output_shapes	
:?
?
,batch_normalization_12/AssignMovingAvg_1/mulMul.batch_normalization_12/AssignMovingAvg_1/sub_1,batch_normalization_12/AssignMovingAvg_1/sub*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_12/moving_variance*
T0
?
<batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_12/moving_variance,batch_normalization_12/AssignMovingAvg_1/mul*
dtype0*9
_class/
-+loc:@batch_normalization_12/moving_variance
?
9batch_normalization_12/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_12/moving_variance=^batch_normalization_12/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_12/moving_variance*
dtype0
x
activation_11/ReluRelu!batch_normalization_12/cond/Merge*
T0*0
_output_shapes
:??????????
?
zero_padding2d_12/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
zero_padding2d_12/PadPadactivation_11/Reluzero_padding2d_12/Pad/paddings*0
_output_shapes
:??????????*
	Tpaddings0*
T0
?
1conv2d_14/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            *#
_class
loc:@conv2d_14/kernel
?
/conv2d_14/kernel/Initializer/random_uniform/minConst*
valueB
 *?Q?*
dtype0*#
_class
loc:@conv2d_14/kernel*
_output_shapes
: 
?
/conv2d_14/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *?Q=*#
_class
loc:@conv2d_14/kernel*
dtype0
?
9conv2d_14/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_14/kernel/Initializer/random_uniform/shape*#
_class
loc:@conv2d_14/kernel*
seed2 *(
_output_shapes
:??*
dtype0*
T0*

seed 
?
/conv2d_14/kernel/Initializer/random_uniform/subSub/conv2d_14/kernel/Initializer/random_uniform/max/conv2d_14/kernel/Initializer/random_uniform/min*#
_class
loc:@conv2d_14/kernel*
_output_shapes
: *
T0
?
/conv2d_14/kernel/Initializer/random_uniform/mulMul9conv2d_14/kernel/Initializer/random_uniform/RandomUniform/conv2d_14/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@conv2d_14/kernel*(
_output_shapes
:??
?
+conv2d_14/kernel/Initializer/random_uniformAdd/conv2d_14/kernel/Initializer/random_uniform/mul/conv2d_14/kernel/Initializer/random_uniform/min*#
_class
loc:@conv2d_14/kernel*(
_output_shapes
:??*
T0
?
conv2d_14/kernelVarHandleOp*
	container *
shape:??*#
_class
loc:@conv2d_14/kernel*
_output_shapes
: *
dtype0*!
shared_nameconv2d_14/kernel
q
1conv2d_14/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_14/kernel*
_output_shapes
: 
?
conv2d_14/kernel/AssignAssignVariableOpconv2d_14/kernel+conv2d_14/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_14/kernel*
dtype0
?
$conv2d_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:??*
dtype0*#
_class
loc:@conv2d_14/kernel
h
conv2d_14/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
z
conv2d_14/Conv2D/ReadVariableOpReadVariableOpconv2d_14/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_14/Conv2DConv2Dzero_padding2d_12/Padconv2d_14/Conv2D/ReadVariableOp*0
_output_shapes
:??????????*
data_formatNHWC*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID*
strides
*
T0*
explicit_paddings
 
?
-batch_normalization_13/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*/
_class%
#!loc:@batch_normalization_13/gamma*
_output_shapes	
:?
?
batch_normalization_13/gammaVarHandleOp*/
_class%
#!loc:@batch_normalization_13/gamma*
dtype0*
_output_shapes
: *
	container *
shape:?*-
shared_namebatch_normalization_13/gamma
?
=batch_normalization_13/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_13/gamma*
_output_shapes
: 
?
#batch_normalization_13/gamma/AssignAssignVariableOpbatch_normalization_13/gamma-batch_normalization_13/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_13/gamma*
dtype0
?
0batch_normalization_13/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_13/gamma*
_output_shapes	
:?*
dtype0*/
_class%
#!loc:@batch_normalization_13/gamma
?
-batch_normalization_13/beta/Initializer/zerosConst*
valueB?*    *.
_class$
" loc:@batch_normalization_13/beta*
dtype0*
_output_shapes	
:?
?
batch_normalization_13/betaVarHandleOp*
_output_shapes
: *.
_class$
" loc:@batch_normalization_13/beta*
	container *,
shared_namebatch_normalization_13/beta*
dtype0*
shape:?
?
<batch_normalization_13/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_13/beta*
_output_shapes
: 
?
"batch_normalization_13/beta/AssignAssignVariableOpbatch_normalization_13/beta-batch_normalization_13/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_13/beta*
dtype0
?
/batch_normalization_13/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_13/beta*
dtype0*.
_class$
" loc:@batch_normalization_13/beta*
_output_shapes	
:?
?
4batch_normalization_13/moving_mean/Initializer/zerosConst*
valueB?*    *5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes	
:?*
dtype0
?
"batch_normalization_13/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*3
shared_name$"batch_normalization_13/moving_mean*5
_class+
)'loc:@batch_normalization_13/moving_mean*
	container *
shape:?
?
Cbatch_normalization_13/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_13/moving_mean*
_output_shapes
: 
?
)batch_normalization_13/moving_mean/AssignAssignVariableOp"batch_normalization_13/moving_mean4batch_normalization_13/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_13/moving_mean*
dtype0
?
6batch_normalization_13/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
dtype0*5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes	
:?
?
7batch_normalization_13/moving_variance/Initializer/onesConst*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_13/moving_variance*
dtype0*
valueB?*  ??
?
&batch_normalization_13/moving_varianceVarHandleOp*
	container *
_output_shapes
: *
shape:?*7
shared_name(&batch_normalization_13/moving_variance*
dtype0*9
_class/
-+loc:@batch_normalization_13/moving_variance
?
Gbatch_normalization_13/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_13/moving_variance*
_output_shapes
: 
?
-batch_normalization_13/moving_variance/AssignAssignVariableOp&batch_normalization_13/moving_variance7batch_normalization_13/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_13/moving_variance*
dtype0
?
:batch_normalization_13/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*9
_class/
-+loc:@batch_normalization_13/moving_variance*
_output_shapes	
:?*
dtype0
{
"batch_normalization_13/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
w
$batch_normalization_13/cond/switch_tIdentity$batch_normalization_13/cond/Switch:1*
T0
*
_output_shapes
: 
u
$batch_normalization_13/cond/switch_fIdentity"batch_normalization_13/cond/Switch*
T0
*
_output_shapes
: 
f
#batch_normalization_13/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
*batch_normalization_13/cond/ReadVariableOpReadVariableOp3batch_normalization_13/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
1batch_normalization_13/cond/ReadVariableOp/SwitchSwitchbatch_normalization_13/gamma#batch_normalization_13/cond/pred_id*
T0*/
_class%
#!loc:@batch_normalization_13/gamma*
_output_shapes
: : 
?
,batch_normalization_13/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_13/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_13/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_13/beta#batch_normalization_13/cond/pred_id*.
_class$
" loc:@batch_normalization_13/beta*
T0*
_output_shapes
: : 
?
!batch_normalization_13/cond/ConstConst%^batch_normalization_13/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
#batch_normalization_13/cond/Const_1Const%^batch_normalization_13/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
*batch_normalization_13/cond/FusedBatchNormFusedBatchNorm3batch_normalization_13/cond/FusedBatchNorm/Switch:1*batch_normalization_13/cond/ReadVariableOp,batch_normalization_13/cond/ReadVariableOp_1!batch_normalization_13/cond/Const#batch_normalization_13/cond/Const_1*
epsilon%?ŧ7*
is_training(*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?
?
1batch_normalization_13/cond/FusedBatchNorm/SwitchSwitchconv2d_14/Conv2D#batch_normalization_13/cond/pred_id*L
_output_shapes:
8:??????????:??????????*#
_class
loc:@conv2d_14/Conv2D*
T0
?
,batch_normalization_13/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_13/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_13/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_13/gamma#batch_normalization_13/cond/pred_id*/
_class%
#!loc:@batch_normalization_13/gamma*
_output_shapes
: : *
T0
?
,batch_normalization_13/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_13/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_13/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_13/beta#batch_normalization_13/cond/pred_id*.
_class$
" loc:@batch_normalization_13/beta*
_output_shapes
: : *
T0
?
;batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:?*
dtype0
?
Bbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_13/moving_mean#batch_normalization_13/cond/pred_id*
T0*
_output_shapes
: : *5
_class+
)'loc:@batch_normalization_13/moving_mean
?
=batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Dbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_13/moving_variance#batch_normalization_13/cond/pred_id*
_output_shapes
: : *9
_class/
-+loc:@batch_normalization_13/moving_variance*
T0
?
,batch_normalization_13/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_13/cond/FusedBatchNorm_1/Switch,batch_normalization_13/cond/ReadVariableOp_2,batch_normalization_13/cond/ReadVariableOp_3;batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1*
is_training( *L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
epsilon%?ŧ7*
T0
?
3batch_normalization_13/cond/FusedBatchNorm_1/SwitchSwitchconv2d_14/Conv2D#batch_normalization_13/cond/pred_id*#
_class
loc:@conv2d_14/Conv2D*L
_output_shapes:
8:??????????:??????????*
T0
?
!batch_normalization_13/cond/MergeMerge,batch_normalization_13/cond/FusedBatchNorm_1*batch_normalization_13/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
N*
T0
?
#batch_normalization_13/cond/Merge_1Merge.batch_normalization_13/cond/FusedBatchNorm_1:1,batch_normalization_13/cond/FusedBatchNorm:1*
N*
_output_shapes
	:?: *
T0
?
#batch_normalization_13/cond/Merge_2Merge.batch_normalization_13/cond/FusedBatchNorm_1:2,batch_normalization_13/cond/FusedBatchNorm:2*
_output_shapes
	:?: *
N*
T0
}
$batch_normalization_13/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
{
&batch_normalization_13/cond_1/switch_tIdentity&batch_normalization_13/cond_1/Switch:1*
_output_shapes
: *
T0

y
&batch_normalization_13/cond_1/switch_fIdentity$batch_normalization_13/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_13/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_13/cond_1/ConstConst'^batch_normalization_13/cond_1/switch_t*
_output_shapes
: *
dtype0*
valueB
 *?p}?
?
%batch_normalization_13/cond_1/Const_1Const'^batch_normalization_13/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#batch_normalization_13/cond_1/MergeMerge%batch_normalization_13/cond_1/Const_1#batch_normalization_13/cond_1/Const*
_output_shapes
: : *
T0*
N
?
,batch_normalization_13/AssignMovingAvg/sub/xConst*
dtype0*5
_class+
)'loc:@batch_normalization_13/moving_mean*
valueB
 *  ??*
_output_shapes
: 
?
*batch_normalization_13/AssignMovingAvg/subSub,batch_normalization_13/AssignMovingAvg/sub/x#batch_normalization_13/cond_1/Merge*
T0*5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes
: 
?
5batch_normalization_13/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_13/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_13/AssignMovingAvg/sub_1Sub5batch_normalization_13/AssignMovingAvg/ReadVariableOp#batch_normalization_13/cond/Merge_1*5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes	
:?*
T0
?
*batch_normalization_13/AssignMovingAvg/mulMul,batch_normalization_13/AssignMovingAvg/sub_1*batch_normalization_13/AssignMovingAvg/sub*
T0*5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes	
:?
?
:batch_normalization_13/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_13/moving_mean*batch_normalization_13/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_13/moving_mean*
dtype0
?
7batch_normalization_13/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_13/moving_mean;^batch_normalization_13/AssignMovingAvg/AssignSubVariableOp*5
_class+
)'loc:@batch_normalization_13/moving_mean*
_output_shapes	
:?*
dtype0
?
.batch_normalization_13/AssignMovingAvg_1/sub/xConst*9
_class/
-+loc:@batch_normalization_13/moving_variance*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
,batch_normalization_13/AssignMovingAvg_1/subSub.batch_normalization_13/AssignMovingAvg_1/sub/x#batch_normalization_13/cond_1/Merge*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_13/moving_variance
?
7batch_normalization_13/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_13/moving_variance*
_output_shapes	
:?*
dtype0
?
.batch_normalization_13/AssignMovingAvg_1/sub_1Sub7batch_normalization_13/AssignMovingAvg_1/ReadVariableOp#batch_normalization_13/cond/Merge_2*9
_class/
-+loc:@batch_normalization_13/moving_variance*
T0*
_output_shapes	
:?
?
,batch_normalization_13/AssignMovingAvg_1/mulMul.batch_normalization_13/AssignMovingAvg_1/sub_1,batch_normalization_13/AssignMovingAvg_1/sub*9
_class/
-+loc:@batch_normalization_13/moving_variance*
_output_shapes	
:?*
T0
?
<batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_13/moving_variance,batch_normalization_13/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_13/moving_variance*
dtype0
?
9batch_normalization_13/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_13/moving_variance=^batch_normalization_13/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_13/moving_variance
x
activation_12/ReluRelu!batch_normalization_13/cond/Merge*
T0*0
_output_shapes
:??????????
?
zero_padding2d_13/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
zero_padding2d_13/PadPadactivation_12/Reluzero_padding2d_13/Pad/paddings*
	Tpaddings0*0
_output_shapes
:??????????*
T0
?
1conv2d_15/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *#
_class
loc:@conv2d_15/kernel*
_output_shapes
:*
dtype0
?
/conv2d_15/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?Q?*
_output_shapes
: *#
_class
loc:@conv2d_15/kernel
?
/conv2d_15/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?Q=*#
_class
loc:@conv2d_15/kernel
?
9conv2d_15/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_15/kernel/Initializer/random_uniform/shape*#
_class
loc:@conv2d_15/kernel*
T0*
dtype0*
seed2 *(
_output_shapes
:??*

seed 
?
/conv2d_15/kernel/Initializer/random_uniform/subSub/conv2d_15/kernel/Initializer/random_uniform/max/conv2d_15/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@conv2d_15/kernel*
_output_shapes
: 
?
/conv2d_15/kernel/Initializer/random_uniform/mulMul9conv2d_15/kernel/Initializer/random_uniform/RandomUniform/conv2d_15/kernel/Initializer/random_uniform/sub*(
_output_shapes
:??*#
_class
loc:@conv2d_15/kernel*
T0
?
+conv2d_15/kernel/Initializer/random_uniformAdd/conv2d_15/kernel/Initializer/random_uniform/mul/conv2d_15/kernel/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_15/kernel
?
conv2d_15/kernelVarHandleOp*
shape:??*
_output_shapes
: *
dtype0*
	container *!
shared_nameconv2d_15/kernel*#
_class
loc:@conv2d_15/kernel
q
1conv2d_15/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_15/kernel*
_output_shapes
: 
?
conv2d_15/kernel/AssignAssignVariableOpconv2d_15/kernel+conv2d_15/kernel/Initializer/random_uniform*
dtype0*#
_class
loc:@conv2d_15/kernel
?
$conv2d_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_15/kernel*
dtype0*(
_output_shapes
:??*#
_class
loc:@conv2d_15/kernel
h
conv2d_15/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
z
conv2d_15/Conv2D/ReadVariableOpReadVariableOpconv2d_15/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_15/Conv2DConv2Dzero_padding2d_13/Padconv2d_15/Conv2D/ReadVariableOp*
strides
*
explicit_paddings
 *
data_formatNHWC*0
_output_shapes
:??????????*
T0*
	dilations
*
use_cudnn_on_gpu(*
paddingVALID
h
	add_5/addAddconv2d_15/Conv2D	add_4/add*0
_output_shapes
:??????????*
T0
?
-batch_normalization_14/gamma/Initializer/onesConst*
_output_shapes	
:?*
valueB?*  ??*
dtype0*/
_class%
#!loc:@batch_normalization_14/gamma
?
batch_normalization_14/gammaVarHandleOp*
_output_shapes
: *
	container *-
shared_namebatch_normalization_14/gamma*
dtype0*/
_class%
#!loc:@batch_normalization_14/gamma*
shape:?
?
=batch_normalization_14/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_14/gamma*
_output_shapes
: 
?
#batch_normalization_14/gamma/AssignAssignVariableOpbatch_normalization_14/gamma-batch_normalization_14/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_14/gamma*
dtype0
?
0batch_normalization_14/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_14/gamma*
dtype0*
_output_shapes	
:?*/
_class%
#!loc:@batch_normalization_14/gamma
?
-batch_normalization_14/beta/Initializer/zerosConst*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_14/beta*
dtype0*
valueB?*    
?
batch_normalization_14/betaVarHandleOp*
	container *
_output_shapes
: *.
_class$
" loc:@batch_normalization_14/beta*
dtype0*,
shared_namebatch_normalization_14/beta*
shape:?
?
<batch_normalization_14/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_14/beta*
_output_shapes
: 
?
"batch_normalization_14/beta/AssignAssignVariableOpbatch_normalization_14/beta-batch_normalization_14/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_14/beta*
dtype0
?
/batch_normalization_14/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_14/beta*
_output_shapes	
:?*
dtype0*.
_class$
" loc:@batch_normalization_14/beta
?
4batch_normalization_14/moving_mean/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_14/moving_mean
?
"batch_normalization_14/moving_meanVarHandleOp*
	container *
shape:?*3
shared_name$"batch_normalization_14/moving_mean*
_output_shapes
: *
dtype0*5
_class+
)'loc:@batch_normalization_14/moving_mean
?
Cbatch_normalization_14/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_14/moving_mean*
_output_shapes
: 
?
)batch_normalization_14/moving_mean/AssignAssignVariableOp"batch_normalization_14/moving_mean4batch_normalization_14/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_14/moving_mean*
dtype0
?
6batch_normalization_14/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*5
_class+
)'loc:@batch_normalization_14/moving_mean*
dtype0*
_output_shapes	
:?
?
7batch_normalization_14/moving_variance/Initializer/onesConst*
dtype0*
valueB?*  ??*9
_class/
-+loc:@batch_normalization_14/moving_variance*
_output_shapes	
:?
?
&batch_normalization_14/moving_varianceVarHandleOp*
dtype0*9
_class/
-+loc:@batch_normalization_14/moving_variance*
_output_shapes
: *7
shared_name(&batch_normalization_14/moving_variance*
shape:?*
	container 
?
Gbatch_normalization_14/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_14/moving_variance*
_output_shapes
: 
?
-batch_normalization_14/moving_variance/AssignAssignVariableOp&batch_normalization_14/moving_variance7batch_normalization_14/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_14/moving_variance*
dtype0
?
:batch_normalization_14/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_14/moving_variance
{
"batch_normalization_14/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
w
$batch_normalization_14/cond/switch_tIdentity$batch_normalization_14/cond/Switch:1*
T0
*
_output_shapes
: 
u
$batch_normalization_14/cond/switch_fIdentity"batch_normalization_14/cond/Switch*
_output_shapes
: *
T0

f
#batch_normalization_14/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
*batch_normalization_14/cond/ReadVariableOpReadVariableOp3batch_normalization_14/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
1batch_normalization_14/cond/ReadVariableOp/SwitchSwitchbatch_normalization_14/gamma#batch_normalization_14/cond/pred_id*
T0*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_14/gamma
?
,batch_normalization_14/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_14/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_14/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_14/beta#batch_normalization_14/cond/pred_id*.
_class$
" loc:@batch_normalization_14/beta*
T0*
_output_shapes
: : 
?
!batch_normalization_14/cond/ConstConst%^batch_normalization_14/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
#batch_normalization_14/cond/Const_1Const%^batch_normalization_14/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
*batch_normalization_14/cond/FusedBatchNormFusedBatchNorm3batch_normalization_14/cond/FusedBatchNorm/Switch:1*batch_normalization_14/cond/ReadVariableOp,batch_normalization_14/cond/ReadVariableOp_1!batch_normalization_14/cond/Const#batch_normalization_14/cond/Const_1*
is_training(*L
_output_shapes:
8:??????????:?:?:?:?*
T0*
data_formatNHWC*
epsilon%?ŧ7
?
1batch_normalization_14/cond/FusedBatchNorm/SwitchSwitch	add_5/add#batch_normalization_14/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_5/add*
T0
?
,batch_normalization_14/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_14/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_14/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_14/gamma#batch_normalization_14/cond/pred_id*
T0*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_14/gamma
?
,batch_normalization_14/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_14/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_14/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_14/beta#batch_normalization_14/cond/pred_id*.
_class$
" loc:@batch_normalization_14/beta*
_output_shapes
: : *
T0
?
;batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Bbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_14/moving_mean#batch_normalization_14/cond/pred_id*5
_class+
)'loc:@batch_normalization_14/moving_mean*
_output_shapes
: : *
T0
?
=batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Dbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_14/moving_variance#batch_normalization_14/cond/pred_id*
_output_shapes
: : *9
_class/
-+loc:@batch_normalization_14/moving_variance*
T0
?
,batch_normalization_14/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_14/cond/FusedBatchNorm_1/Switch,batch_normalization_14/cond/ReadVariableOp_2,batch_normalization_14/cond/ReadVariableOp_3;batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%?ŧ7*
is_training( *L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
T0
?
3batch_normalization_14/cond/FusedBatchNorm_1/SwitchSwitch	add_5/add#batch_normalization_14/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
T0*
_class
loc:@add_5/add
?
!batch_normalization_14/cond/MergeMerge,batch_normalization_14/cond/FusedBatchNorm_1*batch_normalization_14/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
N*
T0
?
#batch_normalization_14/cond/Merge_1Merge.batch_normalization_14/cond/FusedBatchNorm_1:1,batch_normalization_14/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:?: 
?
#batch_normalization_14/cond/Merge_2Merge.batch_normalization_14/cond/FusedBatchNorm_1:2,batch_normalization_14/cond/FusedBatchNorm:2*
_output_shapes
	:?: *
N*
T0
}
$batch_normalization_14/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
{
&batch_normalization_14/cond_1/switch_tIdentity&batch_normalization_14/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_14/cond_1/switch_fIdentity$batch_normalization_14/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_14/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_14/cond_1/ConstConst'^batch_normalization_14/cond_1/switch_t*
valueB
 *?p}?*
dtype0*
_output_shapes
: 
?
%batch_normalization_14/cond_1/Const_1Const'^batch_normalization_14/cond_1/switch_f*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
#batch_normalization_14/cond_1/MergeMerge%batch_normalization_14/cond_1/Const_1#batch_normalization_14/cond_1/Const*
_output_shapes
: : *
T0*
N
?
,batch_normalization_14/AssignMovingAvg/sub/xConst*
_output_shapes
: *
valueB
 *  ??*
dtype0*5
_class+
)'loc:@batch_normalization_14/moving_mean
?
*batch_normalization_14/AssignMovingAvg/subSub,batch_normalization_14/AssignMovingAvg/sub/x#batch_normalization_14/cond_1/Merge*5
_class+
)'loc:@batch_normalization_14/moving_mean*
T0*
_output_shapes
: 
?
5batch_normalization_14/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_14/moving_mean*
_output_shapes	
:?*
dtype0
?
,batch_normalization_14/AssignMovingAvg/sub_1Sub5batch_normalization_14/AssignMovingAvg/ReadVariableOp#batch_normalization_14/cond/Merge_1*5
_class+
)'loc:@batch_normalization_14/moving_mean*
_output_shapes	
:?*
T0
?
*batch_normalization_14/AssignMovingAvg/mulMul,batch_normalization_14/AssignMovingAvg/sub_1*batch_normalization_14/AssignMovingAvg/sub*
T0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_14/moving_mean
?
:batch_normalization_14/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_14/moving_mean*batch_normalization_14/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_14/moving_mean*
dtype0
?
7batch_normalization_14/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_14/moving_mean;^batch_normalization_14/AssignMovingAvg/AssignSubVariableOp*
dtype0*5
_class+
)'loc:@batch_normalization_14/moving_mean*
_output_shapes	
:?
?
.batch_normalization_14/AssignMovingAvg_1/sub/xConst*
dtype0*
valueB
 *  ??*9
_class/
-+loc:@batch_normalization_14/moving_variance*
_output_shapes
: 
?
,batch_normalization_14/AssignMovingAvg_1/subSub.batch_normalization_14/AssignMovingAvg_1/sub/x#batch_normalization_14/cond_1/Merge*9
_class/
-+loc:@batch_normalization_14/moving_variance*
_output_shapes
: *
T0
?
7batch_normalization_14/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_14/moving_variance*
_output_shapes	
:?*
dtype0
?
.batch_normalization_14/AssignMovingAvg_1/sub_1Sub7batch_normalization_14/AssignMovingAvg_1/ReadVariableOp#batch_normalization_14/cond/Merge_2*
_output_shapes	
:?*
T0*9
_class/
-+loc:@batch_normalization_14/moving_variance
?
,batch_normalization_14/AssignMovingAvg_1/mulMul.batch_normalization_14/AssignMovingAvg_1/sub_1,batch_normalization_14/AssignMovingAvg_1/sub*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_14/moving_variance*
T0
?
<batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_14/moving_variance,batch_normalization_14/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_14/moving_variance*
dtype0
?
9batch_normalization_14/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_14/moving_variance=^batch_normalization_14/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_14/moving_variance*
dtype0
x
activation_13/ReluRelu!batch_normalization_14/cond/Merge*0
_output_shapes
:??????????*
T0
?
1conv2d_16/kernel/Initializer/random_uniform/shapeConst*%
valueB"            *
_output_shapes
:*#
_class
loc:@conv2d_16/kernel*
dtype0
?
/conv2d_16/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *q??*#
_class
loc:@conv2d_16/kernel*
_output_shapes
: 
?
/conv2d_16/kernel/Initializer/random_uniform/maxConst*#
_class
loc:@conv2d_16/kernel*
_output_shapes
: *
dtype0*
valueB
 *q?>
?
9conv2d_16/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_16/kernel/Initializer/random_uniform/shape*
dtype0*
T0*#
_class
loc:@conv2d_16/kernel*(
_output_shapes
:??*

seed *
seed2 
?
/conv2d_16/kernel/Initializer/random_uniform/subSub/conv2d_16/kernel/Initializer/random_uniform/max/conv2d_16/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*#
_class
loc:@conv2d_16/kernel
?
/conv2d_16/kernel/Initializer/random_uniform/mulMul9conv2d_16/kernel/Initializer/random_uniform/RandomUniform/conv2d_16/kernel/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_16/kernel
?
+conv2d_16/kernel/Initializer/random_uniformAdd/conv2d_16/kernel/Initializer/random_uniform/mul/conv2d_16/kernel/Initializer/random_uniform/min*(
_output_shapes
:??*#
_class
loc:@conv2d_16/kernel*
T0
?
conv2d_16/kernelVarHandleOp*!
shared_nameconv2d_16/kernel*
_output_shapes
: *
shape:??*
dtype0*
	container *#
_class
loc:@conv2d_16/kernel
q
1conv2d_16/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_16/kernel*
_output_shapes
: 
?
conv2d_16/kernel/AssignAssignVariableOpconv2d_16/kernel+conv2d_16/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_16/kernel*
dtype0
?
$conv2d_16/kernel/Read/ReadVariableOpReadVariableOpconv2d_16/kernel*(
_output_shapes
:??*
dtype0*#
_class
loc:@conv2d_16/kernel
h
conv2d_16/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
z
conv2d_16/Conv2D/ReadVariableOpReadVariableOpconv2d_16/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_16/Conv2DConv2Dactivation_13/Reluconv2d_16/Conv2D/ReadVariableOp*0
_output_shapes
:??????????*
	dilations
*
use_cudnn_on_gpu(*
T0*
data_formatNHWC*
strides
*
paddingVALID*
explicit_paddings
 
?
zero_padding2d_14/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0
?
zero_padding2d_14/PadPadactivation_13/Reluzero_padding2d_14/Pad/paddings*0
_output_shapes
:??????????*
	Tpaddings0*
T0
?
1conv2d_17/kernel/Initializer/random_uniform/shapeConst*
dtype0*#
_class
loc:@conv2d_17/kernel*%
valueB"            *
_output_shapes
:
?
/conv2d_17/kernel/Initializer/random_uniform/minConst*#
_class
loc:@conv2d_17/kernel*
valueB
 *?Q?*
dtype0*
_output_shapes
: 
?
/conv2d_17/kernel/Initializer/random_uniform/maxConst*
dtype0*#
_class
loc:@conv2d_17/kernel*
_output_shapes
: *
valueB
 *?Q=
?
9conv2d_17/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_17/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0*
T0*#
_class
loc:@conv2d_17/kernel*

seed *(
_output_shapes
:??
?
/conv2d_17/kernel/Initializer/random_uniform/subSub/conv2d_17/kernel/Initializer/random_uniform/max/conv2d_17/kernel/Initializer/random_uniform/min*
_output_shapes
: *#
_class
loc:@conv2d_17/kernel*
T0
?
/conv2d_17/kernel/Initializer/random_uniform/mulMul9conv2d_17/kernel/Initializer/random_uniform/RandomUniform/conv2d_17/kernel/Initializer/random_uniform/sub*#
_class
loc:@conv2d_17/kernel*(
_output_shapes
:??*
T0
?
+conv2d_17/kernel/Initializer/random_uniformAdd/conv2d_17/kernel/Initializer/random_uniform/mul/conv2d_17/kernel/Initializer/random_uniform/min*(
_output_shapes
:??*#
_class
loc:@conv2d_17/kernel*
T0
?
conv2d_17/kernelVarHandleOp*#
_class
loc:@conv2d_17/kernel*
dtype0*
_output_shapes
: *
	container *!
shared_nameconv2d_17/kernel*
shape:??
q
1conv2d_17/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_17/kernel*
_output_shapes
: 
?
conv2d_17/kernel/AssignAssignVariableOpconv2d_17/kernel+conv2d_17/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_17/kernel*
dtype0
?
$conv2d_17/kernel/Read/ReadVariableOpReadVariableOpconv2d_17/kernel*#
_class
loc:@conv2d_17/kernel*(
_output_shapes
:??*
dtype0
h
conv2d_17/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
z
conv2d_17/Conv2D/ReadVariableOpReadVariableOpconv2d_17/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_17/Conv2DConv2Dzero_padding2d_14/Padconv2d_17/Conv2D/ReadVariableOp*
T0*
strides
*
data_formatNHWC*
	dilations
*0
_output_shapes
:??????????*
paddingVALID*
explicit_paddings
 *
use_cudnn_on_gpu(
?
-batch_normalization_15/gamma/Initializer/onesConst*
dtype0*/
_class%
#!loc:@batch_normalization_15/gamma*
valueB?*  ??*
_output_shapes	
:?
?
batch_normalization_15/gammaVarHandleOp*
	container *
dtype0*
shape:?*
_output_shapes
: */
_class%
#!loc:@batch_normalization_15/gamma*-
shared_namebatch_normalization_15/gamma
?
=batch_normalization_15/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_15/gamma*
_output_shapes
: 
?
#batch_normalization_15/gamma/AssignAssignVariableOpbatch_normalization_15/gamma-batch_normalization_15/gamma/Initializer/ones*
dtype0*/
_class%
#!loc:@batch_normalization_15/gamma
?
0batch_normalization_15/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_15/gamma*/
_class%
#!loc:@batch_normalization_15/gamma*
_output_shapes	
:?*
dtype0
?
-batch_normalization_15/beta/Initializer/zerosConst*
valueB?*    *.
_class$
" loc:@batch_normalization_15/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_15/betaVarHandleOp*.
_class$
" loc:@batch_normalization_15/beta*,
shared_namebatch_normalization_15/beta*
dtype0*
shape:?*
_output_shapes
: *
	container 
?
<batch_normalization_15/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_15/beta*
_output_shapes
: 
?
"batch_normalization_15/beta/AssignAssignVariableOpbatch_normalization_15/beta-batch_normalization_15/beta/Initializer/zeros*
dtype0*.
_class$
" loc:@batch_normalization_15/beta
?
/batch_normalization_15/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_15/beta*.
_class$
" loc:@batch_normalization_15/beta*
dtype0*
_output_shapes	
:?
?
4batch_normalization_15/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*5
_class+
)'loc:@batch_normalization_15/moving_mean
?
"batch_normalization_15/moving_meanVarHandleOp*3
shared_name$"batch_normalization_15/moving_mean*
dtype0*5
_class+
)'loc:@batch_normalization_15/moving_mean*
	container *
shape:?*
_output_shapes
: 
?
Cbatch_normalization_15/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_15/moving_mean*
_output_shapes
: 
?
)batch_normalization_15/moving_mean/AssignAssignVariableOp"batch_normalization_15/moving_mean4batch_normalization_15/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_15/moving_mean*
dtype0
?
6batch_normalization_15/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
_output_shapes	
:?*
dtype0*5
_class+
)'loc:@batch_normalization_15/moving_mean
?
7batch_normalization_15/moving_variance/Initializer/onesConst*
valueB?*  ??*
dtype0*9
_class/
-+loc:@batch_normalization_15/moving_variance*
_output_shapes	
:?
?
&batch_normalization_15/moving_varianceVarHandleOp*
_output_shapes
: *
	container *
dtype0*9
_class/
-+loc:@batch_normalization_15/moving_variance*
shape:?*7
shared_name(&batch_normalization_15/moving_variance
?
Gbatch_normalization_15/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_15/moving_variance*
_output_shapes
: 
?
-batch_normalization_15/moving_variance/AssignAssignVariableOp&batch_normalization_15/moving_variance7batch_normalization_15/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_15/moving_variance*
dtype0
?
:batch_normalization_15/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
dtype0*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_15/moving_variance
{
"batch_normalization_15/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

w
$batch_normalization_15/cond/switch_tIdentity$batch_normalization_15/cond/Switch:1*
_output_shapes
: *
T0

u
$batch_normalization_15/cond/switch_fIdentity"batch_normalization_15/cond/Switch*
_output_shapes
: *
T0

f
#batch_normalization_15/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
*batch_normalization_15/cond/ReadVariableOpReadVariableOp3batch_normalization_15/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
1batch_normalization_15/cond/ReadVariableOp/SwitchSwitchbatch_normalization_15/gamma#batch_normalization_15/cond/pred_id*
_output_shapes
: : *
T0*/
_class%
#!loc:@batch_normalization_15/gamma
?
,batch_normalization_15/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_15/cond/ReadVariableOp_1/Switch:1*
_output_shapes	
:?*
dtype0
?
3batch_normalization_15/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_15/beta#batch_normalization_15/cond/pred_id*.
_class$
" loc:@batch_normalization_15/beta*
_output_shapes
: : *
T0
?
!batch_normalization_15/cond/ConstConst%^batch_normalization_15/cond/switch_t*
_output_shapes
: *
dtype0*
valueB 
?
#batch_normalization_15/cond/Const_1Const%^batch_normalization_15/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
*batch_normalization_15/cond/FusedBatchNormFusedBatchNorm3batch_normalization_15/cond/FusedBatchNorm/Switch:1*batch_normalization_15/cond/ReadVariableOp,batch_normalization_15/cond/ReadVariableOp_1!batch_normalization_15/cond/Const#batch_normalization_15/cond/Const_1*
epsilon%?ŧ7*
data_formatNHWC*
T0*L
_output_shapes:
8:??????????:?:?:?:?*
is_training(
?
1batch_normalization_15/cond/FusedBatchNorm/SwitchSwitchconv2d_17/Conv2D#batch_normalization_15/cond/pred_id*#
_class
loc:@conv2d_17/Conv2D*L
_output_shapes:
8:??????????:??????????*
T0
?
,batch_normalization_15/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_15/cond/ReadVariableOp_2/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_15/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_15/gamma#batch_normalization_15/cond/pred_id*/
_class%
#!loc:@batch_normalization_15/gamma*
_output_shapes
: : *
T0
?
,batch_normalization_15/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_15/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_15/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_15/beta#batch_normalization_15/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_15/beta
?
;batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Bbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_15/moving_mean#batch_normalization_15/cond/pred_id*
_output_shapes
: : *5
_class+
)'loc:@batch_normalization_15/moving_mean*
T0
?
=batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Dbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_15/moving_variance#batch_normalization_15/cond/pred_id*
_output_shapes
: : *
T0*9
_class/
-+loc:@batch_normalization_15/moving_variance
?
,batch_normalization_15/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_15/cond/FusedBatchNorm_1/Switch,batch_normalization_15/cond/ReadVariableOp_2,batch_normalization_15/cond/ReadVariableOp_3;batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1*
epsilon%?ŧ7*L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
T0*
is_training( 
?
3batch_normalization_15/cond/FusedBatchNorm_1/SwitchSwitchconv2d_17/Conv2D#batch_normalization_15/cond/pred_id*#
_class
loc:@conv2d_17/Conv2D*L
_output_shapes:
8:??????????:??????????*
T0
?
!batch_normalization_15/cond/MergeMerge,batch_normalization_15/cond/FusedBatchNorm_1*batch_normalization_15/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
N*
T0
?
#batch_normalization_15/cond/Merge_1Merge.batch_normalization_15/cond/FusedBatchNorm_1:1,batch_normalization_15/cond/FusedBatchNorm:1*
N*
T0*
_output_shapes
	:?: 
?
#batch_normalization_15/cond/Merge_2Merge.batch_normalization_15/cond/FusedBatchNorm_1:2,batch_normalization_15/cond/FusedBatchNorm:2*
T0*
_output_shapes
	:?: *
N
}
$batch_normalization_15/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

{
&batch_normalization_15/cond_1/switch_tIdentity&batch_normalization_15/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_15/cond_1/switch_fIdentity$batch_normalization_15/cond_1/Switch*
T0
*
_output_shapes
: 
h
%batch_normalization_15/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_15/cond_1/ConstConst'^batch_normalization_15/cond_1/switch_t*
_output_shapes
: *
valueB
 *?p}?*
dtype0
?
%batch_normalization_15/cond_1/Const_1Const'^batch_normalization_15/cond_1/switch_f*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
#batch_normalization_15/cond_1/MergeMerge%batch_normalization_15/cond_1/Const_1#batch_normalization_15/cond_1/Const*
_output_shapes
: : *
T0*
N
?
,batch_normalization_15/AssignMovingAvg/sub/xConst*
valueB
 *  ??*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_15/moving_mean*
dtype0
?
*batch_normalization_15/AssignMovingAvg/subSub,batch_normalization_15/AssignMovingAvg/sub/x#batch_normalization_15/cond_1/Merge*
T0*5
_class+
)'loc:@batch_normalization_15/moving_mean*
_output_shapes
: 
?
5batch_normalization_15/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_15/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_15/AssignMovingAvg/sub_1Sub5batch_normalization_15/AssignMovingAvg/ReadVariableOp#batch_normalization_15/cond/Merge_1*
T0*5
_class+
)'loc:@batch_normalization_15/moving_mean*
_output_shapes	
:?
?
*batch_normalization_15/AssignMovingAvg/mulMul,batch_normalization_15/AssignMovingAvg/sub_1*batch_normalization_15/AssignMovingAvg/sub*
T0*5
_class+
)'loc:@batch_normalization_15/moving_mean*
_output_shapes	
:?
?
:batch_normalization_15/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_15/moving_mean*batch_normalization_15/AssignMovingAvg/mul*
dtype0*5
_class+
)'loc:@batch_normalization_15/moving_mean
?
7batch_normalization_15/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_15/moving_mean;^batch_normalization_15/AssignMovingAvg/AssignSubVariableOp*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_15/moving_mean*
dtype0
?
.batch_normalization_15/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_15/moving_variance*
dtype0
?
,batch_normalization_15/AssignMovingAvg_1/subSub.batch_normalization_15/AssignMovingAvg_1/sub/x#batch_normalization_15/cond_1/Merge*
T0*9
_class/
-+loc:@batch_normalization_15/moving_variance*
_output_shapes
: 
?
7batch_normalization_15/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_15/moving_variance*
_output_shapes	
:?*
dtype0
?
.batch_normalization_15/AssignMovingAvg_1/sub_1Sub7batch_normalization_15/AssignMovingAvg_1/ReadVariableOp#batch_normalization_15/cond/Merge_2*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_15/moving_variance*
T0
?
,batch_normalization_15/AssignMovingAvg_1/mulMul.batch_normalization_15/AssignMovingAvg_1/sub_1,batch_normalization_15/AssignMovingAvg_1/sub*
_output_shapes	
:?*
T0*9
_class/
-+loc:@batch_normalization_15/moving_variance
?
<batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_15/moving_variance,batch_normalization_15/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_15/moving_variance*
dtype0
?
9batch_normalization_15/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_15/moving_variance=^batch_normalization_15/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*9
_class/
-+loc:@batch_normalization_15/moving_variance*
_output_shapes	
:?
x
activation_14/ReluRelu!batch_normalization_15/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_15/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:
?
zero_padding2d_15/PadPadactivation_14/Reluzero_padding2d_15/Pad/paddings*
T0*
	Tpaddings0*0
_output_shapes
:??????????
?
1conv2d_18/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
dtype0*#
_class
loc:@conv2d_18/kernel*%
valueB"            
?
/conv2d_18/kernel/Initializer/random_uniform/minConst*
dtype0*#
_class
loc:@conv2d_18/kernel*
valueB
 *:??*
_output_shapes
: 
?
/conv2d_18/kernel/Initializer/random_uniform/maxConst*
valueB
 *:?=*#
_class
loc:@conv2d_18/kernel*
_output_shapes
: *
dtype0
?
9conv2d_18/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_18/kernel/Initializer/random_uniform/shape*(
_output_shapes
:??*
T0*
seed2 *
dtype0*

seed *#
_class
loc:@conv2d_18/kernel
?
/conv2d_18/kernel/Initializer/random_uniform/subSub/conv2d_18/kernel/Initializer/random_uniform/max/conv2d_18/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@conv2d_18/kernel*
_output_shapes
: 
?
/conv2d_18/kernel/Initializer/random_uniform/mulMul9conv2d_18/kernel/Initializer/random_uniform/RandomUniform/conv2d_18/kernel/Initializer/random_uniform/sub*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_18/kernel
?
+conv2d_18/kernel/Initializer/random_uniformAdd/conv2d_18/kernel/Initializer/random_uniform/mul/conv2d_18/kernel/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_18/kernel
?
conv2d_18/kernelVarHandleOp*
	container *#
_class
loc:@conv2d_18/kernel*
_output_shapes
: *
dtype0*!
shared_nameconv2d_18/kernel*
shape:??
q
1conv2d_18/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_18/kernel*
_output_shapes
: 
?
conv2d_18/kernel/AssignAssignVariableOpconv2d_18/kernel+conv2d_18/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_18/kernel*
dtype0
?
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*#
_class
loc:@conv2d_18/kernel*(
_output_shapes
:??*
dtype0
h
conv2d_18/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
z
conv2d_18/Conv2D/ReadVariableOpReadVariableOpconv2d_18/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_18/Conv2DConv2Dzero_padding2d_15/Padconv2d_18/Conv2D/ReadVariableOp*0
_output_shapes
:??????????*
	dilations
*
data_formatNHWC*
T0*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
o
	add_6/addAddconv2d_18/Conv2Dconv2d_16/Conv2D*0
_output_shapes
:??????????*
T0
?
-batch_normalization_16/gamma/Initializer/onesConst*/
_class%
#!loc:@batch_normalization_16/gamma*
dtype0*
_output_shapes	
:?*
valueB?*  ??
?
batch_normalization_16/gammaVarHandleOp*
shape:?*
dtype0*/
_class%
#!loc:@batch_normalization_16/gamma*
	container *
_output_shapes
: *-
shared_namebatch_normalization_16/gamma
?
=batch_normalization_16/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_16/gamma*
_output_shapes
: 
?
#batch_normalization_16/gamma/AssignAssignVariableOpbatch_normalization_16/gamma-batch_normalization_16/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_16/gamma*
dtype0
?
0batch_normalization_16/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_16/gamma*
dtype0*
_output_shapes	
:?*/
_class%
#!loc:@batch_normalization_16/gamma
?
-batch_normalization_16/beta/Initializer/zerosConst*
valueB?*    *.
_class$
" loc:@batch_normalization_16/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_16/betaVarHandleOp*
_output_shapes
: *.
_class$
" loc:@batch_normalization_16/beta*
shape:?*
	container *
dtype0*,
shared_namebatch_normalization_16/beta
?
<batch_normalization_16/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_16/beta*
_output_shapes
: 
?
"batch_normalization_16/beta/AssignAssignVariableOpbatch_normalization_16/beta-batch_normalization_16/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_16/beta*
dtype0
?
/batch_normalization_16/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_16/beta*
dtype0*
_output_shapes	
:?*.
_class$
" loc:@batch_normalization_16/beta
?
4batch_normalization_16/moving_mean/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    *
dtype0*5
_class+
)'loc:@batch_normalization_16/moving_mean
?
"batch_normalization_16/moving_meanVarHandleOp*3
shared_name$"batch_normalization_16/moving_mean*5
_class+
)'loc:@batch_normalization_16/moving_mean*
	container *
dtype0*
_output_shapes
: *
shape:?
?
Cbatch_normalization_16/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_16/moving_mean*
_output_shapes
: 
?
)batch_normalization_16/moving_mean/AssignAssignVariableOp"batch_normalization_16/moving_mean4batch_normalization_16/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_16/moving_mean*
dtype0
?
6batch_normalization_16/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
_output_shapes	
:?*
dtype0*5
_class+
)'loc:@batch_normalization_16/moving_mean
?
7batch_normalization_16/moving_variance/Initializer/onesConst*
dtype0*9
_class/
-+loc:@batch_normalization_16/moving_variance*
valueB?*  ??*
_output_shapes	
:?
?
&batch_normalization_16/moving_varianceVarHandleOp*
dtype0*
	container *9
_class/
-+loc:@batch_normalization_16/moving_variance*7
shared_name(&batch_normalization_16/moving_variance*
shape:?*
_output_shapes
: 
?
Gbatch_normalization_16/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_16/moving_variance*
_output_shapes
: 
?
-batch_normalization_16/moving_variance/AssignAssignVariableOp&batch_normalization_16/moving_variance7batch_normalization_16/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_16/moving_variance*
dtype0
?
:batch_normalization_16/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*9
_class/
-+loc:@batch_normalization_16/moving_variance*
dtype0*
_output_shapes	
:?
{
"batch_normalization_16/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

w
$batch_normalization_16/cond/switch_tIdentity$batch_normalization_16/cond/Switch:1*
_output_shapes
: *
T0

u
$batch_normalization_16/cond/switch_fIdentity"batch_normalization_16/cond/Switch*
_output_shapes
: *
T0

f
#batch_normalization_16/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
*batch_normalization_16/cond/ReadVariableOpReadVariableOp3batch_normalization_16/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
1batch_normalization_16/cond/ReadVariableOp/SwitchSwitchbatch_normalization_16/gamma#batch_normalization_16/cond/pred_id*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_16/gamma*
T0
?
,batch_normalization_16/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_16/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_16/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_16/beta#batch_normalization_16/cond/pred_id*.
_class$
" loc:@batch_normalization_16/beta*
_output_shapes
: : *
T0
?
!batch_normalization_16/cond/ConstConst%^batch_normalization_16/cond/switch_t*
_output_shapes
: *
valueB *
dtype0
?
#batch_normalization_16/cond/Const_1Const%^batch_normalization_16/cond/switch_t*
dtype0*
_output_shapes
: *
valueB 
?
*batch_normalization_16/cond/FusedBatchNormFusedBatchNorm3batch_normalization_16/cond/FusedBatchNorm/Switch:1*batch_normalization_16/cond/ReadVariableOp,batch_normalization_16/cond/ReadVariableOp_1!batch_normalization_16/cond/Const#batch_normalization_16/cond/Const_1*
data_formatNHWC*
T0*
is_training(*L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%?ŧ7
?
1batch_normalization_16/cond/FusedBatchNorm/SwitchSwitch	add_6/add#batch_normalization_16/cond/pred_id*
_class
loc:@add_6/add*
T0*L
_output_shapes:
8:??????????:??????????
?
,batch_normalization_16/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_16/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_16/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_16/gamma#batch_normalization_16/cond/pred_id*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_16/gamma*
T0
?
,batch_normalization_16/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_16/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_16/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_16/beta#batch_normalization_16/cond/pred_id*.
_class$
" loc:@batch_normalization_16/beta*
_output_shapes
: : *
T0
?
;batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
dtype0*
_output_shapes	
:?
?
Bbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_16/moving_mean#batch_normalization_16/cond/pred_id*
T0*5
_class+
)'loc:@batch_normalization_16/moving_mean*
_output_shapes
: : 
?
=batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Dbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_16/moving_variance#batch_normalization_16/cond/pred_id*
_output_shapes
: : *
T0*9
_class/
-+loc:@batch_normalization_16/moving_variance
?
,batch_normalization_16/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_16/cond/FusedBatchNorm_1/Switch,batch_normalization_16/cond/ReadVariableOp_2,batch_normalization_16/cond/ReadVariableOp_3;batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1*L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC*
T0*
epsilon%?ŧ7*
is_training( 
?
3batch_normalization_16/cond/FusedBatchNorm_1/SwitchSwitch	add_6/add#batch_normalization_16/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
T0*
_class
loc:@add_6/add
?
!batch_normalization_16/cond/MergeMerge,batch_normalization_16/cond/FusedBatchNorm_1*batch_normalization_16/cond/FusedBatchNorm*
T0*
N*2
_output_shapes 
:??????????: 
?
#batch_normalization_16/cond/Merge_1Merge.batch_normalization_16/cond/FusedBatchNorm_1:1,batch_normalization_16/cond/FusedBatchNorm:1*
T0*
_output_shapes
	:?: *
N
?
#batch_normalization_16/cond/Merge_2Merge.batch_normalization_16/cond/FusedBatchNorm_1:2,batch_normalization_16/cond/FusedBatchNorm:2*
T0*
_output_shapes
	:?: *
N
}
$batch_normalization_16/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
{
&batch_normalization_16/cond_1/switch_tIdentity&batch_normalization_16/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_16/cond_1/switch_fIdentity$batch_normalization_16/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_16/cond_1/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
#batch_normalization_16/cond_1/ConstConst'^batch_normalization_16/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *?p}?
?
%batch_normalization_16/cond_1/Const_1Const'^batch_normalization_16/cond_1/switch_f*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
#batch_normalization_16/cond_1/MergeMerge%batch_normalization_16/cond_1/Const_1#batch_normalization_16/cond_1/Const*
T0*
N*
_output_shapes
: : 
?
,batch_normalization_16/AssignMovingAvg/sub/xConst*5
_class+
)'loc:@batch_normalization_16/moving_mean*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
*batch_normalization_16/AssignMovingAvg/subSub,batch_normalization_16/AssignMovingAvg/sub/x#batch_normalization_16/cond_1/Merge*
T0*5
_class+
)'loc:@batch_normalization_16/moving_mean*
_output_shapes
: 
?
5batch_normalization_16/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_16/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_16/AssignMovingAvg/sub_1Sub5batch_normalization_16/AssignMovingAvg/ReadVariableOp#batch_normalization_16/cond/Merge_1*5
_class+
)'loc:@batch_normalization_16/moving_mean*
_output_shapes	
:?*
T0
?
*batch_normalization_16/AssignMovingAvg/mulMul,batch_normalization_16/AssignMovingAvg/sub_1*batch_normalization_16/AssignMovingAvg/sub*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_16/moving_mean*
T0
?
:batch_normalization_16/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_16/moving_mean*batch_normalization_16/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_16/moving_mean*
dtype0
?
7batch_normalization_16/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_16/moving_mean;^batch_normalization_16/AssignMovingAvg/AssignSubVariableOp*5
_class+
)'loc:@batch_normalization_16/moving_mean*
_output_shapes	
:?*
dtype0
?
.batch_normalization_16/AssignMovingAvg_1/sub/xConst*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_16/moving_variance*
valueB
 *  ??*
dtype0
?
,batch_normalization_16/AssignMovingAvg_1/subSub.batch_normalization_16/AssignMovingAvg_1/sub/x#batch_normalization_16/cond_1/Merge*9
_class/
-+loc:@batch_normalization_16/moving_variance*
T0*
_output_shapes
: 
?
7batch_normalization_16/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_16/moving_variance*
dtype0*
_output_shapes	
:?
?
.batch_normalization_16/AssignMovingAvg_1/sub_1Sub7batch_normalization_16/AssignMovingAvg_1/ReadVariableOp#batch_normalization_16/cond/Merge_2*9
_class/
-+loc:@batch_normalization_16/moving_variance*
T0*
_output_shapes	
:?
?
,batch_normalization_16/AssignMovingAvg_1/mulMul.batch_normalization_16/AssignMovingAvg_1/sub_1,batch_normalization_16/AssignMovingAvg_1/sub*9
_class/
-+loc:@batch_normalization_16/moving_variance*
_output_shapes	
:?*
T0
?
<batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_16/moving_variance,batch_normalization_16/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_16/moving_variance*
dtype0
?
9batch_normalization_16/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_16/moving_variance=^batch_normalization_16/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_16/moving_variance
x
activation_15/ReluRelu!batch_normalization_16/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_16/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_16/PadPadactivation_15/Reluzero_padding2d_16/Pad/paddings*
	Tpaddings0*0
_output_shapes
:??????????*
T0
?
1conv2d_19/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0*#
_class
loc:@conv2d_19/kernel
?
/conv2d_19/kernel/Initializer/random_uniform/minConst*#
_class
loc:@conv2d_19/kernel*
dtype0*
valueB
 *:??*
_output_shapes
: 
?
/conv2d_19/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *:?=*#
_class
loc:@conv2d_19/kernel*
dtype0
?
9conv2d_19/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_19/kernel/Initializer/random_uniform/shape*#
_class
loc:@conv2d_19/kernel*(
_output_shapes
:??*

seed *
T0*
seed2 *
dtype0
?
/conv2d_19/kernel/Initializer/random_uniform/subSub/conv2d_19/kernel/Initializer/random_uniform/max/conv2d_19/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@conv2d_19/kernel*
_output_shapes
: 
?
/conv2d_19/kernel/Initializer/random_uniform/mulMul9conv2d_19/kernel/Initializer/random_uniform/RandomUniform/conv2d_19/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@conv2d_19/kernel*(
_output_shapes
:??
?
+conv2d_19/kernel/Initializer/random_uniformAdd/conv2d_19/kernel/Initializer/random_uniform/mul/conv2d_19/kernel/Initializer/random_uniform/min*
T0*(
_output_shapes
:??*#
_class
loc:@conv2d_19/kernel
?
conv2d_19/kernelVarHandleOp*#
_class
loc:@conv2d_19/kernel*
_output_shapes
: *
shape:??*
	container *!
shared_nameconv2d_19/kernel*
dtype0
q
1conv2d_19/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_19/kernel*
_output_shapes
: 
?
conv2d_19/kernel/AssignAssignVariableOpconv2d_19/kernel+conv2d_19/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_19/kernel*
dtype0
?
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*
dtype0*(
_output_shapes
:??*#
_class
loc:@conv2d_19/kernel
h
conv2d_19/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
z
conv2d_19/Conv2D/ReadVariableOpReadVariableOpconv2d_19/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_19/Conv2DConv2Dzero_padding2d_16/Padconv2d_19/Conv2D/ReadVariableOp*
use_cudnn_on_gpu(*
	dilations
*
explicit_paddings
 *
data_formatNHWC*
strides
*0
_output_shapes
:??????????*
T0*
paddingVALID
?
-batch_normalization_17/gamma/Initializer/onesConst*
_output_shapes	
:?*/
_class%
#!loc:@batch_normalization_17/gamma*
dtype0*
valueB?*  ??
?
batch_normalization_17/gammaVarHandleOp*
dtype0*-
shared_namebatch_normalization_17/gamma*/
_class%
#!loc:@batch_normalization_17/gamma*
	container *
_output_shapes
: *
shape:?
?
=batch_normalization_17/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_17/gamma*
_output_shapes
: 
?
#batch_normalization_17/gamma/AssignAssignVariableOpbatch_normalization_17/gamma-batch_normalization_17/gamma/Initializer/ones*
dtype0*/
_class%
#!loc:@batch_normalization_17/gamma
?
0batch_normalization_17/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_17/gamma*
_output_shapes	
:?*/
_class%
#!loc:@batch_normalization_17/gamma*
dtype0
?
-batch_normalization_17/beta/Initializer/zerosConst*.
_class$
" loc:@batch_normalization_17/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
batch_normalization_17/betaVarHandleOp*
dtype0*
_output_shapes
: *,
shared_namebatch_normalization_17/beta*.
_class$
" loc:@batch_normalization_17/beta*
	container *
shape:?
?
<batch_normalization_17/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_17/beta*
_output_shapes
: 
?
"batch_normalization_17/beta/AssignAssignVariableOpbatch_normalization_17/beta-batch_normalization_17/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_17/beta*
dtype0
?
/batch_normalization_17/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_17/beta*
_output_shapes	
:?*
dtype0*.
_class$
" loc:@batch_normalization_17/beta
?
4batch_normalization_17/moving_mean/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*5
_class+
)'loc:@batch_normalization_17/moving_mean
?
"batch_normalization_17/moving_meanVarHandleOp*
_output_shapes
: *
	container *5
_class+
)'loc:@batch_normalization_17/moving_mean*3
shared_name$"batch_normalization_17/moving_mean*
dtype0*
shape:?
?
Cbatch_normalization_17/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_17/moving_mean*
_output_shapes
: 
?
)batch_normalization_17/moving_mean/AssignAssignVariableOp"batch_normalization_17/moving_mean4batch_normalization_17/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_17/moving_mean*
dtype0
?
6batch_normalization_17/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
dtype0*5
_class+
)'loc:@batch_normalization_17/moving_mean*
_output_shapes	
:?
?
7batch_normalization_17/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_17/moving_variance
?
&batch_normalization_17/moving_varianceVarHandleOp*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_17/moving_variance*7
shared_name(&batch_normalization_17/moving_variance*
shape:?*
dtype0*
	container 
?
Gbatch_normalization_17/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_17/moving_variance*
_output_shapes
: 
?
-batch_normalization_17/moving_variance/AssignAssignVariableOp&batch_normalization_17/moving_variance7batch_normalization_17/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_17/moving_variance*
dtype0
?
:batch_normalization_17/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
dtype0*9
_class/
-+loc:@batch_normalization_17/moving_variance*
_output_shapes	
:?
{
"batch_normalization_17/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

w
$batch_normalization_17/cond/switch_tIdentity$batch_normalization_17/cond/Switch:1*
T0
*
_output_shapes
: 
u
$batch_normalization_17/cond/switch_fIdentity"batch_normalization_17/cond/Switch*
T0
*
_output_shapes
: 
f
#batch_normalization_17/cond/pred_idIdentitykeras_learning_phase*
_output_shapes
: *
T0

?
*batch_normalization_17/cond/ReadVariableOpReadVariableOp3batch_normalization_17/cond/ReadVariableOp/Switch:1*
dtype0*
_output_shapes	
:?
?
1batch_normalization_17/cond/ReadVariableOp/SwitchSwitchbatch_normalization_17/gamma#batch_normalization_17/cond/pred_id*
T0*/
_class%
#!loc:@batch_normalization_17/gamma*
_output_shapes
: : 
?
,batch_normalization_17/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_17/cond/ReadVariableOp_1/Switch:1*
_output_shapes	
:?*
dtype0
?
3batch_normalization_17/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_17/beta#batch_normalization_17/cond/pred_id*.
_class$
" loc:@batch_normalization_17/beta*
T0*
_output_shapes
: : 
?
!batch_normalization_17/cond/ConstConst%^batch_normalization_17/cond/switch_t*
valueB *
dtype0*
_output_shapes
: 
?
#batch_normalization_17/cond/Const_1Const%^batch_normalization_17/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
*batch_normalization_17/cond/FusedBatchNormFusedBatchNorm3batch_normalization_17/cond/FusedBatchNorm/Switch:1*batch_normalization_17/cond/ReadVariableOp,batch_normalization_17/cond/ReadVariableOp_1!batch_normalization_17/cond/Const#batch_normalization_17/cond/Const_1*
epsilon%?ŧ7*
is_training(*
T0*L
_output_shapes:
8:??????????:?:?:?:?*
data_formatNHWC
?
1batch_normalization_17/cond/FusedBatchNorm/SwitchSwitchconv2d_19/Conv2D#batch_normalization_17/cond/pred_id*#
_class
loc:@conv2d_19/Conv2D*
T0*L
_output_shapes:
8:??????????:??????????
?
,batch_normalization_17/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_17/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_17/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_17/gamma#batch_normalization_17/cond/pred_id*/
_class%
#!loc:@batch_normalization_17/gamma*
_output_shapes
: : *
T0
?
,batch_normalization_17/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_17/cond/ReadVariableOp_3/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_17/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_17/beta#batch_normalization_17/cond/pred_id*
_output_shapes
: : *
T0*.
_class$
" loc:@batch_normalization_17/beta
?
;batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:?*
dtype0
?
Bbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_17/moving_mean#batch_normalization_17/cond/pred_id*
_output_shapes
: : *
T0*5
_class+
)'loc:@batch_normalization_17/moving_mean
?
=batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
dtype0*
_output_shapes	
:?
?
Dbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_17/moving_variance#batch_normalization_17/cond/pred_id*
T0*
_output_shapes
: : *9
_class/
-+loc:@batch_normalization_17/moving_variance
?
,batch_normalization_17/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_17/cond/FusedBatchNorm_1/Switch,batch_normalization_17/cond/ReadVariableOp_2,batch_normalization_17/cond/ReadVariableOp_3;batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1*
T0*L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%?ŧ7*
is_training( *
data_formatNHWC
?
3batch_normalization_17/cond/FusedBatchNorm_1/SwitchSwitchconv2d_19/Conv2D#batch_normalization_17/cond/pred_id*L
_output_shapes:
8:??????????:??????????*#
_class
loc:@conv2d_19/Conv2D*
T0
?
!batch_normalization_17/cond/MergeMerge,batch_normalization_17/cond/FusedBatchNorm_1*batch_normalization_17/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
N*
T0
?
#batch_normalization_17/cond/Merge_1Merge.batch_normalization_17/cond/FusedBatchNorm_1:1,batch_normalization_17/cond/FusedBatchNorm:1*
_output_shapes
	:?: *
N*
T0
?
#batch_normalization_17/cond/Merge_2Merge.batch_normalization_17/cond/FusedBatchNorm_1:2,batch_normalization_17/cond/FusedBatchNorm:2*
_output_shapes
	:?: *
T0*
N
}
$batch_normalization_17/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
{
&batch_normalization_17/cond_1/switch_tIdentity&batch_normalization_17/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_17/cond_1/switch_fIdentity$batch_normalization_17/cond_1/Switch*
T0
*
_output_shapes
: 
h
%batch_normalization_17/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_17/cond_1/ConstConst'^batch_normalization_17/cond_1/switch_t*
dtype0*
_output_shapes
: *
valueB
 *?p}?
?
%batch_normalization_17/cond_1/Const_1Const'^batch_normalization_17/cond_1/switch_f*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
#batch_normalization_17/cond_1/MergeMerge%batch_normalization_17/cond_1/Const_1#batch_normalization_17/cond_1/Const*
T0*
_output_shapes
: : *
N
?
,batch_normalization_17/AssignMovingAvg/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??*5
_class+
)'loc:@batch_normalization_17/moving_mean
?
*batch_normalization_17/AssignMovingAvg/subSub,batch_normalization_17/AssignMovingAvg/sub/x#batch_normalization_17/cond_1/Merge*
T0*
_output_shapes
: *5
_class+
)'loc:@batch_normalization_17/moving_mean
?
5batch_normalization_17/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_17/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_17/AssignMovingAvg/sub_1Sub5batch_normalization_17/AssignMovingAvg/ReadVariableOp#batch_normalization_17/cond/Merge_1*
_output_shapes	
:?*
T0*5
_class+
)'loc:@batch_normalization_17/moving_mean
?
*batch_normalization_17/AssignMovingAvg/mulMul,batch_normalization_17/AssignMovingAvg/sub_1*batch_normalization_17/AssignMovingAvg/sub*5
_class+
)'loc:@batch_normalization_17/moving_mean*
T0*
_output_shapes	
:?
?
:batch_normalization_17/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_17/moving_mean*batch_normalization_17/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_17/moving_mean*
dtype0
?
7batch_normalization_17/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_17/moving_mean;^batch_normalization_17/AssignMovingAvg/AssignSubVariableOp*5
_class+
)'loc:@batch_normalization_17/moving_mean*
dtype0*
_output_shapes	
:?
?
.batch_normalization_17/AssignMovingAvg_1/sub/xConst*
dtype0*9
_class/
-+loc:@batch_normalization_17/moving_variance*
_output_shapes
: *
valueB
 *  ??
?
,batch_normalization_17/AssignMovingAvg_1/subSub.batch_normalization_17/AssignMovingAvg_1/sub/x#batch_normalization_17/cond_1/Merge*
_output_shapes
: *
T0*9
_class/
-+loc:@batch_normalization_17/moving_variance
?
7batch_normalization_17/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_17/moving_variance*
dtype0*
_output_shapes	
:?
?
.batch_normalization_17/AssignMovingAvg_1/sub_1Sub7batch_normalization_17/AssignMovingAvg_1/ReadVariableOp#batch_normalization_17/cond/Merge_2*
_output_shapes	
:?*
T0*9
_class/
-+loc:@batch_normalization_17/moving_variance
?
,batch_normalization_17/AssignMovingAvg_1/mulMul.batch_normalization_17/AssignMovingAvg_1/sub_1,batch_normalization_17/AssignMovingAvg_1/sub*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_17/moving_variance*
T0
?
<batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_17/moving_variance,batch_normalization_17/AssignMovingAvg_1/mul*9
_class/
-+loc:@batch_normalization_17/moving_variance*
dtype0
?
9batch_normalization_17/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_17/moving_variance=^batch_normalization_17/AssignMovingAvg_1/AssignSubVariableOp*
dtype0*9
_class/
-+loc:@batch_normalization_17/moving_variance*
_output_shapes	
:?
x
activation_16/ReluRelu!batch_normalization_17/cond/Merge*0
_output_shapes
:??????????*
T0
?
zero_padding2d_17/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
zero_padding2d_17/PadPadactivation_16/Reluzero_padding2d_17/Pad/paddings*
T0*0
_output_shapes
:??????????*
	Tpaddings0
?
1conv2d_20/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*#
_class
loc:@conv2d_20/kernel*%
valueB"            
?
/conv2d_20/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *:??*#
_class
loc:@conv2d_20/kernel
?
/conv2d_20/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *#
_class
loc:@conv2d_20/kernel*
dtype0*
valueB
 *:?=
?
9conv2d_20/kernel/Initializer/random_uniform/RandomUniformRandomUniform1conv2d_20/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@conv2d_20/kernel*

seed *(
_output_shapes
:??*
seed2 *
dtype0
?
/conv2d_20/kernel/Initializer/random_uniform/subSub/conv2d_20/kernel/Initializer/random_uniform/max/conv2d_20/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@conv2d_20/kernel*
_output_shapes
: 
?
/conv2d_20/kernel/Initializer/random_uniform/mulMul9conv2d_20/kernel/Initializer/random_uniform/RandomUniform/conv2d_20/kernel/Initializer/random_uniform/sub*#
_class
loc:@conv2d_20/kernel*(
_output_shapes
:??*
T0
?
+conv2d_20/kernel/Initializer/random_uniformAdd/conv2d_20/kernel/Initializer/random_uniform/mul/conv2d_20/kernel/Initializer/random_uniform/min*#
_class
loc:@conv2d_20/kernel*
T0*(
_output_shapes
:??
?
conv2d_20/kernelVarHandleOp*
	container *
dtype0*
_output_shapes
: *#
_class
loc:@conv2d_20/kernel*!
shared_nameconv2d_20/kernel*
shape:??
q
1conv2d_20/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpconv2d_20/kernel*
_output_shapes
: 
?
conv2d_20/kernel/AssignAssignVariableOpconv2d_20/kernel+conv2d_20/kernel/Initializer/random_uniform*#
_class
loc:@conv2d_20/kernel*
dtype0
?
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*(
_output_shapes
:??*#
_class
loc:@conv2d_20/kernel*
dtype0
h
conv2d_20/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
z
conv2d_20/Conv2D/ReadVariableOpReadVariableOpconv2d_20/kernel*
dtype0*(
_output_shapes
:??
?
conv2d_20/Conv2DConv2Dzero_padding2d_17/Padconv2d_20/Conv2D/ReadVariableOp*
explicit_paddings
 *
T0*
paddingVALID*
data_formatNHWC*0
_output_shapes
:??????????*
strides
*
use_cudnn_on_gpu(*
	dilations

h
	add_7/addAddconv2d_20/Conv2D	add_6/add*
T0*0
_output_shapes
:??????????
?
-batch_normalization_18/gamma/Initializer/onesConst*
dtype0*
valueB?*  ??*/
_class%
#!loc:@batch_normalization_18/gamma*
_output_shapes	
:?
?
batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
	container *-
shared_namebatch_normalization_18/gamma*
shape:?*/
_class%
#!loc:@batch_normalization_18/gamma*
dtype0
?
=batch_normalization_18/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_18/gamma*
_output_shapes
: 
?
#batch_normalization_18/gamma/AssignAssignVariableOpbatch_normalization_18/gamma-batch_normalization_18/gamma/Initializer/ones*/
_class%
#!loc:@batch_normalization_18/gamma*
dtype0
?
0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes	
:?*
dtype0*/
_class%
#!loc:@batch_normalization_18/gamma
?
-batch_normalization_18/beta/Initializer/zerosConst*
valueB?*    *.
_class$
" loc:@batch_normalization_18/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_18/betaVarHandleOp*
_output_shapes
: *,
shared_namebatch_normalization_18/beta*
dtype0*
	container *
shape:?*.
_class$
" loc:@batch_normalization_18/beta
?
<batch_normalization_18/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbatch_normalization_18/beta*
_output_shapes
: 
?
"batch_normalization_18/beta/AssignAssignVariableOpbatch_normalization_18/beta-batch_normalization_18/beta/Initializer/zeros*.
_class$
" loc:@batch_normalization_18/beta*
dtype0
?
/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes	
:?*
dtype0*.
_class$
" loc:@batch_normalization_18/beta
?
4batch_normalization_18/moving_mean/Initializer/zerosConst*5
_class+
)'loc:@batch_normalization_18/moving_mean*
_output_shapes	
:?*
dtype0*
valueB?*    
?
"batch_normalization_18/moving_meanVarHandleOp*5
_class+
)'loc:@batch_normalization_18/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes
: *3
shared_name$"batch_normalization_18/moving_mean
?
Cbatch_normalization_18/moving_mean/IsInitialized/VarIsInitializedOpVarIsInitializedOp"batch_normalization_18/moving_mean*
_output_shapes
: 
?
)batch_normalization_18/moving_mean/AssignAssignVariableOp"batch_normalization_18/moving_mean4batch_normalization_18/moving_mean/Initializer/zeros*5
_class+
)'loc:@batch_normalization_18/moving_mean*
dtype0
?
6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes	
:?*
dtype0*5
_class+
)'loc:@batch_normalization_18/moving_mean
?
7batch_normalization_18/moving_variance/Initializer/onesConst*
valueB?*  ??*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_18/moving_variance
?
&batch_normalization_18/moving_varianceVarHandleOp*9
_class/
-+loc:@batch_normalization_18/moving_variance*
dtype0*
	container *7
shared_name(&batch_normalization_18/moving_variance*
shape:?*
_output_shapes
: 
?
Gbatch_normalization_18/moving_variance/IsInitialized/VarIsInitializedOpVarIsInitializedOp&batch_normalization_18/moving_variance*
_output_shapes
: 
?
-batch_normalization_18/moving_variance/AssignAssignVariableOp&batch_normalization_18/moving_variance7batch_normalization_18/moving_variance/Initializer/ones*9
_class/
-+loc:@batch_normalization_18/moving_variance*
dtype0
?
:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_18/moving_variance*
dtype0
{
"batch_normalization_18/cond/SwitchSwitchkeras_learning_phasekeras_learning_phase*
T0
*
_output_shapes
: : 
w
$batch_normalization_18/cond/switch_tIdentity$batch_normalization_18/cond/Switch:1*
_output_shapes
: *
T0

u
$batch_normalization_18/cond/switch_fIdentity"batch_normalization_18/cond/Switch*
T0
*
_output_shapes
: 
f
#batch_normalization_18/cond/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
*batch_normalization_18/cond/ReadVariableOpReadVariableOp3batch_normalization_18/cond/ReadVariableOp/Switch:1*
_output_shapes	
:?*
dtype0
?
1batch_normalization_18/cond/ReadVariableOp/SwitchSwitchbatch_normalization_18/gamma#batch_normalization_18/cond/pred_id*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_18/gamma*
T0
?
,batch_normalization_18/cond/ReadVariableOp_1ReadVariableOp5batch_normalization_18/cond/ReadVariableOp_1/Switch:1*
dtype0*
_output_shapes	
:?
?
3batch_normalization_18/cond/ReadVariableOp_1/SwitchSwitchbatch_normalization_18/beta#batch_normalization_18/cond/pred_id*.
_class$
" loc:@batch_normalization_18/beta*
T0*
_output_shapes
: : 
?
!batch_normalization_18/cond/ConstConst%^batch_normalization_18/cond/switch_t*
dtype0*
valueB *
_output_shapes
: 
?
#batch_normalization_18/cond/Const_1Const%^batch_normalization_18/cond/switch_t*
valueB *
_output_shapes
: *
dtype0
?
*batch_normalization_18/cond/FusedBatchNormFusedBatchNorm3batch_normalization_18/cond/FusedBatchNorm/Switch:1*batch_normalization_18/cond/ReadVariableOp,batch_normalization_18/cond/ReadVariableOp_1!batch_normalization_18/cond/Const#batch_normalization_18/cond/Const_1*
T0*L
_output_shapes:
8:??????????:?:?:?:?*
is_training(*
epsilon%?ŧ7*
data_formatNHWC
?
1batch_normalization_18/cond/FusedBatchNorm/SwitchSwitch	add_7/add#batch_normalization_18/cond/pred_id*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_7/add*
T0
?
,batch_normalization_18/cond/ReadVariableOp_2ReadVariableOp3batch_normalization_18/cond/ReadVariableOp_2/Switch*
dtype0*
_output_shapes	
:?
?
3batch_normalization_18/cond/ReadVariableOp_2/SwitchSwitchbatch_normalization_18/gamma#batch_normalization_18/cond/pred_id*
T0*
_output_shapes
: : */
_class%
#!loc:@batch_normalization_18/gamma
?
,batch_normalization_18/cond/ReadVariableOp_3ReadVariableOp3batch_normalization_18/cond/ReadVariableOp_3/Switch*
_output_shapes	
:?*
dtype0
?
3batch_normalization_18/cond/ReadVariableOp_3/SwitchSwitchbatch_normalization_18/beta#batch_normalization_18/cond/pred_id*
T0*
_output_shapes
: : *.
_class$
" loc:@batch_normalization_18/beta
?
;batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOpReadVariableOpBbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp/Switch*
_output_shapes	
:?*
dtype0
?
Bbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp/SwitchSwitch"batch_normalization_18/moving_mean#batch_normalization_18/cond/pred_id*5
_class+
)'loc:@batch_normalization_18/moving_mean*
T0*
_output_shapes
: : 
?
=batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1ReadVariableOpDbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch*
_output_shapes	
:?*
dtype0
?
Dbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1/SwitchSwitch&batch_normalization_18/moving_variance#batch_normalization_18/cond/pred_id*
_output_shapes
: : *
T0*9
_class/
-+loc:@batch_normalization_18/moving_variance
?
,batch_normalization_18/cond/FusedBatchNorm_1FusedBatchNorm3batch_normalization_18/cond/FusedBatchNorm_1/Switch,batch_normalization_18/cond/ReadVariableOp_2,batch_normalization_18/cond/ReadVariableOp_3;batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp=batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1*L
_output_shapes:
8:??????????:?:?:?:?*
epsilon%?ŧ7*
data_formatNHWC*
is_training( *
T0
?
3batch_normalization_18/cond/FusedBatchNorm_1/SwitchSwitch	add_7/add#batch_normalization_18/cond/pred_id*
T0*L
_output_shapes:
8:??????????:??????????*
_class
loc:@add_7/add
?
!batch_normalization_18/cond/MergeMerge,batch_normalization_18/cond/FusedBatchNorm_1*batch_normalization_18/cond/FusedBatchNorm*2
_output_shapes 
:??????????: *
T0*
N
?
#batch_normalization_18/cond/Merge_1Merge.batch_normalization_18/cond/FusedBatchNorm_1:1,batch_normalization_18/cond/FusedBatchNorm:1*
N*
_output_shapes
	:?: *
T0
?
#batch_normalization_18/cond/Merge_2Merge.batch_normalization_18/cond/FusedBatchNorm_1:2,batch_normalization_18/cond/FusedBatchNorm:2*
T0*
_output_shapes
	:?: *
N
}
$batch_normalization_18/cond_1/SwitchSwitchkeras_learning_phasekeras_learning_phase*
_output_shapes
: : *
T0

{
&batch_normalization_18/cond_1/switch_tIdentity&batch_normalization_18/cond_1/Switch:1*
T0
*
_output_shapes
: 
y
&batch_normalization_18/cond_1/switch_fIdentity$batch_normalization_18/cond_1/Switch*
_output_shapes
: *
T0

h
%batch_normalization_18/cond_1/pred_idIdentitykeras_learning_phase*
T0
*
_output_shapes
: 
?
#batch_normalization_18/cond_1/ConstConst'^batch_normalization_18/cond_1/switch_t*
valueB
 *?p}?*
_output_shapes
: *
dtype0
?
%batch_normalization_18/cond_1/Const_1Const'^batch_normalization_18/cond_1/switch_f*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
#batch_normalization_18/cond_1/MergeMerge%batch_normalization_18/cond_1/Const_1#batch_normalization_18/cond_1/Const*
N*
_output_shapes
: : *
T0
?
,batch_normalization_18/AssignMovingAvg/sub/xConst*
valueB
 *  ??*5
_class+
)'loc:@batch_normalization_18/moving_mean*
_output_shapes
: *
dtype0
?
*batch_normalization_18/AssignMovingAvg/subSub,batch_normalization_18/AssignMovingAvg/sub/x#batch_normalization_18/cond_1/Merge*
_output_shapes
: *
T0*5
_class+
)'loc:@batch_normalization_18/moving_mean
?
5batch_normalization_18/AssignMovingAvg/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
dtype0*
_output_shapes	
:?
?
,batch_normalization_18/AssignMovingAvg/sub_1Sub5batch_normalization_18/AssignMovingAvg/ReadVariableOp#batch_normalization_18/cond/Merge_1*
_output_shapes	
:?*
T0*5
_class+
)'loc:@batch_normalization_18/moving_mean
?
*batch_normalization_18/AssignMovingAvg/mulMul,batch_normalization_18/AssignMovingAvg/sub_1*batch_normalization_18/AssignMovingAvg/sub*
T0*
_output_shapes	
:?*5
_class+
)'loc:@batch_normalization_18/moving_mean
?
:batch_normalization_18/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batch_normalization_18/moving_mean*batch_normalization_18/AssignMovingAvg/mul*5
_class+
)'loc:@batch_normalization_18/moving_mean*
dtype0
?
7batch_normalization_18/AssignMovingAvg/ReadVariableOp_1ReadVariableOp"batch_normalization_18/moving_mean;^batch_normalization_18/AssignMovingAvg/AssignSubVariableOp*5
_class+
)'loc:@batch_normalization_18/moving_mean*
_output_shapes	
:?*
dtype0
?
.batch_normalization_18/AssignMovingAvg_1/sub/xConst*
valueB
 *  ??*
_output_shapes
: *9
_class/
-+loc:@batch_normalization_18/moving_variance*
dtype0
?
,batch_normalization_18/AssignMovingAvg_1/subSub.batch_normalization_18/AssignMovingAvg_1/sub/x#batch_normalization_18/cond_1/Merge*9
_class/
-+loc:@batch_normalization_18/moving_variance*
_output_shapes
: *
T0
?
7batch_normalization_18/AssignMovingAvg_1/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
dtype0*
_output_shapes	
:?
?
.batch_normalization_18/AssignMovingAvg_1/sub_1Sub7batch_normalization_18/AssignMovingAvg_1/ReadVariableOp#batch_normalization_18/cond/Merge_2*
T0*9
_class/
-+loc:@batch_normalization_18/moving_variance*
_output_shapes	
:?
?
,batch_normalization_18/AssignMovingAvg_1/mulMul.batch_normalization_18/AssignMovingAvg_1/sub_1,batch_normalization_18/AssignMovingAvg_1/sub*
T0*
_output_shapes	
:?*9
_class/
-+loc:@batch_normalization_18/moving_variance
?
<batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp&batch_normalization_18/moving_variance,batch_normalization_18/AssignMovingAvg_1/mul*
dtype0*9
_class/
-+loc:@batch_normalization_18/moving_variance
?
9batch_normalization_18/AssignMovingAvg_1/ReadVariableOp_1ReadVariableOp&batch_normalization_18/moving_variance=^batch_normalization_18/AssignMovingAvg_1/AssignSubVariableOp*
_output_shapes	
:?*
dtype0*9
_class/
-+loc:@batch_normalization_18/moving_variance
x
activation_17/ReluRelu!batch_normalization_18/cond/Merge*0
_output_shapes
:??????????*
T0
?
/global_average_pooling2d/Mean/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
?
global_average_pooling2d/MeanMeanactivation_17/Relu/global_average_pooling2d/Mean/reduction_indices*

Tidx0*
	keep_dims( *(
_output_shapes
:??????????*
T0
?
-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"   
   *
_class
loc:@dense/kernel*
_output_shapes
:
?
+dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *??۽*
_class
loc:@dense/kernel*
dtype0
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: *
valueB
 *???=
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
_output_shapes
:	?
*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?
*
T0*
_class
loc:@dense/kernel
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
:	?

?
dense/kernelVarHandleOp*
	container *
shape:	?
*
_output_shapes
: *
shared_namedense/kernel*
dtype0*
_class
loc:@dense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
?
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
_class
loc:@dense/kernel*
dtype0
?
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes
:	?
*
_class
loc:@dense/kernel
?
dense/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense/bias*
dtype0*
_output_shapes
:

?

dense/biasVarHandleOp*
shape:
*
shared_name
dense/bias*
	container *
_output_shapes
: *
dtype0*
_class
loc:@dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
{
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
_class
loc:@dense/bias*
dtype0
?
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
_class
loc:@dense/bias*
dtype0
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?
*
dtype0
?
dense/MatMulMatMulglobal_average_pooling2d/Meandense/MatMul/ReadVariableOp*'
_output_shapes
:?????????
*
T0*
transpose_a( *
transpose_b( 
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0
?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*
data_formatNHWC*'
_output_shapes
:?????????

a
activation_18/SoftmaxSoftmaxdense/BiasAdd*'
_output_shapes
:?????????
*
T0
^
keras_learning_phase_1/inputConst*
dtype0
*
value	B
 Z *
_output_shapes
: 
?
keras_learning_phase_1PlaceholderWithDefaultkeras_learning_phase_1/input*
dtype0
*
shape: *
_output_shapes
: 
]
VarIsInitializedOpVarIsInitializedOpbatch_normalization_11/beta*
_output_shapes
: 
j
VarIsInitializedOp_1VarIsInitializedOp&batch_normalization_15/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_2VarIsInitializedOp"batch_normalization_11/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_3VarIsInitializedOpbatch_normalization_1/gamma*
_output_shapes
: 
j
VarIsInitializedOp_4VarIsInitializedOp&batch_normalization_11/moving_variance*
_output_shapes
: 
S
VarIsInitializedOp_5VarIsInitializedOpconv2d_1/kernel*
_output_shapes
: 
`
VarIsInitializedOp_6VarIsInitializedOpbatch_normalization_12/gamma*
_output_shapes
: 
^
VarIsInitializedOp_7VarIsInitializedOpbatch_normalization_1/beta*
_output_shapes
: 
_
VarIsInitializedOp_8VarIsInitializedOpbatch_normalization_12/beta*
_output_shapes
: 
e
VarIsInitializedOp_9VarIsInitializedOp!batch_normalization_1/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_10VarIsInitializedOpbatch_normalization_8/gamma*
_output_shapes
: 
`
VarIsInitializedOp_11VarIsInitializedOpbatch_normalization_9/gamma*
_output_shapes
: 
j
VarIsInitializedOp_12VarIsInitializedOp%batch_normalization_1/moving_variance*
_output_shapes
: 
g
VarIsInitializedOp_13VarIsInitializedOp"batch_normalization_12/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_14VarIsInitializedOpbatch_normalization_8/beta*
_output_shapes
: 
_
VarIsInitializedOp_15VarIsInitializedOpbatch_normalization_9/beta*
_output_shapes
: 
k
VarIsInitializedOp_16VarIsInitializedOp&batch_normalization_12/moving_variance*
_output_shapes
: 
a
VarIsInitializedOp_17VarIsInitializedOpbatch_normalization_14/gamma*
_output_shapes
: 
f
VarIsInitializedOp_18VarIsInitializedOp!batch_normalization_9/moving_mean*
_output_shapes
: 
f
VarIsInitializedOp_19VarIsInitializedOp!batch_normalization_8/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_20VarIsInitializedOpbatch_normalization_14/beta*
_output_shapes
: 
U
VarIsInitializedOp_21VarIsInitializedOpconv2d_17/kernel*
_output_shapes
: 
j
VarIsInitializedOp_22VarIsInitializedOp%batch_normalization_9/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_23VarIsInitializedOp%batch_normalization_8/moving_variance*
_output_shapes
: 
g
VarIsInitializedOp_24VarIsInitializedOp"batch_normalization_14/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_25VarIsInitializedOpbatch_normalization_4/gamma*
_output_shapes
: 
U
VarIsInitializedOp_26VarIsInitializedOpconv2d_12/kernel*
_output_shapes
: 
k
VarIsInitializedOp_27VarIsInitializedOp&batch_normalization_14/moving_variance*
_output_shapes
: 
_
VarIsInitializedOp_28VarIsInitializedOpbatch_normalization_4/beta*
_output_shapes
: 
`
VarIsInitializedOp_29VarIsInitializedOpbatch_normalization_6/gamma*
_output_shapes
: 
U
VarIsInitializedOp_30VarIsInitializedOpconv2d_16/kernel*
_output_shapes
: 
U
VarIsInitializedOp_31VarIsInitializedOpconv2d_20/kernel*
_output_shapes
: 
f
VarIsInitializedOp_32VarIsInitializedOp!batch_normalization_4/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_33VarIsInitializedOpbatch_normalization_6/beta*
_output_shapes
: 
Q
VarIsInitializedOp_34VarIsInitializedOpdense/kernel*
_output_shapes
: 
j
VarIsInitializedOp_35VarIsInitializedOp%batch_normalization_4/moving_variance*
_output_shapes
: 
O
VarIsInitializedOp_36VarIsInitializedOp
dense/bias*
_output_shapes
: 
U
VarIsInitializedOp_37VarIsInitializedOpconv2d_15/kernel*
_output_shapes
: 
f
VarIsInitializedOp_38VarIsInitializedOp!batch_normalization_6/moving_mean*
_output_shapes
: 
R
VarIsInitializedOp_39VarIsInitializedOpconv2d/kernel*
_output_shapes
: 
j
VarIsInitializedOp_40VarIsInitializedOp%batch_normalization_6/moving_variance*
_output_shapes
: 
U
VarIsInitializedOp_41VarIsInitializedOpconv2d_10/kernel*
_output_shapes
: 
a
VarIsInitializedOp_42VarIsInitializedOpbatch_normalization_15/gamma*
_output_shapes
: 
T
VarIsInitializedOp_43VarIsInitializedOpconv2d_9/kernel*
_output_shapes
: 
`
VarIsInitializedOp_44VarIsInitializedOpbatch_normalization_2/gamma*
_output_shapes
: 
`
VarIsInitializedOp_45VarIsInitializedOpbatch_normalization_15/beta*
_output_shapes
: 
T
VarIsInitializedOp_46VarIsInitializedOpconv2d_2/kernel*
_output_shapes
: 
T
VarIsInitializedOp_47VarIsInitializedOpconv2d_5/kernel*
_output_shapes
: 
g
VarIsInitializedOp_48VarIsInitializedOp"batch_normalization_15/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_49VarIsInitializedOpbatch_normalization_2/beta*
_output_shapes
: 
f
VarIsInitializedOp_50VarIsInitializedOp!batch_normalization_2/moving_mean*
_output_shapes
: 
a
VarIsInitializedOp_51VarIsInitializedOpbatch_normalization_16/gamma*
_output_shapes
: 
a
VarIsInitializedOp_52VarIsInitializedOpbatch_normalization_17/gamma*
_output_shapes
: 
`
VarIsInitializedOp_53VarIsInitializedOpbatch_normalization_7/gamma*
_output_shapes
: 
j
VarIsInitializedOp_54VarIsInitializedOp%batch_normalization_2/moving_variance*
_output_shapes
: 
`
VarIsInitializedOp_55VarIsInitializedOpbatch_normalization_16/beta*
_output_shapes
: 
`
VarIsInitializedOp_56VarIsInitializedOpbatch_normalization_17/beta*
_output_shapes
: 
_
VarIsInitializedOp_57VarIsInitializedOpbatch_normalization_7/beta*
_output_shapes
: 
a
VarIsInitializedOp_58VarIsInitializedOpbatch_normalization_13/gamma*
_output_shapes
: 
g
VarIsInitializedOp_59VarIsInitializedOp"batch_normalization_17/moving_mean*
_output_shapes
: 
f
VarIsInitializedOp_60VarIsInitializedOp!batch_normalization_7/moving_mean*
_output_shapes
: 
g
VarIsInitializedOp_61VarIsInitializedOp"batch_normalization_16/moving_mean*
_output_shapes
: 
k
VarIsInitializedOp_62VarIsInitializedOp&batch_normalization_17/moving_variance*
_output_shapes
: 
j
VarIsInitializedOp_63VarIsInitializedOp%batch_normalization_7/moving_variance*
_output_shapes
: 
`
VarIsInitializedOp_64VarIsInitializedOpbatch_normalization_13/beta*
_output_shapes
: 
k
VarIsInitializedOp_65VarIsInitializedOp&batch_normalization_16/moving_variance*
_output_shapes
: 
a
VarIsInitializedOp_66VarIsInitializedOpbatch_normalization_18/gamma*
_output_shapes
: 
g
VarIsInitializedOp_67VarIsInitializedOp"batch_normalization_13/moving_mean*
_output_shapes
: 
`
VarIsInitializedOp_68VarIsInitializedOpbatch_normalization_18/beta*
_output_shapes
: 
`
VarIsInitializedOp_69VarIsInitializedOpbatch_normalization_3/gamma*
_output_shapes
: 
k
VarIsInitializedOp_70VarIsInitializedOp&batch_normalization_13/moving_variance*
_output_shapes
: 
`
VarIsInitializedOp_71VarIsInitializedOpbatch_normalization_5/gamma*
_output_shapes
: 
g
VarIsInitializedOp_72VarIsInitializedOp"batch_normalization_18/moving_mean*
_output_shapes
: 
_
VarIsInitializedOp_73VarIsInitializedOpbatch_normalization_3/beta*
_output_shapes
: 
U
VarIsInitializedOp_74VarIsInitializedOpconv2d_18/kernel*
_output_shapes
: 
f
VarIsInitializedOp_75VarIsInitializedOp!batch_normalization_3/moving_mean*
_output_shapes
: 
k
VarIsInitializedOp_76VarIsInitializedOp&batch_normalization_18/moving_variance*
_output_shapes
: 
_
VarIsInitializedOp_77VarIsInitializedOpbatch_normalization_5/beta*
_output_shapes
: 
a
VarIsInitializedOp_78VarIsInitializedOpbatch_normalization_10/gamma*
_output_shapes
: 
j
VarIsInitializedOp_79VarIsInitializedOp%batch_normalization_3/moving_variance*
_output_shapes
: 
f
VarIsInitializedOp_80VarIsInitializedOp!batch_normalization_5/moving_mean*
_output_shapes
: 
U
VarIsInitializedOp_81VarIsInitializedOpconv2d_13/kernel*
_output_shapes
: 
`
VarIsInitializedOp_82VarIsInitializedOpbatch_normalization_10/beta*
_output_shapes
: 
j
VarIsInitializedOp_83VarIsInitializedOp%batch_normalization_5/moving_variance*
_output_shapes
: 
T
VarIsInitializedOp_84VarIsInitializedOpconv2d_8/kernel*
_output_shapes
: 
g
VarIsInitializedOp_85VarIsInitializedOp"batch_normalization_10/moving_mean*
_output_shapes
: 
U
VarIsInitializedOp_86VarIsInitializedOpconv2d_19/kernel*
_output_shapes
: 
]
VarIsInitializedOp_87VarIsInitializedOpbatch_normalization/beta*
_output_shapes
: 
T
VarIsInitializedOp_88VarIsInitializedOpconv2d_7/kernel*
_output_shapes
: 
k
VarIsInitializedOp_89VarIsInitializedOp&batch_normalization_10/moving_variance*
_output_shapes
: 
d
VarIsInitializedOp_90VarIsInitializedOpbatch_normalization/moving_mean*
_output_shapes
: 
U
VarIsInitializedOp_91VarIsInitializedOpconv2d_11/kernel*
_output_shapes
: 
U
VarIsInitializedOp_92VarIsInitializedOpconv2d_14/kernel*
_output_shapes
: 
h
VarIsInitializedOp_93VarIsInitializedOp#batch_normalization/moving_variance*
_output_shapes
: 
T
VarIsInitializedOp_94VarIsInitializedOpconv2d_6/kernel*
_output_shapes
: 
T
VarIsInitializedOp_95VarIsInitializedOpconv2d_3/kernel*
_output_shapes
: 
T
VarIsInitializedOp_96VarIsInitializedOpconv2d_4/kernel*
_output_shapes
: 
a
VarIsInitializedOp_97VarIsInitializedOpbatch_normalization_11/gamma*
_output_shapes
: 
?
init_1NoOp ^batch_normalization/beta/Assign'^batch_normalization/moving_mean/Assign+^batch_normalization/moving_variance/Assign"^batch_normalization_1/beta/Assign#^batch_normalization_1/gamma/Assign)^batch_normalization_1/moving_mean/Assign-^batch_normalization_1/moving_variance/Assign#^batch_normalization_10/beta/Assign$^batch_normalization_10/gamma/Assign*^batch_normalization_10/moving_mean/Assign.^batch_normalization_10/moving_variance/Assign#^batch_normalization_11/beta/Assign$^batch_normalization_11/gamma/Assign*^batch_normalization_11/moving_mean/Assign.^batch_normalization_11/moving_variance/Assign#^batch_normalization_12/beta/Assign$^batch_normalization_12/gamma/Assign*^batch_normalization_12/moving_mean/Assign.^batch_normalization_12/moving_variance/Assign#^batch_normalization_13/beta/Assign$^batch_normalization_13/gamma/Assign*^batch_normalization_13/moving_mean/Assign.^batch_normalization_13/moving_variance/Assign#^batch_normalization_14/beta/Assign$^batch_normalization_14/gamma/Assign*^batch_normalization_14/moving_mean/Assign.^batch_normalization_14/moving_variance/Assign#^batch_normalization_15/beta/Assign$^batch_normalization_15/gamma/Assign*^batch_normalization_15/moving_mean/Assign.^batch_normalization_15/moving_variance/Assign#^batch_normalization_16/beta/Assign$^batch_normalization_16/gamma/Assign*^batch_normalization_16/moving_mean/Assign.^batch_normalization_16/moving_variance/Assign#^batch_normalization_17/beta/Assign$^batch_normalization_17/gamma/Assign*^batch_normalization_17/moving_mean/Assign.^batch_normalization_17/moving_variance/Assign#^batch_normalization_18/beta/Assign$^batch_normalization_18/gamma/Assign*^batch_normalization_18/moving_mean/Assign.^batch_normalization_18/moving_variance/Assign"^batch_normalization_2/beta/Assign#^batch_normalization_2/gamma/Assign)^batch_normalization_2/moving_mean/Assign-^batch_normalization_2/moving_variance/Assign"^batch_normalization_3/beta/Assign#^batch_normalization_3/gamma/Assign)^batch_normalization_3/moving_mean/Assign-^batch_normalization_3/moving_variance/Assign"^batch_normalization_4/beta/Assign#^batch_normalization_4/gamma/Assign)^batch_normalization_4/moving_mean/Assign-^batch_normalization_4/moving_variance/Assign"^batch_normalization_5/beta/Assign#^batch_normalization_5/gamma/Assign)^batch_normalization_5/moving_mean/Assign-^batch_normalization_5/moving_variance/Assign"^batch_normalization_6/beta/Assign#^batch_normalization_6/gamma/Assign)^batch_normalization_6/moving_mean/Assign-^batch_normalization_6/moving_variance/Assign"^batch_normalization_7/beta/Assign#^batch_normalization_7/gamma/Assign)^batch_normalization_7/moving_mean/Assign-^batch_normalization_7/moving_variance/Assign"^batch_normalization_8/beta/Assign#^batch_normalization_8/gamma/Assign)^batch_normalization_8/moving_mean/Assign-^batch_normalization_8/moving_variance/Assign"^batch_normalization_9/beta/Assign#^batch_normalization_9/gamma/Assign)^batch_normalization_9/moving_mean/Assign-^batch_normalization_9/moving_variance/Assign^conv2d/kernel/Assign^conv2d_1/kernel/Assign^conv2d_10/kernel/Assign^conv2d_11/kernel/Assign^conv2d_12/kernel/Assign^conv2d_13/kernel/Assign^conv2d_14/kernel/Assign^conv2d_15/kernel/Assign^conv2d_16/kernel/Assign^conv2d_17/kernel/Assign^conv2d_18/kernel/Assign^conv2d_19/kernel/Assign^conv2d_2/kernel/Assign^conv2d_20/kernel/Assign^conv2d_3/kernel/Assign^conv2d_4/kernel/Assign^conv2d_5/kernel/Assign^conv2d_6/kernel/Assign^conv2d_7/kernel/Assign^conv2d_8/kernel/Assign^conv2d_9/kernel/Assign^dense/bias/Assign^dense/kernel/Assign
T
PlaceholderPlaceholder*
_output_shapes
:*
dtype0*
shape:
X
AssignVariableOpAssignVariableOpbatch_normalization/betaPlaceholder*
dtype0
v
ReadVariableOpReadVariableOpbatch_normalization/beta^AssignVariableOp*
dtype0*
_output_shapes
:
V
Placeholder_1Placeholder*
dtype0*
_output_shapes
:*
shape:
c
AssignVariableOp_1AssignVariableOpbatch_normalization/moving_meanPlaceholder_1*
dtype0
?
ReadVariableOp_1ReadVariableOpbatch_normalization/moving_mean^AssignVariableOp_1*
_output_shapes
:*
dtype0
V
Placeholder_2Placeholder*
shape:*
_output_shapes
:*
dtype0
g
AssignVariableOp_2AssignVariableOp#batch_normalization/moving_variancePlaceholder_2*
dtype0
?
ReadVariableOp_2ReadVariableOp#batch_normalization/moving_variance^AssignVariableOp_2*
_output_shapes
:*
dtype0
n
Placeholder_3Placeholder*
dtype0*
shape:@*&
_output_shapes
:@
Q
AssignVariableOp_3AssignVariableOpconv2d/kernelPlaceholder_3*
dtype0
{
ReadVariableOp_3ReadVariableOpconv2d/kernel^AssignVariableOp_3*&
_output_shapes
:@*
dtype0
V
Placeholder_4Placeholder*
dtype0*
shape:@*
_output_shapes
:@
_
AssignVariableOp_4AssignVariableOpbatch_normalization_1/gammaPlaceholder_4*
dtype0
}
ReadVariableOp_4ReadVariableOpbatch_normalization_1/gamma^AssignVariableOp_4*
_output_shapes
:@*
dtype0
V
Placeholder_5Placeholder*
shape:@*
dtype0*
_output_shapes
:@
^
AssignVariableOp_5AssignVariableOpbatch_normalization_1/betaPlaceholder_5*
dtype0
|
ReadVariableOp_5ReadVariableOpbatch_normalization_1/beta^AssignVariableOp_5*
dtype0*
_output_shapes
:@
V
Placeholder_6Placeholder*
shape:@*
dtype0*
_output_shapes
:@
e
AssignVariableOp_6AssignVariableOp!batch_normalization_1/moving_meanPlaceholder_6*
dtype0
?
ReadVariableOp_6ReadVariableOp!batch_normalization_1/moving_mean^AssignVariableOp_6*
dtype0*
_output_shapes
:@
V
Placeholder_7Placeholder*
dtype0*
_output_shapes
:@*
shape:@
i
AssignVariableOp_7AssignVariableOp%batch_normalization_1/moving_variancePlaceholder_7*
dtype0
?
ReadVariableOp_7ReadVariableOp%batch_normalization_1/moving_variance^AssignVariableOp_7*
_output_shapes
:@*
dtype0
V
Placeholder_8Placeholder*
_output_shapes
:@*
shape:@*
dtype0
_
AssignVariableOp_8AssignVariableOpbatch_normalization_2/gammaPlaceholder_8*
dtype0
}
ReadVariableOp_8ReadVariableOpbatch_normalization_2/gamma^AssignVariableOp_8*
_output_shapes
:@*
dtype0
V
Placeholder_9Placeholder*
shape:@*
dtype0*
_output_shapes
:@
^
AssignVariableOp_9AssignVariableOpbatch_normalization_2/betaPlaceholder_9*
dtype0
|
ReadVariableOp_9ReadVariableOpbatch_normalization_2/beta^AssignVariableOp_9*
_output_shapes
:@*
dtype0
W
Placeholder_10Placeholder*
shape:@*
dtype0*
_output_shapes
:@
g
AssignVariableOp_10AssignVariableOp!batch_normalization_2/moving_meanPlaceholder_10*
dtype0
?
ReadVariableOp_10ReadVariableOp!batch_normalization_2/moving_mean^AssignVariableOp_10*
dtype0*
_output_shapes
:@
W
Placeholder_11Placeholder*
shape:@*
dtype0*
_output_shapes
:@
k
AssignVariableOp_11AssignVariableOp%batch_normalization_2/moving_variancePlaceholder_11*
dtype0
?
ReadVariableOp_11ReadVariableOp%batch_normalization_2/moving_variance^AssignVariableOp_11*
dtype0*
_output_shapes
:@
o
Placeholder_12Placeholder*&
_output_shapes
:@@*
dtype0*
shape:@@
U
AssignVariableOp_12AssignVariableOpconv2d_2/kernelPlaceholder_12*
dtype0

ReadVariableOp_12ReadVariableOpconv2d_2/kernel^AssignVariableOp_12*
dtype0*&
_output_shapes
:@@
W
Placeholder_13Placeholder*
_output_shapes
:@*
dtype0*
shape:@
a
AssignVariableOp_13AssignVariableOpbatch_normalization_3/gammaPlaceholder_13*
dtype0

ReadVariableOp_13ReadVariableOpbatch_normalization_3/gamma^AssignVariableOp_13*
dtype0*
_output_shapes
:@
W
Placeholder_14Placeholder*
_output_shapes
:@*
shape:@*
dtype0
`
AssignVariableOp_14AssignVariableOpbatch_normalization_3/betaPlaceholder_14*
dtype0
~
ReadVariableOp_14ReadVariableOpbatch_normalization_3/beta^AssignVariableOp_14*
dtype0*
_output_shapes
:@
W
Placeholder_15Placeholder*
_output_shapes
:@*
dtype0*
shape:@
g
AssignVariableOp_15AssignVariableOp!batch_normalization_3/moving_meanPlaceholder_15*
dtype0
?
ReadVariableOp_15ReadVariableOp!batch_normalization_3/moving_mean^AssignVariableOp_15*
dtype0*
_output_shapes
:@
W
Placeholder_16Placeholder*
dtype0*
shape:@*
_output_shapes
:@
k
AssignVariableOp_16AssignVariableOp%batch_normalization_3/moving_variancePlaceholder_16*
dtype0
?
ReadVariableOp_16ReadVariableOp%batch_normalization_3/moving_variance^AssignVariableOp_16*
_output_shapes
:@*
dtype0
o
Placeholder_17Placeholder*
shape:@@*&
_output_shapes
:@@*
dtype0
U
AssignVariableOp_17AssignVariableOpconv2d_3/kernelPlaceholder_17*
dtype0

ReadVariableOp_17ReadVariableOpconv2d_3/kernel^AssignVariableOp_17*&
_output_shapes
:@@*
dtype0
o
Placeholder_18Placeholder*
dtype0*
shape:@@*&
_output_shapes
:@@
U
AssignVariableOp_18AssignVariableOpconv2d_1/kernelPlaceholder_18*
dtype0

ReadVariableOp_18ReadVariableOpconv2d_1/kernel^AssignVariableOp_18*&
_output_shapes
:@@*
dtype0
W
Placeholder_19Placeholder*
shape:@*
_output_shapes
:@*
dtype0
a
AssignVariableOp_19AssignVariableOpbatch_normalization_4/gammaPlaceholder_19*
dtype0

ReadVariableOp_19ReadVariableOpbatch_normalization_4/gamma^AssignVariableOp_19*
dtype0*
_output_shapes
:@
W
Placeholder_20Placeholder*
shape:@*
dtype0*
_output_shapes
:@
`
AssignVariableOp_20AssignVariableOpbatch_normalization_4/betaPlaceholder_20*
dtype0
~
ReadVariableOp_20ReadVariableOpbatch_normalization_4/beta^AssignVariableOp_20*
_output_shapes
:@*
dtype0
W
Placeholder_21Placeholder*
shape:@*
dtype0*
_output_shapes
:@
g
AssignVariableOp_21AssignVariableOp!batch_normalization_4/moving_meanPlaceholder_21*
dtype0
?
ReadVariableOp_21ReadVariableOp!batch_normalization_4/moving_mean^AssignVariableOp_21*
_output_shapes
:@*
dtype0
W
Placeholder_22Placeholder*
_output_shapes
:@*
shape:@*
dtype0
k
AssignVariableOp_22AssignVariableOp%batch_normalization_4/moving_variancePlaceholder_22*
dtype0
?
ReadVariableOp_22ReadVariableOp%batch_normalization_4/moving_variance^AssignVariableOp_22*
_output_shapes
:@*
dtype0
o
Placeholder_23Placeholder*&
_output_shapes
:@@*
shape:@@*
dtype0
U
AssignVariableOp_23AssignVariableOpconv2d_4/kernelPlaceholder_23*
dtype0

ReadVariableOp_23ReadVariableOpconv2d_4/kernel^AssignVariableOp_23*&
_output_shapes
:@@*
dtype0
W
Placeholder_24Placeholder*
shape:@*
_output_shapes
:@*
dtype0
a
AssignVariableOp_24AssignVariableOpbatch_normalization_5/gammaPlaceholder_24*
dtype0

ReadVariableOp_24ReadVariableOpbatch_normalization_5/gamma^AssignVariableOp_24*
dtype0*
_output_shapes
:@
W
Placeholder_25Placeholder*
_output_shapes
:@*
dtype0*
shape:@
`
AssignVariableOp_25AssignVariableOpbatch_normalization_5/betaPlaceholder_25*
dtype0
~
ReadVariableOp_25ReadVariableOpbatch_normalization_5/beta^AssignVariableOp_25*
dtype0*
_output_shapes
:@
W
Placeholder_26Placeholder*
shape:@*
dtype0*
_output_shapes
:@
g
AssignVariableOp_26AssignVariableOp!batch_normalization_5/moving_meanPlaceholder_26*
dtype0
?
ReadVariableOp_26ReadVariableOp!batch_normalization_5/moving_mean^AssignVariableOp_26*
dtype0*
_output_shapes
:@
W
Placeholder_27Placeholder*
shape:@*
_output_shapes
:@*
dtype0
k
AssignVariableOp_27AssignVariableOp%batch_normalization_5/moving_variancePlaceholder_27*
dtype0
?
ReadVariableOp_27ReadVariableOp%batch_normalization_5/moving_variance^AssignVariableOp_27*
_output_shapes
:@*
dtype0
o
Placeholder_28Placeholder*
shape:@@*&
_output_shapes
:@@*
dtype0
U
AssignVariableOp_28AssignVariableOpconv2d_5/kernelPlaceholder_28*
dtype0

ReadVariableOp_28ReadVariableOpconv2d_5/kernel^AssignVariableOp_28*
dtype0*&
_output_shapes
:@@
W
Placeholder_29Placeholder*
_output_shapes
:@*
shape:@*
dtype0
a
AssignVariableOp_29AssignVariableOpbatch_normalization_6/gammaPlaceholder_29*
dtype0

ReadVariableOp_29ReadVariableOpbatch_normalization_6/gamma^AssignVariableOp_29*
_output_shapes
:@*
dtype0
W
Placeholder_30Placeholder*
shape:@*
dtype0*
_output_shapes
:@
`
AssignVariableOp_30AssignVariableOpbatch_normalization_6/betaPlaceholder_30*
dtype0
~
ReadVariableOp_30ReadVariableOpbatch_normalization_6/beta^AssignVariableOp_30*
dtype0*
_output_shapes
:@
W
Placeholder_31Placeholder*
dtype0*
shape:@*
_output_shapes
:@
g
AssignVariableOp_31AssignVariableOp!batch_normalization_6/moving_meanPlaceholder_31*
dtype0
?
ReadVariableOp_31ReadVariableOp!batch_normalization_6/moving_mean^AssignVariableOp_31*
dtype0*
_output_shapes
:@
W
Placeholder_32Placeholder*
_output_shapes
:@*
dtype0*
shape:@
k
AssignVariableOp_32AssignVariableOp%batch_normalization_6/moving_variancePlaceholder_32*
dtype0
?
ReadVariableOp_32ReadVariableOp%batch_normalization_6/moving_variance^AssignVariableOp_32*
dtype0*
_output_shapes
:@
q
Placeholder_33Placeholder*
dtype0*'
_output_shapes
:@?*
shape:@?
U
AssignVariableOp_33AssignVariableOpconv2d_7/kernelPlaceholder_33*
dtype0
?
ReadVariableOp_33ReadVariableOpconv2d_7/kernel^AssignVariableOp_33*'
_output_shapes
:@?*
dtype0
Y
Placeholder_34Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
a
AssignVariableOp_34AssignVariableOpbatch_normalization_7/gammaPlaceholder_34*
dtype0
?
ReadVariableOp_34ReadVariableOpbatch_normalization_7/gamma^AssignVariableOp_34*
_output_shapes	
:?*
dtype0
Y
Placeholder_35Placeholder*
_output_shapes	
:?*
shape:?*
dtype0
`
AssignVariableOp_35AssignVariableOpbatch_normalization_7/betaPlaceholder_35*
dtype0

ReadVariableOp_35ReadVariableOpbatch_normalization_7/beta^AssignVariableOp_35*
dtype0*
_output_shapes	
:?
Y
Placeholder_36Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
g
AssignVariableOp_36AssignVariableOp!batch_normalization_7/moving_meanPlaceholder_36*
dtype0
?
ReadVariableOp_36ReadVariableOp!batch_normalization_7/moving_mean^AssignVariableOp_36*
_output_shapes	
:?*
dtype0
Y
Placeholder_37Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
k
AssignVariableOp_37AssignVariableOp%batch_normalization_7/moving_variancePlaceholder_37*
dtype0
?
ReadVariableOp_37ReadVariableOp%batch_normalization_7/moving_variance^AssignVariableOp_37*
_output_shapes	
:?*
dtype0
s
Placeholder_38Placeholder*
shape:??*(
_output_shapes
:??*
dtype0
U
AssignVariableOp_38AssignVariableOpconv2d_8/kernelPlaceholder_38*
dtype0
?
ReadVariableOp_38ReadVariableOpconv2d_8/kernel^AssignVariableOp_38*(
_output_shapes
:??*
dtype0
q
Placeholder_39Placeholder*'
_output_shapes
:@?*
shape:@?*
dtype0
U
AssignVariableOp_39AssignVariableOpconv2d_6/kernelPlaceholder_39*
dtype0
?
ReadVariableOp_39ReadVariableOpconv2d_6/kernel^AssignVariableOp_39*
dtype0*'
_output_shapes
:@?
Y
Placeholder_40Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
a
AssignVariableOp_40AssignVariableOpbatch_normalization_8/gammaPlaceholder_40*
dtype0
?
ReadVariableOp_40ReadVariableOpbatch_normalization_8/gamma^AssignVariableOp_40*
dtype0*
_output_shapes	
:?
Y
Placeholder_41Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
`
AssignVariableOp_41AssignVariableOpbatch_normalization_8/betaPlaceholder_41*
dtype0

ReadVariableOp_41ReadVariableOpbatch_normalization_8/beta^AssignVariableOp_41*
_output_shapes	
:?*
dtype0
Y
Placeholder_42Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
g
AssignVariableOp_42AssignVariableOp!batch_normalization_8/moving_meanPlaceholder_42*
dtype0
?
ReadVariableOp_42ReadVariableOp!batch_normalization_8/moving_mean^AssignVariableOp_42*
_output_shapes	
:?*
dtype0
Y
Placeholder_43Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
k
AssignVariableOp_43AssignVariableOp%batch_normalization_8/moving_variancePlaceholder_43*
dtype0
?
ReadVariableOp_43ReadVariableOp%batch_normalization_8/moving_variance^AssignVariableOp_43*
_output_shapes	
:?*
dtype0
s
Placeholder_44Placeholder*
shape:??*(
_output_shapes
:??*
dtype0
U
AssignVariableOp_44AssignVariableOpconv2d_9/kernelPlaceholder_44*
dtype0
?
ReadVariableOp_44ReadVariableOpconv2d_9/kernel^AssignVariableOp_44*
dtype0*(
_output_shapes
:??
Y
Placeholder_45Placeholder*
_output_shapes	
:?*
shape:?*
dtype0
a
AssignVariableOp_45AssignVariableOpbatch_normalization_9/gammaPlaceholder_45*
dtype0
?
ReadVariableOp_45ReadVariableOpbatch_normalization_9/gamma^AssignVariableOp_45*
_output_shapes	
:?*
dtype0
Y
Placeholder_46Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
`
AssignVariableOp_46AssignVariableOpbatch_normalization_9/betaPlaceholder_46*
dtype0

ReadVariableOp_46ReadVariableOpbatch_normalization_9/beta^AssignVariableOp_46*
dtype0*
_output_shapes	
:?
Y
Placeholder_47Placeholder*
_output_shapes	
:?*
shape:?*
dtype0
g
AssignVariableOp_47AssignVariableOp!batch_normalization_9/moving_meanPlaceholder_47*
dtype0
?
ReadVariableOp_47ReadVariableOp!batch_normalization_9/moving_mean^AssignVariableOp_47*
_output_shapes	
:?*
dtype0
Y
Placeholder_48Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
k
AssignVariableOp_48AssignVariableOp%batch_normalization_9/moving_variancePlaceholder_48*
dtype0
?
ReadVariableOp_48ReadVariableOp%batch_normalization_9/moving_variance^AssignVariableOp_48*
_output_shapes	
:?*
dtype0
s
Placeholder_49Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
V
AssignVariableOp_49AssignVariableOpconv2d_10/kernelPlaceholder_49*
dtype0
?
ReadVariableOp_49ReadVariableOpconv2d_10/kernel^AssignVariableOp_49*(
_output_shapes
:??*
dtype0
Y
Placeholder_50Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
b
AssignVariableOp_50AssignVariableOpbatch_normalization_10/gammaPlaceholder_50*
dtype0
?
ReadVariableOp_50ReadVariableOpbatch_normalization_10/gamma^AssignVariableOp_50*
dtype0*
_output_shapes	
:?
Y
Placeholder_51Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
a
AssignVariableOp_51AssignVariableOpbatch_normalization_10/betaPlaceholder_51*
dtype0
?
ReadVariableOp_51ReadVariableOpbatch_normalization_10/beta^AssignVariableOp_51*
_output_shapes	
:?*
dtype0
Y
Placeholder_52Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
h
AssignVariableOp_52AssignVariableOp"batch_normalization_10/moving_meanPlaceholder_52*
dtype0
?
ReadVariableOp_52ReadVariableOp"batch_normalization_10/moving_mean^AssignVariableOp_52*
_output_shapes	
:?*
dtype0
Y
Placeholder_53Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
l
AssignVariableOp_53AssignVariableOp&batch_normalization_10/moving_variancePlaceholder_53*
dtype0
?
ReadVariableOp_53ReadVariableOp&batch_normalization_10/moving_variance^AssignVariableOp_53*
dtype0*
_output_shapes	
:?
s
Placeholder_54Placeholder*
dtype0*(
_output_shapes
:??*
shape:??
V
AssignVariableOp_54AssignVariableOpconv2d_12/kernelPlaceholder_54*
dtype0
?
ReadVariableOp_54ReadVariableOpconv2d_12/kernel^AssignVariableOp_54*(
_output_shapes
:??*
dtype0
Y
Placeholder_55Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
b
AssignVariableOp_55AssignVariableOpbatch_normalization_11/gammaPlaceholder_55*
dtype0
?
ReadVariableOp_55ReadVariableOpbatch_normalization_11/gamma^AssignVariableOp_55*
dtype0*
_output_shapes	
:?
Y
Placeholder_56Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
a
AssignVariableOp_56AssignVariableOpbatch_normalization_11/betaPlaceholder_56*
dtype0
?
ReadVariableOp_56ReadVariableOpbatch_normalization_11/beta^AssignVariableOp_56*
dtype0*
_output_shapes	
:?
Y
Placeholder_57Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
h
AssignVariableOp_57AssignVariableOp"batch_normalization_11/moving_meanPlaceholder_57*
dtype0
?
ReadVariableOp_57ReadVariableOp"batch_normalization_11/moving_mean^AssignVariableOp_57*
dtype0*
_output_shapes	
:?
Y
Placeholder_58Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
l
AssignVariableOp_58AssignVariableOp&batch_normalization_11/moving_variancePlaceholder_58*
dtype0
?
ReadVariableOp_58ReadVariableOp&batch_normalization_11/moving_variance^AssignVariableOp_58*
_output_shapes	
:?*
dtype0
s
Placeholder_59Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
V
AssignVariableOp_59AssignVariableOpconv2d_13/kernelPlaceholder_59*
dtype0
?
ReadVariableOp_59ReadVariableOpconv2d_13/kernel^AssignVariableOp_59*(
_output_shapes
:??*
dtype0
s
Placeholder_60Placeholder*
dtype0*(
_output_shapes
:??*
shape:??
V
AssignVariableOp_60AssignVariableOpconv2d_11/kernelPlaceholder_60*
dtype0
?
ReadVariableOp_60ReadVariableOpconv2d_11/kernel^AssignVariableOp_60*(
_output_shapes
:??*
dtype0
Y
Placeholder_61Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
b
AssignVariableOp_61AssignVariableOpbatch_normalization_12/gammaPlaceholder_61*
dtype0
?
ReadVariableOp_61ReadVariableOpbatch_normalization_12/gamma^AssignVariableOp_61*
dtype0*
_output_shapes	
:?
Y
Placeholder_62Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
a
AssignVariableOp_62AssignVariableOpbatch_normalization_12/betaPlaceholder_62*
dtype0
?
ReadVariableOp_62ReadVariableOpbatch_normalization_12/beta^AssignVariableOp_62*
dtype0*
_output_shapes	
:?
Y
Placeholder_63Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
h
AssignVariableOp_63AssignVariableOp"batch_normalization_12/moving_meanPlaceholder_63*
dtype0
?
ReadVariableOp_63ReadVariableOp"batch_normalization_12/moving_mean^AssignVariableOp_63*
_output_shapes	
:?*
dtype0
Y
Placeholder_64Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
l
AssignVariableOp_64AssignVariableOp&batch_normalization_12/moving_variancePlaceholder_64*
dtype0
?
ReadVariableOp_64ReadVariableOp&batch_normalization_12/moving_variance^AssignVariableOp_64*
dtype0*
_output_shapes	
:?
s
Placeholder_65Placeholder*
shape:??*(
_output_shapes
:??*
dtype0
V
AssignVariableOp_65AssignVariableOpconv2d_14/kernelPlaceholder_65*
dtype0
?
ReadVariableOp_65ReadVariableOpconv2d_14/kernel^AssignVariableOp_65*
dtype0*(
_output_shapes
:??
Y
Placeholder_66Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
b
AssignVariableOp_66AssignVariableOpbatch_normalization_13/gammaPlaceholder_66*
dtype0
?
ReadVariableOp_66ReadVariableOpbatch_normalization_13/gamma^AssignVariableOp_66*
dtype0*
_output_shapes	
:?
Y
Placeholder_67Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
a
AssignVariableOp_67AssignVariableOpbatch_normalization_13/betaPlaceholder_67*
dtype0
?
ReadVariableOp_67ReadVariableOpbatch_normalization_13/beta^AssignVariableOp_67*
_output_shapes	
:?*
dtype0
Y
Placeholder_68Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
h
AssignVariableOp_68AssignVariableOp"batch_normalization_13/moving_meanPlaceholder_68*
dtype0
?
ReadVariableOp_68ReadVariableOp"batch_normalization_13/moving_mean^AssignVariableOp_68*
dtype0*
_output_shapes	
:?
Y
Placeholder_69Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
l
AssignVariableOp_69AssignVariableOp&batch_normalization_13/moving_variancePlaceholder_69*
dtype0
?
ReadVariableOp_69ReadVariableOp&batch_normalization_13/moving_variance^AssignVariableOp_69*
_output_shapes	
:?*
dtype0
s
Placeholder_70Placeholder*
shape:??*(
_output_shapes
:??*
dtype0
V
AssignVariableOp_70AssignVariableOpconv2d_15/kernelPlaceholder_70*
dtype0
?
ReadVariableOp_70ReadVariableOpconv2d_15/kernel^AssignVariableOp_70*
dtype0*(
_output_shapes
:??
Y
Placeholder_71Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
b
AssignVariableOp_71AssignVariableOpbatch_normalization_14/gammaPlaceholder_71*
dtype0
?
ReadVariableOp_71ReadVariableOpbatch_normalization_14/gamma^AssignVariableOp_71*
dtype0*
_output_shapes	
:?
Y
Placeholder_72Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
a
AssignVariableOp_72AssignVariableOpbatch_normalization_14/betaPlaceholder_72*
dtype0
?
ReadVariableOp_72ReadVariableOpbatch_normalization_14/beta^AssignVariableOp_72*
dtype0*
_output_shapes	
:?
Y
Placeholder_73Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
h
AssignVariableOp_73AssignVariableOp"batch_normalization_14/moving_meanPlaceholder_73*
dtype0
?
ReadVariableOp_73ReadVariableOp"batch_normalization_14/moving_mean^AssignVariableOp_73*
_output_shapes	
:?*
dtype0
Y
Placeholder_74Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
l
AssignVariableOp_74AssignVariableOp&batch_normalization_14/moving_variancePlaceholder_74*
dtype0
?
ReadVariableOp_74ReadVariableOp&batch_normalization_14/moving_variance^AssignVariableOp_74*
_output_shapes	
:?*
dtype0
s
Placeholder_75Placeholder*
dtype0*(
_output_shapes
:??*
shape:??
V
AssignVariableOp_75AssignVariableOpconv2d_17/kernelPlaceholder_75*
dtype0
?
ReadVariableOp_75ReadVariableOpconv2d_17/kernel^AssignVariableOp_75*
dtype0*(
_output_shapes
:??
Y
Placeholder_76Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
b
AssignVariableOp_76AssignVariableOpbatch_normalization_15/gammaPlaceholder_76*
dtype0
?
ReadVariableOp_76ReadVariableOpbatch_normalization_15/gamma^AssignVariableOp_76*
_output_shapes	
:?*
dtype0
Y
Placeholder_77Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
a
AssignVariableOp_77AssignVariableOpbatch_normalization_15/betaPlaceholder_77*
dtype0
?
ReadVariableOp_77ReadVariableOpbatch_normalization_15/beta^AssignVariableOp_77*
_output_shapes	
:?*
dtype0
Y
Placeholder_78Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
h
AssignVariableOp_78AssignVariableOp"batch_normalization_15/moving_meanPlaceholder_78*
dtype0
?
ReadVariableOp_78ReadVariableOp"batch_normalization_15/moving_mean^AssignVariableOp_78*
_output_shapes	
:?*
dtype0
Y
Placeholder_79Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
l
AssignVariableOp_79AssignVariableOp&batch_normalization_15/moving_variancePlaceholder_79*
dtype0
?
ReadVariableOp_79ReadVariableOp&batch_normalization_15/moving_variance^AssignVariableOp_79*
dtype0*
_output_shapes	
:?
s
Placeholder_80Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
V
AssignVariableOp_80AssignVariableOpconv2d_18/kernelPlaceholder_80*
dtype0
?
ReadVariableOp_80ReadVariableOpconv2d_18/kernel^AssignVariableOp_80*(
_output_shapes
:??*
dtype0
s
Placeholder_81Placeholder*
shape:??*
dtype0*(
_output_shapes
:??
V
AssignVariableOp_81AssignVariableOpconv2d_16/kernelPlaceholder_81*
dtype0
?
ReadVariableOp_81ReadVariableOpconv2d_16/kernel^AssignVariableOp_81*
dtype0*(
_output_shapes
:??
Y
Placeholder_82Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
b
AssignVariableOp_82AssignVariableOpbatch_normalization_16/gammaPlaceholder_82*
dtype0
?
ReadVariableOp_82ReadVariableOpbatch_normalization_16/gamma^AssignVariableOp_82*
dtype0*
_output_shapes	
:?
Y
Placeholder_83Placeholder*
dtype0*
_output_shapes	
:?*
shape:?
a
AssignVariableOp_83AssignVariableOpbatch_normalization_16/betaPlaceholder_83*
dtype0
?
ReadVariableOp_83ReadVariableOpbatch_normalization_16/beta^AssignVariableOp_83*
dtype0*
_output_shapes	
:?
Y
Placeholder_84Placeholder*
shape:?*
dtype0*
_output_shapes	
:?
h
AssignVariableOp_84AssignVariableOp"batch_normalization_16/moving_meanPlaceholder_84*
dtype0
?
ReadVariableOp_84ReadVariableOp"batch_normalization_16/moving_mean^AssignVariableOp_84*
dtype0*
_output_shapes	
:?
Y
Placeholder_85Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
l
AssignVariableOp_85AssignVariableOp&batch_normalization_16/moving_variancePlaceholder_85*
dtype0
?
ReadVariableOp_85ReadVariableOp&batch_normalization_16/moving_variance^AssignVariableOp_85*
dtype0*
_output_shapes	
:?
s
Placeholder_86Placeholder*(
_output_shapes
:??*
dtype0*
shape:??
V
AssignVariableOp_86AssignVariableOpconv2d_19/kernelPlaceholder_86*
dtype0
?
ReadVariableOp_86ReadVariableOpconv2d_19/kernel^AssignVariableOp_86*(
_output_shapes
:??*
dtype0
Y
Placeholder_87Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
b
AssignVariableOp_87AssignVariableOpbatch_normalization_17/gammaPlaceholder_87*
dtype0
?
ReadVariableOp_87ReadVariableOpbatch_normalization_17/gamma^AssignVariableOp_87*
dtype0*
_output_shapes	
:?
Y
Placeholder_88Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
a
AssignVariableOp_88AssignVariableOpbatch_normalization_17/betaPlaceholder_88*
dtype0
?
ReadVariableOp_88ReadVariableOpbatch_normalization_17/beta^AssignVariableOp_88*
_output_shapes	
:?*
dtype0
Y
Placeholder_89Placeholder*
dtype0*
shape:?*
_output_shapes	
:?
h
AssignVariableOp_89AssignVariableOp"batch_normalization_17/moving_meanPlaceholder_89*
dtype0
?
ReadVariableOp_89ReadVariableOp"batch_normalization_17/moving_mean^AssignVariableOp_89*
dtype0*
_output_shapes	
:?
Y
Placeholder_90Placeholder*
_output_shapes	
:?*
shape:?*
dtype0
l
AssignVariableOp_90AssignVariableOp&batch_normalization_17/moving_variancePlaceholder_90*
dtype0
?
ReadVariableOp_90ReadVariableOp&batch_normalization_17/moving_variance^AssignVariableOp_90*
_output_shapes	
:?*
dtype0
s
Placeholder_91Placeholder*(
_output_shapes
:??*
shape:??*
dtype0
V
AssignVariableOp_91AssignVariableOpconv2d_20/kernelPlaceholder_91*
dtype0
?
ReadVariableOp_91ReadVariableOpconv2d_20/kernel^AssignVariableOp_91*
dtype0*(
_output_shapes
:??
Y
Placeholder_92Placeholder*
shape:?*
_output_shapes	
:?*
dtype0
b
AssignVariableOp_92AssignVariableOpbatch_normalization_18/gammaPlaceholder_92*
dtype0
?
ReadVariableOp_92ReadVariableOpbatch_normalization_18/gamma^AssignVariableOp_92*
dtype0*
_output_shapes	
:?
Y
Placeholder_93Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
a
AssignVariableOp_93AssignVariableOpbatch_normalization_18/betaPlaceholder_93*
dtype0
?
ReadVariableOp_93ReadVariableOpbatch_normalization_18/beta^AssignVariableOp_93*
_output_shapes	
:?*
dtype0
Y
Placeholder_94Placeholder*
_output_shapes	
:?*
dtype0*
shape:?
h
AssignVariableOp_94AssignVariableOp"batch_normalization_18/moving_meanPlaceholder_94*
dtype0
?
ReadVariableOp_94ReadVariableOp"batch_normalization_18/moving_mean^AssignVariableOp_94*
_output_shapes	
:?*
dtype0
Y
Placeholder_95Placeholder*
_output_shapes	
:?*
shape:?*
dtype0
l
AssignVariableOp_95AssignVariableOp&batch_normalization_18/moving_variancePlaceholder_95*
dtype0
?
ReadVariableOp_95ReadVariableOp&batch_normalization_18/moving_variance^AssignVariableOp_95*
dtype0*
_output_shapes	
:?
a
Placeholder_96Placeholder*
dtype0*
shape:	?
*
_output_shapes
:	?

R
AssignVariableOp_96AssignVariableOpdense/kernelPlaceholder_96*
dtype0
u
ReadVariableOp_96ReadVariableOpdense/kernel^AssignVariableOp_96*
dtype0*
_output_shapes
:	?

W
Placeholder_97Placeholder*
shape:
*
dtype0*
_output_shapes
:

P
AssignVariableOp_97AssignVariableOp
dense/biasPlaceholder_97*
dtype0
n
ReadVariableOp_97ReadVariableOp
dense/bias^AssignVariableOp_97*
dtype0*
_output_shapes
:


init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
Y
save/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_01d9c67c70fb42e186efbc85f375d39c/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
_output_shapes
: *
dtype0
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*?
value?B?bBbatch_normalization/betaBbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_10/betaBbatch_normalization_10/gammaB"batch_normalization_10/moving_meanB&batch_normalization_10/moving_varianceBbatch_normalization_11/betaBbatch_normalization_11/gammaB"batch_normalization_11/moving_meanB&batch_normalization_11/moving_varianceBbatch_normalization_12/betaBbatch_normalization_12/gammaB"batch_normalization_12/moving_meanB&batch_normalization_12/moving_varianceBbatch_normalization_13/betaBbatch_normalization_13/gammaB"batch_normalization_13/moving_meanB&batch_normalization_13/moving_varianceBbatch_normalization_14/betaBbatch_normalization_14/gammaB"batch_normalization_14/moving_meanB&batch_normalization_14/moving_varianceBbatch_normalization_15/betaBbatch_normalization_15/gammaB"batch_normalization_15/moving_meanB&batch_normalization_15/moving_varianceBbatch_normalization_16/betaBbatch_normalization_16/gammaB"batch_normalization_16/moving_meanB&batch_normalization_16/moving_varianceBbatch_normalization_17/betaBbatch_normalization_17/gammaB"batch_normalization_17/moving_meanB&batch_normalization_17/moving_varianceBbatch_normalization_18/betaBbatch_normalization_18/gammaB"batch_normalization_18/moving_meanB&batch_normalization_18/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbatch_normalization_3/betaBbatch_normalization_3/gammaB!batch_normalization_3/moving_meanB%batch_normalization_3/moving_varianceBbatch_normalization_4/betaBbatch_normalization_4/gammaB!batch_normalization_4/moving_meanB%batch_normalization_4/moving_varianceBbatch_normalization_5/betaBbatch_normalization_5/gammaB!batch_normalization_5/moving_meanB%batch_normalization_5/moving_varianceBbatch_normalization_6/betaBbatch_normalization_6/gammaB!batch_normalization_6/moving_meanB%batch_normalization_6/moving_varianceBbatch_normalization_7/betaBbatch_normalization_7/gammaB!batch_normalization_7/moving_meanB%batch_normalization_7/moving_varianceBbatch_normalization_8/betaBbatch_normalization_8/gammaB!batch_normalization_8/moving_meanB%batch_normalization_8/moving_varianceBbatch_normalization_9/betaBbatch_normalization_9/gammaB!batch_normalization_9/moving_meanB%batch_normalization_9/moving_varianceBconv2d/kernelBconv2d_1/kernelBconv2d_10/kernelBconv2d_11/kernelBconv2d_12/kernelBconv2d_13/kernelBconv2d_14/kernelBconv2d_15/kernelBconv2d_16/kernelBconv2d_17/kernelBconv2d_18/kernelBconv2d_19/kernelBconv2d_2/kernelBconv2d_20/kernelBconv2d_3/kernelBconv2d_4/kernelBconv2d_5/kernelBconv2d_6/kernelBconv2d_7/kernelBconv2d_8/kernelBconv2d_9/kernelB
dense/biasBdense/kernel*
dtype0
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:b
?'
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp/batch_normalization_10/beta/Read/ReadVariableOp0batch_normalization_10/gamma/Read/ReadVariableOp6batch_normalization_10/moving_mean/Read/ReadVariableOp:batch_normalization_10/moving_variance/Read/ReadVariableOp/batch_normalization_11/beta/Read/ReadVariableOp0batch_normalization_11/gamma/Read/ReadVariableOp6batch_normalization_11/moving_mean/Read/ReadVariableOp:batch_normalization_11/moving_variance/Read/ReadVariableOp/batch_normalization_12/beta/Read/ReadVariableOp0batch_normalization_12/gamma/Read/ReadVariableOp6batch_normalization_12/moving_mean/Read/ReadVariableOp:batch_normalization_12/moving_variance/Read/ReadVariableOp/batch_normalization_13/beta/Read/ReadVariableOp0batch_normalization_13/gamma/Read/ReadVariableOp6batch_normalization_13/moving_mean/Read/ReadVariableOp:batch_normalization_13/moving_variance/Read/ReadVariableOp/batch_normalization_14/beta/Read/ReadVariableOp0batch_normalization_14/gamma/Read/ReadVariableOp6batch_normalization_14/moving_mean/Read/ReadVariableOp:batch_normalization_14/moving_variance/Read/ReadVariableOp/batch_normalization_15/beta/Read/ReadVariableOp0batch_normalization_15/gamma/Read/ReadVariableOp6batch_normalization_15/moving_mean/Read/ReadVariableOp:batch_normalization_15/moving_variance/Read/ReadVariableOp/batch_normalization_16/beta/Read/ReadVariableOp0batch_normalization_16/gamma/Read/ReadVariableOp6batch_normalization_16/moving_mean/Read/ReadVariableOp:batch_normalization_16/moving_variance/Read/ReadVariableOp/batch_normalization_17/beta/Read/ReadVariableOp0batch_normalization_17/gamma/Read/ReadVariableOp6batch_normalization_17/moving_mean/Read/ReadVariableOp:batch_normalization_17/moving_variance/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp.batch_normalization_9/beta/Read/ReadVariableOp/batch_normalization_9/gamma/Read/ReadVariableOp5batch_normalization_9/moving_mean/Read/ReadVariableOp9batch_normalization_9/moving_variance/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp$conv2d_12/kernel/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp$conv2d_14/kernel/Read/ReadVariableOp$conv2d_15/kernel/Read/ReadVariableOp$conv2d_16/kernel/Read/ReadVariableOp$conv2d_17/kernel/Read/ReadVariableOp$conv2d_18/kernel/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp"/device:CPU:0*p
dtypesf
d2b
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:b*?
value?B?bBbatch_normalization/betaBbatch_normalization/moving_meanB#batch_normalization/moving_varianceBbatch_normalization_1/betaBbatch_normalization_1/gammaB!batch_normalization_1/moving_meanB%batch_normalization_1/moving_varianceBbatch_normalization_10/betaBbatch_normalization_10/gammaB"batch_normalization_10/moving_meanB&batch_normalization_10/moving_varianceBbatch_normalization_11/betaBbatch_normalization_11/gammaB"batch_normalization_11/moving_meanB&batch_normalization_11/moving_varianceBbatch_normalization_12/betaBbatch_normalization_12/gammaB"batch_normalization_12/moving_meanB&batch_normalization_12/moving_varianceBbatch_normalization_13/betaBbatch_normalization_13/gammaB"batch_normalization_13/moving_meanB&batch_normalization_13/moving_varianceBbatch_normalization_14/betaBbatch_normalization_14/gammaB"batch_normalization_14/moving_meanB&batch_normalization_14/moving_varianceBbatch_normalization_15/betaBbatch_normalization_15/gammaB"batch_normalization_15/moving_meanB&batch_normalization_15/moving_varianceBbatch_normalization_16/betaBbatch_normalization_16/gammaB"batch_normalization_16/moving_meanB&batch_normalization_16/moving_varianceBbatch_normalization_17/betaBbatch_normalization_17/gammaB"batch_normalization_17/moving_meanB&batch_normalization_17/moving_varianceBbatch_normalization_18/betaBbatch_normalization_18/gammaB"batch_normalization_18/moving_meanB&batch_normalization_18/moving_varianceBbatch_normalization_2/betaBbatch_normalization_2/gammaB!batch_normalization_2/moving_meanB%batch_normalization_2/moving_varianceBbatch_normalization_3/betaBbatch_normalization_3/gammaB!batch_normalization_3/moving_meanB%batch_normalization_3/moving_varianceBbatch_normalization_4/betaBbatch_normalization_4/gammaB!batch_normalization_4/moving_meanB%batch_normalization_4/moving_varianceBbatch_normalization_5/betaBbatch_normalization_5/gammaB!batch_normalization_5/moving_meanB%batch_normalization_5/moving_varianceBbatch_normalization_6/betaBbatch_normalization_6/gammaB!batch_normalization_6/moving_meanB%batch_normalization_6/moving_varianceBbatch_normalization_7/betaBbatch_normalization_7/gammaB!batch_normalization_7/moving_meanB%batch_normalization_7/moving_varianceBbatch_normalization_8/betaBbatch_normalization_8/gammaB!batch_normalization_8/moving_meanB%batch_normalization_8/moving_varianceBbatch_normalization_9/betaBbatch_normalization_9/gammaB!batch_normalization_9/moving_meanB%batch_normalization_9/moving_varianceBconv2d/kernelBconv2d_1/kernelBconv2d_10/kernelBconv2d_11/kernelBconv2d_12/kernelBconv2d_13/kernelBconv2d_14/kernelBconv2d_15/kernelBconv2d_16/kernelBconv2d_17/kernelBconv2d_18/kernelBconv2d_19/kernelBconv2d_2/kernelBconv2d_20/kernelBconv2d_3/kernelBconv2d_4/kernelBconv2d_5/kernelBconv2d_6/kernelBconv2d_7/kernelBconv2d_8/kernelBconv2d_9/kernelB
dense/biasBdense/kernel
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*p
dtypesf
d2b*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
a
save/AssignVariableOpAssignVariableOpbatch_normalization/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
j
save/AssignVariableOp_1AssignVariableOpbatch_normalization/moving_meansave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
_output_shapes
:*
T0
n
save/AssignVariableOp_2AssignVariableOp#batch_normalization/moving_variancesave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
_output_shapes
:*
T0
e
save/AssignVariableOp_3AssignVariableOpbatch_normalization_1/betasave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
f
save/AssignVariableOp_4AssignVariableOpbatch_normalization_1/gammasave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
_output_shapes
:*
T0
l
save/AssignVariableOp_5AssignVariableOp!batch_normalization_1/moving_meansave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
p
save/AssignVariableOp_6AssignVariableOp%batch_normalization_1/moving_variancesave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
_output_shapes
:*
T0
f
save/AssignVariableOp_7AssignVariableOpbatch_normalization_10/betasave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
_output_shapes
:*
T0
g
save/AssignVariableOp_8AssignVariableOpbatch_normalization_10/gammasave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
_output_shapes
:*
T0
n
save/AssignVariableOp_9AssignVariableOp"batch_normalization_10/moving_meansave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
_output_shapes
:*
T0
s
save/AssignVariableOp_10AssignVariableOp&batch_normalization_10/moving_variancesave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
h
save/AssignVariableOp_11AssignVariableOpbatch_normalization_11/betasave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
i
save/AssignVariableOp_12AssignVariableOpbatch_normalization_11/gammasave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
o
save/AssignVariableOp_13AssignVariableOp"batch_normalization_11/moving_meansave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
_output_shapes
:*
T0
s
save/AssignVariableOp_14AssignVariableOp&batch_normalization_11/moving_variancesave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
_output_shapes
:*
T0
h
save/AssignVariableOp_15AssignVariableOpbatch_normalization_12/betasave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
i
save/AssignVariableOp_16AssignVariableOpbatch_normalization_12/gammasave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
o
save/AssignVariableOp_17AssignVariableOp"batch_normalization_12/moving_meansave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
s
save/AssignVariableOp_18AssignVariableOp&batch_normalization_12/moving_variancesave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
_output_shapes
:*
T0
h
save/AssignVariableOp_19AssignVariableOpbatch_normalization_13/betasave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
i
save/AssignVariableOp_20AssignVariableOpbatch_normalization_13/gammasave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:
o
save/AssignVariableOp_21AssignVariableOp"batch_normalization_13/moving_meansave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:
s
save/AssignVariableOp_22AssignVariableOp&batch_normalization_13/moving_variancesave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
h
save/AssignVariableOp_23AssignVariableOpbatch_normalization_14/betasave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:
i
save/AssignVariableOp_24AssignVariableOpbatch_normalization_14/gammasave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
_output_shapes
:*
T0
o
save/AssignVariableOp_25AssignVariableOp"batch_normalization_14/moving_meansave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
_output_shapes
:*
T0
s
save/AssignVariableOp_26AssignVariableOp&batch_normalization_14/moving_variancesave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
_output_shapes
:*
T0
h
save/AssignVariableOp_27AssignVariableOpbatch_normalization_15/betasave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
_output_shapes
:*
T0
i
save/AssignVariableOp_28AssignVariableOpbatch_normalization_15/gammasave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
o
save/AssignVariableOp_29AssignVariableOp"batch_normalization_15/moving_meansave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
_output_shapes
:*
T0
s
save/AssignVariableOp_30AssignVariableOp&batch_normalization_15/moving_variancesave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
h
save/AssignVariableOp_31AssignVariableOpbatch_normalization_16/betasave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
i
save/AssignVariableOp_32AssignVariableOpbatch_normalization_16/gammasave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
o
save/AssignVariableOp_33AssignVariableOp"batch_normalization_16/moving_meansave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
_output_shapes
:*
T0
s
save/AssignVariableOp_34AssignVariableOp&batch_normalization_16/moving_variancesave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
h
save/AssignVariableOp_35AssignVariableOpbatch_normalization_17/betasave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
_output_shapes
:*
T0
i
save/AssignVariableOp_36AssignVariableOpbatch_normalization_17/gammasave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
o
save/AssignVariableOp_37AssignVariableOp"batch_normalization_17/moving_meansave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
_output_shapes
:*
T0
s
save/AssignVariableOp_38AssignVariableOp&batch_normalization_17/moving_variancesave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
h
save/AssignVariableOp_39AssignVariableOpbatch_normalization_18/betasave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
i
save/AssignVariableOp_40AssignVariableOpbatch_normalization_18/gammasave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
_output_shapes
:*
T0
o
save/AssignVariableOp_41AssignVariableOp"batch_normalization_18/moving_meansave/Identity_42*
dtype0
R
save/Identity_43Identitysave/RestoreV2:42*
T0*
_output_shapes
:
s
save/AssignVariableOp_42AssignVariableOp&batch_normalization_18/moving_variancesave/Identity_43*
dtype0
R
save/Identity_44Identitysave/RestoreV2:43*
_output_shapes
:*
T0
g
save/AssignVariableOp_43AssignVariableOpbatch_normalization_2/betasave/Identity_44*
dtype0
R
save/Identity_45Identitysave/RestoreV2:44*
T0*
_output_shapes
:
h
save/AssignVariableOp_44AssignVariableOpbatch_normalization_2/gammasave/Identity_45*
dtype0
R
save/Identity_46Identitysave/RestoreV2:45*
_output_shapes
:*
T0
n
save/AssignVariableOp_45AssignVariableOp!batch_normalization_2/moving_meansave/Identity_46*
dtype0
R
save/Identity_47Identitysave/RestoreV2:46*
_output_shapes
:*
T0
r
save/AssignVariableOp_46AssignVariableOp%batch_normalization_2/moving_variancesave/Identity_47*
dtype0
R
save/Identity_48Identitysave/RestoreV2:47*
_output_shapes
:*
T0
g
save/AssignVariableOp_47AssignVariableOpbatch_normalization_3/betasave/Identity_48*
dtype0
R
save/Identity_49Identitysave/RestoreV2:48*
T0*
_output_shapes
:
h
save/AssignVariableOp_48AssignVariableOpbatch_normalization_3/gammasave/Identity_49*
dtype0
R
save/Identity_50Identitysave/RestoreV2:49*
_output_shapes
:*
T0
n
save/AssignVariableOp_49AssignVariableOp!batch_normalization_3/moving_meansave/Identity_50*
dtype0
R
save/Identity_51Identitysave/RestoreV2:50*
_output_shapes
:*
T0
r
save/AssignVariableOp_50AssignVariableOp%batch_normalization_3/moving_variancesave/Identity_51*
dtype0
R
save/Identity_52Identitysave/RestoreV2:51*
T0*
_output_shapes
:
g
save/AssignVariableOp_51AssignVariableOpbatch_normalization_4/betasave/Identity_52*
dtype0
R
save/Identity_53Identitysave/RestoreV2:52*
_output_shapes
:*
T0
h
save/AssignVariableOp_52AssignVariableOpbatch_normalization_4/gammasave/Identity_53*
dtype0
R
save/Identity_54Identitysave/RestoreV2:53*
_output_shapes
:*
T0
n
save/AssignVariableOp_53AssignVariableOp!batch_normalization_4/moving_meansave/Identity_54*
dtype0
R
save/Identity_55Identitysave/RestoreV2:54*
T0*
_output_shapes
:
r
save/AssignVariableOp_54AssignVariableOp%batch_normalization_4/moving_variancesave/Identity_55*
dtype0
R
save/Identity_56Identitysave/RestoreV2:55*
T0*
_output_shapes
:
g
save/AssignVariableOp_55AssignVariableOpbatch_normalization_5/betasave/Identity_56*
dtype0
R
save/Identity_57Identitysave/RestoreV2:56*
T0*
_output_shapes
:
h
save/AssignVariableOp_56AssignVariableOpbatch_normalization_5/gammasave/Identity_57*
dtype0
R
save/Identity_58Identitysave/RestoreV2:57*
T0*
_output_shapes
:
n
save/AssignVariableOp_57AssignVariableOp!batch_normalization_5/moving_meansave/Identity_58*
dtype0
R
save/Identity_59Identitysave/RestoreV2:58*
_output_shapes
:*
T0
r
save/AssignVariableOp_58AssignVariableOp%batch_normalization_5/moving_variancesave/Identity_59*
dtype0
R
save/Identity_60Identitysave/RestoreV2:59*
T0*
_output_shapes
:
g
save/AssignVariableOp_59AssignVariableOpbatch_normalization_6/betasave/Identity_60*
dtype0
R
save/Identity_61Identitysave/RestoreV2:60*
T0*
_output_shapes
:
h
save/AssignVariableOp_60AssignVariableOpbatch_normalization_6/gammasave/Identity_61*
dtype0
R
save/Identity_62Identitysave/RestoreV2:61*
_output_shapes
:*
T0
n
save/AssignVariableOp_61AssignVariableOp!batch_normalization_6/moving_meansave/Identity_62*
dtype0
R
save/Identity_63Identitysave/RestoreV2:62*
T0*
_output_shapes
:
r
save/AssignVariableOp_62AssignVariableOp%batch_normalization_6/moving_variancesave/Identity_63*
dtype0
R
save/Identity_64Identitysave/RestoreV2:63*
T0*
_output_shapes
:
g
save/AssignVariableOp_63AssignVariableOpbatch_normalization_7/betasave/Identity_64*
dtype0
R
save/Identity_65Identitysave/RestoreV2:64*
T0*
_output_shapes
:
h
save/AssignVariableOp_64AssignVariableOpbatch_normalization_7/gammasave/Identity_65*
dtype0
R
save/Identity_66Identitysave/RestoreV2:65*
T0*
_output_shapes
:
n
save/AssignVariableOp_65AssignVariableOp!batch_normalization_7/moving_meansave/Identity_66*
dtype0
R
save/Identity_67Identitysave/RestoreV2:66*
T0*
_output_shapes
:
r
save/AssignVariableOp_66AssignVariableOp%batch_normalization_7/moving_variancesave/Identity_67*
dtype0
R
save/Identity_68Identitysave/RestoreV2:67*
T0*
_output_shapes
:
g
save/AssignVariableOp_67AssignVariableOpbatch_normalization_8/betasave/Identity_68*
dtype0
R
save/Identity_69Identitysave/RestoreV2:68*
_output_shapes
:*
T0
h
save/AssignVariableOp_68AssignVariableOpbatch_normalization_8/gammasave/Identity_69*
dtype0
R
save/Identity_70Identitysave/RestoreV2:69*
T0*
_output_shapes
:
n
save/AssignVariableOp_69AssignVariableOp!batch_normalization_8/moving_meansave/Identity_70*
dtype0
R
save/Identity_71Identitysave/RestoreV2:70*
T0*
_output_shapes
:
r
save/AssignVariableOp_70AssignVariableOp%batch_normalization_8/moving_variancesave/Identity_71*
dtype0
R
save/Identity_72Identitysave/RestoreV2:71*
_output_shapes
:*
T0
g
save/AssignVariableOp_71AssignVariableOpbatch_normalization_9/betasave/Identity_72*
dtype0
R
save/Identity_73Identitysave/RestoreV2:72*
_output_shapes
:*
T0
h
save/AssignVariableOp_72AssignVariableOpbatch_normalization_9/gammasave/Identity_73*
dtype0
R
save/Identity_74Identitysave/RestoreV2:73*
T0*
_output_shapes
:
n
save/AssignVariableOp_73AssignVariableOp!batch_normalization_9/moving_meansave/Identity_74*
dtype0
R
save/Identity_75Identitysave/RestoreV2:74*
_output_shapes
:*
T0
r
save/AssignVariableOp_74AssignVariableOp%batch_normalization_9/moving_variancesave/Identity_75*
dtype0
R
save/Identity_76Identitysave/RestoreV2:75*
_output_shapes
:*
T0
Z
save/AssignVariableOp_75AssignVariableOpconv2d/kernelsave/Identity_76*
dtype0
R
save/Identity_77Identitysave/RestoreV2:76*
T0*
_output_shapes
:
\
save/AssignVariableOp_76AssignVariableOpconv2d_1/kernelsave/Identity_77*
dtype0
R
save/Identity_78Identitysave/RestoreV2:77*
T0*
_output_shapes
:
]
save/AssignVariableOp_77AssignVariableOpconv2d_10/kernelsave/Identity_78*
dtype0
R
save/Identity_79Identitysave/RestoreV2:78*
_output_shapes
:*
T0
]
save/AssignVariableOp_78AssignVariableOpconv2d_11/kernelsave/Identity_79*
dtype0
R
save/Identity_80Identitysave/RestoreV2:79*
_output_shapes
:*
T0
]
save/AssignVariableOp_79AssignVariableOpconv2d_12/kernelsave/Identity_80*
dtype0
R
save/Identity_81Identitysave/RestoreV2:80*
_output_shapes
:*
T0
]
save/AssignVariableOp_80AssignVariableOpconv2d_13/kernelsave/Identity_81*
dtype0
R
save/Identity_82Identitysave/RestoreV2:81*
_output_shapes
:*
T0
]
save/AssignVariableOp_81AssignVariableOpconv2d_14/kernelsave/Identity_82*
dtype0
R
save/Identity_83Identitysave/RestoreV2:82*
_output_shapes
:*
T0
]
save/AssignVariableOp_82AssignVariableOpconv2d_15/kernelsave/Identity_83*
dtype0
R
save/Identity_84Identitysave/RestoreV2:83*
_output_shapes
:*
T0
]
save/AssignVariableOp_83AssignVariableOpconv2d_16/kernelsave/Identity_84*
dtype0
R
save/Identity_85Identitysave/RestoreV2:84*
_output_shapes
:*
T0
]
save/AssignVariableOp_84AssignVariableOpconv2d_17/kernelsave/Identity_85*
dtype0
R
save/Identity_86Identitysave/RestoreV2:85*
T0*
_output_shapes
:
]
save/AssignVariableOp_85AssignVariableOpconv2d_18/kernelsave/Identity_86*
dtype0
R
save/Identity_87Identitysave/RestoreV2:86*
T0*
_output_shapes
:
]
save/AssignVariableOp_86AssignVariableOpconv2d_19/kernelsave/Identity_87*
dtype0
R
save/Identity_88Identitysave/RestoreV2:87*
T0*
_output_shapes
:
\
save/AssignVariableOp_87AssignVariableOpconv2d_2/kernelsave/Identity_88*
dtype0
R
save/Identity_89Identitysave/RestoreV2:88*
T0*
_output_shapes
:
]
save/AssignVariableOp_88AssignVariableOpconv2d_20/kernelsave/Identity_89*
dtype0
R
save/Identity_90Identitysave/RestoreV2:89*
T0*
_output_shapes
:
\
save/AssignVariableOp_89AssignVariableOpconv2d_3/kernelsave/Identity_90*
dtype0
R
save/Identity_91Identitysave/RestoreV2:90*
_output_shapes
:*
T0
\
save/AssignVariableOp_90AssignVariableOpconv2d_4/kernelsave/Identity_91*
dtype0
R
save/Identity_92Identitysave/RestoreV2:91*
T0*
_output_shapes
:
\
save/AssignVariableOp_91AssignVariableOpconv2d_5/kernelsave/Identity_92*
dtype0
R
save/Identity_93Identitysave/RestoreV2:92*
T0*
_output_shapes
:
\
save/AssignVariableOp_92AssignVariableOpconv2d_6/kernelsave/Identity_93*
dtype0
R
save/Identity_94Identitysave/RestoreV2:93*
T0*
_output_shapes
:
\
save/AssignVariableOp_93AssignVariableOpconv2d_7/kernelsave/Identity_94*
dtype0
R
save/Identity_95Identitysave/RestoreV2:94*
T0*
_output_shapes
:
\
save/AssignVariableOp_94AssignVariableOpconv2d_8/kernelsave/Identity_95*
dtype0
R
save/Identity_96Identitysave/RestoreV2:95*
T0*
_output_shapes
:
\
save/AssignVariableOp_95AssignVariableOpconv2d_9/kernelsave/Identity_96*
dtype0
R
save/Identity_97Identitysave/RestoreV2:96*
T0*
_output_shapes
:
W
save/AssignVariableOp_96AssignVariableOp
dense/biassave/Identity_97*
dtype0
R
save/Identity_98Identitysave/RestoreV2:97*
T0*
_output_shapes
:
Y
save/AssignVariableOp_97AssignVariableOpdense/kernelsave/Identity_98*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_42^save/AssignVariableOp_43^save/AssignVariableOp_44^save/AssignVariableOp_45^save/AssignVariableOp_46^save/AssignVariableOp_47^save/AssignVariableOp_48^save/AssignVariableOp_49^save/AssignVariableOp_5^save/AssignVariableOp_50^save/AssignVariableOp_51^save/AssignVariableOp_52^save/AssignVariableOp_53^save/AssignVariableOp_54^save/AssignVariableOp_55^save/AssignVariableOp_56^save/AssignVariableOp_57^save/AssignVariableOp_58^save/AssignVariableOp_59^save/AssignVariableOp_6^save/AssignVariableOp_60^save/AssignVariableOp_61^save/AssignVariableOp_62^save/AssignVariableOp_63^save/AssignVariableOp_64^save/AssignVariableOp_65^save/AssignVariableOp_66^save/AssignVariableOp_67^save/AssignVariableOp_68^save/AssignVariableOp_69^save/AssignVariableOp_7^save/AssignVariableOp_70^save/AssignVariableOp_71^save/AssignVariableOp_72^save/AssignVariableOp_73^save/AssignVariableOp_74^save/AssignVariableOp_75^save/AssignVariableOp_76^save/AssignVariableOp_77^save/AssignVariableOp_78^save/AssignVariableOp_79^save/AssignVariableOp_8^save/AssignVariableOp_80^save/AssignVariableOp_81^save/AssignVariableOp_82^save/AssignVariableOp_83^save/AssignVariableOp_84^save/AssignVariableOp_85^save/AssignVariableOp_86^save/AssignVariableOp_87^save/AssignVariableOp_88^save/AssignVariableOp_89^save/AssignVariableOp_9^save/AssignVariableOp_90^save/AssignVariableOp_91^save/AssignVariableOp_92^save/AssignVariableOp_93^save/AssignVariableOp_94^save/AssignVariableOp_95^save/AssignVariableOp_96^save/AssignVariableOp_97
-
save/restore_allNoOp^save/restore_shard"&<
save/Const:0save/Identity:0save/restore_all (5 @F8"??
cond_context????
?
"batch_normalization/cond/cond_text"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_t:0 *?
batch_normalization/Const:0
batch_normalization/beta:0
 batch_normalization/cond/Const:0
"batch_normalization/cond/Const_1:0
0batch_normalization/cond/FusedBatchNorm/Switch:1
2batch_normalization/cond/FusedBatchNorm/Switch_1:1
)batch_normalization/cond/FusedBatchNorm:0
)batch_normalization/cond/FusedBatchNorm:1
)batch_normalization/cond/FusedBatchNorm:2
)batch_normalization/cond/FusedBatchNorm:3
)batch_normalization/cond/FusedBatchNorm:4
0batch_normalization/cond/ReadVariableOp/Switch:1
)batch_normalization/cond/ReadVariableOp:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_t:0
data:0H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0:
data:00batch_normalization/cond/FusedBatchNorm/Switch:1N
batch_normalization/beta:00batch_normalization/cond/ReadVariableOp/Switch:1Q
batch_normalization/Const:02batch_normalization/cond/FusedBatchNorm/Switch_1:1
?
$batch_normalization/cond/cond_text_1"batch_normalization/cond/pred_id:0#batch_normalization/cond/switch_f:0*?
batch_normalization/Const:0
batch_normalization/beta:0
Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
:batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp:0
Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
<batch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1:0
2batch_normalization/cond/FusedBatchNorm_1/Switch:0
4batch_normalization/cond/FusedBatchNorm_1/Switch_1:0
+batch_normalization/cond/FusedBatchNorm_1:0
+batch_normalization/cond/FusedBatchNorm_1:1
+batch_normalization/cond/FusedBatchNorm_1:2
+batch_normalization/cond/FusedBatchNorm_1:3
+batch_normalization/cond/FusedBatchNorm_1:4
2batch_normalization/cond/ReadVariableOp_1/Switch:0
+batch_normalization/cond/ReadVariableOp_1:0
"batch_normalization/cond/pred_id:0
#batch_normalization/cond/switch_f:0
!batch_normalization/moving_mean:0
%batch_normalization/moving_variance:0
data:0<
data:02batch_normalization/cond/FusedBatchNorm_1/Switch:0f
!batch_normalization/moving_mean:0Abatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0S
batch_normalization/Const:04batch_normalization/cond/FusedBatchNorm_1/Switch_1:0P
batch_normalization/beta:02batch_normalization/cond/ReadVariableOp_1/Switch:0H
"batch_normalization/cond/pred_id:0"batch_normalization/cond/pred_id:0l
%batch_normalization/moving_variance:0Cbatch_normalization/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?
$batch_normalization/cond_1/cond_text$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_t:0 *?
"batch_normalization/cond_1/Const:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_t:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
?
&batch_normalization/cond_1/cond_text_1$batch_normalization/cond_1/pred_id:0%batch_normalization/cond_1/switch_f:0*?
$batch_normalization/cond_1/Const_1:0
$batch_normalization/cond_1/pred_id:0
%batch_normalization/cond_1/switch_f:0L
$batch_normalization/cond_1/pred_id:0$batch_normalization/cond_1/pred_id:0
?
$batch_normalization_1/cond/cond_text$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_t:0 *?
batch_normalization_1/beta:0
"batch_normalization_1/cond/Const:0
$batch_normalization_1/cond/Const_1:0
2batch_normalization_1/cond/FusedBatchNorm/Switch:1
+batch_normalization_1/cond/FusedBatchNorm:0
+batch_normalization_1/cond/FusedBatchNorm:1
+batch_normalization_1/cond/FusedBatchNorm:2
+batch_normalization_1/cond/FusedBatchNorm:3
+batch_normalization_1/cond/FusedBatchNorm:4
2batch_normalization_1/cond/ReadVariableOp/Switch:1
+batch_normalization_1/cond/ReadVariableOp:0
4batch_normalization_1/cond/ReadVariableOp_1/Switch:1
-batch_normalization_1/cond/ReadVariableOp_1:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_t:0
batch_normalization_1/gamma:0
conv2d/Conv2D:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0S
batch_normalization_1/gamma:02batch_normalization_1/cond/ReadVariableOp/Switch:1E
conv2d/Conv2D:02batch_normalization_1/cond/FusedBatchNorm/Switch:1
?
&batch_normalization_1/cond/cond_text_1$batch_normalization_1/cond/pred_id:0%batch_normalization_1/cond/switch_f:0*?
batch_normalization_1/beta:0
Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_1/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_1/cond/FusedBatchNorm_1:0
-batch_normalization_1/cond/FusedBatchNorm_1:1
-batch_normalization_1/cond/FusedBatchNorm_1:2
-batch_normalization_1/cond/FusedBatchNorm_1:3
-batch_normalization_1/cond/FusedBatchNorm_1:4
4batch_normalization_1/cond/ReadVariableOp_2/Switch:0
-batch_normalization_1/cond/ReadVariableOp_2:0
4batch_normalization_1/cond/ReadVariableOp_3/Switch:0
-batch_normalization_1/cond/ReadVariableOp_3:0
$batch_normalization_1/cond/pred_id:0
%batch_normalization_1/cond/switch_f:0
batch_normalization_1/gamma:0
#batch_normalization_1/moving_mean:0
'batch_normalization_1/moving_variance:0
conv2d/Conv2D:0p
'batch_normalization_1/moving_variance:0Ebatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0U
batch_normalization_1/gamma:04batch_normalization_1/cond/ReadVariableOp_2/Switch:0L
$batch_normalization_1/cond/pred_id:0$batch_normalization_1/cond/pred_id:0G
conv2d/Conv2D:04batch_normalization_1/cond/FusedBatchNorm_1/Switch:0T
batch_normalization_1/beta:04batch_normalization_1/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_1/moving_mean:0Cbatch_normalization_1/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
?
&batch_normalization_1/cond_1/cond_text&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_t:0 *?
$batch_normalization_1/cond_1/Const:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_t:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
?
(batch_normalization_1/cond_1/cond_text_1&batch_normalization_1/cond_1/pred_id:0'batch_normalization_1/cond_1/switch_f:0*?
&batch_normalization_1/cond_1/Const_1:0
&batch_normalization_1/cond_1/pred_id:0
'batch_normalization_1/cond_1/switch_f:0P
&batch_normalization_1/cond_1/pred_id:0&batch_normalization_1/cond_1/pred_id:0
?	
$batch_normalization_2/cond/cond_text$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_t:0 *?
batch_normalization_2/beta:0
"batch_normalization_2/cond/Const:0
$batch_normalization_2/cond/Const_1:0
2batch_normalization_2/cond/FusedBatchNorm/Switch:1
+batch_normalization_2/cond/FusedBatchNorm:0
+batch_normalization_2/cond/FusedBatchNorm:1
+batch_normalization_2/cond/FusedBatchNorm:2
+batch_normalization_2/cond/FusedBatchNorm:3
+batch_normalization_2/cond/FusedBatchNorm:4
2batch_normalization_2/cond/ReadVariableOp/Switch:1
+batch_normalization_2/cond/ReadVariableOp:0
4batch_normalization_2/cond/ReadVariableOp_1/Switch:1
-batch_normalization_2/cond/ReadVariableOp_1:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_t:0
batch_normalization_2/gamma:0
max_pooling2d/MaxPool:0T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0M
max_pooling2d/MaxPool:02batch_normalization_2/cond/FusedBatchNorm/Switch:1S
batch_normalization_2/gamma:02batch_normalization_2/cond/ReadVariableOp/Switch:1
?
&batch_normalization_2/cond/cond_text_1$batch_normalization_2/cond/pred_id:0%batch_normalization_2/cond/switch_f:0*?
batch_normalization_2/beta:0
Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_2/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_2/cond/FusedBatchNorm_1:0
-batch_normalization_2/cond/FusedBatchNorm_1:1
-batch_normalization_2/cond/FusedBatchNorm_1:2
-batch_normalization_2/cond/FusedBatchNorm_1:3
-batch_normalization_2/cond/FusedBatchNorm_1:4
4batch_normalization_2/cond/ReadVariableOp_2/Switch:0
-batch_normalization_2/cond/ReadVariableOp_2:0
4batch_normalization_2/cond/ReadVariableOp_3/Switch:0
-batch_normalization_2/cond/ReadVariableOp_3:0
$batch_normalization_2/cond/pred_id:0
%batch_normalization_2/cond/switch_f:0
batch_normalization_2/gamma:0
#batch_normalization_2/moving_mean:0
'batch_normalization_2/moving_variance:0
max_pooling2d/MaxPool:0p
'batch_normalization_2/moving_variance:0Ebatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0O
max_pooling2d/MaxPool:04batch_normalization_2/cond/FusedBatchNorm_1/Switch:0U
batch_normalization_2/gamma:04batch_normalization_2/cond/ReadVariableOp_2/Switch:0L
$batch_normalization_2/cond/pred_id:0$batch_normalization_2/cond/pred_id:0T
batch_normalization_2/beta:04batch_normalization_2/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_2/moving_mean:0Cbatch_normalization_2/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
?
&batch_normalization_2/cond_1/cond_text&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_t:0 *?
$batch_normalization_2/cond_1/Const:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_t:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
?
(batch_normalization_2/cond_1/cond_text_1&batch_normalization_2/cond_1/pred_id:0'batch_normalization_2/cond_1/switch_f:0*?
&batch_normalization_2/cond_1/Const_1:0
&batch_normalization_2/cond_1/pred_id:0
'batch_normalization_2/cond_1/switch_f:0P
&batch_normalization_2/cond_1/pred_id:0&batch_normalization_2/cond_1/pred_id:0
?
$batch_normalization_3/cond/cond_text$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_t:0 *?
batch_normalization_3/beta:0
"batch_normalization_3/cond/Const:0
$batch_normalization_3/cond/Const_1:0
2batch_normalization_3/cond/FusedBatchNorm/Switch:1
+batch_normalization_3/cond/FusedBatchNorm:0
+batch_normalization_3/cond/FusedBatchNorm:1
+batch_normalization_3/cond/FusedBatchNorm:2
+batch_normalization_3/cond/FusedBatchNorm:3
+batch_normalization_3/cond/FusedBatchNorm:4
2batch_normalization_3/cond/ReadVariableOp/Switch:1
+batch_normalization_3/cond/ReadVariableOp:0
4batch_normalization_3/cond/ReadVariableOp_1/Switch:1
-batch_normalization_3/cond/ReadVariableOp_1:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_t:0
batch_normalization_3/gamma:0
conv2d_2/Conv2D:0S
batch_normalization_3/gamma:02batch_normalization_3/cond/ReadVariableOp/Switch:1T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0G
conv2d_2/Conv2D:02batch_normalization_3/cond/FusedBatchNorm/Switch:1
?
&batch_normalization_3/cond/cond_text_1$batch_normalization_3/cond/pred_id:0%batch_normalization_3/cond/switch_f:0*?
batch_normalization_3/beta:0
Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_3/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_3/cond/FusedBatchNorm_1:0
-batch_normalization_3/cond/FusedBatchNorm_1:1
-batch_normalization_3/cond/FusedBatchNorm_1:2
-batch_normalization_3/cond/FusedBatchNorm_1:3
-batch_normalization_3/cond/FusedBatchNorm_1:4
4batch_normalization_3/cond/ReadVariableOp_2/Switch:0
-batch_normalization_3/cond/ReadVariableOp_2:0
4batch_normalization_3/cond/ReadVariableOp_3/Switch:0
-batch_normalization_3/cond/ReadVariableOp_3:0
$batch_normalization_3/cond/pred_id:0
%batch_normalization_3/cond/switch_f:0
batch_normalization_3/gamma:0
#batch_normalization_3/moving_mean:0
'batch_normalization_3/moving_variance:0
conv2d_2/Conv2D:0I
conv2d_2/Conv2D:04batch_normalization_3/cond/FusedBatchNorm_1/Switch:0U
batch_normalization_3/gamma:04batch_normalization_3/cond/ReadVariableOp_2/Switch:0L
$batch_normalization_3/cond/pred_id:0$batch_normalization_3/cond/pred_id:0T
batch_normalization_3/beta:04batch_normalization_3/cond/ReadVariableOp_3/Switch:0j
#batch_normalization_3/moving_mean:0Cbatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_3/moving_variance:0Ebatch_normalization_3/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?
&batch_normalization_3/cond_1/cond_text&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_t:0 *?
$batch_normalization_3/cond_1/Const:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_t:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
?
(batch_normalization_3/cond_1/cond_text_1&batch_normalization_3/cond_1/pred_id:0'batch_normalization_3/cond_1/switch_f:0*?
&batch_normalization_3/cond_1/Const_1:0
&batch_normalization_3/cond_1/pred_id:0
'batch_normalization_3/cond_1/switch_f:0P
&batch_normalization_3/cond_1/pred_id:0&batch_normalization_3/cond_1/pred_id:0
?
$batch_normalization_4/cond/cond_text$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_t:0 *?
	add/add:0
batch_normalization_4/beta:0
"batch_normalization_4/cond/Const:0
$batch_normalization_4/cond/Const_1:0
2batch_normalization_4/cond/FusedBatchNorm/Switch:1
+batch_normalization_4/cond/FusedBatchNorm:0
+batch_normalization_4/cond/FusedBatchNorm:1
+batch_normalization_4/cond/FusedBatchNorm:2
+batch_normalization_4/cond/FusedBatchNorm:3
+batch_normalization_4/cond/FusedBatchNorm:4
2batch_normalization_4/cond/ReadVariableOp/Switch:1
+batch_normalization_4/cond/ReadVariableOp:0
4batch_normalization_4/cond/ReadVariableOp_1/Switch:1
-batch_normalization_4/cond/ReadVariableOp_1:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_t:0
batch_normalization_4/gamma:0T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_1/Switch:1?
	add/add:02batch_normalization_4/cond/FusedBatchNorm/Switch:1L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0S
batch_normalization_4/gamma:02batch_normalization_4/cond/ReadVariableOp/Switch:1
?
&batch_normalization_4/cond/cond_text_1$batch_normalization_4/cond/pred_id:0%batch_normalization_4/cond/switch_f:0*?
	add/add:0
batch_normalization_4/beta:0
Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_4/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_4/cond/FusedBatchNorm_1:0
-batch_normalization_4/cond/FusedBatchNorm_1:1
-batch_normalization_4/cond/FusedBatchNorm_1:2
-batch_normalization_4/cond/FusedBatchNorm_1:3
-batch_normalization_4/cond/FusedBatchNorm_1:4
4batch_normalization_4/cond/ReadVariableOp_2/Switch:0
-batch_normalization_4/cond/ReadVariableOp_2:0
4batch_normalization_4/cond/ReadVariableOp_3/Switch:0
-batch_normalization_4/cond/ReadVariableOp_3:0
$batch_normalization_4/cond/pred_id:0
%batch_normalization_4/cond/switch_f:0
batch_normalization_4/gamma:0
#batch_normalization_4/moving_mean:0
'batch_normalization_4/moving_variance:0j
#batch_normalization_4/moving_mean:0Cbatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0L
$batch_normalization_4/cond/pred_id:0$batch_normalization_4/cond/pred_id:0p
'batch_normalization_4/moving_variance:0Ebatch_normalization_4/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0A
	add/add:04batch_normalization_4/cond/FusedBatchNorm_1/Switch:0T
batch_normalization_4/beta:04batch_normalization_4/cond/ReadVariableOp_3/Switch:0U
batch_normalization_4/gamma:04batch_normalization_4/cond/ReadVariableOp_2/Switch:0
?
&batch_normalization_4/cond_1/cond_text&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_t:0 *?
$batch_normalization_4/cond_1/Const:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_t:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0
?
(batch_normalization_4/cond_1/cond_text_1&batch_normalization_4/cond_1/pred_id:0'batch_normalization_4/cond_1/switch_f:0*?
&batch_normalization_4/cond_1/Const_1:0
&batch_normalization_4/cond_1/pred_id:0
'batch_normalization_4/cond_1/switch_f:0P
&batch_normalization_4/cond_1/pred_id:0&batch_normalization_4/cond_1/pred_id:0
?
$batch_normalization_5/cond/cond_text$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_t:0 *?
batch_normalization_5/beta:0
"batch_normalization_5/cond/Const:0
$batch_normalization_5/cond/Const_1:0
2batch_normalization_5/cond/FusedBatchNorm/Switch:1
+batch_normalization_5/cond/FusedBatchNorm:0
+batch_normalization_5/cond/FusedBatchNorm:1
+batch_normalization_5/cond/FusedBatchNorm:2
+batch_normalization_5/cond/FusedBatchNorm:3
+batch_normalization_5/cond/FusedBatchNorm:4
2batch_normalization_5/cond/ReadVariableOp/Switch:1
+batch_normalization_5/cond/ReadVariableOp:0
4batch_normalization_5/cond/ReadVariableOp_1/Switch:1
-batch_normalization_5/cond/ReadVariableOp_1:0
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_t:0
batch_normalization_5/gamma:0
conv2d_4/Conv2D:0G
conv2d_4/Conv2D:02batch_normalization_5/cond/FusedBatchNorm/Switch:1T
batch_normalization_5/beta:04batch_normalization_5/cond/ReadVariableOp_1/Switch:1S
batch_normalization_5/gamma:02batch_normalization_5/cond/ReadVariableOp/Switch:1L
$batch_normalization_5/cond/pred_id:0$batch_normalization_5/cond/pred_id:0
?
&batch_normalization_5/cond/cond_text_1$batch_normalization_5/cond/pred_id:0%batch_normalization_5/cond/switch_f:0*?
batch_normalization_5/beta:0
Cbatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_5/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_5/cond/FusedBatchNorm_1:0
-batch_normalization_5/cond/FusedBatchNorm_1:1
-batch_normalization_5/cond/FusedBatchNorm_1:2
-batch_normalization_5/cond/FusedBatchNorm_1:3
-batch_normalization_5/cond/FusedBatchNorm_1:4
4batch_normalization_5/cond/ReadVariableOp_2/Switch:0
-batch_normalization_5/cond/ReadVariableOp_2:0
4batch_normalization_5/cond/ReadVariableOp_3/Switch:0
-batch_normalization_5/cond/ReadVariableOp_3:0
$batch_normalization_5/cond/pred_id:0
%batch_normalization_5/cond/switch_f:0
batch_normalization_5/gamma:0
#batch_normalization_5/moving_mean:0
'batch_normalization_5/moving_variance:0
conv2d_4/Conv2D:0j
#batch_normalization_5/moving_mean:0Cbatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_5/moving_variance:0Ebatch_normalization_5/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0T
batch_normalization_5/beta:04batch_normalization_5/cond/ReadVariableOp_3/Switch:0U
batch_normalization_5/gamma:04batch_normalization_5/cond/ReadVariableOp_2/Switch:0L
$batch_normalization_5/cond/pred_id:0$batch_normalization_5/cond/pred_id:0I
conv2d_4/Conv2D:04batch_normalization_5/cond/FusedBatchNorm_1/Switch:0
?
&batch_normalization_5/cond_1/cond_text&batch_normalization_5/cond_1/pred_id:0'batch_normalization_5/cond_1/switch_t:0 *?
$batch_normalization_5/cond_1/Const:0
&batch_normalization_5/cond_1/pred_id:0
'batch_normalization_5/cond_1/switch_t:0P
&batch_normalization_5/cond_1/pred_id:0&batch_normalization_5/cond_1/pred_id:0
?
(batch_normalization_5/cond_1/cond_text_1&batch_normalization_5/cond_1/pred_id:0'batch_normalization_5/cond_1/switch_f:0*?
&batch_normalization_5/cond_1/Const_1:0
&batch_normalization_5/cond_1/pred_id:0
'batch_normalization_5/cond_1/switch_f:0P
&batch_normalization_5/cond_1/pred_id:0&batch_normalization_5/cond_1/pred_id:0
?
$batch_normalization_6/cond/cond_text$batch_normalization_6/cond/pred_id:0%batch_normalization_6/cond/switch_t:0 *?
add_1/add:0
batch_normalization_6/beta:0
"batch_normalization_6/cond/Const:0
$batch_normalization_6/cond/Const_1:0
2batch_normalization_6/cond/FusedBatchNorm/Switch:1
+batch_normalization_6/cond/FusedBatchNorm:0
+batch_normalization_6/cond/FusedBatchNorm:1
+batch_normalization_6/cond/FusedBatchNorm:2
+batch_normalization_6/cond/FusedBatchNorm:3
+batch_normalization_6/cond/FusedBatchNorm:4
2batch_normalization_6/cond/ReadVariableOp/Switch:1
+batch_normalization_6/cond/ReadVariableOp:0
4batch_normalization_6/cond/ReadVariableOp_1/Switch:1
-batch_normalization_6/cond/ReadVariableOp_1:0
$batch_normalization_6/cond/pred_id:0
%batch_normalization_6/cond/switch_t:0
batch_normalization_6/gamma:0A
add_1/add:02batch_normalization_6/cond/FusedBatchNorm/Switch:1T
batch_normalization_6/beta:04batch_normalization_6/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_6/cond/pred_id:0$batch_normalization_6/cond/pred_id:0S
batch_normalization_6/gamma:02batch_normalization_6/cond/ReadVariableOp/Switch:1
?
&batch_normalization_6/cond/cond_text_1$batch_normalization_6/cond/pred_id:0%batch_normalization_6/cond/switch_f:0*?
add_1/add:0
batch_normalization_6/beta:0
Cbatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_6/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_6/cond/FusedBatchNorm_1:0
-batch_normalization_6/cond/FusedBatchNorm_1:1
-batch_normalization_6/cond/FusedBatchNorm_1:2
-batch_normalization_6/cond/FusedBatchNorm_1:3
-batch_normalization_6/cond/FusedBatchNorm_1:4
4batch_normalization_6/cond/ReadVariableOp_2/Switch:0
-batch_normalization_6/cond/ReadVariableOp_2:0
4batch_normalization_6/cond/ReadVariableOp_3/Switch:0
-batch_normalization_6/cond/ReadVariableOp_3:0
$batch_normalization_6/cond/pred_id:0
%batch_normalization_6/cond/switch_f:0
batch_normalization_6/gamma:0
#batch_normalization_6/moving_mean:0
'batch_normalization_6/moving_variance:0U
batch_normalization_6/gamma:04batch_normalization_6/cond/ReadVariableOp_2/Switch:0C
add_1/add:04batch_normalization_6/cond/FusedBatchNorm_1/Switch:0j
#batch_normalization_6/moving_mean:0Cbatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0L
$batch_normalization_6/cond/pred_id:0$batch_normalization_6/cond/pred_id:0T
batch_normalization_6/beta:04batch_normalization_6/cond/ReadVariableOp_3/Switch:0p
'batch_normalization_6/moving_variance:0Ebatch_normalization_6/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?
&batch_normalization_6/cond_1/cond_text&batch_normalization_6/cond_1/pred_id:0'batch_normalization_6/cond_1/switch_t:0 *?
$batch_normalization_6/cond_1/Const:0
&batch_normalization_6/cond_1/pred_id:0
'batch_normalization_6/cond_1/switch_t:0P
&batch_normalization_6/cond_1/pred_id:0&batch_normalization_6/cond_1/pred_id:0
?
(batch_normalization_6/cond_1/cond_text_1&batch_normalization_6/cond_1/pred_id:0'batch_normalization_6/cond_1/switch_f:0*?
&batch_normalization_6/cond_1/Const_1:0
&batch_normalization_6/cond_1/pred_id:0
'batch_normalization_6/cond_1/switch_f:0P
&batch_normalization_6/cond_1/pred_id:0&batch_normalization_6/cond_1/pred_id:0
?
$batch_normalization_7/cond/cond_text$batch_normalization_7/cond/pred_id:0%batch_normalization_7/cond/switch_t:0 *?
batch_normalization_7/beta:0
"batch_normalization_7/cond/Const:0
$batch_normalization_7/cond/Const_1:0
2batch_normalization_7/cond/FusedBatchNorm/Switch:1
+batch_normalization_7/cond/FusedBatchNorm:0
+batch_normalization_7/cond/FusedBatchNorm:1
+batch_normalization_7/cond/FusedBatchNorm:2
+batch_normalization_7/cond/FusedBatchNorm:3
+batch_normalization_7/cond/FusedBatchNorm:4
2batch_normalization_7/cond/ReadVariableOp/Switch:1
+batch_normalization_7/cond/ReadVariableOp:0
4batch_normalization_7/cond/ReadVariableOp_1/Switch:1
-batch_normalization_7/cond/ReadVariableOp_1:0
$batch_normalization_7/cond/pred_id:0
%batch_normalization_7/cond/switch_t:0
batch_normalization_7/gamma:0
conv2d_7/Conv2D:0G
conv2d_7/Conv2D:02batch_normalization_7/cond/FusedBatchNorm/Switch:1L
$batch_normalization_7/cond/pred_id:0$batch_normalization_7/cond/pred_id:0S
batch_normalization_7/gamma:02batch_normalization_7/cond/ReadVariableOp/Switch:1T
batch_normalization_7/beta:04batch_normalization_7/cond/ReadVariableOp_1/Switch:1
?
&batch_normalization_7/cond/cond_text_1$batch_normalization_7/cond/pred_id:0%batch_normalization_7/cond/switch_f:0*?
batch_normalization_7/beta:0
Cbatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_7/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_7/cond/FusedBatchNorm_1:0
-batch_normalization_7/cond/FusedBatchNorm_1:1
-batch_normalization_7/cond/FusedBatchNorm_1:2
-batch_normalization_7/cond/FusedBatchNorm_1:3
-batch_normalization_7/cond/FusedBatchNorm_1:4
4batch_normalization_7/cond/ReadVariableOp_2/Switch:0
-batch_normalization_7/cond/ReadVariableOp_2:0
4batch_normalization_7/cond/ReadVariableOp_3/Switch:0
-batch_normalization_7/cond/ReadVariableOp_3:0
$batch_normalization_7/cond/pred_id:0
%batch_normalization_7/cond/switch_f:0
batch_normalization_7/gamma:0
#batch_normalization_7/moving_mean:0
'batch_normalization_7/moving_variance:0
conv2d_7/Conv2D:0U
batch_normalization_7/gamma:04batch_normalization_7/cond/ReadVariableOp_2/Switch:0I
conv2d_7/Conv2D:04batch_normalization_7/cond/FusedBatchNorm_1/Switch:0j
#batch_normalization_7/moving_mean:0Cbatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_7/moving_variance:0Ebatch_normalization_7/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0L
$batch_normalization_7/cond/pred_id:0$batch_normalization_7/cond/pred_id:0T
batch_normalization_7/beta:04batch_normalization_7/cond/ReadVariableOp_3/Switch:0
?
&batch_normalization_7/cond_1/cond_text&batch_normalization_7/cond_1/pred_id:0'batch_normalization_7/cond_1/switch_t:0 *?
$batch_normalization_7/cond_1/Const:0
&batch_normalization_7/cond_1/pred_id:0
'batch_normalization_7/cond_1/switch_t:0P
&batch_normalization_7/cond_1/pred_id:0&batch_normalization_7/cond_1/pred_id:0
?
(batch_normalization_7/cond_1/cond_text_1&batch_normalization_7/cond_1/pred_id:0'batch_normalization_7/cond_1/switch_f:0*?
&batch_normalization_7/cond_1/Const_1:0
&batch_normalization_7/cond_1/pred_id:0
'batch_normalization_7/cond_1/switch_f:0P
&batch_normalization_7/cond_1/pred_id:0&batch_normalization_7/cond_1/pred_id:0
?
$batch_normalization_8/cond/cond_text$batch_normalization_8/cond/pred_id:0%batch_normalization_8/cond/switch_t:0 *?
add_2/add:0
batch_normalization_8/beta:0
"batch_normalization_8/cond/Const:0
$batch_normalization_8/cond/Const_1:0
2batch_normalization_8/cond/FusedBatchNorm/Switch:1
+batch_normalization_8/cond/FusedBatchNorm:0
+batch_normalization_8/cond/FusedBatchNorm:1
+batch_normalization_8/cond/FusedBatchNorm:2
+batch_normalization_8/cond/FusedBatchNorm:3
+batch_normalization_8/cond/FusedBatchNorm:4
2batch_normalization_8/cond/ReadVariableOp/Switch:1
+batch_normalization_8/cond/ReadVariableOp:0
4batch_normalization_8/cond/ReadVariableOp_1/Switch:1
-batch_normalization_8/cond/ReadVariableOp_1:0
$batch_normalization_8/cond/pred_id:0
%batch_normalization_8/cond/switch_t:0
batch_normalization_8/gamma:0T
batch_normalization_8/beta:04batch_normalization_8/cond/ReadVariableOp_1/Switch:1L
$batch_normalization_8/cond/pred_id:0$batch_normalization_8/cond/pred_id:0S
batch_normalization_8/gamma:02batch_normalization_8/cond/ReadVariableOp/Switch:1A
add_2/add:02batch_normalization_8/cond/FusedBatchNorm/Switch:1
?
&batch_normalization_8/cond/cond_text_1$batch_normalization_8/cond/pred_id:0%batch_normalization_8/cond/switch_f:0*?
add_2/add:0
batch_normalization_8/beta:0
Cbatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_8/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_8/cond/FusedBatchNorm_1:0
-batch_normalization_8/cond/FusedBatchNorm_1:1
-batch_normalization_8/cond/FusedBatchNorm_1:2
-batch_normalization_8/cond/FusedBatchNorm_1:3
-batch_normalization_8/cond/FusedBatchNorm_1:4
4batch_normalization_8/cond/ReadVariableOp_2/Switch:0
-batch_normalization_8/cond/ReadVariableOp_2:0
4batch_normalization_8/cond/ReadVariableOp_3/Switch:0
-batch_normalization_8/cond/ReadVariableOp_3:0
$batch_normalization_8/cond/pred_id:0
%batch_normalization_8/cond/switch_f:0
batch_normalization_8/gamma:0
#batch_normalization_8/moving_mean:0
'batch_normalization_8/moving_variance:0j
#batch_normalization_8/moving_mean:0Cbatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0p
'batch_normalization_8/moving_variance:0Ebatch_normalization_8/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0L
$batch_normalization_8/cond/pred_id:0$batch_normalization_8/cond/pred_id:0U
batch_normalization_8/gamma:04batch_normalization_8/cond/ReadVariableOp_2/Switch:0C
add_2/add:04batch_normalization_8/cond/FusedBatchNorm_1/Switch:0T
batch_normalization_8/beta:04batch_normalization_8/cond/ReadVariableOp_3/Switch:0
?
&batch_normalization_8/cond_1/cond_text&batch_normalization_8/cond_1/pred_id:0'batch_normalization_8/cond_1/switch_t:0 *?
$batch_normalization_8/cond_1/Const:0
&batch_normalization_8/cond_1/pred_id:0
'batch_normalization_8/cond_1/switch_t:0P
&batch_normalization_8/cond_1/pred_id:0&batch_normalization_8/cond_1/pred_id:0
?
(batch_normalization_8/cond_1/cond_text_1&batch_normalization_8/cond_1/pred_id:0'batch_normalization_8/cond_1/switch_f:0*?
&batch_normalization_8/cond_1/Const_1:0
&batch_normalization_8/cond_1/pred_id:0
'batch_normalization_8/cond_1/switch_f:0P
&batch_normalization_8/cond_1/pred_id:0&batch_normalization_8/cond_1/pred_id:0
?
$batch_normalization_9/cond/cond_text$batch_normalization_9/cond/pred_id:0%batch_normalization_9/cond/switch_t:0 *?
batch_normalization_9/beta:0
"batch_normalization_9/cond/Const:0
$batch_normalization_9/cond/Const_1:0
2batch_normalization_9/cond/FusedBatchNorm/Switch:1
+batch_normalization_9/cond/FusedBatchNorm:0
+batch_normalization_9/cond/FusedBatchNorm:1
+batch_normalization_9/cond/FusedBatchNorm:2
+batch_normalization_9/cond/FusedBatchNorm:3
+batch_normalization_9/cond/FusedBatchNorm:4
2batch_normalization_9/cond/ReadVariableOp/Switch:1
+batch_normalization_9/cond/ReadVariableOp:0
4batch_normalization_9/cond/ReadVariableOp_1/Switch:1
-batch_normalization_9/cond/ReadVariableOp_1:0
$batch_normalization_9/cond/pred_id:0
%batch_normalization_9/cond/switch_t:0
batch_normalization_9/gamma:0
conv2d_9/Conv2D:0S
batch_normalization_9/gamma:02batch_normalization_9/cond/ReadVariableOp/Switch:1T
batch_normalization_9/beta:04batch_normalization_9/cond/ReadVariableOp_1/Switch:1G
conv2d_9/Conv2D:02batch_normalization_9/cond/FusedBatchNorm/Switch:1L
$batch_normalization_9/cond/pred_id:0$batch_normalization_9/cond/pred_id:0
?
&batch_normalization_9/cond/cond_text_1$batch_normalization_9/cond/pred_id:0%batch_normalization_9/cond/switch_f:0*?
batch_normalization_9/beta:0
Cbatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
<batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp:0
Ebatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
>batch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1:0
4batch_normalization_9/cond/FusedBatchNorm_1/Switch:0
-batch_normalization_9/cond/FusedBatchNorm_1:0
-batch_normalization_9/cond/FusedBatchNorm_1:1
-batch_normalization_9/cond/FusedBatchNorm_1:2
-batch_normalization_9/cond/FusedBatchNorm_1:3
-batch_normalization_9/cond/FusedBatchNorm_1:4
4batch_normalization_9/cond/ReadVariableOp_2/Switch:0
-batch_normalization_9/cond/ReadVariableOp_2:0
4batch_normalization_9/cond/ReadVariableOp_3/Switch:0
-batch_normalization_9/cond/ReadVariableOp_3:0
$batch_normalization_9/cond/pred_id:0
%batch_normalization_9/cond/switch_f:0
batch_normalization_9/gamma:0
#batch_normalization_9/moving_mean:0
'batch_normalization_9/moving_variance:0
conv2d_9/Conv2D:0p
'batch_normalization_9/moving_variance:0Ebatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0T
batch_normalization_9/beta:04batch_normalization_9/cond/ReadVariableOp_3/Switch:0U
batch_normalization_9/gamma:04batch_normalization_9/cond/ReadVariableOp_2/Switch:0I
conv2d_9/Conv2D:04batch_normalization_9/cond/FusedBatchNorm_1/Switch:0j
#batch_normalization_9/moving_mean:0Cbatch_normalization_9/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0L
$batch_normalization_9/cond/pred_id:0$batch_normalization_9/cond/pred_id:0
?
&batch_normalization_9/cond_1/cond_text&batch_normalization_9/cond_1/pred_id:0'batch_normalization_9/cond_1/switch_t:0 *?
$batch_normalization_9/cond_1/Const:0
&batch_normalization_9/cond_1/pred_id:0
'batch_normalization_9/cond_1/switch_t:0P
&batch_normalization_9/cond_1/pred_id:0&batch_normalization_9/cond_1/pred_id:0
?
(batch_normalization_9/cond_1/cond_text_1&batch_normalization_9/cond_1/pred_id:0'batch_normalization_9/cond_1/switch_f:0*?
&batch_normalization_9/cond_1/Const_1:0
&batch_normalization_9/cond_1/pred_id:0
'batch_normalization_9/cond_1/switch_f:0P
&batch_normalization_9/cond_1/pred_id:0&batch_normalization_9/cond_1/pred_id:0
?	
%batch_normalization_10/cond/cond_text%batch_normalization_10/cond/pred_id:0&batch_normalization_10/cond/switch_t:0 *?
add_3/add:0
batch_normalization_10/beta:0
#batch_normalization_10/cond/Const:0
%batch_normalization_10/cond/Const_1:0
3batch_normalization_10/cond/FusedBatchNorm/Switch:1
,batch_normalization_10/cond/FusedBatchNorm:0
,batch_normalization_10/cond/FusedBatchNorm:1
,batch_normalization_10/cond/FusedBatchNorm:2
,batch_normalization_10/cond/FusedBatchNorm:3
,batch_normalization_10/cond/FusedBatchNorm:4
3batch_normalization_10/cond/ReadVariableOp/Switch:1
,batch_normalization_10/cond/ReadVariableOp:0
5batch_normalization_10/cond/ReadVariableOp_1/Switch:1
.batch_normalization_10/cond/ReadVariableOp_1:0
%batch_normalization_10/cond/pred_id:0
&batch_normalization_10/cond/switch_t:0
batch_normalization_10/gamma:0U
batch_normalization_10/gamma:03batch_normalization_10/cond/ReadVariableOp/Switch:1V
batch_normalization_10/beta:05batch_normalization_10/cond/ReadVariableOp_1/Switch:1N
%batch_normalization_10/cond/pred_id:0%batch_normalization_10/cond/pred_id:0B
add_3/add:03batch_normalization_10/cond/FusedBatchNorm/Switch:1
?
'batch_normalization_10/cond/cond_text_1%batch_normalization_10/cond/pred_id:0&batch_normalization_10/cond/switch_f:0*?
add_3/add:0
batch_normalization_10/beta:0
Dbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_10/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_10/cond/FusedBatchNorm_1:0
.batch_normalization_10/cond/FusedBatchNorm_1:1
.batch_normalization_10/cond/FusedBatchNorm_1:2
.batch_normalization_10/cond/FusedBatchNorm_1:3
.batch_normalization_10/cond/FusedBatchNorm_1:4
5batch_normalization_10/cond/ReadVariableOp_2/Switch:0
.batch_normalization_10/cond/ReadVariableOp_2:0
5batch_normalization_10/cond/ReadVariableOp_3/Switch:0
.batch_normalization_10/cond/ReadVariableOp_3:0
%batch_normalization_10/cond/pred_id:0
&batch_normalization_10/cond/switch_f:0
batch_normalization_10/gamma:0
$batch_normalization_10/moving_mean:0
(batch_normalization_10/moving_variance:0r
(batch_normalization_10/moving_variance:0Fbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0V
batch_normalization_10/beta:05batch_normalization_10/cond/ReadVariableOp_3/Switch:0l
$batch_normalization_10/moving_mean:0Dbatch_normalization_10/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0W
batch_normalization_10/gamma:05batch_normalization_10/cond/ReadVariableOp_2/Switch:0D
add_3/add:05batch_normalization_10/cond/FusedBatchNorm_1/Switch:0N
%batch_normalization_10/cond/pred_id:0%batch_normalization_10/cond/pred_id:0
?
'batch_normalization_10/cond_1/cond_text'batch_normalization_10/cond_1/pred_id:0(batch_normalization_10/cond_1/switch_t:0 *?
%batch_normalization_10/cond_1/Const:0
'batch_normalization_10/cond_1/pred_id:0
(batch_normalization_10/cond_1/switch_t:0R
'batch_normalization_10/cond_1/pred_id:0'batch_normalization_10/cond_1/pred_id:0
?
)batch_normalization_10/cond_1/cond_text_1'batch_normalization_10/cond_1/pred_id:0(batch_normalization_10/cond_1/switch_f:0*?
'batch_normalization_10/cond_1/Const_1:0
'batch_normalization_10/cond_1/pred_id:0
(batch_normalization_10/cond_1/switch_f:0R
'batch_normalization_10/cond_1/pred_id:0'batch_normalization_10/cond_1/pred_id:0
?	
%batch_normalization_11/cond/cond_text%batch_normalization_11/cond/pred_id:0&batch_normalization_11/cond/switch_t:0 *?
batch_normalization_11/beta:0
#batch_normalization_11/cond/Const:0
%batch_normalization_11/cond/Const_1:0
3batch_normalization_11/cond/FusedBatchNorm/Switch:1
,batch_normalization_11/cond/FusedBatchNorm:0
,batch_normalization_11/cond/FusedBatchNorm:1
,batch_normalization_11/cond/FusedBatchNorm:2
,batch_normalization_11/cond/FusedBatchNorm:3
,batch_normalization_11/cond/FusedBatchNorm:4
3batch_normalization_11/cond/ReadVariableOp/Switch:1
,batch_normalization_11/cond/ReadVariableOp:0
5batch_normalization_11/cond/ReadVariableOp_1/Switch:1
.batch_normalization_11/cond/ReadVariableOp_1:0
%batch_normalization_11/cond/pred_id:0
&batch_normalization_11/cond/switch_t:0
batch_normalization_11/gamma:0
conv2d_12/Conv2D:0N
%batch_normalization_11/cond/pred_id:0%batch_normalization_11/cond/pred_id:0U
batch_normalization_11/gamma:03batch_normalization_11/cond/ReadVariableOp/Switch:1I
conv2d_12/Conv2D:03batch_normalization_11/cond/FusedBatchNorm/Switch:1V
batch_normalization_11/beta:05batch_normalization_11/cond/ReadVariableOp_1/Switch:1
?
'batch_normalization_11/cond/cond_text_1%batch_normalization_11/cond/pred_id:0&batch_normalization_11/cond/switch_f:0*?
batch_normalization_11/beta:0
Dbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_11/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_11/cond/FusedBatchNorm_1:0
.batch_normalization_11/cond/FusedBatchNorm_1:1
.batch_normalization_11/cond/FusedBatchNorm_1:2
.batch_normalization_11/cond/FusedBatchNorm_1:3
.batch_normalization_11/cond/FusedBatchNorm_1:4
5batch_normalization_11/cond/ReadVariableOp_2/Switch:0
.batch_normalization_11/cond/ReadVariableOp_2:0
5batch_normalization_11/cond/ReadVariableOp_3/Switch:0
.batch_normalization_11/cond/ReadVariableOp_3:0
%batch_normalization_11/cond/pred_id:0
&batch_normalization_11/cond/switch_f:0
batch_normalization_11/gamma:0
$batch_normalization_11/moving_mean:0
(batch_normalization_11/moving_variance:0
conv2d_12/Conv2D:0N
%batch_normalization_11/cond/pred_id:0%batch_normalization_11/cond/pred_id:0r
(batch_normalization_11/moving_variance:0Fbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0K
conv2d_12/Conv2D:05batch_normalization_11/cond/FusedBatchNorm_1/Switch:0W
batch_normalization_11/gamma:05batch_normalization_11/cond/ReadVariableOp_2/Switch:0V
batch_normalization_11/beta:05batch_normalization_11/cond/ReadVariableOp_3/Switch:0l
$batch_normalization_11/moving_mean:0Dbatch_normalization_11/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
?
'batch_normalization_11/cond_1/cond_text'batch_normalization_11/cond_1/pred_id:0(batch_normalization_11/cond_1/switch_t:0 *?
%batch_normalization_11/cond_1/Const:0
'batch_normalization_11/cond_1/pred_id:0
(batch_normalization_11/cond_1/switch_t:0R
'batch_normalization_11/cond_1/pred_id:0'batch_normalization_11/cond_1/pred_id:0
?
)batch_normalization_11/cond_1/cond_text_1'batch_normalization_11/cond_1/pred_id:0(batch_normalization_11/cond_1/switch_f:0*?
'batch_normalization_11/cond_1/Const_1:0
'batch_normalization_11/cond_1/pred_id:0
(batch_normalization_11/cond_1/switch_f:0R
'batch_normalization_11/cond_1/pred_id:0'batch_normalization_11/cond_1/pred_id:0
?	
%batch_normalization_12/cond/cond_text%batch_normalization_12/cond/pred_id:0&batch_normalization_12/cond/switch_t:0 *?
add_4/add:0
batch_normalization_12/beta:0
#batch_normalization_12/cond/Const:0
%batch_normalization_12/cond/Const_1:0
3batch_normalization_12/cond/FusedBatchNorm/Switch:1
,batch_normalization_12/cond/FusedBatchNorm:0
,batch_normalization_12/cond/FusedBatchNorm:1
,batch_normalization_12/cond/FusedBatchNorm:2
,batch_normalization_12/cond/FusedBatchNorm:3
,batch_normalization_12/cond/FusedBatchNorm:4
3batch_normalization_12/cond/ReadVariableOp/Switch:1
,batch_normalization_12/cond/ReadVariableOp:0
5batch_normalization_12/cond/ReadVariableOp_1/Switch:1
.batch_normalization_12/cond/ReadVariableOp_1:0
%batch_normalization_12/cond/pred_id:0
&batch_normalization_12/cond/switch_t:0
batch_normalization_12/gamma:0V
batch_normalization_12/beta:05batch_normalization_12/cond/ReadVariableOp_1/Switch:1U
batch_normalization_12/gamma:03batch_normalization_12/cond/ReadVariableOp/Switch:1N
%batch_normalization_12/cond/pred_id:0%batch_normalization_12/cond/pred_id:0B
add_4/add:03batch_normalization_12/cond/FusedBatchNorm/Switch:1
?
'batch_normalization_12/cond/cond_text_1%batch_normalization_12/cond/pred_id:0&batch_normalization_12/cond/switch_f:0*?
add_4/add:0
batch_normalization_12/beta:0
Dbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_12/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_12/cond/FusedBatchNorm_1:0
.batch_normalization_12/cond/FusedBatchNorm_1:1
.batch_normalization_12/cond/FusedBatchNorm_1:2
.batch_normalization_12/cond/FusedBatchNorm_1:3
.batch_normalization_12/cond/FusedBatchNorm_1:4
5batch_normalization_12/cond/ReadVariableOp_2/Switch:0
.batch_normalization_12/cond/ReadVariableOp_2:0
5batch_normalization_12/cond/ReadVariableOp_3/Switch:0
.batch_normalization_12/cond/ReadVariableOp_3:0
%batch_normalization_12/cond/pred_id:0
&batch_normalization_12/cond/switch_f:0
batch_normalization_12/gamma:0
$batch_normalization_12/moving_mean:0
(batch_normalization_12/moving_variance:0W
batch_normalization_12/gamma:05batch_normalization_12/cond/ReadVariableOp_2/Switch:0V
batch_normalization_12/beta:05batch_normalization_12/cond/ReadVariableOp_3/Switch:0D
add_4/add:05batch_normalization_12/cond/FusedBatchNorm_1/Switch:0r
(batch_normalization_12/moving_variance:0Fbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0l
$batch_normalization_12/moving_mean:0Dbatch_normalization_12/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0N
%batch_normalization_12/cond/pred_id:0%batch_normalization_12/cond/pred_id:0
?
'batch_normalization_12/cond_1/cond_text'batch_normalization_12/cond_1/pred_id:0(batch_normalization_12/cond_1/switch_t:0 *?
%batch_normalization_12/cond_1/Const:0
'batch_normalization_12/cond_1/pred_id:0
(batch_normalization_12/cond_1/switch_t:0R
'batch_normalization_12/cond_1/pred_id:0'batch_normalization_12/cond_1/pred_id:0
?
)batch_normalization_12/cond_1/cond_text_1'batch_normalization_12/cond_1/pred_id:0(batch_normalization_12/cond_1/switch_f:0*?
'batch_normalization_12/cond_1/Const_1:0
'batch_normalization_12/cond_1/pred_id:0
(batch_normalization_12/cond_1/switch_f:0R
'batch_normalization_12/cond_1/pred_id:0'batch_normalization_12/cond_1/pred_id:0
?	
%batch_normalization_13/cond/cond_text%batch_normalization_13/cond/pred_id:0&batch_normalization_13/cond/switch_t:0 *?
batch_normalization_13/beta:0
#batch_normalization_13/cond/Const:0
%batch_normalization_13/cond/Const_1:0
3batch_normalization_13/cond/FusedBatchNorm/Switch:1
,batch_normalization_13/cond/FusedBatchNorm:0
,batch_normalization_13/cond/FusedBatchNorm:1
,batch_normalization_13/cond/FusedBatchNorm:2
,batch_normalization_13/cond/FusedBatchNorm:3
,batch_normalization_13/cond/FusedBatchNorm:4
3batch_normalization_13/cond/ReadVariableOp/Switch:1
,batch_normalization_13/cond/ReadVariableOp:0
5batch_normalization_13/cond/ReadVariableOp_1/Switch:1
.batch_normalization_13/cond/ReadVariableOp_1:0
%batch_normalization_13/cond/pred_id:0
&batch_normalization_13/cond/switch_t:0
batch_normalization_13/gamma:0
conv2d_14/Conv2D:0I
conv2d_14/Conv2D:03batch_normalization_13/cond/FusedBatchNorm/Switch:1N
%batch_normalization_13/cond/pred_id:0%batch_normalization_13/cond/pred_id:0U
batch_normalization_13/gamma:03batch_normalization_13/cond/ReadVariableOp/Switch:1V
batch_normalization_13/beta:05batch_normalization_13/cond/ReadVariableOp_1/Switch:1
?
'batch_normalization_13/cond/cond_text_1%batch_normalization_13/cond/pred_id:0&batch_normalization_13/cond/switch_f:0*?
batch_normalization_13/beta:0
Dbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_13/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_13/cond/FusedBatchNorm_1:0
.batch_normalization_13/cond/FusedBatchNorm_1:1
.batch_normalization_13/cond/FusedBatchNorm_1:2
.batch_normalization_13/cond/FusedBatchNorm_1:3
.batch_normalization_13/cond/FusedBatchNorm_1:4
5batch_normalization_13/cond/ReadVariableOp_2/Switch:0
.batch_normalization_13/cond/ReadVariableOp_2:0
5batch_normalization_13/cond/ReadVariableOp_3/Switch:0
.batch_normalization_13/cond/ReadVariableOp_3:0
%batch_normalization_13/cond/pred_id:0
&batch_normalization_13/cond/switch_f:0
batch_normalization_13/gamma:0
$batch_normalization_13/moving_mean:0
(batch_normalization_13/moving_variance:0
conv2d_14/Conv2D:0K
conv2d_14/Conv2D:05batch_normalization_13/cond/FusedBatchNorm_1/Switch:0N
%batch_normalization_13/cond/pred_id:0%batch_normalization_13/cond/pred_id:0r
(batch_normalization_13/moving_variance:0Fbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0V
batch_normalization_13/beta:05batch_normalization_13/cond/ReadVariableOp_3/Switch:0l
$batch_normalization_13/moving_mean:0Dbatch_normalization_13/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0W
batch_normalization_13/gamma:05batch_normalization_13/cond/ReadVariableOp_2/Switch:0
?
'batch_normalization_13/cond_1/cond_text'batch_normalization_13/cond_1/pred_id:0(batch_normalization_13/cond_1/switch_t:0 *?
%batch_normalization_13/cond_1/Const:0
'batch_normalization_13/cond_1/pred_id:0
(batch_normalization_13/cond_1/switch_t:0R
'batch_normalization_13/cond_1/pred_id:0'batch_normalization_13/cond_1/pred_id:0
?
)batch_normalization_13/cond_1/cond_text_1'batch_normalization_13/cond_1/pred_id:0(batch_normalization_13/cond_1/switch_f:0*?
'batch_normalization_13/cond_1/Const_1:0
'batch_normalization_13/cond_1/pred_id:0
(batch_normalization_13/cond_1/switch_f:0R
'batch_normalization_13/cond_1/pred_id:0'batch_normalization_13/cond_1/pred_id:0
?	
%batch_normalization_14/cond/cond_text%batch_normalization_14/cond/pred_id:0&batch_normalization_14/cond/switch_t:0 *?
add_5/add:0
batch_normalization_14/beta:0
#batch_normalization_14/cond/Const:0
%batch_normalization_14/cond/Const_1:0
3batch_normalization_14/cond/FusedBatchNorm/Switch:1
,batch_normalization_14/cond/FusedBatchNorm:0
,batch_normalization_14/cond/FusedBatchNorm:1
,batch_normalization_14/cond/FusedBatchNorm:2
,batch_normalization_14/cond/FusedBatchNorm:3
,batch_normalization_14/cond/FusedBatchNorm:4
3batch_normalization_14/cond/ReadVariableOp/Switch:1
,batch_normalization_14/cond/ReadVariableOp:0
5batch_normalization_14/cond/ReadVariableOp_1/Switch:1
.batch_normalization_14/cond/ReadVariableOp_1:0
%batch_normalization_14/cond/pred_id:0
&batch_normalization_14/cond/switch_t:0
batch_normalization_14/gamma:0U
batch_normalization_14/gamma:03batch_normalization_14/cond/ReadVariableOp/Switch:1B
add_5/add:03batch_normalization_14/cond/FusedBatchNorm/Switch:1V
batch_normalization_14/beta:05batch_normalization_14/cond/ReadVariableOp_1/Switch:1N
%batch_normalization_14/cond/pred_id:0%batch_normalization_14/cond/pred_id:0
?
'batch_normalization_14/cond/cond_text_1%batch_normalization_14/cond/pred_id:0&batch_normalization_14/cond/switch_f:0*?
add_5/add:0
batch_normalization_14/beta:0
Dbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_14/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_14/cond/FusedBatchNorm_1:0
.batch_normalization_14/cond/FusedBatchNorm_1:1
.batch_normalization_14/cond/FusedBatchNorm_1:2
.batch_normalization_14/cond/FusedBatchNorm_1:3
.batch_normalization_14/cond/FusedBatchNorm_1:4
5batch_normalization_14/cond/ReadVariableOp_2/Switch:0
.batch_normalization_14/cond/ReadVariableOp_2:0
5batch_normalization_14/cond/ReadVariableOp_3/Switch:0
.batch_normalization_14/cond/ReadVariableOp_3:0
%batch_normalization_14/cond/pred_id:0
&batch_normalization_14/cond/switch_f:0
batch_normalization_14/gamma:0
$batch_normalization_14/moving_mean:0
(batch_normalization_14/moving_variance:0l
$batch_normalization_14/moving_mean:0Dbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0r
(batch_normalization_14/moving_variance:0Fbatch_normalization_14/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0D
add_5/add:05batch_normalization_14/cond/FusedBatchNorm_1/Switch:0V
batch_normalization_14/beta:05batch_normalization_14/cond/ReadVariableOp_3/Switch:0N
%batch_normalization_14/cond/pred_id:0%batch_normalization_14/cond/pred_id:0W
batch_normalization_14/gamma:05batch_normalization_14/cond/ReadVariableOp_2/Switch:0
?
'batch_normalization_14/cond_1/cond_text'batch_normalization_14/cond_1/pred_id:0(batch_normalization_14/cond_1/switch_t:0 *?
%batch_normalization_14/cond_1/Const:0
'batch_normalization_14/cond_1/pred_id:0
(batch_normalization_14/cond_1/switch_t:0R
'batch_normalization_14/cond_1/pred_id:0'batch_normalization_14/cond_1/pred_id:0
?
)batch_normalization_14/cond_1/cond_text_1'batch_normalization_14/cond_1/pred_id:0(batch_normalization_14/cond_1/switch_f:0*?
'batch_normalization_14/cond_1/Const_1:0
'batch_normalization_14/cond_1/pred_id:0
(batch_normalization_14/cond_1/switch_f:0R
'batch_normalization_14/cond_1/pred_id:0'batch_normalization_14/cond_1/pred_id:0
?	
%batch_normalization_15/cond/cond_text%batch_normalization_15/cond/pred_id:0&batch_normalization_15/cond/switch_t:0 *?
batch_normalization_15/beta:0
#batch_normalization_15/cond/Const:0
%batch_normalization_15/cond/Const_1:0
3batch_normalization_15/cond/FusedBatchNorm/Switch:1
,batch_normalization_15/cond/FusedBatchNorm:0
,batch_normalization_15/cond/FusedBatchNorm:1
,batch_normalization_15/cond/FusedBatchNorm:2
,batch_normalization_15/cond/FusedBatchNorm:3
,batch_normalization_15/cond/FusedBatchNorm:4
3batch_normalization_15/cond/ReadVariableOp/Switch:1
,batch_normalization_15/cond/ReadVariableOp:0
5batch_normalization_15/cond/ReadVariableOp_1/Switch:1
.batch_normalization_15/cond/ReadVariableOp_1:0
%batch_normalization_15/cond/pred_id:0
&batch_normalization_15/cond/switch_t:0
batch_normalization_15/gamma:0
conv2d_17/Conv2D:0V
batch_normalization_15/beta:05batch_normalization_15/cond/ReadVariableOp_1/Switch:1N
%batch_normalization_15/cond/pred_id:0%batch_normalization_15/cond/pred_id:0I
conv2d_17/Conv2D:03batch_normalization_15/cond/FusedBatchNorm/Switch:1U
batch_normalization_15/gamma:03batch_normalization_15/cond/ReadVariableOp/Switch:1
?
'batch_normalization_15/cond/cond_text_1%batch_normalization_15/cond/pred_id:0&batch_normalization_15/cond/switch_f:0*?
batch_normalization_15/beta:0
Dbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_15/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_15/cond/FusedBatchNorm_1:0
.batch_normalization_15/cond/FusedBatchNorm_1:1
.batch_normalization_15/cond/FusedBatchNorm_1:2
.batch_normalization_15/cond/FusedBatchNorm_1:3
.batch_normalization_15/cond/FusedBatchNorm_1:4
5batch_normalization_15/cond/ReadVariableOp_2/Switch:0
.batch_normalization_15/cond/ReadVariableOp_2:0
5batch_normalization_15/cond/ReadVariableOp_3/Switch:0
.batch_normalization_15/cond/ReadVariableOp_3:0
%batch_normalization_15/cond/pred_id:0
&batch_normalization_15/cond/switch_f:0
batch_normalization_15/gamma:0
$batch_normalization_15/moving_mean:0
(batch_normalization_15/moving_variance:0
conv2d_17/Conv2D:0N
%batch_normalization_15/cond/pred_id:0%batch_normalization_15/cond/pred_id:0K
conv2d_17/Conv2D:05batch_normalization_15/cond/FusedBatchNorm_1/Switch:0V
batch_normalization_15/beta:05batch_normalization_15/cond/ReadVariableOp_3/Switch:0l
$batch_normalization_15/moving_mean:0Dbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0r
(batch_normalization_15/moving_variance:0Fbatch_normalization_15/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0W
batch_normalization_15/gamma:05batch_normalization_15/cond/ReadVariableOp_2/Switch:0
?
'batch_normalization_15/cond_1/cond_text'batch_normalization_15/cond_1/pred_id:0(batch_normalization_15/cond_1/switch_t:0 *?
%batch_normalization_15/cond_1/Const:0
'batch_normalization_15/cond_1/pred_id:0
(batch_normalization_15/cond_1/switch_t:0R
'batch_normalization_15/cond_1/pred_id:0'batch_normalization_15/cond_1/pred_id:0
?
)batch_normalization_15/cond_1/cond_text_1'batch_normalization_15/cond_1/pred_id:0(batch_normalization_15/cond_1/switch_f:0*?
'batch_normalization_15/cond_1/Const_1:0
'batch_normalization_15/cond_1/pred_id:0
(batch_normalization_15/cond_1/switch_f:0R
'batch_normalization_15/cond_1/pred_id:0'batch_normalization_15/cond_1/pred_id:0
?	
%batch_normalization_16/cond/cond_text%batch_normalization_16/cond/pred_id:0&batch_normalization_16/cond/switch_t:0 *?
add_6/add:0
batch_normalization_16/beta:0
#batch_normalization_16/cond/Const:0
%batch_normalization_16/cond/Const_1:0
3batch_normalization_16/cond/FusedBatchNorm/Switch:1
,batch_normalization_16/cond/FusedBatchNorm:0
,batch_normalization_16/cond/FusedBatchNorm:1
,batch_normalization_16/cond/FusedBatchNorm:2
,batch_normalization_16/cond/FusedBatchNorm:3
,batch_normalization_16/cond/FusedBatchNorm:4
3batch_normalization_16/cond/ReadVariableOp/Switch:1
,batch_normalization_16/cond/ReadVariableOp:0
5batch_normalization_16/cond/ReadVariableOp_1/Switch:1
.batch_normalization_16/cond/ReadVariableOp_1:0
%batch_normalization_16/cond/pred_id:0
&batch_normalization_16/cond/switch_t:0
batch_normalization_16/gamma:0V
batch_normalization_16/beta:05batch_normalization_16/cond/ReadVariableOp_1/Switch:1U
batch_normalization_16/gamma:03batch_normalization_16/cond/ReadVariableOp/Switch:1N
%batch_normalization_16/cond/pred_id:0%batch_normalization_16/cond/pred_id:0B
add_6/add:03batch_normalization_16/cond/FusedBatchNorm/Switch:1
?
'batch_normalization_16/cond/cond_text_1%batch_normalization_16/cond/pred_id:0&batch_normalization_16/cond/switch_f:0*?
add_6/add:0
batch_normalization_16/beta:0
Dbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_16/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_16/cond/FusedBatchNorm_1:0
.batch_normalization_16/cond/FusedBatchNorm_1:1
.batch_normalization_16/cond/FusedBatchNorm_1:2
.batch_normalization_16/cond/FusedBatchNorm_1:3
.batch_normalization_16/cond/FusedBatchNorm_1:4
5batch_normalization_16/cond/ReadVariableOp_2/Switch:0
.batch_normalization_16/cond/ReadVariableOp_2:0
5batch_normalization_16/cond/ReadVariableOp_3/Switch:0
.batch_normalization_16/cond/ReadVariableOp_3:0
%batch_normalization_16/cond/pred_id:0
&batch_normalization_16/cond/switch_f:0
batch_normalization_16/gamma:0
$batch_normalization_16/moving_mean:0
(batch_normalization_16/moving_variance:0V
batch_normalization_16/beta:05batch_normalization_16/cond/ReadVariableOp_3/Switch:0r
(batch_normalization_16/moving_variance:0Fbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0l
$batch_normalization_16/moving_mean:0Dbatch_normalization_16/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0W
batch_normalization_16/gamma:05batch_normalization_16/cond/ReadVariableOp_2/Switch:0N
%batch_normalization_16/cond/pred_id:0%batch_normalization_16/cond/pred_id:0D
add_6/add:05batch_normalization_16/cond/FusedBatchNorm_1/Switch:0
?
'batch_normalization_16/cond_1/cond_text'batch_normalization_16/cond_1/pred_id:0(batch_normalization_16/cond_1/switch_t:0 *?
%batch_normalization_16/cond_1/Const:0
'batch_normalization_16/cond_1/pred_id:0
(batch_normalization_16/cond_1/switch_t:0R
'batch_normalization_16/cond_1/pred_id:0'batch_normalization_16/cond_1/pred_id:0
?
)batch_normalization_16/cond_1/cond_text_1'batch_normalization_16/cond_1/pred_id:0(batch_normalization_16/cond_1/switch_f:0*?
'batch_normalization_16/cond_1/Const_1:0
'batch_normalization_16/cond_1/pred_id:0
(batch_normalization_16/cond_1/switch_f:0R
'batch_normalization_16/cond_1/pred_id:0'batch_normalization_16/cond_1/pred_id:0
?	
%batch_normalization_17/cond/cond_text%batch_normalization_17/cond/pred_id:0&batch_normalization_17/cond/switch_t:0 *?
batch_normalization_17/beta:0
#batch_normalization_17/cond/Const:0
%batch_normalization_17/cond/Const_1:0
3batch_normalization_17/cond/FusedBatchNorm/Switch:1
,batch_normalization_17/cond/FusedBatchNorm:0
,batch_normalization_17/cond/FusedBatchNorm:1
,batch_normalization_17/cond/FusedBatchNorm:2
,batch_normalization_17/cond/FusedBatchNorm:3
,batch_normalization_17/cond/FusedBatchNorm:4
3batch_normalization_17/cond/ReadVariableOp/Switch:1
,batch_normalization_17/cond/ReadVariableOp:0
5batch_normalization_17/cond/ReadVariableOp_1/Switch:1
.batch_normalization_17/cond/ReadVariableOp_1:0
%batch_normalization_17/cond/pred_id:0
&batch_normalization_17/cond/switch_t:0
batch_normalization_17/gamma:0
conv2d_19/Conv2D:0I
conv2d_19/Conv2D:03batch_normalization_17/cond/FusedBatchNorm/Switch:1U
batch_normalization_17/gamma:03batch_normalization_17/cond/ReadVariableOp/Switch:1N
%batch_normalization_17/cond/pred_id:0%batch_normalization_17/cond/pred_id:0V
batch_normalization_17/beta:05batch_normalization_17/cond/ReadVariableOp_1/Switch:1
?
'batch_normalization_17/cond/cond_text_1%batch_normalization_17/cond/pred_id:0&batch_normalization_17/cond/switch_f:0*?
batch_normalization_17/beta:0
Dbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_17/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_17/cond/FusedBatchNorm_1:0
.batch_normalization_17/cond/FusedBatchNorm_1:1
.batch_normalization_17/cond/FusedBatchNorm_1:2
.batch_normalization_17/cond/FusedBatchNorm_1:3
.batch_normalization_17/cond/FusedBatchNorm_1:4
5batch_normalization_17/cond/ReadVariableOp_2/Switch:0
.batch_normalization_17/cond/ReadVariableOp_2:0
5batch_normalization_17/cond/ReadVariableOp_3/Switch:0
.batch_normalization_17/cond/ReadVariableOp_3:0
%batch_normalization_17/cond/pred_id:0
&batch_normalization_17/cond/switch_f:0
batch_normalization_17/gamma:0
$batch_normalization_17/moving_mean:0
(batch_normalization_17/moving_variance:0
conv2d_19/Conv2D:0W
batch_normalization_17/gamma:05batch_normalization_17/cond/ReadVariableOp_2/Switch:0r
(batch_normalization_17/moving_variance:0Fbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0N
%batch_normalization_17/cond/pred_id:0%batch_normalization_17/cond/pred_id:0K
conv2d_19/Conv2D:05batch_normalization_17/cond/FusedBatchNorm_1/Switch:0l
$batch_normalization_17/moving_mean:0Dbatch_normalization_17/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0V
batch_normalization_17/beta:05batch_normalization_17/cond/ReadVariableOp_3/Switch:0
?
'batch_normalization_17/cond_1/cond_text'batch_normalization_17/cond_1/pred_id:0(batch_normalization_17/cond_1/switch_t:0 *?
%batch_normalization_17/cond_1/Const:0
'batch_normalization_17/cond_1/pred_id:0
(batch_normalization_17/cond_1/switch_t:0R
'batch_normalization_17/cond_1/pred_id:0'batch_normalization_17/cond_1/pred_id:0
?
)batch_normalization_17/cond_1/cond_text_1'batch_normalization_17/cond_1/pred_id:0(batch_normalization_17/cond_1/switch_f:0*?
'batch_normalization_17/cond_1/Const_1:0
'batch_normalization_17/cond_1/pred_id:0
(batch_normalization_17/cond_1/switch_f:0R
'batch_normalization_17/cond_1/pred_id:0'batch_normalization_17/cond_1/pred_id:0
?	
%batch_normalization_18/cond/cond_text%batch_normalization_18/cond/pred_id:0&batch_normalization_18/cond/switch_t:0 *?
add_7/add:0
batch_normalization_18/beta:0
#batch_normalization_18/cond/Const:0
%batch_normalization_18/cond/Const_1:0
3batch_normalization_18/cond/FusedBatchNorm/Switch:1
,batch_normalization_18/cond/FusedBatchNorm:0
,batch_normalization_18/cond/FusedBatchNorm:1
,batch_normalization_18/cond/FusedBatchNorm:2
,batch_normalization_18/cond/FusedBatchNorm:3
,batch_normalization_18/cond/FusedBatchNorm:4
3batch_normalization_18/cond/ReadVariableOp/Switch:1
,batch_normalization_18/cond/ReadVariableOp:0
5batch_normalization_18/cond/ReadVariableOp_1/Switch:1
.batch_normalization_18/cond/ReadVariableOp_1:0
%batch_normalization_18/cond/pred_id:0
&batch_normalization_18/cond/switch_t:0
batch_normalization_18/gamma:0N
%batch_normalization_18/cond/pred_id:0%batch_normalization_18/cond/pred_id:0B
add_7/add:03batch_normalization_18/cond/FusedBatchNorm/Switch:1V
batch_normalization_18/beta:05batch_normalization_18/cond/ReadVariableOp_1/Switch:1U
batch_normalization_18/gamma:03batch_normalization_18/cond/ReadVariableOp/Switch:1
?
'batch_normalization_18/cond/cond_text_1%batch_normalization_18/cond/pred_id:0&batch_normalization_18/cond/switch_f:0*?
add_7/add:0
batch_normalization_18/beta:0
Dbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
=batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp:0
Fbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0
?batch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1:0
5batch_normalization_18/cond/FusedBatchNorm_1/Switch:0
.batch_normalization_18/cond/FusedBatchNorm_1:0
.batch_normalization_18/cond/FusedBatchNorm_1:1
.batch_normalization_18/cond/FusedBatchNorm_1:2
.batch_normalization_18/cond/FusedBatchNorm_1:3
.batch_normalization_18/cond/FusedBatchNorm_1:4
5batch_normalization_18/cond/ReadVariableOp_2/Switch:0
.batch_normalization_18/cond/ReadVariableOp_2:0
5batch_normalization_18/cond/ReadVariableOp_3/Switch:0
.batch_normalization_18/cond/ReadVariableOp_3:0
%batch_normalization_18/cond/pred_id:0
&batch_normalization_18/cond/switch_f:0
batch_normalization_18/gamma:0
$batch_normalization_18/moving_mean:0
(batch_normalization_18/moving_variance:0r
(batch_normalization_18/moving_variance:0Fbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp_1/Switch:0N
%batch_normalization_18/cond/pred_id:0%batch_normalization_18/cond/pred_id:0W
batch_normalization_18/gamma:05batch_normalization_18/cond/ReadVariableOp_2/Switch:0V
batch_normalization_18/beta:05batch_normalization_18/cond/ReadVariableOp_3/Switch:0D
add_7/add:05batch_normalization_18/cond/FusedBatchNorm_1/Switch:0l
$batch_normalization_18/moving_mean:0Dbatch_normalization_18/cond/FusedBatchNorm_1/ReadVariableOp/Switch:0
?
'batch_normalization_18/cond_1/cond_text'batch_normalization_18/cond_1/pred_id:0(batch_normalization_18/cond_1/switch_t:0 *?
%batch_normalization_18/cond_1/Const:0
'batch_normalization_18/cond_1/pred_id:0
(batch_normalization_18/cond_1/switch_t:0R
'batch_normalization_18/cond_1/pred_id:0'batch_normalization_18/cond_1/pred_id:0
?
)batch_normalization_18/cond_1/cond_text_1'batch_normalization_18/cond_1/pred_id:0(batch_normalization_18/cond_1/switch_f:0*?
'batch_normalization_18/cond_1/Const_1:0
'batch_normalization_18/cond_1/pred_id:0
(batch_normalization_18/cond_1/switch_f:0R
'batch_normalization_18/cond_1/pred_id:0'batch_normalization_18/cond_1/pred_id:0"??
	variables????
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
?
!batch_normalization/moving_mean:0&batch_normalization/moving_mean/Assign5batch_normalization/moving_mean/Read/ReadVariableOp:0(23batch_normalization/moving_mean/Initializer/zeros:0@H
?
%batch_normalization/moving_variance:0*batch_normalization/moving_variance/Assign9batch_normalization/moving_variance/Read/ReadVariableOp:0(26batch_normalization/moving_variance/Initializer/ones:0@H
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
#batch_normalization_1/moving_mean:0(batch_normalization_1/moving_mean/Assign7batch_normalization_1/moving_mean/Read/ReadVariableOp:0(25batch_normalization_1/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_1/moving_variance:0,batch_normalization_1/moving_variance/Assign;batch_normalization_1/moving_variance/Read/ReadVariableOp:0(28batch_normalization_1/moving_variance/Initializer/ones:0@H
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
#batch_normalization_2/moving_mean:0(batch_normalization_2/moving_mean/Assign7batch_normalization_2/moving_mean/Read/ReadVariableOp:0(25batch_normalization_2/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_2/moving_variance:0,batch_normalization_2/moving_variance/Assign;batch_normalization_2/moving_variance/Read/ReadVariableOp:0(28batch_normalization_2/moving_variance/Initializer/ones:0@H
?
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
?
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
#batch_normalization_3/moving_mean:0(batch_normalization_3/moving_mean/Assign7batch_normalization_3/moving_mean/Read/ReadVariableOp:0(25batch_normalization_3/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_3/moving_variance:0,batch_normalization_3/moving_variance/Assign;batch_normalization_3/moving_variance/Read/ReadVariableOp:0(28batch_normalization_3/moving_variance/Initializer/ones:0@H
?
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
?
#batch_normalization_4/moving_mean:0(batch_normalization_4/moving_mean/Assign7batch_normalization_4/moving_mean/Read/ReadVariableOp:0(25batch_normalization_4/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_4/moving_variance:0,batch_normalization_4/moving_variance/Assign;batch_normalization_4/moving_variance/Read/ReadVariableOp:0(28batch_normalization_4/moving_variance/Initializer/ones:0@H
?
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
?
batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign1batch_normalization_5/gamma/Read/ReadVariableOp:0(2.batch_normalization_5/gamma/Initializer/ones:08
?
batch_normalization_5/beta:0!batch_normalization_5/beta/Assign0batch_normalization_5/beta/Read/ReadVariableOp:0(2.batch_normalization_5/beta/Initializer/zeros:08
?
#batch_normalization_5/moving_mean:0(batch_normalization_5/moving_mean/Assign7batch_normalization_5/moving_mean/Read/ReadVariableOp:0(25batch_normalization_5/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_5/moving_variance:0,batch_normalization_5/moving_variance/Assign;batch_normalization_5/moving_variance/Read/ReadVariableOp:0(28batch_normalization_5/moving_variance/Initializer/ones:0@H
?
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
?
batch_normalization_6/gamma:0"batch_normalization_6/gamma/Assign1batch_normalization_6/gamma/Read/ReadVariableOp:0(2.batch_normalization_6/gamma/Initializer/ones:08
?
batch_normalization_6/beta:0!batch_normalization_6/beta/Assign0batch_normalization_6/beta/Read/ReadVariableOp:0(2.batch_normalization_6/beta/Initializer/zeros:08
?
#batch_normalization_6/moving_mean:0(batch_normalization_6/moving_mean/Assign7batch_normalization_6/moving_mean/Read/ReadVariableOp:0(25batch_normalization_6/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_6/moving_variance:0,batch_normalization_6/moving_variance/Assign;batch_normalization_6/moving_variance/Read/ReadVariableOp:0(28batch_normalization_6/moving_variance/Initializer/ones:0@H
?
conv2d_6/kernel:0conv2d_6/kernel/Assign%conv2d_6/kernel/Read/ReadVariableOp:0(2,conv2d_6/kernel/Initializer/random_uniform:08
?
conv2d_7/kernel:0conv2d_7/kernel/Assign%conv2d_7/kernel/Read/ReadVariableOp:0(2,conv2d_7/kernel/Initializer/random_uniform:08
?
batch_normalization_7/gamma:0"batch_normalization_7/gamma/Assign1batch_normalization_7/gamma/Read/ReadVariableOp:0(2.batch_normalization_7/gamma/Initializer/ones:08
?
batch_normalization_7/beta:0!batch_normalization_7/beta/Assign0batch_normalization_7/beta/Read/ReadVariableOp:0(2.batch_normalization_7/beta/Initializer/zeros:08
?
#batch_normalization_7/moving_mean:0(batch_normalization_7/moving_mean/Assign7batch_normalization_7/moving_mean/Read/ReadVariableOp:0(25batch_normalization_7/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_7/moving_variance:0,batch_normalization_7/moving_variance/Assign;batch_normalization_7/moving_variance/Read/ReadVariableOp:0(28batch_normalization_7/moving_variance/Initializer/ones:0@H
?
conv2d_8/kernel:0conv2d_8/kernel/Assign%conv2d_8/kernel/Read/ReadVariableOp:0(2,conv2d_8/kernel/Initializer/random_uniform:08
?
batch_normalization_8/gamma:0"batch_normalization_8/gamma/Assign1batch_normalization_8/gamma/Read/ReadVariableOp:0(2.batch_normalization_8/gamma/Initializer/ones:08
?
batch_normalization_8/beta:0!batch_normalization_8/beta/Assign0batch_normalization_8/beta/Read/ReadVariableOp:0(2.batch_normalization_8/beta/Initializer/zeros:08
?
#batch_normalization_8/moving_mean:0(batch_normalization_8/moving_mean/Assign7batch_normalization_8/moving_mean/Read/ReadVariableOp:0(25batch_normalization_8/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_8/moving_variance:0,batch_normalization_8/moving_variance/Assign;batch_normalization_8/moving_variance/Read/ReadVariableOp:0(28batch_normalization_8/moving_variance/Initializer/ones:0@H
?
conv2d_9/kernel:0conv2d_9/kernel/Assign%conv2d_9/kernel/Read/ReadVariableOp:0(2,conv2d_9/kernel/Initializer/random_uniform:08
?
batch_normalization_9/gamma:0"batch_normalization_9/gamma/Assign1batch_normalization_9/gamma/Read/ReadVariableOp:0(2.batch_normalization_9/gamma/Initializer/ones:08
?
batch_normalization_9/beta:0!batch_normalization_9/beta/Assign0batch_normalization_9/beta/Read/ReadVariableOp:0(2.batch_normalization_9/beta/Initializer/zeros:08
?
#batch_normalization_9/moving_mean:0(batch_normalization_9/moving_mean/Assign7batch_normalization_9/moving_mean/Read/ReadVariableOp:0(25batch_normalization_9/moving_mean/Initializer/zeros:0@H
?
'batch_normalization_9/moving_variance:0,batch_normalization_9/moving_variance/Assign;batch_normalization_9/moving_variance/Read/ReadVariableOp:0(28batch_normalization_9/moving_variance/Initializer/ones:0@H
?
conv2d_10/kernel:0conv2d_10/kernel/Assign&conv2d_10/kernel/Read/ReadVariableOp:0(2-conv2d_10/kernel/Initializer/random_uniform:08
?
batch_normalization_10/gamma:0#batch_normalization_10/gamma/Assign2batch_normalization_10/gamma/Read/ReadVariableOp:0(2/batch_normalization_10/gamma/Initializer/ones:08
?
batch_normalization_10/beta:0"batch_normalization_10/beta/Assign1batch_normalization_10/beta/Read/ReadVariableOp:0(2/batch_normalization_10/beta/Initializer/zeros:08
?
$batch_normalization_10/moving_mean:0)batch_normalization_10/moving_mean/Assign8batch_normalization_10/moving_mean/Read/ReadVariableOp:0(26batch_normalization_10/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_10/moving_variance:0-batch_normalization_10/moving_variance/Assign<batch_normalization_10/moving_variance/Read/ReadVariableOp:0(29batch_normalization_10/moving_variance/Initializer/ones:0@H
?
conv2d_11/kernel:0conv2d_11/kernel/Assign&conv2d_11/kernel/Read/ReadVariableOp:0(2-conv2d_11/kernel/Initializer/random_uniform:08
?
conv2d_12/kernel:0conv2d_12/kernel/Assign&conv2d_12/kernel/Read/ReadVariableOp:0(2-conv2d_12/kernel/Initializer/random_uniform:08
?
batch_normalization_11/gamma:0#batch_normalization_11/gamma/Assign2batch_normalization_11/gamma/Read/ReadVariableOp:0(2/batch_normalization_11/gamma/Initializer/ones:08
?
batch_normalization_11/beta:0"batch_normalization_11/beta/Assign1batch_normalization_11/beta/Read/ReadVariableOp:0(2/batch_normalization_11/beta/Initializer/zeros:08
?
$batch_normalization_11/moving_mean:0)batch_normalization_11/moving_mean/Assign8batch_normalization_11/moving_mean/Read/ReadVariableOp:0(26batch_normalization_11/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_11/moving_variance:0-batch_normalization_11/moving_variance/Assign<batch_normalization_11/moving_variance/Read/ReadVariableOp:0(29batch_normalization_11/moving_variance/Initializer/ones:0@H
?
conv2d_13/kernel:0conv2d_13/kernel/Assign&conv2d_13/kernel/Read/ReadVariableOp:0(2-conv2d_13/kernel/Initializer/random_uniform:08
?
batch_normalization_12/gamma:0#batch_normalization_12/gamma/Assign2batch_normalization_12/gamma/Read/ReadVariableOp:0(2/batch_normalization_12/gamma/Initializer/ones:08
?
batch_normalization_12/beta:0"batch_normalization_12/beta/Assign1batch_normalization_12/beta/Read/ReadVariableOp:0(2/batch_normalization_12/beta/Initializer/zeros:08
?
$batch_normalization_12/moving_mean:0)batch_normalization_12/moving_mean/Assign8batch_normalization_12/moving_mean/Read/ReadVariableOp:0(26batch_normalization_12/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_12/moving_variance:0-batch_normalization_12/moving_variance/Assign<batch_normalization_12/moving_variance/Read/ReadVariableOp:0(29batch_normalization_12/moving_variance/Initializer/ones:0@H
?
conv2d_14/kernel:0conv2d_14/kernel/Assign&conv2d_14/kernel/Read/ReadVariableOp:0(2-conv2d_14/kernel/Initializer/random_uniform:08
?
batch_normalization_13/gamma:0#batch_normalization_13/gamma/Assign2batch_normalization_13/gamma/Read/ReadVariableOp:0(2/batch_normalization_13/gamma/Initializer/ones:08
?
batch_normalization_13/beta:0"batch_normalization_13/beta/Assign1batch_normalization_13/beta/Read/ReadVariableOp:0(2/batch_normalization_13/beta/Initializer/zeros:08
?
$batch_normalization_13/moving_mean:0)batch_normalization_13/moving_mean/Assign8batch_normalization_13/moving_mean/Read/ReadVariableOp:0(26batch_normalization_13/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_13/moving_variance:0-batch_normalization_13/moving_variance/Assign<batch_normalization_13/moving_variance/Read/ReadVariableOp:0(29batch_normalization_13/moving_variance/Initializer/ones:0@H
?
conv2d_15/kernel:0conv2d_15/kernel/Assign&conv2d_15/kernel/Read/ReadVariableOp:0(2-conv2d_15/kernel/Initializer/random_uniform:08
?
batch_normalization_14/gamma:0#batch_normalization_14/gamma/Assign2batch_normalization_14/gamma/Read/ReadVariableOp:0(2/batch_normalization_14/gamma/Initializer/ones:08
?
batch_normalization_14/beta:0"batch_normalization_14/beta/Assign1batch_normalization_14/beta/Read/ReadVariableOp:0(2/batch_normalization_14/beta/Initializer/zeros:08
?
$batch_normalization_14/moving_mean:0)batch_normalization_14/moving_mean/Assign8batch_normalization_14/moving_mean/Read/ReadVariableOp:0(26batch_normalization_14/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_14/moving_variance:0-batch_normalization_14/moving_variance/Assign<batch_normalization_14/moving_variance/Read/ReadVariableOp:0(29batch_normalization_14/moving_variance/Initializer/ones:0@H
?
conv2d_16/kernel:0conv2d_16/kernel/Assign&conv2d_16/kernel/Read/ReadVariableOp:0(2-conv2d_16/kernel/Initializer/random_uniform:08
?
conv2d_17/kernel:0conv2d_17/kernel/Assign&conv2d_17/kernel/Read/ReadVariableOp:0(2-conv2d_17/kernel/Initializer/random_uniform:08
?
batch_normalization_15/gamma:0#batch_normalization_15/gamma/Assign2batch_normalization_15/gamma/Read/ReadVariableOp:0(2/batch_normalization_15/gamma/Initializer/ones:08
?
batch_normalization_15/beta:0"batch_normalization_15/beta/Assign1batch_normalization_15/beta/Read/ReadVariableOp:0(2/batch_normalization_15/beta/Initializer/zeros:08
?
$batch_normalization_15/moving_mean:0)batch_normalization_15/moving_mean/Assign8batch_normalization_15/moving_mean/Read/ReadVariableOp:0(26batch_normalization_15/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_15/moving_variance:0-batch_normalization_15/moving_variance/Assign<batch_normalization_15/moving_variance/Read/ReadVariableOp:0(29batch_normalization_15/moving_variance/Initializer/ones:0@H
?
conv2d_18/kernel:0conv2d_18/kernel/Assign&conv2d_18/kernel/Read/ReadVariableOp:0(2-conv2d_18/kernel/Initializer/random_uniform:08
?
batch_normalization_16/gamma:0#batch_normalization_16/gamma/Assign2batch_normalization_16/gamma/Read/ReadVariableOp:0(2/batch_normalization_16/gamma/Initializer/ones:08
?
batch_normalization_16/beta:0"batch_normalization_16/beta/Assign1batch_normalization_16/beta/Read/ReadVariableOp:0(2/batch_normalization_16/beta/Initializer/zeros:08
?
$batch_normalization_16/moving_mean:0)batch_normalization_16/moving_mean/Assign8batch_normalization_16/moving_mean/Read/ReadVariableOp:0(26batch_normalization_16/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_16/moving_variance:0-batch_normalization_16/moving_variance/Assign<batch_normalization_16/moving_variance/Read/ReadVariableOp:0(29batch_normalization_16/moving_variance/Initializer/ones:0@H
?
conv2d_19/kernel:0conv2d_19/kernel/Assign&conv2d_19/kernel/Read/ReadVariableOp:0(2-conv2d_19/kernel/Initializer/random_uniform:08
?
batch_normalization_17/gamma:0#batch_normalization_17/gamma/Assign2batch_normalization_17/gamma/Read/ReadVariableOp:0(2/batch_normalization_17/gamma/Initializer/ones:08
?
batch_normalization_17/beta:0"batch_normalization_17/beta/Assign1batch_normalization_17/beta/Read/ReadVariableOp:0(2/batch_normalization_17/beta/Initializer/zeros:08
?
$batch_normalization_17/moving_mean:0)batch_normalization_17/moving_mean/Assign8batch_normalization_17/moving_mean/Read/ReadVariableOp:0(26batch_normalization_17/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_17/moving_variance:0-batch_normalization_17/moving_variance/Assign<batch_normalization_17/moving_variance/Read/ReadVariableOp:0(29batch_normalization_17/moving_variance/Initializer/ones:0@H
?
conv2d_20/kernel:0conv2d_20/kernel/Assign&conv2d_20/kernel/Read/ReadVariableOp:0(2-conv2d_20/kernel/Initializer/random_uniform:08
?
batch_normalization_18/gamma:0#batch_normalization_18/gamma/Assign2batch_normalization_18/gamma/Read/ReadVariableOp:0(2/batch_normalization_18/gamma/Initializer/ones:08
?
batch_normalization_18/beta:0"batch_normalization_18/beta/Assign1batch_normalization_18/beta/Read/ReadVariableOp:0(2/batch_normalization_18/beta/Initializer/zeros:08
?
$batch_normalization_18/moving_mean:0)batch_normalization_18/moving_mean/Assign8batch_normalization_18/moving_mean/Read/ReadVariableOp:0(26batch_normalization_18/moving_mean/Initializer/zeros:0@H
?
(batch_normalization_18/moving_variance:0-batch_normalization_18/moving_variance/Assign<batch_normalization_18/moving_variance/Read/ReadVariableOp:0(29batch_normalization_18/moving_variance/Initializer/ones:0@H
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08")
saved_model_main_op

legacy_init_op"?J
trainable_variables?J?J
?
batch_normalization/beta:0batch_normalization/beta/Assign.batch_normalization/beta/Read/ReadVariableOp:0(2,batch_normalization/beta/Initializer/zeros:08
|
conv2d/kernel:0conv2d/kernel/Assign#conv2d/kernel/Read/ReadVariableOp:0(2*conv2d/kernel/Initializer/random_uniform:08
?
batch_normalization_1/gamma:0"batch_normalization_1/gamma/Assign1batch_normalization_1/gamma/Read/ReadVariableOp:0(2.batch_normalization_1/gamma/Initializer/ones:08
?
batch_normalization_1/beta:0!batch_normalization_1/beta/Assign0batch_normalization_1/beta/Read/ReadVariableOp:0(2.batch_normalization_1/beta/Initializer/zeros:08
?
batch_normalization_2/gamma:0"batch_normalization_2/gamma/Assign1batch_normalization_2/gamma/Read/ReadVariableOp:0(2.batch_normalization_2/gamma/Initializer/ones:08
?
batch_normalization_2/beta:0!batch_normalization_2/beta/Assign0batch_normalization_2/beta/Read/ReadVariableOp:0(2.batch_normalization_2/beta/Initializer/zeros:08
?
conv2d_1/kernel:0conv2d_1/kernel/Assign%conv2d_1/kernel/Read/ReadVariableOp:0(2,conv2d_1/kernel/Initializer/random_uniform:08
?
conv2d_2/kernel:0conv2d_2/kernel/Assign%conv2d_2/kernel/Read/ReadVariableOp:0(2,conv2d_2/kernel/Initializer/random_uniform:08
?
batch_normalization_3/gamma:0"batch_normalization_3/gamma/Assign1batch_normalization_3/gamma/Read/ReadVariableOp:0(2.batch_normalization_3/gamma/Initializer/ones:08
?
batch_normalization_3/beta:0!batch_normalization_3/beta/Assign0batch_normalization_3/beta/Read/ReadVariableOp:0(2.batch_normalization_3/beta/Initializer/zeros:08
?
conv2d_3/kernel:0conv2d_3/kernel/Assign%conv2d_3/kernel/Read/ReadVariableOp:0(2,conv2d_3/kernel/Initializer/random_uniform:08
?
batch_normalization_4/gamma:0"batch_normalization_4/gamma/Assign1batch_normalization_4/gamma/Read/ReadVariableOp:0(2.batch_normalization_4/gamma/Initializer/ones:08
?
batch_normalization_4/beta:0!batch_normalization_4/beta/Assign0batch_normalization_4/beta/Read/ReadVariableOp:0(2.batch_normalization_4/beta/Initializer/zeros:08
?
conv2d_4/kernel:0conv2d_4/kernel/Assign%conv2d_4/kernel/Read/ReadVariableOp:0(2,conv2d_4/kernel/Initializer/random_uniform:08
?
batch_normalization_5/gamma:0"batch_normalization_5/gamma/Assign1batch_normalization_5/gamma/Read/ReadVariableOp:0(2.batch_normalization_5/gamma/Initializer/ones:08
?
batch_normalization_5/beta:0!batch_normalization_5/beta/Assign0batch_normalization_5/beta/Read/ReadVariableOp:0(2.batch_normalization_5/beta/Initializer/zeros:08
?
conv2d_5/kernel:0conv2d_5/kernel/Assign%conv2d_5/kernel/Read/ReadVariableOp:0(2,conv2d_5/kernel/Initializer/random_uniform:08
?
batch_normalization_6/gamma:0"batch_normalization_6/gamma/Assign1batch_normalization_6/gamma/Read/ReadVariableOp:0(2.batch_normalization_6/gamma/Initializer/ones:08
?
batch_normalization_6/beta:0!batch_normalization_6/beta/Assign0batch_normalization_6/beta/Read/ReadVariableOp:0(2.batch_normalization_6/beta/Initializer/zeros:08
?
conv2d_6/kernel:0conv2d_6/kernel/Assign%conv2d_6/kernel/Read/ReadVariableOp:0(2,conv2d_6/kernel/Initializer/random_uniform:08
?
conv2d_7/kernel:0conv2d_7/kernel/Assign%conv2d_7/kernel/Read/ReadVariableOp:0(2,conv2d_7/kernel/Initializer/random_uniform:08
?
batch_normalization_7/gamma:0"batch_normalization_7/gamma/Assign1batch_normalization_7/gamma/Read/ReadVariableOp:0(2.batch_normalization_7/gamma/Initializer/ones:08
?
batch_normalization_7/beta:0!batch_normalization_7/beta/Assign0batch_normalization_7/beta/Read/ReadVariableOp:0(2.batch_normalization_7/beta/Initializer/zeros:08
?
conv2d_8/kernel:0conv2d_8/kernel/Assign%conv2d_8/kernel/Read/ReadVariableOp:0(2,conv2d_8/kernel/Initializer/random_uniform:08
?
batch_normalization_8/gamma:0"batch_normalization_8/gamma/Assign1batch_normalization_8/gamma/Read/ReadVariableOp:0(2.batch_normalization_8/gamma/Initializer/ones:08
?
batch_normalization_8/beta:0!batch_normalization_8/beta/Assign0batch_normalization_8/beta/Read/ReadVariableOp:0(2.batch_normalization_8/beta/Initializer/zeros:08
?
conv2d_9/kernel:0conv2d_9/kernel/Assign%conv2d_9/kernel/Read/ReadVariableOp:0(2,conv2d_9/kernel/Initializer/random_uniform:08
?
batch_normalization_9/gamma:0"batch_normalization_9/gamma/Assign1batch_normalization_9/gamma/Read/ReadVariableOp:0(2.batch_normalization_9/gamma/Initializer/ones:08
?
batch_normalization_9/beta:0!batch_normalization_9/beta/Assign0batch_normalization_9/beta/Read/ReadVariableOp:0(2.batch_normalization_9/beta/Initializer/zeros:08
?
conv2d_10/kernel:0conv2d_10/kernel/Assign&conv2d_10/kernel/Read/ReadVariableOp:0(2-conv2d_10/kernel/Initializer/random_uniform:08
?
batch_normalization_10/gamma:0#batch_normalization_10/gamma/Assign2batch_normalization_10/gamma/Read/ReadVariableOp:0(2/batch_normalization_10/gamma/Initializer/ones:08
?
batch_normalization_10/beta:0"batch_normalization_10/beta/Assign1batch_normalization_10/beta/Read/ReadVariableOp:0(2/batch_normalization_10/beta/Initializer/zeros:08
?
conv2d_11/kernel:0conv2d_11/kernel/Assign&conv2d_11/kernel/Read/ReadVariableOp:0(2-conv2d_11/kernel/Initializer/random_uniform:08
?
conv2d_12/kernel:0conv2d_12/kernel/Assign&conv2d_12/kernel/Read/ReadVariableOp:0(2-conv2d_12/kernel/Initializer/random_uniform:08
?
batch_normalization_11/gamma:0#batch_normalization_11/gamma/Assign2batch_normalization_11/gamma/Read/ReadVariableOp:0(2/batch_normalization_11/gamma/Initializer/ones:08
?
batch_normalization_11/beta:0"batch_normalization_11/beta/Assign1batch_normalization_11/beta/Read/ReadVariableOp:0(2/batch_normalization_11/beta/Initializer/zeros:08
?
conv2d_13/kernel:0conv2d_13/kernel/Assign&conv2d_13/kernel/Read/ReadVariableOp:0(2-conv2d_13/kernel/Initializer/random_uniform:08
?
batch_normalization_12/gamma:0#batch_normalization_12/gamma/Assign2batch_normalization_12/gamma/Read/ReadVariableOp:0(2/batch_normalization_12/gamma/Initializer/ones:08
?
batch_normalization_12/beta:0"batch_normalization_12/beta/Assign1batch_normalization_12/beta/Read/ReadVariableOp:0(2/batch_normalization_12/beta/Initializer/zeros:08
?
conv2d_14/kernel:0conv2d_14/kernel/Assign&conv2d_14/kernel/Read/ReadVariableOp:0(2-conv2d_14/kernel/Initializer/random_uniform:08
?
batch_normalization_13/gamma:0#batch_normalization_13/gamma/Assign2batch_normalization_13/gamma/Read/ReadVariableOp:0(2/batch_normalization_13/gamma/Initializer/ones:08
?
batch_normalization_13/beta:0"batch_normalization_13/beta/Assign1batch_normalization_13/beta/Read/ReadVariableOp:0(2/batch_normalization_13/beta/Initializer/zeros:08
?
conv2d_15/kernel:0conv2d_15/kernel/Assign&conv2d_15/kernel/Read/ReadVariableOp:0(2-conv2d_15/kernel/Initializer/random_uniform:08
?
batch_normalization_14/gamma:0#batch_normalization_14/gamma/Assign2batch_normalization_14/gamma/Read/ReadVariableOp:0(2/batch_normalization_14/gamma/Initializer/ones:08
?
batch_normalization_14/beta:0"batch_normalization_14/beta/Assign1batch_normalization_14/beta/Read/ReadVariableOp:0(2/batch_normalization_14/beta/Initializer/zeros:08
?
conv2d_16/kernel:0conv2d_16/kernel/Assign&conv2d_16/kernel/Read/ReadVariableOp:0(2-conv2d_16/kernel/Initializer/random_uniform:08
?
conv2d_17/kernel:0conv2d_17/kernel/Assign&conv2d_17/kernel/Read/ReadVariableOp:0(2-conv2d_17/kernel/Initializer/random_uniform:08
?
batch_normalization_15/gamma:0#batch_normalization_15/gamma/Assign2batch_normalization_15/gamma/Read/ReadVariableOp:0(2/batch_normalization_15/gamma/Initializer/ones:08
?
batch_normalization_15/beta:0"batch_normalization_15/beta/Assign1batch_normalization_15/beta/Read/ReadVariableOp:0(2/batch_normalization_15/beta/Initializer/zeros:08
?
conv2d_18/kernel:0conv2d_18/kernel/Assign&conv2d_18/kernel/Read/ReadVariableOp:0(2-conv2d_18/kernel/Initializer/random_uniform:08
?
batch_normalization_16/gamma:0#batch_normalization_16/gamma/Assign2batch_normalization_16/gamma/Read/ReadVariableOp:0(2/batch_normalization_16/gamma/Initializer/ones:08
?
batch_normalization_16/beta:0"batch_normalization_16/beta/Assign1batch_normalization_16/beta/Read/ReadVariableOp:0(2/batch_normalization_16/beta/Initializer/zeros:08
?
conv2d_19/kernel:0conv2d_19/kernel/Assign&conv2d_19/kernel/Read/ReadVariableOp:0(2-conv2d_19/kernel/Initializer/random_uniform:08
?
batch_normalization_17/gamma:0#batch_normalization_17/gamma/Assign2batch_normalization_17/gamma/Read/ReadVariableOp:0(2/batch_normalization_17/gamma/Initializer/ones:08
?
batch_normalization_17/beta:0"batch_normalization_17/beta/Assign1batch_normalization_17/beta/Read/ReadVariableOp:0(2/batch_normalization_17/beta/Initializer/zeros:08
?
conv2d_20/kernel:0conv2d_20/kernel/Assign&conv2d_20/kernel/Read/ReadVariableOp:0(2-conv2d_20/kernel/Initializer/random_uniform:08
?
batch_normalization_18/gamma:0#batch_normalization_18/gamma/Assign2batch_normalization_18/gamma/Read/ReadVariableOp:0(2/batch_normalization_18/gamma/Initializer/ones:08
?
batch_normalization_18/beta:0"batch_normalization_18/beta/Assign1batch_normalization_18/beta/Read/ReadVariableOp:0(2/batch_normalization_18/beta/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08*?
predict?
/
inputs%
data:0?????????@@8
scores.
activation_18/Softmax:0?????????
tensorflow/serving/predict