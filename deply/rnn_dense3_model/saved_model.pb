у╢
цм
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
Т
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
Б
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.5.0-dev202010312v1.12.1-44987-g5b76abd4498щЇ
Д
rnn_dense3_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
шА*$
shared_namernn_dense3_1/kernel
}
'rnn_dense3_1/kernel/Read/ReadVariableOpReadVariableOprnn_dense3_1/kernel* 
_output_shapes
:
шА*
dtype0
{
rnn_dense3_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*"
shared_namernn_dense3_1/bias
t
%rnn_dense3_1/bias/Read/ReadVariableOpReadVariableOprnn_dense3_1/bias*
_output_shapes	
:А*
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
в
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
В
rnn_dense3_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *$
shared_namernn_dense3_2/kernel
{
'rnn_dense3_2/kernel/Read/ReadVariableOpReadVariableOprnn_dense3_2/kernel*
_output_shapes

:@ *
dtype0
z
rnn_dense3_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_namernn_dense3_2/bias
s
%rnn_dense3_2/bias/Read/ReadVariableOpReadVariableOprnn_dense3_2/bias*
_output_shapes
: *
dtype0
О
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma
З
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta
Е
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean
У
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
в
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance
Ы
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0
В
rnn_dense3_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_namernn_dense3_3/kernel
{
'rnn_dense3_3/kernel/Read/ReadVariableOpReadVariableOprnn_dense3_3/kernel*
_output_shapes

: *
dtype0
z
rnn_dense3_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namernn_dense3_3/bias
s
%rnn_dense3_3/bias/Read/ReadVariableOpReadVariableOprnn_dense3_3/bias*
_output_shapes
:*
dtype0
Ж
rnn_dense3_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namernn_dense3_out/kernel

)rnn_dense3_out/kernel/Read/ReadVariableOpReadVariableOprnn_dense3_out/kernel*
_output_shapes

:*
dtype0
~
rnn_dense3_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namernn_dense3_out/bias
w
'rnn_dense3_out/bias/Read/ReadVariableOpReadVariableOprnn_dense3_out/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
д
$rnn_dense3_r1/simple_rnn_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$rnn_dense3_r1/simple_rnn_cell/kernel
Э
8rnn_dense3_r1/simple_rnn_cell/kernel/Read/ReadVariableOpReadVariableOp$rnn_dense3_r1/simple_rnn_cell/kernel*
_output_shapes

:@*
dtype0
╕
.rnn_dense3_r1/simple_rnn_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*?
shared_name0.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel
▒
Brnn_dense3_r1/simple_rnn_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel*
_output_shapes

:@@*
dtype0
Ь
"rnn_dense3_r1/simple_rnn_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"rnn_dense3_r1/simple_rnn_cell/bias
Х
6rnn_dense3_r1/simple_rnn_cell/bias/Read/ReadVariableOpReadVariableOp"rnn_dense3_r1/simple_rnn_cell/bias*
_output_shapes
:@*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
Т
Adam/rnn_dense3_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
шА*+
shared_nameAdam/rnn_dense3_1/kernel/m
Л
.Adam/rnn_dense3_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_1/kernel/m* 
_output_shapes
:
шА*
dtype0
Й
Adam/rnn_dense3_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameAdam/rnn_dense3_1/bias/m
В
,Adam/rnn_dense3_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_1/bias/m*
_output_shapes	
:А*
dtype0
Ь
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m
Х
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m
У
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/m
Х
6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/m
У
5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
Р
Adam/rnn_dense3_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *+
shared_nameAdam/rnn_dense3_2/kernel/m
Й
.Adam/rnn_dense3_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_2/kernel/m*
_output_shapes

:@ *
dtype0
И
Adam/rnn_dense3_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/rnn_dense3_2/bias/m
Б
,Adam/rnn_dense3_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_2/bias/m*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/m
Х
6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/m*
_output_shapes
: *
dtype0
Ъ
!Adam/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/m
У
5Adam/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/m*
_output_shapes
: *
dtype0
Р
Adam/rnn_dense3_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/rnn_dense3_3/kernel/m
Й
.Adam/rnn_dense3_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_3/kernel/m*
_output_shapes

: *
dtype0
И
Adam/rnn_dense3_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/rnn_dense3_3/bias/m
Б
,Adam/rnn_dense3_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_3/bias/m*
_output_shapes
:*
dtype0
Ф
Adam/rnn_dense3_out/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/rnn_dense3_out/kernel/m
Н
0Adam/rnn_dense3_out/kernel/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_out/kernel/m*
_output_shapes

:*
dtype0
М
Adam/rnn_dense3_out/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/rnn_dense3_out/bias/m
Е
.Adam/rnn_dense3_out/bias/m/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_out/bias/m*
_output_shapes
:*
dtype0
▓
+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m
л
?Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m*
_output_shapes

:@*
dtype0
╞
5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*F
shared_name75Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m
┐
IAdam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m*
_output_shapes

:@@*
dtype0
к
)Adam/rnn_dense3_r1/simple_rnn_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/rnn_dense3_r1/simple_rnn_cell/bias/m
г
=Adam/rnn_dense3_r1/simple_rnn_cell/bias/m/Read/ReadVariableOpReadVariableOp)Adam/rnn_dense3_r1/simple_rnn_cell/bias/m*
_output_shapes
:@*
dtype0
Т
Adam/rnn_dense3_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
шА*+
shared_nameAdam/rnn_dense3_1/kernel/v
Л
.Adam/rnn_dense3_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_1/kernel/v* 
_output_shapes
:
шА*
dtype0
Й
Adam/rnn_dense3_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*)
shared_nameAdam/rnn_dense3_1/bias/v
В
,Adam/rnn_dense3_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_1/bias/v*
_output_shapes	
:А*
dtype0
Ь
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v
Х
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:*
dtype0
Ъ
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v
У
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:*
dtype0
Ь
"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_2/gamma/v
Х
6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
Ъ
!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_2/beta/v
У
5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
Р
Adam/rnn_dense3_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *+
shared_nameAdam/rnn_dense3_2/kernel/v
Й
.Adam/rnn_dense3_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_2/kernel/v*
_output_shapes

:@ *
dtype0
И
Adam/rnn_dense3_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/rnn_dense3_2/bias/v
Б
,Adam/rnn_dense3_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_2/bias/v*
_output_shapes
: *
dtype0
Ь
"Adam/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_3/gamma/v
Х
6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_3/gamma/v*
_output_shapes
: *
dtype0
Ъ
!Adam/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_3/beta/v
У
5Adam/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_3/beta/v*
_output_shapes
: *
dtype0
Р
Adam/rnn_dense3_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/rnn_dense3_3/kernel/v
Й
.Adam/rnn_dense3_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_3/kernel/v*
_output_shapes

: *
dtype0
И
Adam/rnn_dense3_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/rnn_dense3_3/bias/v
Б
,Adam/rnn_dense3_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_3/bias/v*
_output_shapes
:*
dtype0
Ф
Adam/rnn_dense3_out/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/rnn_dense3_out/kernel/v
Н
0Adam/rnn_dense3_out/kernel/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_out/kernel/v*
_output_shapes

:*
dtype0
М
Adam/rnn_dense3_out/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/rnn_dense3_out/bias/v
Е
.Adam/rnn_dense3_out/bias/v/Read/ReadVariableOpReadVariableOpAdam/rnn_dense3_out/bias/v*
_output_shapes
:*
dtype0
▓
+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*<
shared_name-+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v
л
?Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v*
_output_shapes

:@*
dtype0
╞
5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*F
shared_name75Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v
┐
IAdam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v*
_output_shapes

:@@*
dtype0
к
)Adam/rnn_dense3_r1/simple_rnn_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)Adam/rnn_dense3_r1/simple_rnn_cell/bias/v
г
=Adam/rnn_dense3_r1/simple_rnn_cell/bias/v/Read/ReadVariableOpReadVariableOp)Adam/rnn_dense3_r1/simple_rnn_cell/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
зo
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*тn
value╪nB╒n B╬n
▀
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
 	keras_api
Ч
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&	variables
'regularization_losses
(trainable_variables
)	keras_api
l
*cell
+
state_spec
,	variables
-regularization_losses
.trainable_variables
/	keras_api
R
0	variables
1regularization_losses
2trainable_variables
3	keras_api
Ч
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
h

=kernel
>bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
R
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
Ч
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
Ф
\iter

]beta_1

^beta_2
	_decay
`learning_ratem└m┴"m┬#m├5m─6m┼=m╞>m╟Hm╚Im╔Pm╩Qm╦Vm╠Wm═am╬bm╧cm╨v╤v╥"v╙#v╘5v╒6v╓=v╫>v╪Hv┘Iv┌Pv█Qv▄Vv▌Wv▐av▀bvрcvс
о
0
1
"2
#3
$4
%5
a6
b7
c8
59
610
711
812
=13
>14
H15
I16
J17
K18
P19
Q20
V21
W22
 
~
0
1
"2
#3
a4
b5
c6
57
68
=9
>10
H11
I12
P13
Q14
V15
W16
н
dlayer_metrics
	variables
regularization_losses
enon_trainable_variables

flayers
trainable_variables
glayer_regularization_losses
hmetrics
 
_]
VARIABLE_VALUErnn_dense3_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErnn_dense3_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
ilayer_metrics
	variables
regularization_losses
jnon_trainable_variables

klayers
trainable_variables
llayer_regularization_losses
mmetrics
 
 
 
н
nlayer_metrics
	variables
regularization_losses
onon_trainable_variables

players
trainable_variables
qlayer_regularization_losses
rmetrics
 
 
 
н
slayer_metrics
	variables
regularization_losses
tnon_trainable_variables

ulayers
trainable_variables
vlayer_regularization_losses
wmetrics
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
$2
%3
 

"0
#1
н
xlayer_metrics
&	variables
'regularization_losses
ynon_trainable_variables

zlayers
(trainable_variables
{layer_regularization_losses
|metrics


akernel
brecurrent_kernel
cbias
}	variables
~regularization_losses
trainable_variables
А	keras_api
 

a0
b1
c2
 

a0
b1
c2
┐
Бlayer_metrics
Вmetrics
,	variables
-regularization_losses
Гnon_trainable_variables
Дlayers
.trainable_variables
 Еlayer_regularization_losses
Жstates
 
 
 
▓
Зlayer_metrics
0	variables
1regularization_losses
Иnon_trainable_variables
Йlayers
2trainable_variables
 Кlayer_regularization_losses
Лmetrics
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

50
61
72
83
 

50
61
▓
Мlayer_metrics
9	variables
:regularization_losses
Нnon_trainable_variables
Оlayers
;trainable_variables
 Пlayer_regularization_losses
Рmetrics
_]
VARIABLE_VALUErnn_dense3_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErnn_dense3_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
▓
Сlayer_metrics
?	variables
@regularization_losses
Тnon_trainable_variables
Уlayers
Atrainable_variables
 Фlayer_regularization_losses
Хmetrics
 
 
 
▓
Цlayer_metrics
C	variables
Dregularization_losses
Чnon_trainable_variables
Шlayers
Etrainable_variables
 Щlayer_regularization_losses
Ъmetrics
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
J2
K3
 

H0
I1
▓
Ыlayer_metrics
L	variables
Mregularization_losses
Ьnon_trainable_variables
Эlayers
Ntrainable_variables
 Юlayer_regularization_losses
Яmetrics
_]
VARIABLE_VALUErnn_dense3_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErnn_dense3_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
▓
аlayer_metrics
R	variables
Sregularization_losses
бnon_trainable_variables
вlayers
Ttrainable_variables
 гlayer_regularization_losses
дmetrics
a_
VARIABLE_VALUErnn_dense3_out/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUErnn_dense3_out/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 

V0
W1
▓
еlayer_metrics
X	variables
Yregularization_losses
жnon_trainable_variables
зlayers
Ztrainable_variables
 иlayer_regularization_losses
йmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$rnn_dense3_r1/simple_rnn_cell/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"rnn_dense3_r1/simple_rnn_cell/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
*
$0
%1
72
83
J4
K5
V
0
1
2
3
4
5
6
7
	8

9
10
11
 

к0
л1
м2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

$0
%1
 
 
 

a0
b1
c2
 

a0
b1
c2
▓
нlayer_metrics
}	variables
~regularization_losses
оnon_trainable_variables
пlayers
trainable_variables
 ░layer_regularization_losses
▒metrics
 
 
 

*0
 
 
 
 
 
 
 
 

70
81
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
8

▓total

│count
┤	variables
╡	keras_api
I

╢total

╖count
╕
_fn_kwargs
╣	variables
║	keras_api
I

╗total

╝count
╜
_fn_kwargs
╛	variables
┐	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

▓0
│1

┤	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

╢0
╖1

╣	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

╗0
╝1

╛	variables
ГА
VARIABLE_VALUEAdam/rnn_dense3_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/rnn_dense3_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/rnn_dense3_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/rnn_dense3_out/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/rnn_dense3_out/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUE5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/rnn_dense3_r1/simple_rnn_cell/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/rnn_dense3_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/rnn_dense3_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"Adam/batch_normalization_3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE!Adam/batch_normalization_3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/rnn_dense3_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/rnn_dense3_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdam/rnn_dense3_out/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdam/rnn_dense3_out/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUE+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUE5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUE)Adam/rnn_dense3_r1/simple_rnn_cell/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
З
"serving_default_rnn_dense3_1_inputPlaceholder*(
_output_shapes
:         ш*
dtype0*
shape:         ш
С
StatefulPartitionedCallStatefulPartitionedCall"serving_default_rnn_dense3_1_inputrnn_dense3_1/kernelrnn_dense3_1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta$rnn_dense3_r1/simple_rnn_cell/kernel"rnn_dense3_r1/simple_rnn_cell/bias.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betarnn_dense3_2/kernelrnn_dense3_2/bias%batch_normalization_3/moving_variancebatch_normalization_3/gamma!batch_normalization_3/moving_meanbatch_normalization_3/betarnn_dense3_3/kernelrnn_dense3_3/biasrnn_dense3_out/kernelrnn_dense3_out/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_84647
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╠
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'rnn_dense3_1/kernel/Read/ReadVariableOp%rnn_dense3_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp'rnn_dense3_2/kernel/Read/ReadVariableOp%rnn_dense3_2/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp'rnn_dense3_3/kernel/Read/ReadVariableOp%rnn_dense3_3/bias/Read/ReadVariableOp)rnn_dense3_out/kernel/Read/ReadVariableOp'rnn_dense3_out/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp8rnn_dense3_r1/simple_rnn_cell/kernel/Read/ReadVariableOpBrnn_dense3_r1/simple_rnn_cell/recurrent_kernel/Read/ReadVariableOp6rnn_dense3_r1/simple_rnn_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp.Adam/rnn_dense3_1/kernel/m/Read/ReadVariableOp,Adam/rnn_dense3_1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp.Adam/rnn_dense3_2/kernel/m/Read/ReadVariableOp,Adam/rnn_dense3_2/bias/m/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_3/beta/m/Read/ReadVariableOp.Adam/rnn_dense3_3/kernel/m/Read/ReadVariableOp,Adam/rnn_dense3_3/bias/m/Read/ReadVariableOp0Adam/rnn_dense3_out/kernel/m/Read/ReadVariableOp.Adam/rnn_dense3_out/bias/m/Read/ReadVariableOp?Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m/Read/ReadVariableOpIAdam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m/Read/ReadVariableOp=Adam/rnn_dense3_r1/simple_rnn_cell/bias/m/Read/ReadVariableOp.Adam/rnn_dense3_1/kernel/v/Read/ReadVariableOp,Adam/rnn_dense3_1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp.Adam/rnn_dense3_2/kernel/v/Read/ReadVariableOp,Adam/rnn_dense3_2/bias/v/Read/ReadVariableOp6Adam/batch_normalization_3/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_3/beta/v/Read/ReadVariableOp.Adam/rnn_dense3_3/kernel/v/Read/ReadVariableOp,Adam/rnn_dense3_3/bias/v/Read/ReadVariableOp0Adam/rnn_dense3_out/kernel/v/Read/ReadVariableOp.Adam/rnn_dense3_out/bias/v/Read/ReadVariableOp?Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v/Read/ReadVariableOpIAdam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v/Read/ReadVariableOp=Adam/rnn_dense3_r1/simple_rnn_cell/bias/v/Read/ReadVariableOpConst*Q
TinJ
H2F	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_86113
ў
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamernn_dense3_1/kernelrnn_dense3_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancernn_dense3_2/kernelrnn_dense3_2/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_variancernn_dense3_3/kernelrnn_dense3_3/biasrnn_dense3_out/kernelrnn_dense3_out/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate$rnn_dense3_r1/simple_rnn_cell/kernel.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel"rnn_dense3_r1/simple_rnn_cell/biastotalcounttotal_1count_1total_2count_2Adam/rnn_dense3_1/kernel/mAdam/rnn_dense3_1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/rnn_dense3_2/kernel/mAdam/rnn_dense3_2/bias/m"Adam/batch_normalization_3/gamma/m!Adam/batch_normalization_3/beta/mAdam/rnn_dense3_3/kernel/mAdam/rnn_dense3_3/bias/mAdam/rnn_dense3_out/kernel/mAdam/rnn_dense3_out/bias/m+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m)Adam/rnn_dense3_r1/simple_rnn_cell/bias/mAdam/rnn_dense3_1/kernel/vAdam/rnn_dense3_1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/rnn_dense3_2/kernel/vAdam/rnn_dense3_2/bias/v"Adam/batch_normalization_3/gamma/v!Adam/batch_normalization_3/beta/vAdam/rnn_dense3_3/kernel/vAdam/rnn_dense3_3/bias/vAdam/rnn_dense3_out/kernel/vAdam/rnn_dense3_out/bias/v+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v5Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v)Adam/rnn_dense3_r1/simple_rnn_cell/bias/v*P
TinI
G2E*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_86327Є╢
┘2
щ
while_body_83761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_simple_rnn_cell_matmul_readvariableop_resource_0;
7while_simple_rnn_cell_biasadd_readvariableop_resource_0<
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_simple_rnn_cell_matmul_readvariableop_resource9
5while_simple_rnn_cell_biasadd_readvariableop_resource:
6while_simple_rnn_cell_matmul_1_readvariableop_resourceИв,while/simple_rnn_cell/BiasAdd/ReadVariableOpв+while/simple_rnn_cell/MatMul/ReadVariableOpв-while/simple_rnn_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╤
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02-
+while/simple_rnn_cell/MatMul/ReadVariableOp▀
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/MatMul╨
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02.
,while/simple_rnn_cell/BiasAdd/ReadVariableOp┘
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/BiasAdd╫
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02/
-while/simple_rnn_cell/MatMul_1/ReadVariableOp╚
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
while/simple_rnn_cell/MatMul_1├
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/addС
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/Tanhт
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ы
while/IdentityIdentitywhile/add_1:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity■
while/Identity_1Identitywhile_while_maximum_iterations-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1э
while/Identity_2Identitywhile/add:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Т
E
)__inference_dropout_2_layer_call_fn_85636

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
у
Б
,__inference_rnn_dense3_3_layer_call_fn_85867

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_842542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
й
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_83604

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         А2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         А2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         А2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╦
е
while_cond_83760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_83760___redundant_placeholder03
/while_while_cond_83760___redundant_placeholder13
/while_while_cond_83760___redundant_placeholder23
/while_while_cond_83760___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
°/
┼
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85672

inputs
assignmovingavg_85647
assignmovingavg_1_85653)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85647*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_85647*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85647*
_output_shapes
:@2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85647*
_output_shapes
:@2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_85647AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85647*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85653*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_85653*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85653*
_output_shapes
:@2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85653*
_output_shapes
:@2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_85653AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85653*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1│
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╕
╥
*__inference_rnn_dense3_layer_call_fn_84473
rnn_dense3_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallrnn_dense3_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_844242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
Ю
b
)__inference_dropout_2_layer_call_fn_85631

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╔
и
5__inference_batch_normalization_1_layer_call_fn_85363

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836782
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ч	
т
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_85877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
р
^
B__inference_reshape_layer_call_and_return_conditional_losses_83584

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:         А2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
│
и
5__inference_batch_normalization_3_layer_call_fn_85834

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_841862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
│0
┼
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85317

inputs
assignmovingavg_85292
assignmovingavg_1_85298)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85292*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_85292*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85292*
_output_shapes
:2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85292*
_output_shapes
:2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_85292AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85292*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85298*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_85298*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85298*
_output_shapes
:2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85298*
_output_shapes
:2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_85298AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85298*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2
batchnorm/add_1╕
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
у
Б
,__inference_rnn_dense3_2_layer_call_fn_85738

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_841042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ч	
т
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_84280

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╠B
Н
rnn_dense3_r1_while_body_849948
4rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter>
:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations#
rnn_dense3_r1_while_placeholder%
!rnn_dense3_r1_while_placeholder_1%
!rnn_dense3_r1_while_placeholder_27
3rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1_0s
ornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0H
Drnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0I
Ernn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0J
Frnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0 
rnn_dense3_r1_while_identity"
rnn_dense3_r1_while_identity_1"
rnn_dense3_r1_while_identity_2"
rnn_dense3_r1_while_identity_3"
rnn_dense3_r1_while_identity_45
1rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1q
mrnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorF
Brnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceG
Crnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourceH
Drnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceИв:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpв9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpв;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp▀
Ernn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Ernn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeз
7rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0rnn_dense3_r1_while_placeholderNrnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype029
7rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem√
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpDrnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02;
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpЧ
*rnn_dense3_r1/while/simple_rnn_cell/MatMulMatMul>rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem:item:0Arnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2,
*rnn_dense3_r1/while/simple_rnn_cell/MatMul·
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpErnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02<
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpС
+rnn_dense3_r1/while/simple_rnn_cell/BiasAddBiasAdd4rnn_dense3_r1/while/simple_rnn_cell/MatMul:product:0Brnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2-
+rnn_dense3_r1/while/simple_rnn_cell/BiasAddБ
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpFrnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02=
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpА
,rnn_dense3_r1/while/simple_rnn_cell/MatMul_1MatMul!rnn_dense3_r1_while_placeholder_2Crnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2.
,rnn_dense3_r1/while/simple_rnn_cell/MatMul_1√
'rnn_dense3_r1/while/simple_rnn_cell/addAddV24rnn_dense3_r1/while/simple_rnn_cell/BiasAdd:output:06rnn_dense3_r1/while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2)
'rnn_dense3_r1/while/simple_rnn_cell/add╗
(rnn_dense3_r1/while/simple_rnn_cell/TanhTanh+rnn_dense3_r1/while/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2*
(rnn_dense3_r1/while/simple_rnn_cell/Tanhи
8rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!rnn_dense3_r1_while_placeholder_1rnn_dense3_r1_while_placeholder,rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemx
rnn_dense3_r1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_dense3_r1/while/add/yб
rnn_dense3_r1/while/addAddV2rnn_dense3_r1_while_placeholder"rnn_dense3_r1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/add|
rnn_dense3_r1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_dense3_r1/while/add_1/y╝
rnn_dense3_r1/while/add_1AddV24rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter$rnn_dense3_r1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/add_1┐
rnn_dense3_r1/while/IdentityIdentityrnn_dense3_r1/while/add_1:z:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/Identityр
rnn_dense3_r1/while/Identity_1Identity:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_1┴
rnn_dense3_r1/while/Identity_2Identityrnn_dense3_r1/while/add:z:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_2ю
rnn_dense3_r1/while/Identity_3IdentityHrnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_3у
rnn_dense3_r1/while/Identity_4Identity,rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2 
rnn_dense3_r1/while/Identity_4"E
rnn_dense3_r1_while_identity%rnn_dense3_r1/while/Identity:output:0"I
rnn_dense3_r1_while_identity_1'rnn_dense3_r1/while/Identity_1:output:0"I
rnn_dense3_r1_while_identity_2'rnn_dense3_r1/while/Identity_2:output:0"I
rnn_dense3_r1_while_identity_3'rnn_dense3_r1/while/Identity_3:output:0"I
rnn_dense3_r1_while_identity_4'rnn_dense3_r1/while/Identity_4:output:0"h
1rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_13rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1_0"М
Crnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourceErnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0"О
Drnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceFrnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0"К
Brnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceDrnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0"р
mrnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2x
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp2v
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp2z
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
ч	
р
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_85729

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:          2
TanhН
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╟
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_85626

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╦
е
while_cond_85408
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_85408___redundant_placeholder03
/while_while_cond_85408___redundant_placeholder13
/while_while_cond_85408___redundant_placeholder23
/while_while_cond_85408___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╡
и
5__inference_batch_normalization_2_layer_call_fn_85718

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840562
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
у

п
rnn_dense3_r1_while_cond_847488
4rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter>
:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations#
rnn_dense3_r1_while_placeholder%
!rnn_dense3_r1_while_placeholder_1%
!rnn_dense3_r1_while_placeholder_2:
6rnn_dense3_r1_while_less_rnn_dense3_r1_strided_slice_1O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84748___redundant_placeholder0O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84748___redundant_placeholder1O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84748___redundant_placeholder2O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84748___redundant_placeholder3 
rnn_dense3_r1_while_identity
╢
rnn_dense3_r1/while/LessLessrnn_dense3_r1_while_placeholder6rnn_dense3_r1_while_less_rnn_dense3_r1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/LessЗ
rnn_dense3_r1/while/IdentityIdentityrnn_dense3_r1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_dense3_r1/while/Identity"E
rnn_dense3_r1_while_identity%rnn_dense3_r1/while/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
ч	
р
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_84104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:          2
TanhН
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Т
╦
#__inference_signature_wrapper_84647
rnn_dense3_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallrnn_dense3_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_835402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
°К
┼ 
__inference__traced_save_86113
file_prefix2
.savev2_rnn_dense3_1_kernel_read_readvariableop0
,savev2_rnn_dense3_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop2
.savev2_rnn_dense3_2_kernel_read_readvariableop0
,savev2_rnn_dense3_2_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop2
.savev2_rnn_dense3_3_kernel_read_readvariableop0
,savev2_rnn_dense3_3_bias_read_readvariableop4
0savev2_rnn_dense3_out_kernel_read_readvariableop2
.savev2_rnn_dense3_out_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopC
?savev2_rnn_dense3_r1_simple_rnn_cell_kernel_read_readvariableopM
Isavev2_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_read_readvariableopA
=savev2_rnn_dense3_r1_simple_rnn_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop9
5savev2_adam_rnn_dense3_1_kernel_m_read_readvariableop7
3savev2_adam_rnn_dense3_1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop9
5savev2_adam_rnn_dense3_2_kernel_m_read_readvariableop7
3savev2_adam_rnn_dense3_2_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_m_read_readvariableop9
5savev2_adam_rnn_dense3_3_kernel_m_read_readvariableop7
3savev2_adam_rnn_dense3_3_bias_m_read_readvariableop;
7savev2_adam_rnn_dense3_out_kernel_m_read_readvariableop9
5savev2_adam_rnn_dense3_out_bias_m_read_readvariableopJ
Fsavev2_adam_rnn_dense3_r1_simple_rnn_cell_kernel_m_read_readvariableopT
Psavev2_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_m_read_readvariableopH
Dsavev2_adam_rnn_dense3_r1_simple_rnn_cell_bias_m_read_readvariableop9
5savev2_adam_rnn_dense3_1_kernel_v_read_readvariableop7
3savev2_adam_rnn_dense3_1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop9
5savev2_adam_rnn_dense3_2_kernel_v_read_readvariableop7
3savev2_adam_rnn_dense3_2_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_3_beta_v_read_readvariableop9
5savev2_adam_rnn_dense3_3_kernel_v_read_readvariableop7
3savev2_adam_rnn_dense3_3_bias_v_read_readvariableop;
7savev2_adam_rnn_dense3_out_kernel_v_read_readvariableop9
5savev2_adam_rnn_dense3_out_bias_v_read_readvariableopJ
Fsavev2_adam_rnn_dense3_r1_simple_rnn_cell_kernel_v_read_readvariableopT
Psavev2_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_v_read_readvariableopH
Dsavev2_adam_rnn_dense3_r1_simple_rnn_cell_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameс$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*є#
valueщ#Bц#EB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesХ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*Я
valueХBТEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╚
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_rnn_dense3_1_kernel_read_readvariableop,savev2_rnn_dense3_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop.savev2_rnn_dense3_2_kernel_read_readvariableop,savev2_rnn_dense3_2_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop.savev2_rnn_dense3_3_kernel_read_readvariableop,savev2_rnn_dense3_3_bias_read_readvariableop0savev2_rnn_dense3_out_kernel_read_readvariableop.savev2_rnn_dense3_out_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop?savev2_rnn_dense3_r1_simple_rnn_cell_kernel_read_readvariableopIsavev2_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_read_readvariableop=savev2_rnn_dense3_r1_simple_rnn_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop5savev2_adam_rnn_dense3_1_kernel_m_read_readvariableop3savev2_adam_rnn_dense3_1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop5savev2_adam_rnn_dense3_2_kernel_m_read_readvariableop3savev2_adam_rnn_dense3_2_bias_m_read_readvariableop=savev2_adam_batch_normalization_3_gamma_m_read_readvariableop<savev2_adam_batch_normalization_3_beta_m_read_readvariableop5savev2_adam_rnn_dense3_3_kernel_m_read_readvariableop3savev2_adam_rnn_dense3_3_bias_m_read_readvariableop7savev2_adam_rnn_dense3_out_kernel_m_read_readvariableop5savev2_adam_rnn_dense3_out_bias_m_read_readvariableopFsavev2_adam_rnn_dense3_r1_simple_rnn_cell_kernel_m_read_readvariableopPsavev2_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_m_read_readvariableopDsavev2_adam_rnn_dense3_r1_simple_rnn_cell_bias_m_read_readvariableop5savev2_adam_rnn_dense3_1_kernel_v_read_readvariableop3savev2_adam_rnn_dense3_1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop5savev2_adam_rnn_dense3_2_kernel_v_read_readvariableop3savev2_adam_rnn_dense3_2_bias_v_read_readvariableop=savev2_adam_batch_normalization_3_gamma_v_read_readvariableop<savev2_adam_batch_normalization_3_beta_v_read_readvariableop5savev2_adam_rnn_dense3_3_kernel_v_read_readvariableop3savev2_adam_rnn_dense3_3_bias_v_read_readvariableop7savev2_adam_rnn_dense3_out_kernel_v_read_readvariableop5savev2_adam_rnn_dense3_out_bias_v_read_readvariableopFsavev2_adam_rnn_dense3_r1_simple_rnn_cell_kernel_v_read_readvariableopPsavev2_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_v_read_readvariableopDsavev2_adam_rnn_dense3_r1_simple_rnn_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *S
dtypesI
G2E	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╓
_input_shapes─
┴: :
шА:А:::::@:@:@:@:@ : : : : : : :::: : : : : :@:@@:@: : : : : : :
шА:А:::@:@:@ : : : : ::::@:@@:@:
шА:А:::@:@:@ : : : : ::::@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
шА:!

_output_shapes	
:А: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :&#"
 
_output_shapes
:
шА:!$

_output_shapes	
:А: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:@: (

_output_shapes
:@:$) 

_output_shapes

:@ : *

_output_shapes
: : +

_output_shapes
: : ,

_output_shapes
: :$- 

_output_shapes

: : .

_output_shapes
::$/ 

_output_shapes

:: 0

_output_shapes
::$1 

_output_shapes

:@:$2 

_output_shapes

:@@: 3

_output_shapes
:@:&4"
 
_output_shapes
:
шА:!5

_output_shapes	
:А: 6

_output_shapes
:: 7

_output_shapes
:: 8

_output_shapes
:@: 9

_output_shapes
:@:$: 

_output_shapes

:@ : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :$> 

_output_shapes

: : ?

_output_shapes
::$@ 

_output_shapes

:: A

_output_shapes
::$B 

_output_shapes

:@:$C 

_output_shapes

:@@: D

_output_shapes
:@:E

_output_shapes
: 
А
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_83982

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
л?
с
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84359
rnn_dense3_1_input
rnn_dense3_1_84300
rnn_dense3_1_84302
batch_normalization_1_84307
batch_normalization_1_84309
batch_normalization_1_84311
batch_normalization_1_84313
rnn_dense3_r1_84316
rnn_dense3_r1_84318
rnn_dense3_r1_84320
batch_normalization_2_84324
batch_normalization_2_84326
batch_normalization_2_84328
batch_normalization_2_84330
rnn_dense3_2_84333
rnn_dense3_2_84335
batch_normalization_3_84339
batch_normalization_3_84341
batch_normalization_3_84343
batch_normalization_3_84345
rnn_dense3_3_84348
rnn_dense3_3_84350
rnn_dense3_out_84353
rnn_dense3_out_84355
identityИв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв$rnn_dense3_1/StatefulPartitionedCallв$rnn_dense3_2/StatefulPartitionedCallв$rnn_dense3_3/StatefulPartitionedCallв&rnn_dense3_out/StatefulPartitionedCallв%rnn_dense3_r1/StatefulPartitionedCall▓
$rnn_dense3_1/StatefulPartitionedCallStatefulPartitionedCallrnn_dense3_1_inputrnn_dense3_1_84300rnn_dense3_1_84302*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_835552&
$rnn_dense3_1/StatefulPartitionedCall№
reshape/PartitionedCallPartitionedCall-rnn_dense3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_835842
reshape/PartitionedCallї
dropout_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836092
dropout_1/PartitionedCall▒
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_84307batch_normalization_1_84309batch_normalization_1_84311batch_normalization_1_84313*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836782/
-batch_normalization_1/StatefulPartitionedCallё
%rnn_dense3_r1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0rnn_dense3_r1_84316rnn_dense3_r1_84318rnn_dense3_r1_84320*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_839392'
%rnn_dense3_r1/StatefulPartitionedCall■
dropout_2/PartitionedCallPartitionedCall.rnn_dense3_r1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839872
dropout_2/PartitionedCallм
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0batch_normalization_2_84324batch_normalization_2_84326batch_normalization_2_84328batch_normalization_2_84330*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840562/
-batch_normalization_2/StatefulPartitionedCall╒
$rnn_dense3_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0rnn_dense3_2_84333rnn_dense3_2_84335*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_841042&
$rnn_dense3_2/StatefulPartitionedCall¤
dropout_3/PartitionedCallPartitionedCall-rnn_dense3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841372
dropout_3/PartitionedCallм
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0batch_normalization_3_84339batch_normalization_3_84341batch_normalization_3_84343batch_normalization_3_84345*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_842062/
-batch_normalization_3/StatefulPartitionedCall╒
$rnn_dense3_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0rnn_dense3_3_84348rnn_dense3_3_84350*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_842542&
$rnn_dense3_3/StatefulPartitionedCall╓
&rnn_dense3_out/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_3/StatefulPartitionedCall:output:0rnn_dense3_out_84353rnn_dense3_out_84355*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_842802(
&rnn_dense3_out/StatefulPartitionedCall┘
IdentityIdentity/rnn_dense3_out/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall%^rnn_dense3_1/StatefulPartitionedCall%^rnn_dense3_2/StatefulPartitionedCall%^rnn_dense3_3/StatefulPartitionedCall'^rnn_dense3_out/StatefulPartitionedCall&^rnn_dense3_r1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2L
$rnn_dense3_1/StatefulPartitionedCall$rnn_dense3_1/StatefulPartitionedCall2L
$rnn_dense3_2/StatefulPartitionedCall$rnn_dense3_2/StatefulPartitionedCall2L
$rnn_dense3_3/StatefulPartitionedCall$rnn_dense3_3/StatefulPartitionedCall2P
&rnn_dense3_out/StatefulPartitionedCall&rnn_dense3_out/StatefulPartitionedCall2N
%rnn_dense3_r1/StatefulPartitionedCall%rnn_dense3_r1/StatefulPartitionedCall:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
П
С
-__inference_rnn_dense3_r1_layer_call_fn_85609

inputs
unknown
	unknown_0
	unknown_1
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_839392
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_83609

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         А2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:         А:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╟
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_83987

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:         @2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
А
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_85621

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
в
А
)rnn_dense3_rnn_dense3_r1_while_cond_83419N
Jrnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_loop_counterT
Prnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_maximum_iterations.
*rnn_dense3_rnn_dense3_r1_while_placeholder0
,rnn_dense3_rnn_dense3_r1_while_placeholder_10
,rnn_dense3_rnn_dense3_r1_while_placeholder_2P
Lrnn_dense3_rnn_dense3_r1_while_less_rnn_dense3_rnn_dense3_r1_strided_slice_1e
arnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_cond_83419___redundant_placeholder0e
arnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_cond_83419___redundant_placeholder1e
arnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_cond_83419___redundant_placeholder2e
arnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_cond_83419___redundant_placeholder3+
'rnn_dense3_rnn_dense3_r1_while_identity
э
#rnn_dense3/rnn_dense3_r1/while/LessLess*rnn_dense3_rnn_dense3_r1_while_placeholderLrnn_dense3_rnn_dense3_r1_while_less_rnn_dense3_rnn_dense3_r1_strided_slice_1*
T0*
_output_shapes
: 2%
#rnn_dense3/rnn_dense3_r1/while/Lessи
'rnn_dense3/rnn_dense3_r1/while/IdentityIdentity'rnn_dense3/rnn_dense3_r1/while/Less:z:0*
T0
*
_output_shapes
: 2)
'rnn_dense3/rnn_dense3_r1/while/Identity"[
'rnn_dense3_rnn_dense3_r1_while_identity0rnn_dense3/rnn_dense3_r1/while/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
МG
Й
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85475

inputs2
.simple_rnn_cell_matmul_readvariableop_resource3
/simple_rnn_cell_biasadd_readvariableop_resource4
0simple_rnn_cell_matmul_1_readvariableop_resource
identityИв&simple_rnn_cell/BiasAdd/ReadVariableOpв%simple_rnn_cell/MatMul/ReadVariableOpв'simple_rnn_cell/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:А         2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2╜
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%simple_rnn_cell/MatMul/ReadVariableOp╡
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul╝
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&simple_rnn_cell/BiasAdd/ReadVariableOp┴
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/BiasAdd├
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'simple_rnn_cell/MatMul_1/ReadVariableOp▒
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul_1л
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/add
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/TanhП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_85409*
condR
while_cond_85408*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
transpose_1я
IdentityIdentitystrided_slice_3:output:0'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
█
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_85271

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         А2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         А2

Identity_1"!

identity_1Identity_1:output:0*+
_input_shapes
:         А:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╫Р
ь
 __inference__wrapped_model_83540
rnn_dense3_1_input:
6rnn_dense3_rnn_dense3_1_matmul_readvariableop_resource;
7rnn_dense3_rnn_dense3_1_biasadd_readvariableop_resourceF
Brnn_dense3_batch_normalization_1_batchnorm_readvariableop_resourceJ
Frnn_dense3_batch_normalization_1_batchnorm_mul_readvariableop_resourceH
Drnn_dense3_batch_normalization_1_batchnorm_readvariableop_1_resourceH
Drnn_dense3_batch_normalization_1_batchnorm_readvariableop_2_resourceK
Grnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resourceL
Hrnn_dense3_rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resourceM
Irnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resourceF
Brnn_dense3_batch_normalization_2_batchnorm_readvariableop_resourceJ
Frnn_dense3_batch_normalization_2_batchnorm_mul_readvariableop_resourceH
Drnn_dense3_batch_normalization_2_batchnorm_readvariableop_1_resourceH
Drnn_dense3_batch_normalization_2_batchnorm_readvariableop_2_resource:
6rnn_dense3_rnn_dense3_2_matmul_readvariableop_resource;
7rnn_dense3_rnn_dense3_2_biasadd_readvariableop_resourceF
Brnn_dense3_batch_normalization_3_batchnorm_readvariableop_resourceJ
Frnn_dense3_batch_normalization_3_batchnorm_mul_readvariableop_resourceH
Drnn_dense3_batch_normalization_3_batchnorm_readvariableop_1_resourceH
Drnn_dense3_batch_normalization_3_batchnorm_readvariableop_2_resource:
6rnn_dense3_rnn_dense3_3_matmul_readvariableop_resource;
7rnn_dense3_rnn_dense3_3_biasadd_readvariableop_resource<
8rnn_dense3_rnn_dense3_out_matmul_readvariableop_resource=
9rnn_dense3_rnn_dense3_out_biasadd_readvariableop_resource
identityИв9rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOpв;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1в;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2в=rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOpв9rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOpв;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1в;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2в=rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOpв9rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOpв;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1в;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2в=rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOpв.rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOpв-rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOpв.rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOpв-rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOpв.rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOpв-rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOpв0rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOpв/rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOpв?rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpв>rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpв@rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpвrnn_dense3/rnn_dense3_r1/while╫
-rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOpReadVariableOp6rnn_dense3_rnn_dense3_1_matmul_readvariableop_resource* 
_output_shapes
:
шА*
dtype02/
-rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOp╚
rnn_dense3/rnn_dense3_1/MatMulMatMulrnn_dense3_1_input5rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2 
rnn_dense3/rnn_dense3_1/MatMul╒
.rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOpReadVariableOp7rnn_dense3_rnn_dense3_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype020
.rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOpт
rnn_dense3/rnn_dense3_1/BiasAddBiasAdd(rnn_dense3/rnn_dense3_1/MatMul:product:06rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2!
rnn_dense3/rnn_dense3_1/BiasAddб
rnn_dense3/rnn_dense3_1/TanhTanh(rnn_dense3/rnn_dense3_1/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
rnn_dense3/rnn_dense3_1/TanhД
rnn_dense3/reshape/ShapeShape rnn_dense3/rnn_dense3_1/Tanh:y:0*
T0*
_output_shapes
:2
rnn_dense3/reshape/ShapeЪ
&rnn_dense3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&rnn_dense3/reshape/strided_slice/stackЮ
(rnn_dense3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(rnn_dense3/reshape/strided_slice/stack_1Ю
(rnn_dense3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(rnn_dense3/reshape/strided_slice/stack_2╘
 rnn_dense3/reshape/strided_sliceStridedSlice!rnn_dense3/reshape/Shape:output:0/rnn_dense3/reshape/strided_slice/stack:output:01rnn_dense3/reshape/strided_slice/stack_1:output:01rnn_dense3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 rnn_dense3/reshape/strided_sliceЛ
"rnn_dense3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А2$
"rnn_dense3/reshape/Reshape/shape/1К
"rnn_dense3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"rnn_dense3/reshape/Reshape/shape/2 
 rnn_dense3/reshape/Reshape/shapePack)rnn_dense3/reshape/strided_slice:output:0+rnn_dense3/reshape/Reshape/shape/1:output:0+rnn_dense3/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2"
 rnn_dense3/reshape/Reshape/shape╟
rnn_dense3/reshape/ReshapeReshape rnn_dense3/rnn_dense3_1/Tanh:y:0)rnn_dense3/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:         А2
rnn_dense3/reshape/Reshapeж
rnn_dense3/dropout_1/IdentityIdentity#rnn_dense3/reshape/Reshape:output:0*
T0*,
_output_shapes
:         А2
rnn_dense3/dropout_1/Identityї
9rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpBrnn_dense3_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02;
9rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOpй
0rnn_dense3/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0rnn_dense3/batch_normalization_1/batchnorm/add/yМ
.rnn_dense3/batch_normalization_1/batchnorm/addAddV2Arnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp:value:09rnn_dense3/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:20
.rnn_dense3/batch_normalization_1/batchnorm/add╞
0rnn_dense3/batch_normalization_1/batchnorm/RsqrtRsqrt2rnn_dense3/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:22
0rnn_dense3/batch_normalization_1/batchnorm/RsqrtБ
=rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpFrnn_dense3_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02?
=rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOpЙ
.rnn_dense3/batch_normalization_1/batchnorm/mulMul4rnn_dense3/batch_normalization_1/batchnorm/Rsqrt:y:0Ernn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:20
.rnn_dense3/batch_normalization_1/batchnorm/mul■
0rnn_dense3/batch_normalization_1/batchnorm/mul_1Mul&rnn_dense3/dropout_1/Identity:output:02rnn_dense3/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А22
0rnn_dense3/batch_normalization_1/batchnorm/mul_1√
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpDrnn_dense3_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02=
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1Й
0rnn_dense3/batch_normalization_1/batchnorm/mul_2MulCrnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1:value:02rnn_dense3/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:22
0rnn_dense3/batch_normalization_1/batchnorm/mul_2√
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpDrnn_dense3_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02=
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2З
.rnn_dense3/batch_normalization_1/batchnorm/subSubCrnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2:value:04rnn_dense3/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:20
.rnn_dense3/batch_normalization_1/batchnorm/subО
0rnn_dense3/batch_normalization_1/batchnorm/add_1AddV24rnn_dense3/batch_normalization_1/batchnorm/mul_1:z:02rnn_dense3/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А22
0rnn_dense3/batch_normalization_1/batchnorm/add_1д
rnn_dense3/rnn_dense3_r1/ShapeShape4rnn_dense3/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2 
rnn_dense3/rnn_dense3_r1/Shapeж
,rnn_dense3/rnn_dense3_r1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,rnn_dense3/rnn_dense3_r1/strided_slice/stackк
.rnn_dense3/rnn_dense3_r1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.rnn_dense3/rnn_dense3_r1/strided_slice/stack_1к
.rnn_dense3/rnn_dense3_r1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.rnn_dense3/rnn_dense3_r1/strided_slice/stack_2°
&rnn_dense3/rnn_dense3_r1/strided_sliceStridedSlice'rnn_dense3/rnn_dense3_r1/Shape:output:05rnn_dense3/rnn_dense3_r1/strided_slice/stack:output:07rnn_dense3/rnn_dense3_r1/strided_slice/stack_1:output:07rnn_dense3/rnn_dense3_r1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&rnn_dense3/rnn_dense3_r1/strided_sliceО
$rnn_dense3/rnn_dense3_r1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2&
$rnn_dense3/rnn_dense3_r1/zeros/mul/y╨
"rnn_dense3/rnn_dense3_r1/zeros/mulMul/rnn_dense3/rnn_dense3_r1/strided_slice:output:0-rnn_dense3/rnn_dense3_r1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2$
"rnn_dense3/rnn_dense3_r1/zeros/mulС
%rnn_dense3/rnn_dense3_r1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2'
%rnn_dense3/rnn_dense3_r1/zeros/Less/y╦
#rnn_dense3/rnn_dense3_r1/zeros/LessLess&rnn_dense3/rnn_dense3_r1/zeros/mul:z:0.rnn_dense3/rnn_dense3_r1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2%
#rnn_dense3/rnn_dense3_r1/zeros/LessФ
'rnn_dense3/rnn_dense3_r1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2)
'rnn_dense3/rnn_dense3_r1/zeros/packed/1ч
%rnn_dense3/rnn_dense3_r1/zeros/packedPack/rnn_dense3/rnn_dense3_r1/strided_slice:output:00rnn_dense3/rnn_dense3_r1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2'
%rnn_dense3/rnn_dense3_r1/zeros/packedС
$rnn_dense3/rnn_dense3_r1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$rnn_dense3/rnn_dense3_r1/zeros/Const┘
rnn_dense3/rnn_dense3_r1/zerosFill.rnn_dense3/rnn_dense3_r1/zeros/packed:output:0-rnn_dense3/rnn_dense3_r1/zeros/Const:output:0*
T0*'
_output_shapes
:         @2 
rnn_dense3/rnn_dense3_r1/zerosз
'rnn_dense3/rnn_dense3_r1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'rnn_dense3/rnn_dense3_r1/transpose/permЇ
"rnn_dense3/rnn_dense3_r1/transpose	Transpose4rnn_dense3/batch_normalization_1/batchnorm/add_1:z:00rnn_dense3/rnn_dense3_r1/transpose/perm:output:0*
T0*,
_output_shapes
:А         2$
"rnn_dense3/rnn_dense3_r1/transposeЪ
 rnn_dense3/rnn_dense3_r1/Shape_1Shape&rnn_dense3/rnn_dense3_r1/transpose:y:0*
T0*
_output_shapes
:2"
 rnn_dense3/rnn_dense3_r1/Shape_1к
.rnn_dense3/rnn_dense3_r1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.rnn_dense3/rnn_dense3_r1/strided_slice_1/stackо
0rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_1о
0rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_2Д
(rnn_dense3/rnn_dense3_r1/strided_slice_1StridedSlice)rnn_dense3/rnn_dense3_r1/Shape_1:output:07rnn_dense3/rnn_dense3_r1/strided_slice_1/stack:output:09rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_1:output:09rnn_dense3/rnn_dense3_r1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(rnn_dense3/rnn_dense3_r1/strided_slice_1╖
4rnn_dense3/rnn_dense3_r1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         26
4rnn_dense3/rnn_dense3_r1/TensorArrayV2/element_shapeЦ
&rnn_dense3/rnn_dense3_r1/TensorArrayV2TensorListReserve=rnn_dense3/rnn_dense3_r1/TensorArrayV2/element_shape:output:01rnn_dense3/rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02(
&rnn_dense3/rnn_dense3_r1/TensorArrayV2ё
Nrnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2P
Nrnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape▄
@rnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor&rnn_dense3/rnn_dense3_r1/transpose:y:0Wrnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02B
@rnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorк
.rnn_dense3/rnn_dense3_r1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.rnn_dense3/rnn_dense3_r1/strided_slice_2/stackо
0rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_1о
0rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_2Т
(rnn_dense3/rnn_dense3_r1/strided_slice_2StridedSlice&rnn_dense3/rnn_dense3_r1/transpose:y:07rnn_dense3/rnn_dense3_r1/strided_slice_2/stack:output:09rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_1:output:09rnn_dense3/rnn_dense3_r1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2*
(rnn_dense3/rnn_dense3_r1/strided_slice_2И
>rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpGrnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02@
>rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpЩ
/rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMulMatMul1rnn_dense3/rnn_dense3_r1/strided_slice_2:output:0Frnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @21
/rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMulЗ
?rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpHrnn_dense3_rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02A
?rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpе
0rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAddBiasAdd9rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul:product:0Grnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @22
0rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAddО
@rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpIrnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02B
@rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpХ
1rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1MatMul'rnn_dense3/rnn_dense3_r1/zeros:output:0Hrnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @23
1rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1П
,rnn_dense3/rnn_dense3_r1/simple_rnn_cell/addAddV29rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd:output:0;rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2.
,rnn_dense3/rnn_dense3_r1/simple_rnn_cell/add╩
-rnn_dense3/rnn_dense3_r1/simple_rnn_cell/TanhTanh0rnn_dense3/rnn_dense3_r1/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2/
-rnn_dense3/rnn_dense3_r1/simple_rnn_cell/Tanh┴
6rnn_dense3/rnn_dense3_r1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   28
6rnn_dense3/rnn_dense3_r1/TensorArrayV2_1/element_shapeЬ
(rnn_dense3/rnn_dense3_r1/TensorArrayV2_1TensorListReserve?rnn_dense3/rnn_dense3_r1/TensorArrayV2_1/element_shape:output:01rnn_dense3/rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02*
(rnn_dense3/rnn_dense3_r1/TensorArrayV2_1А
rnn_dense3/rnn_dense3_r1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_dense3/rnn_dense3_r1/time▒
1rnn_dense3/rnn_dense3_r1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         23
1rnn_dense3/rnn_dense3_r1/while/maximum_iterationsЬ
+rnn_dense3/rnn_dense3_r1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2-
+rnn_dense3/rnn_dense3_r1/while/loop_counterЪ
rnn_dense3/rnn_dense3_r1/whileWhile4rnn_dense3/rnn_dense3_r1/while/loop_counter:output:0:rnn_dense3/rnn_dense3_r1/while/maximum_iterations:output:0&rnn_dense3/rnn_dense3_r1/time:output:01rnn_dense3/rnn_dense3_r1/TensorArrayV2_1:handle:0'rnn_dense3/rnn_dense3_r1/zeros:output:01rnn_dense3/rnn_dense3_r1/strided_slice_1:output:0Prnn_dense3/rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor:output_handle:0Grnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resourceHrnn_dense3_rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resourceIrnn_dense3_rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*5
body-R+
)rnn_dense3_rnn_dense3_r1_while_body_83420*5
cond-R+
)rnn_dense3_rnn_dense3_r1_while_cond_83419*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2 
rnn_dense3/rnn_dense3_r1/whileч
Irnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2K
Irnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shape═
;rnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStackTensorListStack'rnn_dense3/rnn_dense3_r1/while:output:3Rrnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype02=
;rnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack│
.rnn_dense3/rnn_dense3_r1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         20
.rnn_dense3/rnn_dense3_r1/strided_slice_3/stackо
0rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_1о
0rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_2░
(rnn_dense3/rnn_dense3_r1/strided_slice_3StridedSliceDrnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:07rnn_dense3/rnn_dense3_r1/strided_slice_3/stack:output:09rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_1:output:09rnn_dense3/rnn_dense3_r1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2*
(rnn_dense3/rnn_dense3_r1/strided_slice_3л
)rnn_dense3/rnn_dense3_r1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)rnn_dense3/rnn_dense3_r1/transpose_1/permК
$rnn_dense3/rnn_dense3_r1/transpose_1	TransposeDrnn_dense3/rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:02rnn_dense3/rnn_dense3_r1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2&
$rnn_dense3/rnn_dense3_r1/transpose_1п
rnn_dense3/dropout_2/IdentityIdentity1rnn_dense3/rnn_dense3_r1/strided_slice_3:output:0*
T0*'
_output_shapes
:         @2
rnn_dense3/dropout_2/Identityї
9rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpBrnn_dense3_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02;
9rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOpй
0rnn_dense3/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0rnn_dense3/batch_normalization_2/batchnorm/add/yМ
.rnn_dense3/batch_normalization_2/batchnorm/addAddV2Arnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp:value:09rnn_dense3/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@20
.rnn_dense3/batch_normalization_2/batchnorm/add╞
0rnn_dense3/batch_normalization_2/batchnorm/RsqrtRsqrt2rnn_dense3/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@22
0rnn_dense3/batch_normalization_2/batchnorm/RsqrtБ
=rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpFrnn_dense3_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02?
=rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOpЙ
.rnn_dense3/batch_normalization_2/batchnorm/mulMul4rnn_dense3/batch_normalization_2/batchnorm/Rsqrt:y:0Ernn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@20
.rnn_dense3/batch_normalization_2/batchnorm/mul∙
0rnn_dense3/batch_normalization_2/batchnorm/mul_1Mul&rnn_dense3/dropout_2/Identity:output:02rnn_dense3/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @22
0rnn_dense3/batch_normalization_2/batchnorm/mul_1√
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpDrnn_dense3_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02=
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1Й
0rnn_dense3/batch_normalization_2/batchnorm/mul_2MulCrnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1:value:02rnn_dense3/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@22
0rnn_dense3/batch_normalization_2/batchnorm/mul_2√
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpDrnn_dense3_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02=
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2З
.rnn_dense3/batch_normalization_2/batchnorm/subSubCrnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2:value:04rnn_dense3/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@20
.rnn_dense3/batch_normalization_2/batchnorm/subЙ
0rnn_dense3/batch_normalization_2/batchnorm/add_1AddV24rnn_dense3/batch_normalization_2/batchnorm/mul_1:z:02rnn_dense3/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @22
0rnn_dense3/batch_normalization_2/batchnorm/add_1╒
-rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOpReadVariableOp6rnn_dense3_rnn_dense3_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOpщ
rnn_dense3/rnn_dense3_2/MatMulMatMul4rnn_dense3/batch_normalization_2/batchnorm/add_1:z:05rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2 
rnn_dense3/rnn_dense3_2/MatMul╘
.rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOpReadVariableOp7rnn_dense3_rnn_dense3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOpс
rnn_dense3/rnn_dense3_2/BiasAddBiasAdd(rnn_dense3/rnn_dense3_2/MatMul:product:06rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2!
rnn_dense3/rnn_dense3_2/BiasAddа
rnn_dense3/rnn_dense3_2/TanhTanh(rnn_dense3/rnn_dense3_2/BiasAdd:output:0*
T0*'
_output_shapes
:          2
rnn_dense3/rnn_dense3_2/TanhЮ
rnn_dense3/dropout_3/IdentityIdentity rnn_dense3/rnn_dense3_2/Tanh:y:0*
T0*'
_output_shapes
:          2
rnn_dense3/dropout_3/Identityї
9rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOpReadVariableOpBrnn_dense3_batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02;
9rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOpй
0rnn_dense3/batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:22
0rnn_dense3/batch_normalization_3/batchnorm/add/yМ
.rnn_dense3/batch_normalization_3/batchnorm/addAddV2Arnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp:value:09rnn_dense3/batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 20
.rnn_dense3/batch_normalization_3/batchnorm/add╞
0rnn_dense3/batch_normalization_3/batchnorm/RsqrtRsqrt2rnn_dense3/batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 22
0rnn_dense3/batch_normalization_3/batchnorm/RsqrtБ
=rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpFrnn_dense3_batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02?
=rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOpЙ
.rnn_dense3/batch_normalization_3/batchnorm/mulMul4rnn_dense3/batch_normalization_3/batchnorm/Rsqrt:y:0Ernn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 20
.rnn_dense3/batch_normalization_3/batchnorm/mul∙
0rnn_dense3/batch_normalization_3/batchnorm/mul_1Mul&rnn_dense3/dropout_3/Identity:output:02rnn_dense3/batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:          22
0rnn_dense3/batch_normalization_3/batchnorm/mul_1√
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOpDrnn_dense3_batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02=
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1Й
0rnn_dense3/batch_normalization_3/batchnorm/mul_2MulCrnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1:value:02rnn_dense3/batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 22
0rnn_dense3/batch_normalization_3/batchnorm/mul_2√
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOpDrnn_dense3_batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02=
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2З
.rnn_dense3/batch_normalization_3/batchnorm/subSubCrnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2:value:04rnn_dense3/batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 20
.rnn_dense3/batch_normalization_3/batchnorm/subЙ
0rnn_dense3/batch_normalization_3/batchnorm/add_1AddV24rnn_dense3/batch_normalization_3/batchnorm/mul_1:z:02rnn_dense3/batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:          22
0rnn_dense3/batch_normalization_3/batchnorm/add_1╒
-rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOpReadVariableOp6rnn_dense3_rnn_dense3_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOpщ
rnn_dense3/rnn_dense3_3/MatMulMatMul4rnn_dense3/batch_normalization_3/batchnorm/add_1:z:05rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
rnn_dense3/rnn_dense3_3/MatMul╘
.rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOpReadVariableOp7rnn_dense3_rnn_dense3_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOpс
rnn_dense3/rnn_dense3_3/BiasAddBiasAdd(rnn_dense3/rnn_dense3_3/MatMul:product:06rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
rnn_dense3/rnn_dense3_3/BiasAddа
rnn_dense3/rnn_dense3_3/TanhTanh(rnn_dense3/rnn_dense3_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rnn_dense3/rnn_dense3_3/Tanh█
/rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOpReadVariableOp8rnn_dense3_rnn_dense3_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype021
/rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOp█
 rnn_dense3/rnn_dense3_out/MatMulMatMul rnn_dense3/rnn_dense3_3/Tanh:y:07rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 rnn_dense3/rnn_dense3_out/MatMul┌
0rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOpReadVariableOp9rnn_dense3_rnn_dense3_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOpщ
!rnn_dense3/rnn_dense3_out/BiasAddBiasAdd*rnn_dense3/rnn_dense3_out/MatMul:product:08rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!rnn_dense3/rnn_dense3_out/BiasAdd╒
IdentityIdentity*rnn_dense3/rnn_dense3_out/BiasAdd:output:0:^rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp<^rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1<^rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2>^rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOp:^rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp<^rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1<^rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2>^rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOp:^rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp<^rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1<^rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2>^rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOp/^rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOp.^rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOp/^rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOp.^rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOp/^rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOp.^rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOp1^rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOp0^rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOp@^rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp?^rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpA^rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp^rnn_dense3/rnn_dense3_r1/while*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2v
9rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp9rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp2z
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_1;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_12z
;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_2;rnn_dense3/batch_normalization_1/batchnorm/ReadVariableOp_22~
=rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOp=rnn_dense3/batch_normalization_1/batchnorm/mul/ReadVariableOp2v
9rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp9rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp2z
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_1;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_12z
;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_2;rnn_dense3/batch_normalization_2/batchnorm/ReadVariableOp_22~
=rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOp=rnn_dense3/batch_normalization_2/batchnorm/mul/ReadVariableOp2v
9rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp9rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp2z
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_1;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_12z
;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_2;rnn_dense3/batch_normalization_3/batchnorm/ReadVariableOp_22~
=rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOp=rnn_dense3/batch_normalization_3/batchnorm/mul/ReadVariableOp2`
.rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOp.rnn_dense3/rnn_dense3_1/BiasAdd/ReadVariableOp2^
-rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOp-rnn_dense3/rnn_dense3_1/MatMul/ReadVariableOp2`
.rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOp.rnn_dense3/rnn_dense3_2/BiasAdd/ReadVariableOp2^
-rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOp-rnn_dense3/rnn_dense3_2/MatMul/ReadVariableOp2`
.rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOp.rnn_dense3/rnn_dense3_3/BiasAdd/ReadVariableOp2^
-rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOp-rnn_dense3/rnn_dense3_3/MatMul/ReadVariableOp2d
0rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOp0rnn_dense3/rnn_dense3_out/BiasAdd/ReadVariableOp2b
/rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOp/rnn_dense3/rnn_dense3_out/MatMul/ReadVariableOp2В
?rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp?rnn_dense3/rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp2А
>rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp>rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp2Д
@rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp@rnn_dense3/rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp2@
rnn_dense3/rnn_dense3_r1/whilernn_dense3/rnn_dense3_r1/while:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
╙
З
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_84206

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1█
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
▓
b
)__inference_dropout_1_layer_call_fn_85276

inputs
identityИвStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836042
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╙
З
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85821

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1█
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
й
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_85266

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         А2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         А*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         А2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         А2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         А2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ф
╞
*__inference_rnn_dense3_layer_call_fn_85165

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21
identityИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *3
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_844242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
р
^
B__inference_reshape_layer_call_and_return_conditional_losses_85249

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:         А2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╡
и
5__inference_batch_normalization_3_layer_call_fn_85847

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_842062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ч
Б
,__inference_rnn_dense3_1_layer_call_fn_85236

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_835552
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
│0
┼
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83658

inputs
assignmovingavg_83633
assignmovingavg_1_83639)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         А2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/83633*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_83633*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/83633*
_output_shapes
:2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/83633*
_output_shapes
:2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_83633AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/83633*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/83639*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_83639*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/83639*
_output_shapes
:2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/83639*
_output_shapes
:2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_83639AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/83639*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2
batchnorm/add_1╕
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
ж
E
)__inference_dropout_1_layer_call_fn_85281

inputs
identity╟
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836092
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*+
_input_shapes
:         А:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
А
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_84132

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘2
щ
while_body_85521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_simple_rnn_cell_matmul_readvariableop_resource_0;
7while_simple_rnn_cell_biasadd_readvariableop_resource_0<
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_simple_rnn_cell_matmul_readvariableop_resource9
5while_simple_rnn_cell_biasadd_readvariableop_resource:
6while_simple_rnn_cell_matmul_1_readvariableop_resourceИв,while/simple_rnn_cell/BiasAdd/ReadVariableOpв+while/simple_rnn_cell/MatMul/ReadVariableOpв-while/simple_rnn_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╤
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02-
+while/simple_rnn_cell/MatMul/ReadVariableOp▀
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/MatMul╨
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02.
,while/simple_rnn_cell/BiasAdd/ReadVariableOp┘
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/BiasAdd╫
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02/
-while/simple_rnn_cell/MatMul_1/ReadVariableOp╚
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
while/simple_rnn_cell/MatMul_1├
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/addС
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/Tanhт
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ы
while/IdentityIdentitywhile/add_1:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity■
while/Identity_1Identitywhile_while_maximum_iterations-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1э
while/Identity_2Identitywhile/add:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Т
E
)__inference_dropout_3_layer_call_fn_85765

inputs
identity┬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841372
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╠B
Н
rnn_dense3_r1_while_body_847498
4rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter>
:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations#
rnn_dense3_r1_while_placeholder%
!rnn_dense3_r1_while_placeholder_1%
!rnn_dense3_r1_while_placeholder_27
3rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1_0s
ornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0H
Drnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0I
Ernn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0J
Frnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0 
rnn_dense3_r1_while_identity"
rnn_dense3_r1_while_identity_1"
rnn_dense3_r1_while_identity_2"
rnn_dense3_r1_while_identity_3"
rnn_dense3_r1_while_identity_45
1rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1q
mrnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorF
Brnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceG
Crnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourceH
Drnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceИв:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpв9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpв;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp▀
Ernn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2G
Ernn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeз
7rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0rnn_dense3_r1_while_placeholderNrnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype029
7rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem√
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpDrnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02;
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpЧ
*rnn_dense3_r1/while/simple_rnn_cell/MatMulMatMul>rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem:item:0Arnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2,
*rnn_dense3_r1/while/simple_rnn_cell/MatMul·
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpErnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02<
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpС
+rnn_dense3_r1/while/simple_rnn_cell/BiasAddBiasAdd4rnn_dense3_r1/while/simple_rnn_cell/MatMul:product:0Brnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2-
+rnn_dense3_r1/while/simple_rnn_cell/BiasAddБ
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpFrnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02=
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpА
,rnn_dense3_r1/while/simple_rnn_cell/MatMul_1MatMul!rnn_dense3_r1_while_placeholder_2Crnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2.
,rnn_dense3_r1/while/simple_rnn_cell/MatMul_1√
'rnn_dense3_r1/while/simple_rnn_cell/addAddV24rnn_dense3_r1/while/simple_rnn_cell/BiasAdd:output:06rnn_dense3_r1/while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2)
'rnn_dense3_r1/while/simple_rnn_cell/add╗
(rnn_dense3_r1/while/simple_rnn_cell/TanhTanh+rnn_dense3_r1/while/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2*
(rnn_dense3_r1/while/simple_rnn_cell/Tanhи
8rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem!rnn_dense3_r1_while_placeholder_1rnn_dense3_r1_while_placeholder,rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02:
8rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemx
rnn_dense3_r1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_dense3_r1/while/add/yб
rnn_dense3_r1/while/addAddV2rnn_dense3_r1_while_placeholder"rnn_dense3_r1/while/add/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/add|
rnn_dense3_r1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
rnn_dense3_r1/while/add_1/y╝
rnn_dense3_r1/while/add_1AddV24rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter$rnn_dense3_r1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/add_1┐
rnn_dense3_r1/while/IdentityIdentityrnn_dense3_r1/while/add_1:z:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/Identityр
rnn_dense3_r1/while/Identity_1Identity:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_1┴
rnn_dense3_r1/while/Identity_2Identityrnn_dense3_r1/while/add:z:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_2ю
rnn_dense3_r1/while/Identity_3IdentityHrnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2 
rnn_dense3_r1/while/Identity_3у
rnn_dense3_r1/while/Identity_4Identity,rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0;^rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:^rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp<^rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2 
rnn_dense3_r1/while/Identity_4"E
rnn_dense3_r1_while_identity%rnn_dense3_r1/while/Identity:output:0"I
rnn_dense3_r1_while_identity_1'rnn_dense3_r1/while/Identity_1:output:0"I
rnn_dense3_r1_while_identity_2'rnn_dense3_r1/while/Identity_2:output:0"I
rnn_dense3_r1_while_identity_3'rnn_dense3_r1/while/Identity_3:output:0"I
rnn_dense3_r1_while_identity_4'rnn_dense3_r1/while/Identity_4:output:0"h
1rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_13rnn_dense3_r1_while_rnn_dense3_r1_strided_slice_1_0"М
Crnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourceErnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0"О
Drnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceFrnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0"К
Brnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceDrnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0"р
mrnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorornn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2x
:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp2v
9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp9rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp2z
;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp;rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Ё	
р
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_85227

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
шА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         А2
TanhО
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
╦
е
while_cond_85520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_85520___redundant_placeholder03
/while_while_cond_85520___redundant_placeholder13
/while_while_cond_85520___redundant_placeholder23
/while_while_cond_85520___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
у

п
rnn_dense3_r1_while_cond_849938
4rnn_dense3_r1_while_rnn_dense3_r1_while_loop_counter>
:rnn_dense3_r1_while_rnn_dense3_r1_while_maximum_iterations#
rnn_dense3_r1_while_placeholder%
!rnn_dense3_r1_while_placeholder_1%
!rnn_dense3_r1_while_placeholder_2:
6rnn_dense3_r1_while_less_rnn_dense3_r1_strided_slice_1O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84993___redundant_placeholder0O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84993___redundant_placeholder1O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84993___redundant_placeholder2O
Krnn_dense3_r1_while_rnn_dense3_r1_while_cond_84993___redundant_placeholder3 
rnn_dense3_r1_while_identity
╢
rnn_dense3_r1/while/LessLessrnn_dense3_r1_while_placeholder6rnn_dense3_r1_while_less_rnn_dense3_r1_strided_slice_1*
T0*
_output_shapes
: 2
rnn_dense3_r1/while/LessЗ
rnn_dense3_r1/while/IdentityIdentityrnn_dense3_r1/while/Less:z:0*
T0
*
_output_shapes
: 2
rnn_dense3_r1/while/Identity"E
rnn_dense3_r1_while_identity%rnn_dense3_r1/while/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╛
╥
*__inference_rnn_dense3_layer_call_fn_84586
rnn_dense3_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21
identityИвStatefulPartitionedCallЮ
StatefulPartitionedCallStatefulPartitionedCallrnn_dense3_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_845372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
╟
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_85755

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:          2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
А
c
D__inference_dropout_3_layer_call_and_return_conditional_losses_85750

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape┤
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2
dropout/GreaterEqual/y╛
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ю
b
)__inference_dropout_3_layer_call_fn_85760

inputs
identityИвStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841322
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*&
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘2
щ
while_body_83873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_simple_rnn_cell_matmul_readvariableop_resource_0;
7while_simple_rnn_cell_biasadd_readvariableop_resource_0<
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_simple_rnn_cell_matmul_readvariableop_resource9
5while_simple_rnn_cell_biasadd_readvariableop_resource:
6while_simple_rnn_cell_matmul_1_readvariableop_resourceИв,while/simple_rnn_cell/BiasAdd/ReadVariableOpв+while/simple_rnn_cell/MatMul/ReadVariableOpв-while/simple_rnn_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╤
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02-
+while/simple_rnn_cell/MatMul/ReadVariableOp▀
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/MatMul╨
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02.
,while/simple_rnn_cell/BiasAdd/ReadVariableOp┘
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/BiasAdd╫
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02/
-while/simple_rnn_cell/MatMul_1/ReadVariableOp╚
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
while/simple_rnn_cell/MatMul_1├
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/addС
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/Tanhт
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ы
while/IdentityIdentitywhile/add_1:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity■
while/Identity_1Identitywhile_while_maximum_iterations-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1э
while/Identity_2Identitywhile/add:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
°/
┼
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_84186

inputs
assignmovingavg_84161
assignmovingavg_1_84167)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:          2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/84161*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_84161*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/84161*
_output_shapes
: 2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/84161*
_output_shapes
: 2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_84161AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/84161*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/84167*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_84167*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/84167*
_output_shapes
: 2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/84167*
_output_shapes
: 2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_84167AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/84167*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1│
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╟
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_84137

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:          2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*&
_input_shapes
:          :O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╙
З
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_84056

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1█
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ь
З
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_83678

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2
batchnorm/add_1р
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
│
и
5__inference_batch_normalization_2_layer_call_fn_85705

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
°/
┼
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85801

inputs
assignmovingavg_85776
assignmovingavg_1_85782)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

: 2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:          2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85776*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_85776*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85776*
_output_shapes
: 2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/85776*
_output_shapes
: 2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_85776AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/85776*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85782*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_85782*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85782*
_output_shapes
: 2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/85782*
_output_shapes
: 2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_85782AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/85782*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:          2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:          2
batchnorm/add_1│
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ч
Г
.__inference_rnn_dense3_out_layer_call_fn_85886

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_842802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 ў
╛
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84915

inputs/
+rnn_dense3_1_matmul_readvariableop_resource0
,rnn_dense3_1_biasadd_readvariableop_resource/
+batch_normalization_1_assignmovingavg_846821
-batch_normalization_1_assignmovingavg_1_84688?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource@
<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resourceA
=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resourceB
>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource/
+batch_normalization_2_assignmovingavg_848301
-batch_normalization_2_assignmovingavg_1_84836?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource/
+rnn_dense3_2_matmul_readvariableop_resource0
,rnn_dense3_2_biasadd_readvariableop_resource/
+batch_normalization_3_assignmovingavg_848771
-batch_normalization_3_assignmovingavg_1_84883?
;batch_normalization_3_batchnorm_mul_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource/
+rnn_dense3_3_matmul_readvariableop_resource0
,rnn_dense3_3_biasadd_readvariableop_resource1
-rnn_dense3_out_matmul_readvariableop_resource2
.rnn_dense3_out_biasadd_readvariableop_resource
identityИв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв4batch_normalization_1/AssignMovingAvg/ReadVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_1/batchnorm/ReadVariableOpв2batch_normalization_1/batchnorm/mul/ReadVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв4batch_normalization_2/AssignMovingAvg/ReadVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpв6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв2batch_normalization_2/batchnorm/mul/ReadVariableOpв9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpв4batch_normalization_3/AssignMovingAvg/ReadVariableOpв;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpв6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв2batch_normalization_3/batchnorm/mul/ReadVariableOpв#rnn_dense3_1/BiasAdd/ReadVariableOpв"rnn_dense3_1/MatMul/ReadVariableOpв#rnn_dense3_2/BiasAdd/ReadVariableOpв"rnn_dense3_2/MatMul/ReadVariableOpв#rnn_dense3_3/BiasAdd/ReadVariableOpв"rnn_dense3_3/MatMul/ReadVariableOpв%rnn_dense3_out/BiasAdd/ReadVariableOpв$rnn_dense3_out/MatMul/ReadVariableOpв4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpв3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpв5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpвrnn_dense3_r1/while╢
"rnn_dense3_1/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_1_matmul_readvariableop_resource* 
_output_shapes
:
шА*
dtype02$
"rnn_dense3_1/MatMul/ReadVariableOpЫ
rnn_dense3_1/MatMulMatMulinputs*rnn_dense3_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/MatMul┤
#rnn_dense3_1/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#rnn_dense3_1/BiasAdd/ReadVariableOp╢
rnn_dense3_1/BiasAddBiasAddrnn_dense3_1/MatMul:product:0+rnn_dense3_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/BiasAddА
rnn_dense3_1/TanhTanhrnn_dense3_1/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/Tanhc
reshape/ShapeShapernn_dense3_1/Tanh:y:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЫ
reshape/ReshapeReshapernn_dense3_1/Tanh:y:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:         А2
reshape/Reshapew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout_1/dropout/Constи
dropout_1/dropout/MulMulreshape/Reshape:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:         А2
dropout_1/dropout/Mulz
dropout_1/dropout/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape╫
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:         А*
dtype020
.dropout_1/dropout/random_uniform/RandomUniformЙ
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2"
 dropout_1/dropout/GreaterEqual/yы
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         А2 
dropout_1/dropout/GreaterEqualв
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         А2
dropout_1/dropout/Castз
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:         А2
dropout_1/dropout/Mul_1╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesъ
"batch_normalization_1/moments/meanMeandropout_1/dropout/Mul_1:z:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2$
"batch_normalization_1/moments/mean┬
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:2,
*batch_normalization_1/moments/StopGradientА
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencedropout_1/dropout/Mul_1:z:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         А21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesО
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2(
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1Н
+batch_normalization_1/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/84682*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decay╘
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_1_assignmovingavg_84682*
_output_shapes
:*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp▐
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/84682*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/sub╒
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/84682*
_output_shapes
:2+
)batch_normalization_1/AssignMovingAvg/mul▒
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_1_assignmovingavg_84682-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/84682*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpУ
-batch_normalization_1/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/84688*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decay┌
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_1_84688*
_output_shapes
:*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/84688*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/sub▀
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/84688*
_output_shapes
:2-
+batch_normalization_1/AssignMovingAvg_1/mul╜
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_1_84688/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/84688*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y┌
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul╥
%batch_normalization_1/batchnorm/mul_1Muldropout_1/dropout/Mul_1:z:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А2'
%batch_normalization_1/batchnorm/mul_1╙
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┘
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2'
%batch_normalization_1/batchnorm/add_1Г
rnn_dense3_r1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
rnn_dense3_r1/ShapeР
!rnn_dense3_r1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!rnn_dense3_r1/strided_slice/stackФ
#rnn_dense3_r1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#rnn_dense3_r1/strided_slice/stack_1Ф
#rnn_dense3_r1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#rnn_dense3_r1/strided_slice/stack_2╢
rnn_dense3_r1/strided_sliceStridedSlicernn_dense3_r1/Shape:output:0*rnn_dense3_r1/strided_slice/stack:output:0,rnn_dense3_r1/strided_slice/stack_1:output:0,rnn_dense3_r1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_dense3_r1/strided_slicex
rnn_dense3_r1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_dense3_r1/zeros/mul/yд
rnn_dense3_r1/zeros/mulMul$rnn_dense3_r1/strided_slice:output:0"rnn_dense3_r1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/zeros/mul{
rnn_dense3_r1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn_dense3_r1/zeros/Less/yЯ
rnn_dense3_r1/zeros/LessLessrnn_dense3_r1/zeros/mul:z:0#rnn_dense3_r1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/zeros/Less~
rnn_dense3_r1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_dense3_r1/zeros/packed/1╗
rnn_dense3_r1/zeros/packedPack$rnn_dense3_r1/strided_slice:output:0%rnn_dense3_r1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_dense3_r1/zeros/packed{
rnn_dense3_r1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_dense3_r1/zeros/Constн
rnn_dense3_r1/zerosFill#rnn_dense3_r1/zeros/packed:output:0"rnn_dense3_r1/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
rnn_dense3_r1/zerosС
rnn_dense3_r1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_dense3_r1/transpose/perm╚
rnn_dense3_r1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0%rnn_dense3_r1/transpose/perm:output:0*
T0*,
_output_shapes
:А         2
rnn_dense3_r1/transposey
rnn_dense3_r1/Shape_1Shapernn_dense3_r1/transpose:y:0*
T0*
_output_shapes
:2
rnn_dense3_r1/Shape_1Ф
#rnn_dense3_r1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rnn_dense3_r1/strided_slice_1/stackШ
%rnn_dense3_r1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_1/stack_1Ш
%rnn_dense3_r1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_1/stack_2┬
rnn_dense3_r1/strided_slice_1StridedSlicernn_dense3_r1/Shape_1:output:0,rnn_dense3_r1/strided_slice_1/stack:output:0.rnn_dense3_r1/strided_slice_1/stack_1:output:0.rnn_dense3_r1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_dense3_r1/strided_slice_1б
)rnn_dense3_r1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)rnn_dense3_r1/TensorArrayV2/element_shapeъ
rnn_dense3_r1/TensorArrayV2TensorListReserve2rnn_dense3_r1/TensorArrayV2/element_shape:output:0&rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_dense3_r1/TensorArrayV2█
Crnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape░
5rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_dense3_r1/transpose:y:0Lrnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorФ
#rnn_dense3_r1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rnn_dense3_r1/strided_slice_2/stackШ
%rnn_dense3_r1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_2/stack_1Ш
%rnn_dense3_r1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_2/stack_2╨
rnn_dense3_r1/strided_slice_2StridedSlicernn_dense3_r1/transpose:y:0,rnn_dense3_r1/strided_slice_2/stack:output:0.rnn_dense3_r1/strided_slice_2/stack_1:output:0.rnn_dense3_r1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
rnn_dense3_r1/strided_slice_2ч
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype025
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpэ
$rnn_dense3_r1/simple_rnn_cell/MatMulMatMul&rnn_dense3_r1/strided_slice_2:output:0;rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2&
$rnn_dense3_r1/simple_rnn_cell/MatMulц
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp∙
%rnn_dense3_r1/simple_rnn_cell/BiasAddBiasAdd.rnn_dense3_r1/simple_rnn_cell/MatMul:product:0<rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2'
%rnn_dense3_r1/simple_rnn_cell/BiasAddэ
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype027
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpщ
&rnn_dense3_r1/simple_rnn_cell/MatMul_1MatMulrnn_dense3_r1/zeros:output:0=rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2(
&rnn_dense3_r1/simple_rnn_cell/MatMul_1у
!rnn_dense3_r1/simple_rnn_cell/addAddV2.rnn_dense3_r1/simple_rnn_cell/BiasAdd:output:00rnn_dense3_r1/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2#
!rnn_dense3_r1/simple_rnn_cell/addй
"rnn_dense3_r1/simple_rnn_cell/TanhTanh%rnn_dense3_r1/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2$
"rnn_dense3_r1/simple_rnn_cell/Tanhл
+rnn_dense3_r1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_dense3_r1/TensorArrayV2_1/element_shapeЁ
rnn_dense3_r1/TensorArrayV2_1TensorListReserve4rnn_dense3_r1/TensorArrayV2_1/element_shape:output:0&rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_dense3_r1/TensorArrayV2_1j
rnn_dense3_r1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_dense3_r1/timeЫ
&rnn_dense3_r1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2(
&rnn_dense3_r1/while/maximum_iterationsЖ
 rnn_dense3_r1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 rnn_dense3_r1/while/loop_counterА
rnn_dense3_r1/whileWhile)rnn_dense3_r1/while/loop_counter:output:0/rnn_dense3_r1/while/maximum_iterations:output:0rnn_dense3_r1/time:output:0&rnn_dense3_r1/TensorArrayV2_1:handle:0rnn_dense3_r1/zeros:output:0&rnn_dense3_r1/strided_slice_1:output:0Ernn_dense3_r1/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resource=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resource>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	**
body"R 
rnn_dense3_r1_while_body_84749**
cond"R 
rnn_dense3_r1_while_cond_84748*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
rnn_dense3_r1/while╤
>rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shapeб
0rnn_dense3_r1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_dense3_r1/while:output:3Grnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype022
0rnn_dense3_r1/TensorArrayV2Stack/TensorListStackЭ
#rnn_dense3_r1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2%
#rnn_dense3_r1/strided_slice_3/stackШ
%rnn_dense3_r1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%rnn_dense3_r1/strided_slice_3/stack_1Ш
%rnn_dense3_r1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_3/stack_2ю
rnn_dense3_r1/strided_slice_3StridedSlice9rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:0,rnn_dense3_r1/strided_slice_3/stack:output:0.rnn_dense3_r1/strided_slice_3/stack_1:output:0.rnn_dense3_r1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
rnn_dense3_r1/strided_slice_3Х
rnn_dense3_r1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
rnn_dense3_r1/transpose_1/perm▐
rnn_dense3_r1/transpose_1	Transpose9rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:0'rnn_dense3_r1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
rnn_dense3_r1/transpose_1w
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout_2/dropout/Const▒
dropout_2/dropout/MulMul&rnn_dense3_r1/strided_slice_3:output:0 dropout_2/dropout/Const:output:0*
T0*'
_output_shapes
:         @2
dropout_2/dropout/MulИ
dropout_2/dropout/ShapeShape&rnn_dense3_r1/strided_slice_3:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape╥
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*'
_output_shapes
:         @*
dtype020
.dropout_2/dropout/random_uniform/RandomUniformЙ
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2"
 dropout_2/dropout/GreaterEqual/yц
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         @2 
dropout_2/dropout/GreaterEqualЭ
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         @2
dropout_2/dropout/Castв
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*'
_output_shapes
:         @2
dropout_2/dropout/Mul_1╢
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_2/moments/mean/reduction_indicesц
"batch_normalization_2/moments/meanMeandropout_2/dropout/Mul_1:z:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2$
"batch_normalization_2/moments/mean╛
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:@2,
*batch_normalization_2/moments/StopGradient√
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedropout_2/dropout/Mul_1:z:03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         @21
/batch_normalization_2/moments/SquaredDifference╛
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_2/moments/variance/reduction_indicesК
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2(
&batch_normalization_2/moments/variance┬
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╩
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1Н
+batch_normalization_2/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/84830*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_2/AssignMovingAvg/decay╘
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_2_assignmovingavg_84830*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp▐
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/84830*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/sub╒
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/84830*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul▒
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_2_assignmovingavg_84830-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/84830*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpУ
-batch_normalization_2/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/84836*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decay┌
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_1_84836*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/84836*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/sub▀
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/84836*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mul╜
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_1_84836/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/84836*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/y┌
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul═
%batch_normalization_2/batchnorm/mul_1Muldropout_2/dropout/Mul_1:z:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_2/batchnorm/add_1┤
"rnn_dense3_2/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02$
"rnn_dense3_2/MatMul/ReadVariableOp╜
rnn_dense3_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0*rnn_dense3_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/MatMul│
#rnn_dense3_2/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#rnn_dense3_2/BiasAdd/ReadVariableOp╡
rnn_dense3_2/BiasAddBiasAddrnn_dense3_2/MatMul:product:0+rnn_dense3_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/BiasAdd
rnn_dense3_2/TanhTanhrnn_dense3_2/BiasAdd:output:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/Tanhw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *в╝Ж?2
dropout_3/dropout/Constа
dropout_3/dropout/MulMulrnn_dense3_2/Tanh:y:0 dropout_3/dropout/Const:output:0*
T0*'
_output_shapes
:          2
dropout_3/dropout/Mulw
dropout_3/dropout/ShapeShapernn_dense3_2/Tanh:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shape╥
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*'
_output_shapes
:          *
dtype020
.dropout_3/dropout/random_uniform/RandomUniformЙ
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L=2"
 dropout_3/dropout/GreaterEqual/yц
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:          2 
dropout_3/dropout/GreaterEqualЭ
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:          2
dropout_3/dropout/Castв
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*'
_output_shapes
:          2
dropout_3/dropout/Mul_1╢
4batch_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 26
4batch_normalization_3/moments/mean/reduction_indicesц
"batch_normalization_3/moments/meanMeandropout_3/dropout/Mul_1:z:0=batch_normalization_3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2$
"batch_normalization_3/moments/mean╛
*batch_normalization_3/moments/StopGradientStopGradient+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes

: 2,
*batch_normalization_3/moments/StopGradient√
/batch_normalization_3/moments/SquaredDifferenceSquaredDifferencedropout_3/dropout/Mul_1:z:03batch_normalization_3/moments/StopGradient:output:0*
T0*'
_output_shapes
:          21
/batch_normalization_3/moments/SquaredDifference╛
8batch_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2:
8batch_normalization_3/moments/variance/reduction_indicesК
&batch_normalization_3/moments/varianceMean3batch_normalization_3/moments/SquaredDifference:z:0Abatch_normalization_3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

: *
	keep_dims(2(
&batch_normalization_3/moments/variance┬
%batch_normalization_3/moments/SqueezeSqueeze+batch_normalization_3/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_3/moments/Squeeze╩
'batch_normalization_3/moments/Squeeze_1Squeeze/batch_normalization_3/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_3/moments/Squeeze_1Н
+batch_normalization_3/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/84877*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_3/AssignMovingAvg/decay╘
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_84877*
_output_shapes
: *
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp▐
)batch_normalization_3/AssignMovingAvg/subSub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_3/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/84877*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub╒
)batch_normalization_3/AssignMovingAvg/mulMul-batch_normalization_3/AssignMovingAvg/sub:z:04batch_normalization_3/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/84877*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/mul▒
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_84877-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/84877*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpУ
-batch_normalization_3/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/84883*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_3/AssignMovingAvg_1/decay┌
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_84883*
_output_shapes
: *
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpш
+batch_normalization_3/AssignMovingAvg_1/subSub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_3/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/84883*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/sub▀
+batch_normalization_3/AssignMovingAvg_1/mulMul/batch_normalization_3/AssignMovingAvg_1/sub:z:06batch_normalization_3/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/84883*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/mul╜
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_84883/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/84883*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/y┌
#batch_normalization_3/batchnorm/addAddV20batch_normalization_3/moments/Squeeze_1:output:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mul═
%batch_normalization_3/batchnorm/mul_1Muldropout_3/dropout/Mul_1:z:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_3/batchnorm/mul_1╙
%batch_normalization_3/batchnorm/mul_2Mul.batch_normalization_3/moments/Squeeze:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOp┘
#batch_normalization_3/batchnorm/subSub6batch_normalization_3/batchnorm/ReadVariableOp:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_3/batchnorm/add_1┤
"rnn_dense3_3/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"rnn_dense3_3/MatMul/ReadVariableOp╜
rnn_dense3_3/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0*rnn_dense3_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/MatMul│
#rnn_dense3_3/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#rnn_dense3_3/BiasAdd/ReadVariableOp╡
rnn_dense3_3/BiasAddBiasAddrnn_dense3_3/MatMul:product:0+rnn_dense3_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/BiasAdd
rnn_dense3_3/TanhTanhrnn_dense3_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/Tanh║
$rnn_dense3_out/MatMul/ReadVariableOpReadVariableOp-rnn_dense3_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$rnn_dense3_out/MatMul/ReadVariableOpп
rnn_dense3_out/MatMulMatMulrnn_dense3_3/Tanh:y:0,rnn_dense3_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_out/MatMul╣
%rnn_dense3_out/BiasAdd/ReadVariableOpReadVariableOp.rnn_dense3_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%rnn_dense3_out/BiasAdd/ReadVariableOp╜
rnn_dense3_out/BiasAddBiasAddrnn_dense3_out/MatMul:product:0-rnn_dense3_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_out/BiasAdd╬
IdentityIdentityrnn_dense3_out/BiasAdd:output:0:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp3^batch_normalization_3/batchnorm/mul/ReadVariableOp$^rnn_dense3_1/BiasAdd/ReadVariableOp#^rnn_dense3_1/MatMul/ReadVariableOp$^rnn_dense3_2/BiasAdd/ReadVariableOp#^rnn_dense3_2/MatMul/ReadVariableOp$^rnn_dense3_3/BiasAdd/ReadVariableOp#^rnn_dense3_3/MatMul/ReadVariableOp&^rnn_dense3_out/BiasAdd/ReadVariableOp%^rnn_dense3_out/MatMul/ReadVariableOp5^rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp4^rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp6^rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp^rnn_dense3_r1/while*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2J
#rnn_dense3_1/BiasAdd/ReadVariableOp#rnn_dense3_1/BiasAdd/ReadVariableOp2H
"rnn_dense3_1/MatMul/ReadVariableOp"rnn_dense3_1/MatMul/ReadVariableOp2J
#rnn_dense3_2/BiasAdd/ReadVariableOp#rnn_dense3_2/BiasAdd/ReadVariableOp2H
"rnn_dense3_2/MatMul/ReadVariableOp"rnn_dense3_2/MatMul/ReadVariableOp2J
#rnn_dense3_3/BiasAdd/ReadVariableOp#rnn_dense3_3/BiasAdd/ReadVariableOp2H
"rnn_dense3_3/MatMul/ReadVariableOp"rnn_dense3_3/MatMul/ReadVariableOp2N
%rnn_dense3_out/BiasAdd/ReadVariableOp%rnn_dense3_out/BiasAdd/ReadVariableOp2L
$rnn_dense3_out/MatMul/ReadVariableOp$rnn_dense3_out/MatMul/ReadVariableOp2l
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp2j
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp2n
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp2*
rnn_dense3_r1/whilernn_dense3_r1/while:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
зд
Ж(
!__inference__traced_restore_86327
file_prefix(
$assignvariableop_rnn_dense3_1_kernel(
$assignvariableop_1_rnn_dense3_1_bias2
.assignvariableop_2_batch_normalization_1_gamma1
-assignvariableop_3_batch_normalization_1_beta8
4assignvariableop_4_batch_normalization_1_moving_mean<
8assignvariableop_5_batch_normalization_1_moving_variance2
.assignvariableop_6_batch_normalization_2_gamma1
-assignvariableop_7_batch_normalization_2_beta8
4assignvariableop_8_batch_normalization_2_moving_mean<
8assignvariableop_9_batch_normalization_2_moving_variance+
'assignvariableop_10_rnn_dense3_2_kernel)
%assignvariableop_11_rnn_dense3_2_bias3
/assignvariableop_12_batch_normalization_3_gamma2
.assignvariableop_13_batch_normalization_3_beta9
5assignvariableop_14_batch_normalization_3_moving_mean=
9assignvariableop_15_batch_normalization_3_moving_variance+
'assignvariableop_16_rnn_dense3_3_kernel)
%assignvariableop_17_rnn_dense3_3_bias-
)assignvariableop_18_rnn_dense3_out_kernel+
'assignvariableop_19_rnn_dense3_out_bias!
assignvariableop_20_adam_iter#
assignvariableop_21_adam_beta_1#
assignvariableop_22_adam_beta_2"
assignvariableop_23_adam_decay*
&assignvariableop_24_adam_learning_rate<
8assignvariableop_25_rnn_dense3_r1_simple_rnn_cell_kernelF
Bassignvariableop_26_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel:
6assignvariableop_27_rnn_dense3_r1_simple_rnn_cell_bias
assignvariableop_28_total
assignvariableop_29_count
assignvariableop_30_total_1
assignvariableop_31_count_1
assignvariableop_32_total_2
assignvariableop_33_count_22
.assignvariableop_34_adam_rnn_dense3_1_kernel_m0
,assignvariableop_35_adam_rnn_dense3_1_bias_m:
6assignvariableop_36_adam_batch_normalization_1_gamma_m9
5assignvariableop_37_adam_batch_normalization_1_beta_m:
6assignvariableop_38_adam_batch_normalization_2_gamma_m9
5assignvariableop_39_adam_batch_normalization_2_beta_m2
.assignvariableop_40_adam_rnn_dense3_2_kernel_m0
,assignvariableop_41_adam_rnn_dense3_2_bias_m:
6assignvariableop_42_adam_batch_normalization_3_gamma_m9
5assignvariableop_43_adam_batch_normalization_3_beta_m2
.assignvariableop_44_adam_rnn_dense3_3_kernel_m0
,assignvariableop_45_adam_rnn_dense3_3_bias_m4
0assignvariableop_46_adam_rnn_dense3_out_kernel_m2
.assignvariableop_47_adam_rnn_dense3_out_bias_mC
?assignvariableop_48_adam_rnn_dense3_r1_simple_rnn_cell_kernel_mM
Iassignvariableop_49_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_mA
=assignvariableop_50_adam_rnn_dense3_r1_simple_rnn_cell_bias_m2
.assignvariableop_51_adam_rnn_dense3_1_kernel_v0
,assignvariableop_52_adam_rnn_dense3_1_bias_v:
6assignvariableop_53_adam_batch_normalization_1_gamma_v9
5assignvariableop_54_adam_batch_normalization_1_beta_v:
6assignvariableop_55_adam_batch_normalization_2_gamma_v9
5assignvariableop_56_adam_batch_normalization_2_beta_v2
.assignvariableop_57_adam_rnn_dense3_2_kernel_v0
,assignvariableop_58_adam_rnn_dense3_2_bias_v:
6assignvariableop_59_adam_batch_normalization_3_gamma_v9
5assignvariableop_60_adam_batch_normalization_3_beta_v2
.assignvariableop_61_adam_rnn_dense3_3_kernel_v0
,assignvariableop_62_adam_rnn_dense3_3_bias_v4
0assignvariableop_63_adam_rnn_dense3_out_kernel_v2
.assignvariableop_64_adam_rnn_dense3_out_bias_vC
?assignvariableop_65_adam_rnn_dense3_r1_simple_rnn_cell_kernel_vM
Iassignvariableop_66_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_vA
=assignvariableop_67_adam_rnn_dense3_r1_simple_rnn_cell_bias_v
identity_69ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9ч$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*є#
valueщ#Bц#EB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЫ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:E*
dtype0*Я
valueХBТEB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЗ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*S
dtypesI
G2E	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityг
AssignVariableOpAssignVariableOp$assignvariableop_rnn_dense3_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1й
AssignVariableOp_1AssignVariableOp$assignvariableop_1_rnn_dense3_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3▓
AssignVariableOp_3AssignVariableOp-assignvariableop_3_batch_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╣
AssignVariableOp_4AssignVariableOp4assignvariableop_4_batch_normalization_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╜
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_normalization_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6│
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_2_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▓
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_2_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_2_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╜
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_2_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10п
AssignVariableOp_10AssignVariableOp'assignvariableop_10_rnn_dense3_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11н
AssignVariableOp_11AssignVariableOp%assignvariableop_11_rnn_dense3_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╖
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_3_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╢
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_3_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14╜
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_3_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┴
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_3_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16п
AssignVariableOp_16AssignVariableOp'assignvariableop_16_rnn_dense3_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17н
AssignVariableOp_17AssignVariableOp%assignvariableop_17_rnn_dense3_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18▒
AssignVariableOp_18AssignVariableOp)assignvariableop_18_rnn_dense3_out_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19п
AssignVariableOp_19AssignVariableOp'assignvariableop_19_rnn_dense3_out_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20е
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21з
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22з
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23ж
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24о
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25└
AssignVariableOp_25AssignVariableOp8assignvariableop_25_rnn_dense3_r1_simple_rnn_cell_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╩
AssignVariableOp_26AssignVariableOpBassignvariableop_26_rnn_dense3_r1_simple_rnn_cell_recurrent_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_rnn_dense3_r1_simple_rnn_cell_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28б
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29б
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30г
AssignVariableOp_30AssignVariableOpassignvariableop_30_total_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31г
AssignVariableOp_31AssignVariableOpassignvariableop_31_count_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32г
AssignVariableOp_32AssignVariableOpassignvariableop_32_total_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33г
AssignVariableOp_33AssignVariableOpassignvariableop_33_count_2Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34╢
AssignVariableOp_34AssignVariableOp.assignvariableop_34_adam_rnn_dense3_1_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35┤
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_rnn_dense3_1_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36╛
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_batch_normalization_1_gamma_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╜
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_batch_normalization_1_beta_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╛
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_batch_normalization_2_gamma_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39╜
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_batch_normalization_2_beta_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40╢
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_rnn_dense3_2_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41┤
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_rnn_dense3_2_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╛
AssignVariableOp_42AssignVariableOp6assignvariableop_42_adam_batch_normalization_3_gamma_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╜
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adam_batch_normalization_3_beta_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╢
AssignVariableOp_44AssignVariableOp.assignvariableop_44_adam_rnn_dense3_3_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45┤
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_rnn_dense3_3_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╕
AssignVariableOp_46AssignVariableOp0assignvariableop_46_adam_rnn_dense3_out_kernel_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╢
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_rnn_dense3_out_bias_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48╟
AssignVariableOp_48AssignVariableOp?assignvariableop_48_adam_rnn_dense3_r1_simple_rnn_cell_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49╤
AssignVariableOp_49AssignVariableOpIassignvariableop_49_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50┼
AssignVariableOp_50AssignVariableOp=assignvariableop_50_adam_rnn_dense3_r1_simple_rnn_cell_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51╢
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_rnn_dense3_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52┤
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_rnn_dense3_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53╛
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_1_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54╜
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_1_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55╛
AssignVariableOp_55AssignVariableOp6assignvariableop_55_adam_batch_normalization_2_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56╜
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_batch_normalization_2_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57╢
AssignVariableOp_57AssignVariableOp.assignvariableop_57_adam_rnn_dense3_2_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58┤
AssignVariableOp_58AssignVariableOp,assignvariableop_58_adam_rnn_dense3_2_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59╛
AssignVariableOp_59AssignVariableOp6assignvariableop_59_adam_batch_normalization_3_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60╜
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_batch_normalization_3_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61╢
AssignVariableOp_61AssignVariableOp.assignvariableop_61_adam_rnn_dense3_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62┤
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_rnn_dense3_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63╕
AssignVariableOp_63AssignVariableOp0assignvariableop_63_adam_rnn_dense3_out_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64╢
AssignVariableOp_64AssignVariableOp.assignvariableop_64_adam_rnn_dense3_out_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65╟
AssignVariableOp_65AssignVariableOp?assignvariableop_65_adam_rnn_dense3_r1_simple_rnn_cell_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66╤
AssignVariableOp_66AssignVariableOpIassignvariableop_66_adam_rnn_dense3_r1_simple_rnn_cell_recurrent_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67┼
AssignVariableOp_67AssignVariableOp=assignvariableop_67_adam_rnn_dense3_r1_simple_rnn_cell_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_679
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╢
Identity_68Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_68й
Identity_69IdentityIdentity_68:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_69"#
identity_69Identity_69:output:0*з
_input_shapesХ
Т: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
МG
Й
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_83827

inputs2
.simple_rnn_cell_matmul_readvariableop_resource3
/simple_rnn_cell_biasadd_readvariableop_resource4
0simple_rnn_cell_matmul_1_readvariableop_resource
identityИв&simple_rnn_cell/BiasAdd/ReadVariableOpв%simple_rnn_cell/MatMul/ReadVariableOpв'simple_rnn_cell/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:А         2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2╜
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%simple_rnn_cell/MatMul/ReadVariableOp╡
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul╝
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&simple_rnn_cell/BiasAdd/ReadVariableOp┴
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/BiasAdd├
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'simple_rnn_cell/MatMul_1/ReadVariableOp▒
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul_1л
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/add
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/TanhП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_83761*
condR
while_cond_83760*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
transpose_1я
IdentityIdentitystrided_slice_3:output:0'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
°/
┼
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_84036

inputs
assignmovingavg_84011
assignmovingavg_1_84017)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpвAssignMovingAvg/ReadVariableOpв%AssignMovingAvg_1/AssignSubVariableOpв AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indicesП
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/mean|
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@2
moments/StopGradientд
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         @2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices▓
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(2
moments/varianceА
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeИ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1╦
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/84011*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayТ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_84011*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOpЁ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/84011*
_output_shapes
:@2
AssignMovingAvg/subч
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/84011*
_output_shapes
:@2
AssignMovingAvg/mulн
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_84011AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/84011*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp╤
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/84017*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayШ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_84017*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp·
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/84017*
_output_shapes
:@2
AssignMovingAvg_1/subё
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/84017*
_output_shapes
:@2
AssignMovingAvg_1/mul╣
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_84017AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/84017*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1│
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
¤C
═	
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84297
rnn_dense3_1_input
rnn_dense3_1_83566
rnn_dense3_1_83568
batch_normalization_1_83706
batch_normalization_1_83708
batch_normalization_1_83710
batch_normalization_1_83712
rnn_dense3_r1_83963
rnn_dense3_r1_83965
rnn_dense3_r1_83967
batch_normalization_2_84084
batch_normalization_2_84086
batch_normalization_2_84088
batch_normalization_2_84090
rnn_dense3_2_84115
rnn_dense3_2_84117
batch_normalization_3_84234
batch_normalization_3_84236
batch_normalization_3_84238
batch_normalization_3_84240
rnn_dense3_3_84265
rnn_dense3_3_84267
rnn_dense3_out_84291
rnn_dense3_out_84293
identityИв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallв$rnn_dense3_1/StatefulPartitionedCallв$rnn_dense3_2/StatefulPartitionedCallв$rnn_dense3_3/StatefulPartitionedCallв&rnn_dense3_out/StatefulPartitionedCallв%rnn_dense3_r1/StatefulPartitionedCall▓
$rnn_dense3_1/StatefulPartitionedCallStatefulPartitionedCallrnn_dense3_1_inputrnn_dense3_1_83566rnn_dense3_1_83568*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_835552&
$rnn_dense3_1/StatefulPartitionedCall№
reshape/PartitionedCallPartitionedCall-rnn_dense3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_835842
reshape/PartitionedCallН
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836042#
!dropout_1/StatefulPartitionedCall╖
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_83706batch_normalization_1_83708batch_normalization_1_83710batch_normalization_1_83712*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836582/
-batch_normalization_1/StatefulPartitionedCallё
%rnn_dense3_r1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0rnn_dense3_r1_83963rnn_dense3_r1_83965rnn_dense3_r1_83967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_838272'
%rnn_dense3_r1/StatefulPartitionedCall║
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall.rnn_dense3_r1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839822#
!dropout_2/StatefulPartitionedCall▓
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0batch_normalization_2_84084batch_normalization_2_84086batch_normalization_2_84088batch_normalization_2_84090*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840362/
-batch_normalization_2/StatefulPartitionedCall╒
$rnn_dense3_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0rnn_dense3_2_84115rnn_dense3_2_84117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_841042&
$rnn_dense3_2/StatefulPartitionedCall╣
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841322#
!dropout_3/StatefulPartitionedCall▓
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0batch_normalization_3_84234batch_normalization_3_84236batch_normalization_3_84238batch_normalization_3_84240*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_841862/
-batch_normalization_3/StatefulPartitionedCall╒
$rnn_dense3_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0rnn_dense3_3_84265rnn_dense3_3_84267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_842542&
$rnn_dense3_3/StatefulPartitionedCall╓
&rnn_dense3_out/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_3/StatefulPartitionedCall:output:0rnn_dense3_out_84291rnn_dense3_out_84293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_842802(
&rnn_dense3_out/StatefulPartitionedCall┼
IdentityIdentity/rnn_dense3_out/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall%^rnn_dense3_1/StatefulPartitionedCall%^rnn_dense3_2/StatefulPartitionedCall%^rnn_dense3_3/StatefulPartitionedCall'^rnn_dense3_out/StatefulPartitionedCall&^rnn_dense3_r1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2L
$rnn_dense3_1/StatefulPartitionedCall$rnn_dense3_1/StatefulPartitionedCall2L
$rnn_dense3_2/StatefulPartitionedCall$rnn_dense3_2/StatefulPartitionedCall2L
$rnn_dense3_3/StatefulPartitionedCall$rnn_dense3_3/StatefulPartitionedCall2P
&rnn_dense3_out/StatefulPartitionedCall&rnn_dense3_out/StatefulPartitionedCall2N
%rnn_dense3_r1/StatefulPartitionedCall%rnn_dense3_r1/StatefulPartitionedCall:\ X
(
_output_shapes
:         ш
,
_user_specified_namernn_dense3_1_input
МG
Й
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85587

inputs2
.simple_rnn_cell_matmul_readvariableop_resource3
/simple_rnn_cell_biasadd_readvariableop_resource4
0simple_rnn_cell_matmul_1_readvariableop_resource
identityИв&simple_rnn_cell/BiasAdd/ReadVariableOpв%simple_rnn_cell/MatMul/ReadVariableOpв'simple_rnn_cell/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:А         2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2╜
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%simple_rnn_cell/MatMul/ReadVariableOp╡
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul╝
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&simple_rnn_cell/BiasAdd/ReadVariableOp┴
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/BiasAdd├
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'simple_rnn_cell/MatMul_1/ReadVariableOp▒
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul_1л
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/add
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/TanhП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_85521*
condR
while_cond_85520*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
transpose_1я
IdentityIdentitystrided_slice_3:output:0'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
┘2
щ
while_body_85409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_simple_rnn_cell_matmul_readvariableop_resource_0;
7while_simple_rnn_cell_biasadd_readvariableop_resource_0<
8while_simple_rnn_cell_matmul_1_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_simple_rnn_cell_matmul_readvariableop_resource9
5while_simple_rnn_cell_biasadd_readvariableop_resource:
6while_simple_rnn_cell_matmul_1_readvariableop_resourceИв,while/simple_rnn_cell/BiasAdd/ReadVariableOpв+while/simple_rnn_cell/MatMul/ReadVariableOpв-while/simple_rnn_cell/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╤
+while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp6while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02-
+while/simple_rnn_cell/MatMul/ReadVariableOp▀
while/simple_rnn_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:03while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/MatMul╨
,while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp7while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02.
,while/simple_rnn_cell/BiasAdd/ReadVariableOp┘
while/simple_rnn_cell/BiasAddBiasAdd&while/simple_rnn_cell/MatMul:product:04while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/BiasAdd╫
-while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp8while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02/
-while/simple_rnn_cell/MatMul_1/ReadVariableOp╚
while/simple_rnn_cell/MatMul_1MatMulwhile_placeholder_25while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2 
while/simple_rnn_cell/MatMul_1├
while/simple_rnn_cell/addAddV2&while/simple_rnn_cell/BiasAdd:output:0(while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/addС
while/simple_rnn_cell/TanhTanhwhile/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
while/simple_rnn_cell/Tanhт
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1ы
while/IdentityIdentitywhile/add_1:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity■
while/Identity_1Identitywhile_while_maximum_iterations-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1э
while/Identity_2Identitywhile/add:z:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3П
while/Identity_4Identitywhile/simple_rnn_cell/Tanh:y:0-^while/simple_rnn_cell/BiasAdd/ReadVariableOp,^while/simple_rnn_cell/MatMul/ReadVariableOp.^while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2
while/Identity_4")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"p
5while_simple_rnn_cell_biasadd_readvariableop_resource7while_simple_rnn_cell_biasadd_readvariableop_resource_0"r
6while_simple_rnn_cell_matmul_1_readvariableop_resource8while_simple_rnn_cell_matmul_1_readvariableop_resource_0"n
4while_simple_rnn_cell_matmul_readvariableop_resource6while_simple_rnn_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2\
,while/simple_rnn_cell/BiasAdd/ReadVariableOp,while/simple_rnn_cell/BiasAdd/ReadVariableOp2Z
+while/simple_rnn_cell/MatMul/ReadVariableOp+while/simple_rnn_cell/MatMul/ReadVariableOp2^
-while/simple_rnn_cell/MatMul_1/ReadVariableOp-while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
ь
З
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85337

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         А2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2
batchnorm/add_1р
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
ч	
р
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_85858

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhН
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
Ъ
C
'__inference_reshape_layer_call_fn_85254

inputs
identity┼
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_835842
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Лх
А
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_85114

inputs/
+rnn_dense3_1_matmul_readvariableop_resource0
,rnn_dense3_1_biasadd_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource@
<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resourceA
=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resourceB
>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource/
+rnn_dense3_2_matmul_readvariableop_resource0
,rnn_dense3_2_biasadd_readvariableop_resource;
7batch_normalization_3_batchnorm_readvariableop_resource?
;batch_normalization_3_batchnorm_mul_readvariableop_resource=
9batch_normalization_3_batchnorm_readvariableop_1_resource=
9batch_normalization_3_batchnorm_readvariableop_2_resource/
+rnn_dense3_3_matmul_readvariableop_resource0
,rnn_dense3_3_biasadd_readvariableop_resource1
-rnn_dense3_out_matmul_readvariableop_resource2
.rnn_dense3_out_biasadd_readvariableop_resource
identityИв.batch_normalization_1/batchnorm/ReadVariableOpв0batch_normalization_1/batchnorm/ReadVariableOp_1в0batch_normalization_1/batchnorm/ReadVariableOp_2в2batch_normalization_1/batchnorm/mul/ReadVariableOpв.batch_normalization_2/batchnorm/ReadVariableOpв0batch_normalization_2/batchnorm/ReadVariableOp_1в0batch_normalization_2/batchnorm/ReadVariableOp_2в2batch_normalization_2/batchnorm/mul/ReadVariableOpв.batch_normalization_3/batchnorm/ReadVariableOpв0batch_normalization_3/batchnorm/ReadVariableOp_1в0batch_normalization_3/batchnorm/ReadVariableOp_2в2batch_normalization_3/batchnorm/mul/ReadVariableOpв#rnn_dense3_1/BiasAdd/ReadVariableOpв"rnn_dense3_1/MatMul/ReadVariableOpв#rnn_dense3_2/BiasAdd/ReadVariableOpв"rnn_dense3_2/MatMul/ReadVariableOpв#rnn_dense3_3/BiasAdd/ReadVariableOpв"rnn_dense3_3/MatMul/ReadVariableOpв%rnn_dense3_out/BiasAdd/ReadVariableOpв$rnn_dense3_out/MatMul/ReadVariableOpв4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpв3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpв5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpвrnn_dense3_r1/while╢
"rnn_dense3_1/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_1_matmul_readvariableop_resource* 
_output_shapes
:
шА*
dtype02$
"rnn_dense3_1/MatMul/ReadVariableOpЫ
rnn_dense3_1/MatMulMatMulinputs*rnn_dense3_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/MatMul┤
#rnn_dense3_1/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#rnn_dense3_1/BiasAdd/ReadVariableOp╢
rnn_dense3_1/BiasAddBiasAddrnn_dense3_1/MatMul:product:0+rnn_dense3_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/BiasAddА
rnn_dense3_1/TanhTanhrnn_dense3_1/BiasAdd:output:0*
T0*(
_output_shapes
:         А2
rnn_dense3_1/Tanhc
reshape/ShapeShapernn_dense3_1/Tanh:y:0*
T0*
_output_shapes
:2
reshape/ShapeД
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackИ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1И
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2Т
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :А2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2╚
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЫ
reshape/ReshapeReshapernn_dense3_1/Tanh:y:0reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:         А2
reshape/ReshapeЕ
dropout_1/IdentityIdentityreshape/Reshape:output:0*
T0*,
_output_shapes
:         А2
dropout_1/Identity╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/mul╥
%batch_normalization_1/batchnorm/mul_1Muldropout_1/Identity:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         А2'
%batch_normalization_1/batchnorm/mul_1┌
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▌
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:2'
%batch_normalization_1/batchnorm/mul_2┌
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2█
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         А2'
%batch_normalization_1/batchnorm/add_1Г
rnn_dense3_r1/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
rnn_dense3_r1/ShapeР
!rnn_dense3_r1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!rnn_dense3_r1/strided_slice/stackФ
#rnn_dense3_r1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#rnn_dense3_r1/strided_slice/stack_1Ф
#rnn_dense3_r1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#rnn_dense3_r1/strided_slice/stack_2╢
rnn_dense3_r1/strided_sliceStridedSlicernn_dense3_r1/Shape:output:0*rnn_dense3_r1/strided_slice/stack:output:0,rnn_dense3_r1/strided_slice/stack_1:output:0,rnn_dense3_r1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_dense3_r1/strided_slicex
rnn_dense3_r1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
rnn_dense3_r1/zeros/mul/yд
rnn_dense3_r1/zeros/mulMul$rnn_dense3_r1/strided_slice:output:0"rnn_dense3_r1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/zeros/mul{
rnn_dense3_r1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
rnn_dense3_r1/zeros/Less/yЯ
rnn_dense3_r1/zeros/LessLessrnn_dense3_r1/zeros/mul:z:0#rnn_dense3_r1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rnn_dense3_r1/zeros/Less~
rnn_dense3_r1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
rnn_dense3_r1/zeros/packed/1╗
rnn_dense3_r1/zeros/packedPack$rnn_dense3_r1/strided_slice:output:0%rnn_dense3_r1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rnn_dense3_r1/zeros/packed{
rnn_dense3_r1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rnn_dense3_r1/zeros/Constн
rnn_dense3_r1/zerosFill#rnn_dense3_r1/zeros/packed:output:0"rnn_dense3_r1/zeros/Const:output:0*
T0*'
_output_shapes
:         @2
rnn_dense3_r1/zerosС
rnn_dense3_r1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
rnn_dense3_r1/transpose/perm╚
rnn_dense3_r1/transpose	Transpose)batch_normalization_1/batchnorm/add_1:z:0%rnn_dense3_r1/transpose/perm:output:0*
T0*,
_output_shapes
:А         2
rnn_dense3_r1/transposey
rnn_dense3_r1/Shape_1Shapernn_dense3_r1/transpose:y:0*
T0*
_output_shapes
:2
rnn_dense3_r1/Shape_1Ф
#rnn_dense3_r1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rnn_dense3_r1/strided_slice_1/stackШ
%rnn_dense3_r1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_1/stack_1Ш
%rnn_dense3_r1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_1/stack_2┬
rnn_dense3_r1/strided_slice_1StridedSlicernn_dense3_r1/Shape_1:output:0,rnn_dense3_r1/strided_slice_1/stack:output:0.rnn_dense3_r1/strided_slice_1/stack_1:output:0.rnn_dense3_r1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rnn_dense3_r1/strided_slice_1б
)rnn_dense3_r1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)rnn_dense3_r1/TensorArrayV2/element_shapeъ
rnn_dense3_r1/TensorArrayV2TensorListReserve2rnn_dense3_r1/TensorArrayV2/element_shape:output:0&rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_dense3_r1/TensorArrayV2█
Crnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape░
5rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorrnn_dense3_r1/transpose:y:0Lrnn_dense3_r1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type027
5rnn_dense3_r1/TensorArrayUnstack/TensorListFromTensorФ
#rnn_dense3_r1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rnn_dense3_r1/strided_slice_2/stackШ
%rnn_dense3_r1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_2/stack_1Ш
%rnn_dense3_r1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_2/stack_2╨
rnn_dense3_r1/strided_slice_2StridedSlicernn_dense3_r1/transpose:y:0,rnn_dense3_r1/strided_slice_2/stack:output:0.rnn_dense3_r1/strided_slice_2/stack_1:output:0.rnn_dense3_r1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
rnn_dense3_r1/strided_slice_2ч
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype025
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOpэ
$rnn_dense3_r1/simple_rnn_cell/MatMulMatMul&rnn_dense3_r1/strided_slice_2:output:0;rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2&
$rnn_dense3_r1/simple_rnn_cell/MatMulц
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp∙
%rnn_dense3_r1/simple_rnn_cell/BiasAddBiasAdd.rnn_dense3_r1/simple_rnn_cell/MatMul:product:0<rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2'
%rnn_dense3_r1/simple_rnn_cell/BiasAddэ
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype027
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOpщ
&rnn_dense3_r1/simple_rnn_cell/MatMul_1MatMulrnn_dense3_r1/zeros:output:0=rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2(
&rnn_dense3_r1/simple_rnn_cell/MatMul_1у
!rnn_dense3_r1/simple_rnn_cell/addAddV2.rnn_dense3_r1/simple_rnn_cell/BiasAdd:output:00rnn_dense3_r1/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2#
!rnn_dense3_r1/simple_rnn_cell/addй
"rnn_dense3_r1/simple_rnn_cell/TanhTanh%rnn_dense3_r1/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2$
"rnn_dense3_r1/simple_rnn_cell/Tanhл
+rnn_dense3_r1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2-
+rnn_dense3_r1/TensorArrayV2_1/element_shapeЁ
rnn_dense3_r1/TensorArrayV2_1TensorListReserve4rnn_dense3_r1/TensorArrayV2_1/element_shape:output:0&rnn_dense3_r1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
rnn_dense3_r1/TensorArrayV2_1j
rnn_dense3_r1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
rnn_dense3_r1/timeЫ
&rnn_dense3_r1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2(
&rnn_dense3_r1/while/maximum_iterationsЖ
 rnn_dense3_r1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2"
 rnn_dense3_r1/while/loop_counterА
rnn_dense3_r1/whileWhile)rnn_dense3_r1/while/loop_counter:output:0/rnn_dense3_r1/while/maximum_iterations:output:0rnn_dense3_r1/time:output:0&rnn_dense3_r1/TensorArrayV2_1:handle:0rnn_dense3_r1/zeros:output:0&rnn_dense3_r1/strided_slice_1:output:0Ernn_dense3_r1/TensorArrayUnstack/TensorListFromTensor:output_handle:0<rnn_dense3_r1_simple_rnn_cell_matmul_readvariableop_resource=rnn_dense3_r1_simple_rnn_cell_biasadd_readvariableop_resource>rnn_dense3_r1_simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	**
body"R 
rnn_dense3_r1_while_body_84994**
cond"R 
rnn_dense3_r1_while_cond_84993*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
rnn_dense3_r1/while╤
>rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2@
>rnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shapeб
0rnn_dense3_r1/TensorArrayV2Stack/TensorListStackTensorListStackrnn_dense3_r1/while:output:3Grnn_dense3_r1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype022
0rnn_dense3_r1/TensorArrayV2Stack/TensorListStackЭ
#rnn_dense3_r1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2%
#rnn_dense3_r1/strided_slice_3/stackШ
%rnn_dense3_r1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%rnn_dense3_r1/strided_slice_3/stack_1Ш
%rnn_dense3_r1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rnn_dense3_r1/strided_slice_3/stack_2ю
rnn_dense3_r1/strided_slice_3StridedSlice9rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:0,rnn_dense3_r1/strided_slice_3/stack:output:0.rnn_dense3_r1/strided_slice_3/stack_1:output:0.rnn_dense3_r1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
rnn_dense3_r1/strided_slice_3Х
rnn_dense3_r1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
rnn_dense3_r1/transpose_1/perm▐
rnn_dense3_r1/transpose_1	Transpose9rnn_dense3_r1/TensorArrayV2Stack/TensorListStack:tensor:0'rnn_dense3_r1/transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
rnn_dense3_r1/transpose_1О
dropout_2/IdentityIdentity&rnn_dense3_r1/strided_slice_3:output:0*
T0*'
_output_shapes
:         @2
dropout_2/Identity╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul═
%batch_normalization_2/batchnorm/mul_1Muldropout_2/Identity:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub▌
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2'
%batch_normalization_2/batchnorm/add_1┤
"rnn_dense3_2/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02$
"rnn_dense3_2/MatMul/ReadVariableOp╜
rnn_dense3_2/MatMulMatMul)batch_normalization_2/batchnorm/add_1:z:0*rnn_dense3_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/MatMul│
#rnn_dense3_2/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#rnn_dense3_2/BiasAdd/ReadVariableOp╡
rnn_dense3_2/BiasAddBiasAddrnn_dense3_2/MatMul:product:0+rnn_dense3_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/BiasAdd
rnn_dense3_2/TanhTanhrnn_dense3_2/BiasAdd:output:0*
T0*'
_output_shapes
:          2
rnn_dense3_2/Tanh}
dropout_3/IdentityIdentityrnn_dense3_2/Tanh:y:0*
T0*'
_output_shapes
:          2
dropout_3/Identity╘
.batch_normalization_3/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_3/batchnorm/ReadVariableOpУ
%batch_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_3/batchnorm/add/yр
#batch_normalization_3/batchnorm/addAddV26batch_normalization_3/batchnorm/ReadVariableOp:value:0.batch_normalization_3/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/addе
%batch_normalization_3/batchnorm/RsqrtRsqrt'batch_normalization_3/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/Rsqrtр
2batch_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_3/batchnorm/mul/ReadVariableOp▌
#batch_normalization_3/batchnorm/mulMul)batch_normalization_3/batchnorm/Rsqrt:y:0:batch_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/mul═
%batch_normalization_3/batchnorm/mul_1Muldropout_3/Identity:output:0'batch_normalization_3/batchnorm/mul:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_3/batchnorm/mul_1┌
0batch_normalization_3/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_1▌
%batch_normalization_3/batchnorm/mul_2Mul8batch_normalization_3/batchnorm/ReadVariableOp_1:value:0'batch_normalization_3/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_3/batchnorm/mul_2┌
0batch_normalization_3/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_3_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_3/batchnorm/ReadVariableOp_2█
#batch_normalization_3/batchnorm/subSub8batch_normalization_3/batchnorm/ReadVariableOp_2:value:0)batch_normalization_3/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_3/batchnorm/sub▌
%batch_normalization_3/batchnorm/add_1AddV2)batch_normalization_3/batchnorm/mul_1:z:0'batch_normalization_3/batchnorm/sub:z:0*
T0*'
_output_shapes
:          2'
%batch_normalization_3/batchnorm/add_1┤
"rnn_dense3_3/MatMul/ReadVariableOpReadVariableOp+rnn_dense3_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"rnn_dense3_3/MatMul/ReadVariableOp╜
rnn_dense3_3/MatMulMatMul)batch_normalization_3/batchnorm/add_1:z:0*rnn_dense3_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/MatMul│
#rnn_dense3_3/BiasAdd/ReadVariableOpReadVariableOp,rnn_dense3_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#rnn_dense3_3/BiasAdd/ReadVariableOp╡
rnn_dense3_3/BiasAddBiasAddrnn_dense3_3/MatMul:product:0+rnn_dense3_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/BiasAdd
rnn_dense3_3/TanhTanhrnn_dense3_3/BiasAdd:output:0*
T0*'
_output_shapes
:         2
rnn_dense3_3/Tanh║
$rnn_dense3_out/MatMul/ReadVariableOpReadVariableOp-rnn_dense3_out_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$rnn_dense3_out/MatMul/ReadVariableOpп
rnn_dense3_out/MatMulMatMulrnn_dense3_3/Tanh:y:0,rnn_dense3_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_out/MatMul╣
%rnn_dense3_out/BiasAdd/ReadVariableOpReadVariableOp.rnn_dense3_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%rnn_dense3_out/BiasAdd/ReadVariableOp╜
rnn_dense3_out/BiasAddBiasAddrnn_dense3_out/MatMul:product:0-rnn_dense3_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
rnn_dense3_out/BiasAdd┬	
IdentityIdentityrnn_dense3_out/BiasAdd:output:0/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp/^batch_normalization_3/batchnorm/ReadVariableOp1^batch_normalization_3/batchnorm/ReadVariableOp_11^batch_normalization_3/batchnorm/ReadVariableOp_23^batch_normalization_3/batchnorm/mul/ReadVariableOp$^rnn_dense3_1/BiasAdd/ReadVariableOp#^rnn_dense3_1/MatMul/ReadVariableOp$^rnn_dense3_2/BiasAdd/ReadVariableOp#^rnn_dense3_2/MatMul/ReadVariableOp$^rnn_dense3_3/BiasAdd/ReadVariableOp#^rnn_dense3_3/MatMul/ReadVariableOp&^rnn_dense3_out/BiasAdd/ReadVariableOp%^rnn_dense3_out/MatMul/ReadVariableOp5^rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp4^rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp6^rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp^rnn_dense3_r1/while*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2`
.batch_normalization_3/batchnorm/ReadVariableOp.batch_normalization_3/batchnorm/ReadVariableOp2d
0batch_normalization_3/batchnorm/ReadVariableOp_10batch_normalization_3/batchnorm/ReadVariableOp_12d
0batch_normalization_3/batchnorm/ReadVariableOp_20batch_normalization_3/batchnorm/ReadVariableOp_22h
2batch_normalization_3/batchnorm/mul/ReadVariableOp2batch_normalization_3/batchnorm/mul/ReadVariableOp2J
#rnn_dense3_1/BiasAdd/ReadVariableOp#rnn_dense3_1/BiasAdd/ReadVariableOp2H
"rnn_dense3_1/MatMul/ReadVariableOp"rnn_dense3_1/MatMul/ReadVariableOp2J
#rnn_dense3_2/BiasAdd/ReadVariableOp#rnn_dense3_2/BiasAdd/ReadVariableOp2H
"rnn_dense3_2/MatMul/ReadVariableOp"rnn_dense3_2/MatMul/ReadVariableOp2J
#rnn_dense3_3/BiasAdd/ReadVariableOp#rnn_dense3_3/BiasAdd/ReadVariableOp2H
"rnn_dense3_3/MatMul/ReadVariableOp"rnn_dense3_3/MatMul/ReadVariableOp2N
%rnn_dense3_out/BiasAdd/ReadVariableOp%rnn_dense3_out/BiasAdd/ReadVariableOp2L
$rnn_dense3_out/MatMul/ReadVariableOp$rnn_dense3_out/MatMul/ReadVariableOp2l
4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp4rnn_dense3_r1/simple_rnn_cell/BiasAdd/ReadVariableOp2j
3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp3rnn_dense3_r1/simple_rnn_cell/MatMul/ReadVariableOp2n
5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp5rnn_dense3_r1/simple_rnn_cell/MatMul_1/ReadVariableOp2*
rnn_dense3_r1/whilernn_dense3_r1/while:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
П
С
-__inference_rnn_dense3_r1_layer_call_fn_85598

inputs
unknown
	unknown_0
	unknown_1
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_838272
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
МG
Й
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_83939

inputs2
.simple_rnn_cell_matmul_readvariableop_resource3
/simple_rnn_cell_biasadd_readvariableop_resource4
0simple_rnn_cell_matmul_1_readvariableop_resource
identityИв&simple_rnn_cell/BiasAdd/ReadVariableOpв%simple_rnn_cell/MatMul/ReadVariableOpв'simple_rnn_cell/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :@2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :@2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @2
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:А         2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask2
strided_slice_2╜
%simple_rnn_cell/MatMul/ReadVariableOpReadVariableOp.simple_rnn_cell_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%simple_rnn_cell/MatMul/ReadVariableOp╡
simple_rnn_cell/MatMulMatMulstrided_slice_2:output:0-simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul╝
&simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOp/simple_rnn_cell_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&simple_rnn_cell/BiasAdd/ReadVariableOp┴
simple_rnn_cell/BiasAddBiasAdd simple_rnn_cell/MatMul:product:0.simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/BiasAdd├
'simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOp0simple_rnn_cell_matmul_1_readvariableop_resource*
_output_shapes

:@@*
dtype02)
'simple_rnn_cell/MatMul_1/ReadVariableOp▒
simple_rnn_cell/MatMul_1MatMulzeros:output:0/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/MatMul_1л
simple_rnn_cell/addAddV2 simple_rnn_cell/BiasAdd:output:0"simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/add
simple_rnn_cell/TanhTanhsimple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @2
simple_rnn_cell/TanhП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╝
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.simple_rnn_cell_matmul_readvariableop_resource/simple_rnn_cell_biasadd_readvariableop_resource0simple_rnn_cell_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :         @: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_83873*
condR
while_cond_83872*8
output_shapes'
%: : : : :         @: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    @   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:А         @*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         @*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         А@2
transpose_1я
IdentityIdentitystrided_slice_3:output:0'^simple_rnn_cell/BiasAdd/ReadVariableOp&^simple_rnn_cell/MatMul/ReadVariableOp(^simple_rnn_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:         А:::2P
&simple_rnn_cell/BiasAdd/ReadVariableOp&simple_rnn_cell/BiasAdd/ReadVariableOp2N
%simple_rnn_cell/MatMul/ReadVariableOp%simple_rnn_cell/MatMul/ReadVariableOp2R
'simple_rnn_cell/MatMul_1/ReadVariableOp'simple_rnn_cell/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
З?
╒
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84537

inputs
rnn_dense3_1_84478
rnn_dense3_1_84480
batch_normalization_1_84485
batch_normalization_1_84487
batch_normalization_1_84489
batch_normalization_1_84491
rnn_dense3_r1_84494
rnn_dense3_r1_84496
rnn_dense3_r1_84498
batch_normalization_2_84502
batch_normalization_2_84504
batch_normalization_2_84506
batch_normalization_2_84508
rnn_dense3_2_84511
rnn_dense3_2_84513
batch_normalization_3_84517
batch_normalization_3_84519
batch_normalization_3_84521
batch_normalization_3_84523
rnn_dense3_3_84526
rnn_dense3_3_84528
rnn_dense3_out_84531
rnn_dense3_out_84533
identityИв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв$rnn_dense3_1/StatefulPartitionedCallв$rnn_dense3_2/StatefulPartitionedCallв$rnn_dense3_3/StatefulPartitionedCallв&rnn_dense3_out/StatefulPartitionedCallв%rnn_dense3_r1/StatefulPartitionedCallж
$rnn_dense3_1/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_dense3_1_84478rnn_dense3_1_84480*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_835552&
$rnn_dense3_1/StatefulPartitionedCall№
reshape/PartitionedCallPartitionedCall-rnn_dense3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_835842
reshape/PartitionedCallї
dropout_1/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836092
dropout_1/PartitionedCall▒
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0batch_normalization_1_84485batch_normalization_1_84487batch_normalization_1_84489batch_normalization_1_84491*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836782/
-batch_normalization_1/StatefulPartitionedCallё
%rnn_dense3_r1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0rnn_dense3_r1_84494rnn_dense3_r1_84496rnn_dense3_r1_84498*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_839392'
%rnn_dense3_r1/StatefulPartitionedCall■
dropout_2/PartitionedCallPartitionedCall.rnn_dense3_r1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839872
dropout_2/PartitionedCallм
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0batch_normalization_2_84502batch_normalization_2_84504batch_normalization_2_84506batch_normalization_2_84508*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840562/
-batch_normalization_2/StatefulPartitionedCall╒
$rnn_dense3_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0rnn_dense3_2_84511rnn_dense3_2_84513*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_841042&
$rnn_dense3_2/StatefulPartitionedCall¤
dropout_3/PartitionedCallPartitionedCall-rnn_dense3_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841372
dropout_3/PartitionedCallм
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0batch_normalization_3_84517batch_normalization_3_84519batch_normalization_3_84521batch_normalization_3_84523*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_842062/
-batch_normalization_3/StatefulPartitionedCall╒
$rnn_dense3_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0rnn_dense3_3_84526rnn_dense3_3_84528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_842542&
$rnn_dense3_3/StatefulPartitionedCall╓
&rnn_dense3_out/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_3/StatefulPartitionedCall:output:0rnn_dense3_out_84531rnn_dense3_out_84533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_842802(
&rnn_dense3_out/StatefulPartitionedCall┘
IdentityIdentity/rnn_dense3_out/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall%^rnn_dense3_1/StatefulPartitionedCall%^rnn_dense3_2/StatefulPartitionedCall%^rnn_dense3_3/StatefulPartitionedCall'^rnn_dense3_out/StatefulPartitionedCall&^rnn_dense3_r1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2L
$rnn_dense3_1/StatefulPartitionedCall$rnn_dense3_1/StatefulPartitionedCall2L
$rnn_dense3_2/StatefulPartitionedCall$rnn_dense3_2/StatefulPartitionedCall2L
$rnn_dense3_3/StatefulPartitionedCall$rnn_dense3_3/StatefulPartitionedCall2P
&rnn_dense3_out/StatefulPartitionedCall&rnn_dense3_out/StatefulPartitionedCall2N
%rnn_dense3_r1/StatefulPartitionedCall%rnn_dense3_r1/StatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
ФO
█
)rnn_dense3_rnn_dense3_r1_while_body_83420N
Jrnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_loop_counterT
Prnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_maximum_iterations.
*rnn_dense3_rnn_dense3_r1_while_placeholder0
,rnn_dense3_rnn_dense3_r1_while_placeholder_10
,rnn_dense3_rnn_dense3_r1_while_placeholder_2M
Irnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_strided_slice_1_0К
Еrnn_dense3_rnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0S
Ornn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0T
Prnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0U
Qrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0+
'rnn_dense3_rnn_dense3_r1_while_identity-
)rnn_dense3_rnn_dense3_r1_while_identity_1-
)rnn_dense3_rnn_dense3_r1_while_identity_2-
)rnn_dense3_rnn_dense3_r1_while_identity_3-
)rnn_dense3_rnn_dense3_r1_while_identity_4K
Grnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_strided_slice_1И
Гrnn_dense3_rnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorQ
Mrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceR
Nrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourceS
Ornn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceИвErnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpвDrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpвFrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpї
Prnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       2R
Prnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shapeъ
Brnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemЕrnn_dense3_rnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0*rnn_dense3_rnn_dense3_r1_while_placeholderYrnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype02D
Brnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItemЬ
Drnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpReadVariableOpOrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0*
_output_shapes

:@*
dtype02F
Drnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp├
5rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMulMatMulIrnn_dense3/rnn_dense3_r1/while/TensorArrayV2Read/TensorListGetItem:item:0Lrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @27
5rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMulЫ
Ernn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpReadVariableOpPrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0*
_output_shapes
:@*
dtype02G
Ernn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp╜
6rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAddBiasAdd?rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul:product:0Mrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @28
6rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAddв
Frnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpReadVariableOpQrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0*
_output_shapes

:@@*
dtype02H
Frnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpм
7rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1MatMul,rnn_dense3_rnn_dense3_r1_while_placeholder_2Nrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @29
7rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1з
2rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/addAddV2?rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd:output:0Arnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1:product:0*
T0*'
_output_shapes
:         @24
2rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/add▄
3rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/TanhTanh6rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/add:z:0*
T0*'
_output_shapes
:         @25
3rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/Tanh▀
Crnn_dense3/rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem,rnn_dense3_rnn_dense3_r1_while_placeholder_1*rnn_dense3_rnn_dense3_r1_while_placeholder7rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0*
_output_shapes
: *
element_dtype02E
Crnn_dense3/rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItemО
$rnn_dense3/rnn_dense3_r1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$rnn_dense3/rnn_dense3_r1/while/add/y═
"rnn_dense3/rnn_dense3_r1/while/addAddV2*rnn_dense3_rnn_dense3_r1_while_placeholder-rnn_dense3/rnn_dense3_r1/while/add/y:output:0*
T0*
_output_shapes
: 2$
"rnn_dense3/rnn_dense3_r1/while/addТ
&rnn_dense3/rnn_dense3_r1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&rnn_dense3/rnn_dense3_r1/while/add_1/yє
$rnn_dense3/rnn_dense3_r1/while/add_1AddV2Jrnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_loop_counter/rnn_dense3/rnn_dense3_r1/while/add_1/y:output:0*
T0*
_output_shapes
: 2&
$rnn_dense3/rnn_dense3_r1/while/add_1Б
'rnn_dense3/rnn_dense3_r1/while/IdentityIdentity(rnn_dense3/rnn_dense3_r1/while/add_1:z:0F^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpE^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpG^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2)
'rnn_dense3/rnn_dense3_r1/while/Identityн
)rnn_dense3/rnn_dense3_r1/while/Identity_1IdentityPrnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_while_maximum_iterationsF^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpE^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpG^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)rnn_dense3/rnn_dense3_r1/while/Identity_1Г
)rnn_dense3/rnn_dense3_r1/while/Identity_2Identity&rnn_dense3/rnn_dense3_r1/while/add:z:0F^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpE^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpG^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)rnn_dense3/rnn_dense3_r1/while/Identity_2░
)rnn_dense3/rnn_dense3_r1/while/Identity_3IdentitySrnn_dense3/rnn_dense3_r1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0F^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpE^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpG^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2+
)rnn_dense3/rnn_dense3_r1/while/Identity_3е
)rnn_dense3/rnn_dense3_r1/while/Identity_4Identity7rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/Tanh:y:0F^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpE^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpG^rnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @2+
)rnn_dense3/rnn_dense3_r1/while/Identity_4"[
'rnn_dense3_rnn_dense3_r1_while_identity0rnn_dense3/rnn_dense3_r1/while/Identity:output:0"_
)rnn_dense3_rnn_dense3_r1_while_identity_12rnn_dense3/rnn_dense3_r1/while/Identity_1:output:0"_
)rnn_dense3_rnn_dense3_r1_while_identity_22rnn_dense3/rnn_dense3_r1/while/Identity_2:output:0"_
)rnn_dense3_rnn_dense3_r1_while_identity_32rnn_dense3/rnn_dense3_r1/while/Identity_3:output:0"_
)rnn_dense3_rnn_dense3_r1_while_identity_42rnn_dense3/rnn_dense3_r1/while/Identity_4:output:0"Ф
Grnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_strided_slice_1Irnn_dense3_rnn_dense3_r1_while_rnn_dense3_rnn_dense3_r1_strided_slice_1_0"в
Nrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resourcePrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_biasadd_readvariableop_resource_0"д
Ornn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resourceQrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_1_readvariableop_resource_0"а
Mrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resourceOrnn_dense3_rnn_dense3_r1_while_simple_rnn_cell_matmul_readvariableop_resource_0"О
Гrnn_dense3_rnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensorЕrnn_dense3_rnn_dense3_r1_while_tensorarrayv2read_tensorlistgetitem_rnn_dense3_rnn_dense3_r1_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :         @: : :::2О
Ernn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOpErnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/BiasAdd/ReadVariableOp2М
Drnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOpDrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul/ReadVariableOp2Р
Frnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOpFrnn_dense3/rnn_dense3_r1/while/simple_rnn_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
: 
Ъ
╞
*__inference_rnn_dense3_layer_call_fn_85216

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_845372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
ч	
р
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_84254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhН
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┘C
┴	
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84424

inputs
rnn_dense3_1_84365
rnn_dense3_1_84367
batch_normalization_1_84372
batch_normalization_1_84374
batch_normalization_1_84376
batch_normalization_1_84378
rnn_dense3_r1_84381
rnn_dense3_r1_84383
rnn_dense3_r1_84385
batch_normalization_2_84389
batch_normalization_2_84391
batch_normalization_2_84393
batch_normalization_2_84395
rnn_dense3_2_84398
rnn_dense3_2_84400
batch_normalization_3_84404
batch_normalization_3_84406
batch_normalization_3_84408
batch_normalization_3_84410
rnn_dense3_3_84413
rnn_dense3_3_84415
rnn_dense3_out_84418
rnn_dense3_out_84420
identityИв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallв-batch_normalization_3/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallв!dropout_3/StatefulPartitionedCallв$rnn_dense3_1/StatefulPartitionedCallв$rnn_dense3_2/StatefulPartitionedCallв$rnn_dense3_3/StatefulPartitionedCallв&rnn_dense3_out/StatefulPartitionedCallв%rnn_dense3_r1/StatefulPartitionedCallж
$rnn_dense3_1/StatefulPartitionedCallStatefulPartitionedCallinputsrnn_dense3_1_84365rnn_dense3_1_84367*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_835552&
$rnn_dense3_1/StatefulPartitionedCall№
reshape/PartitionedCallPartitionedCall-rnn_dense3_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_835842
reshape/PartitionedCallН
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_836042#
!dropout_1/StatefulPartitionedCall╖
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0batch_normalization_1_84372batch_normalization_1_84374batch_normalization_1_84376batch_normalization_1_84378*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836582/
-batch_normalization_1/StatefulPartitionedCallё
%rnn_dense3_r1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0rnn_dense3_r1_84381rnn_dense3_r1_84383rnn_dense3_r1_84385*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_838272'
%rnn_dense3_r1/StatefulPartitionedCall║
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall.rnn_dense3_r1/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_839822#
!dropout_2/StatefulPartitionedCall▓
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0batch_normalization_2_84389batch_normalization_2_84391batch_normalization_2_84393batch_normalization_2_84395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_840362/
-batch_normalization_2/StatefulPartitionedCall╒
$rnn_dense3_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0rnn_dense3_2_84398rnn_dense3_2_84400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_841042&
$rnn_dense3_2/StatefulPartitionedCall╣
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_2/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_841322#
!dropout_3/StatefulPartitionedCall▓
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0batch_normalization_3_84404batch_normalization_3_84406batch_normalization_3_84408batch_normalization_3_84410*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_841862/
-batch_normalization_3/StatefulPartitionedCall╒
$rnn_dense3_3/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0rnn_dense3_3_84413rnn_dense3_3_84415*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_842542&
$rnn_dense3_3/StatefulPartitionedCall╓
&rnn_dense3_out/StatefulPartitionedCallStatefulPartitionedCall-rnn_dense3_3/StatefulPartitionedCall:output:0rnn_dense3_out_84418rnn_dense3_out_84420*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_842802(
&rnn_dense3_out/StatefulPartitionedCall┼
IdentityIdentity/rnn_dense3_out/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall%^rnn_dense3_1/StatefulPartitionedCall%^rnn_dense3_2/StatefulPartitionedCall%^rnn_dense3_3/StatefulPartitionedCall'^rnn_dense3_out/StatefulPartitionedCall&^rnn_dense3_r1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesr
p:         ш:::::::::::::::::::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2L
$rnn_dense3_1/StatefulPartitionedCall$rnn_dense3_1/StatefulPartitionedCall2L
$rnn_dense3_2/StatefulPartitionedCall$rnn_dense3_2/StatefulPartitionedCall2L
$rnn_dense3_3/StatefulPartitionedCall$rnn_dense3_3/StatefulPartitionedCall2P
&rnn_dense3_out/StatefulPartitionedCall&rnn_dense3_out/StatefulPartitionedCall2N
%rnn_dense3_r1/StatefulPartitionedCall%rnn_dense3_r1/StatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
Ё	
р
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_83555

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
шА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         А2
TanhО
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
╦
е
while_cond_83872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_83872___redundant_placeholder03
/while_while_cond_83872___redundant_placeholder13
/while_while_cond_83872___redundant_placeholder23
/while_while_cond_83872___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :         @: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         @:

_output_shapes
: :

_output_shapes
:
╟
и
5__inference_batch_normalization_1_layer_call_fn_85350

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_836582
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         А2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         А::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╙
З
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85692

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulv
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subЕ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         @2
batchnorm/add_1█
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs"║L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╚
serving_default┤
R
rnn_dense3_1_input<
$serving_default_rnn_dense3_1_input:0         шB
rnn_dense3_out0
StatefulPartitionedCall:0         tensorflow/serving/predict:Пж
─\
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+т&call_and_return_all_conditional_losses
у_default_save_signature
ф__call__"ИX
_tf_keras_sequentialщW{"class_name": "Sequential", "name": "rnn_dense3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "rnn_dense3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rnn_dense3_1_input"}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [128, 1]}}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "SimpleRNN", "config": {"name": "rnn_dense3_r1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_3", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "rnn_dense3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "rnn_dense3_1_input"}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [128, 1]}}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "SimpleRNN", "config": {"name": "rnn_dense3_r1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_3", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "rnn_dense3_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}, {"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9900000095367432, "epsilon": 1e-05, "amsgrad": false}}}}
°

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"╤
_tf_keras_layer╖{"class_name": "Dense", "name": "rnn_dense3_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_dense3_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "units": 128, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
Ї
	variables
regularization_losses
trainable_variables
	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"у
_tf_keras_layer╔{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [128, 1]}}}
ш
	variables
regularization_losses
trainable_variables
 	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"╫
_tf_keras_layer╜{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
╖	
!axis
	"gamma
#beta
$moving_mean
%moving_variance
&	variables
'regularization_losses
(trainable_variables
)	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 1]}}
 

*cell
+
state_spec
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"╘	
_tf_keras_rnn_layer╢	{"class_name": "SimpleRNN", "name": "rnn_dense3_r1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_dense3_r1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 1]}}
ш
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+я&call_and_return_all_conditional_losses
Ё__call__"╫
_tf_keras_layer╜{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
┤	
4axis
	5gamma
6beta
7moving_mean
8moving_variance
9	variables
:regularization_losses
;trainable_variables
<	keras_api
+ё&call_and_return_all_conditional_losses
Є__call__"▐
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
№

=kernel
>bias
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"╒
_tf_keras_layer╗{"class_name": "Dense", "name": "rnn_dense3_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_dense3_2", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ш
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
+ї&call_and_return_all_conditional_losses
Ў__call__"╫
_tf_keras_layer╜{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.05, "noise_shape": null, "seed": null}}
┤	
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+ў&call_and_return_all_conditional_losses
°__call__"▐
_tf_keras_layer─{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [1], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {"1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
№

Pkernel
Qbias
R	variables
Sregularization_losses
Ttrainable_variables
U	keras_api
+∙&call_and_return_all_conditional_losses
·__call__"╒
_tf_keras_layer╗{"class_name": "Dense", "name": "rnn_dense3_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_dense3_3", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Б

Vkernel
Wbias
X	variables
Yregularization_losses
Ztrainable_variables
[	keras_api
+√&call_and_return_all_conditional_losses
№__call__"┌
_tf_keras_layer└{"class_name": "Dense", "name": "rnn_dense3_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "rnn_dense3_out", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
з
\iter

]beta_1

^beta_2
	_decay
`learning_ratem└m┴"m┬#m├5m─6m┼=m╞>m╟Hm╚Im╔Pm╩Qm╦Vm╠Wm═am╬bm╧cm╨v╤v╥"v╙#v╘5v╒6v╓=v╫>v╪Hv┘Iv┌Pv█Qv▄Vv▌Wv▐av▀bvрcvс"
	optimizer
╬
0
1
"2
#3
$4
%5
a6
b7
c8
59
610
711
812
=13
>14
H15
I16
J17
K18
P19
Q20
V21
W22"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
0
1
"2
#3
a4
b5
c6
57
68
=9
>10
H11
I12
P13
Q14
V15
W16"
trackable_list_wrapper
╬
dlayer_metrics
	variables
regularization_losses
enon_trainable_variables

flayers
trainable_variables
glayer_regularization_losses
hmetrics
ф__call__
у_default_save_signature
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
-
¤serving_default"
signature_map
':%
шА2rnn_dense3_1/kernel
 :А2rnn_dense3_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
ilayer_metrics
	variables
regularization_losses
jnon_trainable_variables

klayers
trainable_variables
llayer_regularization_losses
mmetrics
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
nlayer_metrics
	variables
regularization_losses
onon_trainable_variables

players
trainable_variables
qlayer_regularization_losses
rmetrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
slayer_metrics
	variables
regularization_losses
tnon_trainable_variables

ulayers
trainable_variables
vlayer_regularization_losses
wmetrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
<
"0
#1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
░
xlayer_metrics
&	variables
'regularization_losses
ynon_trainable_variables

zlayers
(trainable_variables
{layer_regularization_losses
|metrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
ц

akernel
brecurrent_kernel
cbias
}	variables
~regularization_losses
trainable_variables
А	keras_api
+■&call_and_return_all_conditional_losses
 __call__"и
_tf_keras_layerО{"class_name": "SimpleRNNCell", "name": "simple_rnn_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "simple_rnn_cell", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0}}
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
┬
Бlayer_metrics
Вmetrics
,	variables
-regularization_losses
Гnon_trainable_variables
Дlayers
.trainable_variables
 Еlayer_regularization_losses
Жstates
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Зlayer_metrics
0	variables
1regularization_losses
Иnon_trainable_variables
Йlayers
2trainable_variables
 Кlayer_regularization_losses
Лmetrics
Ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
<
50
61
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
╡
Мlayer_metrics
9	variables
:regularization_losses
Нnon_trainable_variables
Оlayers
;trainable_variables
 Пlayer_regularization_losses
Рmetrics
Є__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
%:#@ 2rnn_dense3_2/kernel
: 2rnn_dense3_2/bias
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
╡
Сlayer_metrics
?	variables
@regularization_losses
Тnon_trainable_variables
Уlayers
Atrainable_variables
 Фlayer_regularization_losses
Хmetrics
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Цlayer_metrics
C	variables
Dregularization_losses
Чnon_trainable_variables
Шlayers
Etrainable_variables
 Щlayer_regularization_losses
Ъmetrics
Ў__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
<
H0
I1
J2
K3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
╡
Ыlayer_metrics
L	variables
Mregularization_losses
Ьnon_trainable_variables
Эlayers
Ntrainable_variables
 Юlayer_regularization_losses
Яmetrics
°__call__
+ў&call_and_return_all_conditional_losses
'ў"call_and_return_conditional_losses"
_generic_user_object
%:# 2rnn_dense3_3/kernel
:2rnn_dense3_3/bias
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
╡
аlayer_metrics
R	variables
Sregularization_losses
бnon_trainable_variables
вlayers
Ttrainable_variables
 гlayer_regularization_losses
дmetrics
·__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
':%2rnn_dense3_out/kernel
!:2rnn_dense3_out/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
╡
еlayer_metrics
X	variables
Yregularization_losses
жnon_trainable_variables
зlayers
Ztrainable_variables
 иlayer_regularization_losses
йmetrics
№__call__
+√&call_and_return_all_conditional_losses
'√"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
6:4@2$rnn_dense3_r1/simple_rnn_cell/kernel
@:>@@2.rnn_dense3_r1/simple_rnn_cell/recurrent_kernel
0:.@2"rnn_dense3_r1/simple_rnn_cell/bias
 "
trackable_dict_wrapper
J
$0
%1
72
83
J4
K5"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
8
к0
л1
м2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
a0
b1
c2"
trackable_list_wrapper
╡
нlayer_metrics
}	variables
~regularization_losses
оnon_trainable_variables
пlayers
trainable_variables
 ░layer_regularization_losses
▒metrics
 __call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┐

▓total

│count
┤	variables
╡	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
°

╢total

╖count
╕
_fn_kwargs
╣	variables
║	keras_api"м
_tf_keras_metricС{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}}
∙

╗total

╝count
╜
_fn_kwargs
╛	variables
┐	keras_api"н
_tf_keras_metricТ{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
▓0
│1"
trackable_list_wrapper
.
┤	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╢0
╖1"
trackable_list_wrapper
.
╣	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
╗0
╝1"
trackable_list_wrapper
.
╛	variables"
_generic_user_object
,:*
шА2Adam/rnn_dense3_1/kernel/m
%:#А2Adam/rnn_dense3_1/bias/m
.:,2"Adam/batch_normalization_1/gamma/m
-:+2!Adam/batch_normalization_1/beta/m
.:,@2"Adam/batch_normalization_2/gamma/m
-:+@2!Adam/batch_normalization_2/beta/m
*:(@ 2Adam/rnn_dense3_2/kernel/m
$:" 2Adam/rnn_dense3_2/bias/m
.:, 2"Adam/batch_normalization_3/gamma/m
-:+ 2!Adam/batch_normalization_3/beta/m
*:( 2Adam/rnn_dense3_3/kernel/m
$:"2Adam/rnn_dense3_3/bias/m
,:*2Adam/rnn_dense3_out/kernel/m
&:$2Adam/rnn_dense3_out/bias/m
;:9@2+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/m
E:C@@25Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/m
5:3@2)Adam/rnn_dense3_r1/simple_rnn_cell/bias/m
,:*
шА2Adam/rnn_dense3_1/kernel/v
%:#А2Adam/rnn_dense3_1/bias/v
.:,2"Adam/batch_normalization_1/gamma/v
-:+2!Adam/batch_normalization_1/beta/v
.:,@2"Adam/batch_normalization_2/gamma/v
-:+@2!Adam/batch_normalization_2/beta/v
*:(@ 2Adam/rnn_dense3_2/kernel/v
$:" 2Adam/rnn_dense3_2/bias/v
.:, 2"Adam/batch_normalization_3/gamma/v
-:+ 2!Adam/batch_normalization_3/beta/v
*:( 2Adam/rnn_dense3_3/kernel/v
$:"2Adam/rnn_dense3_3/bias/v
,:*2Adam/rnn_dense3_out/kernel/v
&:$2Adam/rnn_dense3_out/bias/v
;:9@2+Adam/rnn_dense3_r1/simple_rnn_cell/kernel/v
E:C@@25Adam/rnn_dense3_r1/simple_rnn_cell/recurrent_kernel/v
5:3@2)Adam/rnn_dense3_r1/simple_rnn_cell/bias/v
т2▀
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_85114
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84297
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84915
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84359└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
 __inference__wrapped_model_83540┬
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *2в/
-К*
rnn_dense3_1_input         ш
Ў2є
*__inference_rnn_dense3_layer_call_fn_84586
*__inference_rnn_dense3_layer_call_fn_85216
*__inference_rnn_dense3_layer_call_fn_85165
*__inference_rnn_dense3_layer_call_fn_84473└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ё2ю
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_85227в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_rnn_dense3_1_layer_call_fn_85236в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_reshape_layer_call_and_return_conditional_losses_85249в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_reshape_layer_call_fn_85254в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
D__inference_dropout_1_layer_call_and_return_conditional_losses_85271
D__inference_dropout_1_layer_call_and_return_conditional_losses_85266┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Р2Н
)__inference_dropout_1_layer_call_fn_85276
)__inference_dropout_1_layer_call_fn_85281┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85337
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85317┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_1_layer_call_fn_85363
5__inference_batch_normalization_1_layer_call_fn_85350┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85587
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85475╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╣2╢
-__inference_rnn_dense3_r1_layer_call_fn_85609
-__inference_rnn_dense3_r1_layer_call_fn_85598╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╞2├
D__inference_dropout_2_layer_call_and_return_conditional_losses_85626
D__inference_dropout_2_layer_call_and_return_conditional_losses_85621┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Р2Н
)__inference_dropout_2_layer_call_fn_85636
)__inference_dropout_2_layer_call_fn_85631┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85672
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85692┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_2_layer_call_fn_85718
5__inference_batch_normalization_2_layer_call_fn_85705┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ё2ю
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_85729в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_rnn_dense3_2_layer_call_fn_85738в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞2├
D__inference_dropout_3_layer_call_and_return_conditional_losses_85755
D__inference_dropout_3_layer_call_and_return_conditional_losses_85750┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Р2Н
)__inference_dropout_3_layer_call_fn_85765
)__inference_dropout_3_layer_call_fn_85760┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85801
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85821┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2е
5__inference_batch_normalization_3_layer_call_fn_85834
5__inference_batch_normalization_3_layer_call_fn_85847┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ё2ю
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_85858в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙
,__inference_rnn_dense3_3_layer_call_fn_85867в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
є2Ё
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_85877в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_rnn_dense3_out_layer_call_fn_85886в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒B╥
#__inference_signature_wrapper_84647rnn_dense3_1_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
─2┴╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 ╜
 __inference__wrapped_model_83540Ш%"$#acb8576=>KHJIPQVW<в9
2в/
-К*
rnn_dense3_1_input         ш
к "?к<
:
rnn_dense3_out(К%
rnn_dense3_out         └
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85317l$%"#8в5
.в+
%К"
inputs         А
p
к "*в'
 К
0         А
Ъ └
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_85337l%"$#8в5
.в+
%К"
inputs         А
p 
к "*в'
 К
0         А
Ъ Ш
5__inference_batch_normalization_1_layer_call_fn_85350_$%"#8в5
.в+
%К"
inputs         А
p
к "К         АШ
5__inference_batch_normalization_1_layer_call_fn_85363_%"$#8в5
.в+
%К"
inputs         А
p 
к "К         А╢
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85672b78563в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ ╢
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_85692b85763в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ О
5__inference_batch_normalization_2_layer_call_fn_85705U78563в0
)в&
 К
inputs         @
p
к "К         @О
5__inference_batch_normalization_2_layer_call_fn_85718U85763в0
)в&
 К
inputs         @
p 
к "К         @╢
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85801bJKHI3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ ╢
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_85821bKHJI3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ О
5__inference_batch_normalization_3_layer_call_fn_85834UJKHI3в0
)в&
 К
inputs          
p
к "К          О
5__inference_batch_normalization_3_layer_call_fn_85847UKHJI3в0
)в&
 К
inputs          
p 
к "К          о
D__inference_dropout_1_layer_call_and_return_conditional_losses_85266f8в5
.в+
%К"
inputs         А
p
к "*в'
 К
0         А
Ъ о
D__inference_dropout_1_layer_call_and_return_conditional_losses_85271f8в5
.в+
%К"
inputs         А
p 
к "*в'
 К
0         А
Ъ Ж
)__inference_dropout_1_layer_call_fn_85276Y8в5
.в+
%К"
inputs         А
p
к "К         АЖ
)__inference_dropout_1_layer_call_fn_85281Y8в5
.в+
%К"
inputs         А
p 
к "К         Ад
D__inference_dropout_2_layer_call_and_return_conditional_losses_85621\3в0
)в&
 К
inputs         @
p
к "%в"
К
0         @
Ъ д
D__inference_dropout_2_layer_call_and_return_conditional_losses_85626\3в0
)в&
 К
inputs         @
p 
к "%в"
К
0         @
Ъ |
)__inference_dropout_2_layer_call_fn_85631O3в0
)в&
 К
inputs         @
p
к "К         @|
)__inference_dropout_2_layer_call_fn_85636O3в0
)в&
 К
inputs         @
p 
к "К         @д
D__inference_dropout_3_layer_call_and_return_conditional_losses_85750\3в0
)в&
 К
inputs          
p
к "%в"
К
0          
Ъ д
D__inference_dropout_3_layer_call_and_return_conditional_losses_85755\3в0
)в&
 К
inputs          
p 
к "%в"
К
0          
Ъ |
)__inference_dropout_3_layer_call_fn_85760O3в0
)в&
 К
inputs          
p
к "К          |
)__inference_dropout_3_layer_call_fn_85765O3в0
)в&
 К
inputs          
p 
к "К          д
B__inference_reshape_layer_call_and_return_conditional_losses_85249^0в-
&в#
!К
inputs         А
к "*в'
 К
0         А
Ъ |
'__inference_reshape_layer_call_fn_85254Q0в-
&в#
!К
inputs         А
к "К         Ай
G__inference_rnn_dense3_1_layer_call_and_return_conditional_losses_85227^0в-
&в#
!К
inputs         ш
к "&в#
К
0         А
Ъ Б
,__inference_rnn_dense3_1_layer_call_fn_85236Q0в-
&в#
!К
inputs         ш
к "К         Аз
G__inference_rnn_dense3_2_layer_call_and_return_conditional_losses_85729\=>/в,
%в"
 К
inputs         @
к "%в"
К
0          
Ъ 
,__inference_rnn_dense3_2_layer_call_fn_85738O=>/в,
%в"
 К
inputs         @
к "К          з
G__inference_rnn_dense3_3_layer_call_and_return_conditional_losses_85858\PQ/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ 
,__inference_rnn_dense3_3_layer_call_fn_85867OPQ/в,
%в"
 К
inputs          
к "К         ╨
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84297Ж$%"#acb7856=>JKHIPQVWDвA
:в7
-К*
rnn_dense3_1_input         ш
p

 
к "%в"
К
0         
Ъ ╨
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84359Ж%"$#acb8576=>KHJIPQVWDвA
:в7
-К*
rnn_dense3_1_input         ш
p 

 
к "%в"
К
0         
Ъ ├
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_84915z$%"#acb7856=>JKHIPQVW8в5
.в+
!К
inputs         ш
p

 
к "%в"
К
0         
Ъ ├
E__inference_rnn_dense3_layer_call_and_return_conditional_losses_85114z%"$#acb8576=>KHJIPQVW8в5
.в+
!К
inputs         ш
p 

 
к "%в"
К
0         
Ъ з
*__inference_rnn_dense3_layer_call_fn_84473y$%"#acb7856=>JKHIPQVWDвA
:в7
-К*
rnn_dense3_1_input         ш
p

 
к "К         з
*__inference_rnn_dense3_layer_call_fn_84586y%"$#acb8576=>KHJIPQVWDвA
:в7
-К*
rnn_dense3_1_input         ш
p 

 
к "К         Ы
*__inference_rnn_dense3_layer_call_fn_85165m$%"#acb7856=>JKHIPQVW8в5
.в+
!К
inputs         ш
p

 
к "К         Ы
*__inference_rnn_dense3_layer_call_fn_85216m%"$#acb8576=>KHJIPQVW8в5
.в+
!К
inputs         ш
p 

 
к "К         й
I__inference_rnn_dense3_out_layer_call_and_return_conditional_losses_85877\VW/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ Б
.__inference_rnn_dense3_out_layer_call_fn_85886OVW/в,
%в"
 К
inputs         
к "К         ║
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85475nacb@в=
6в3
%К"
inputs         А

 
p

 
к "%в"
К
0         @
Ъ ║
H__inference_rnn_dense3_r1_layer_call_and_return_conditional_losses_85587nacb@в=
6в3
%К"
inputs         А

 
p 

 
к "%в"
К
0         @
Ъ Т
-__inference_rnn_dense3_r1_layer_call_fn_85598aacb@в=
6в3
%К"
inputs         А

 
p

 
к "К         @Т
-__inference_rnn_dense3_r1_layer_call_fn_85609aacb@в=
6в3
%К"
inputs         А

 
p 

 
к "К         @╓
#__inference_signature_wrapper_84647о%"$#acb8576=>KHJIPQVWRвO
в 
HкE
C
rnn_dense3_1_input-К*
rnn_dense3_1_input         ш"?к<
:
rnn_dense3_out(К%
rnn_dense3_out         