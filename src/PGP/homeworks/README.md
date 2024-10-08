# PGP homeworks

## homework01.cpp - instanced indirected rendering
<img style="float:left"  src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/01/before.png" width="512px">
<img style="float:right" src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/01/after.png" width="512px">
<br>
Use indirect call and replace the other draw calls in this file.
Use single glMultiDrawArraysIndirect call to render everything.

You have to create draw indirect buffer that contains correct values (draw commands).
You will need these OpenGL functions:
```
glMultiDrawArraysIndirect
glCreateBuffers
glNamedBufferData
glBindBuffer
```
submit homework1.cpp

## homework02.cpp - geometry shaders
<img style="float:left"  src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/02/before.png" width="512px">
<img style="float:right" src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/02/after.png" width="512px">
<br>
Replace point rendering with Czech flag rendering.
Change shaders in order to replace points with Czech flags.
Do not touch glDrawArrays draw call!
Do not touch vertex shader!

## homework03.cpp - tessellation
<img style="float:left"  src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/03/before.png" width="512px">
<img style="float:right" src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/03/after.png" width="512px">
<br>
Transfrom the cube to sphere using tessellation.

## homework04.cpp - atomic counters
<img style="float:left"  src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/04/before.png" width="512px">
<img style="float:right" src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/04/after.png" width="512px">
<br>
Use atomic instruction inside fragment shader and
count the number of rasterized fragments - i.e. increment the counter by 1 for every execution of fragment shader
Write the number of rasterized fragments of this program
into nofRasterizedSamples variable inside onDraw function.

You may have to create and clear buffer
and use atomic instruction for addition.

Help:
```
glCreateBuffers
glClearNamedBufferData
glBindBufferBase
glGetNamedBufferSubData
layout(binding=0,std430)buffer Counter{uint counter;};
atomicAdd
```


## homework05.cpp - compute shader
<img style="float:left"  src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/05/before.png" width="512px">
<img style="float:right" src="https://git.fit.vutbr.cz/imilet/FitGraphics/raw/branch/master/resources/images/pgp/homework/05/after.png" width="512px">
<br>
Reimplement compute shader.
Each thread should compute mask value and store it into the buffer.
Look at reference images.
Hint: It is fractal image that is composed of this pattern:<br>
▯▯<br>
▮▯<br>
