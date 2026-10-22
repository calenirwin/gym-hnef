"""Generate original seamless material tiles. Development tool: Python + Pillow.
Usage: python generate-materials.py OUTPUT_DIRECTORY
The products consume static WebP files, not this generator.
"""
import sys
from pathlib import Path
import random, math
from PIL import Image
N=256
rng=random.Random(2810)
noise=[[rng.uniform(-1,1) for x in range(N)] for y in range(N)]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
for name,base,strength,woven in [('ink-linen',(13,49,67),8,True),('warm-paper',(230,221,195),10,False),('board-paper',(209,193,149),12,False)]:
 im=Image.new('RGB',(N,N));pixels=[]
 for y in range(N):
  for x in range(N):
   broad=math.sin(x*math.tau/64+math.sin(y*math.tau/128))*.16+math.sin(y*math.tau/128)*.12
   if woven:
    fiber=(math.cos(x*math.tau/4)*.45+math.cos(y*math.tau/4)*.35+noise[y][x]*.35)
   else:
    fiber=noise[y][x]*.65+noise[y][(x-1)%N]*.22+noise[(y-1)%N][x]*.13
   v=(fiber+broad)*strength
   pixels.append(tuple(max(0,min(255,round(c+v))) for c in base))
 im.putdata(pixels);im.save(out/(name+'.webp'),lossless=True)
