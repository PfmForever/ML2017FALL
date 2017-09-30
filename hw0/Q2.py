import sys
from PIL import Image
filename = sys.argv[1]
im = Image.open(filename)
w,h = im.size
im2 = Image.new( "RGB", (w,h) )
for w1 in range(w):
    for h1 in range(h):
        r,g,b = im.getpixel((w1,h1))
        r = r//2
        g = g//2
        b = b//2
        im2.putpixel((w1,h1),(r,g,b))
im2.save('Q2.png','png')
