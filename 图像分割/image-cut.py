from PIL import Image


def loadImage1():
    # 读取图片
    im = Image.open("./data/lena.png")
    return im


def loadImage2():
    # 读取图片
    im = Image.open("./data/lena_modified.png")
    return im


if __name__ == '__main__':
    data1 = loadImage1()
    data2 = loadImage2()
    height = data1.size[0]
    width = data1.size[1]

    # 输出图片的像素值
    for i in range(0, height):
        for j in range(0, width):
            if data1.getpixel((i, j)) == data2.getpixel((i, j)):
                data2.putpixel((i, j), 255)
    data2.save("./data/ans_two.png")
