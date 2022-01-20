import cv2

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

import glob
from PIL import Image
import time

# same pixel value ratio
def pixel_equal(image1, image2, x, y):
    """
    判断两个像素是否相同
    :param image1: 图片1
    :param image2: 图片2
    :param x: 位置x
    :param y: 位置y
    :return: 像素是否相同
    """
    # 取两个图片像素点
    pixel_1 = image1.load()[x, y]
    pixel_2 = image2.load()[x, y]
    threshold = 3
    # 比较每个像素点的RGB值是否在阈值范围内，若两张图片的RGB值都在某一阈值内，则我们认为它的像素点是一样的
    if abs(pixel_1[0] - pixel_2[0]) < threshold and abs(pixel_1[1] - pixel_2[1]) < threshold and abs(pixel_1[2] - pixel_2[2]) < threshold:
        return True
    else:
        return False

def compare(image1, image2):
    """
    进行比较
    :param image1:图片1
    :param image2: 图片2
    :return:
    """
    right_num = 0  # 记录相同像素点个数
    false_num = 0  # 记录不同像素点个数
    all_num = 0  # 记录所有像素点个数
    for i in range(image1.size[0]):
        for j in range(image1.size[1]):
            if image1.load()[i, j][0] != 255:
                if pixel_equal(image1, image2, i, j):
                    right_num += 1
                else:
                    false_num += 1
                all_num += 1

    same_rate = right_num / all_num * 100  # 相同像素点比例
    return same_rate


# change intensity in per text-pixel
def pixel_calculate(image1, image2, x, y):
    # 取两个图片像素点
    pixel_1 = image1.load()[x, y]
    pixel_2 = image2.load()[x, y]
    difference_value = abs(pixel_1[0] - pixel_2[0]) + abs(pixel_1[1] - pixel_2[1]) + abs(pixel_1[2] - pixel_2[2])
    return difference_value

def get_change_ratio(image1, image2):
    change_intensity = 0  # 记录像素值变化量
    all_num = 0  # 记录所有像素点个数
    for i in range(image1.size[0]):
        for j in range(image1.size[1]):
            if image1.load()[i, j][0] != 255:
                each_change_intensity = pixel_calculate(image1, image2, i, j)
                # print(each_change_intensity)
                change_intensity += each_change_intensity
                all_num += 1

    change_intensity_rate = change_intensity / all_num  # 相同像素点比例
    return change_intensity_rate



if __name__ == "__main__":
    image_input_dir = "./doc_ieee_image"
    image_encoded_dir = "./encoded_ieee_image/ieee_combined"
    image_input_list = glob.glob(image_input_dir + '/*')
    image_encoded_list = glob.glob(image_encoded_dir + '/*')

    # 计算PSNR和SSIM值
    count_b = count_g = count_r = 0
    count_ssim = 0
    for image_idx in range(len(image_input_list)):
        # origin_image
        origin_image = cv2.imread(filename=image_input_list[image_idx], flags=cv2.IMREAD_UNCHANGED)

        # noised_image
        noised_image = cv2.imread(filename=image_encoded_list[image_idx], flags=cv2.IMREAD_UNCHANGED)

        # psnr 计算
        b1, g1, r1 = cv2.split(origin_image)
        b2, g2, r2 = cv2.split(noised_image)

        psnr_b = compare_psnr(b1, b2)
        psnr_g = compare_psnr(g1, g2)
        psnr_r = compare_psnr(r1, r2)
        count_b += psnr_b
        count_g += psnr_g
        count_r += psnr_r

        # ssim 计算
        ssim = compare_ssim(origin_image, noised_image, multichannel=True)
        count_ssim += ssim

    count_average_b, count_average_g, count_average_r = count_b / len(image_input_list), count_g / len(
        image_input_list), count_r / len(image_input_list)
    print("psnr_b：{0:.2f}, psnr_g：{1:.2f}, psnr_r：{2:.2f}, psnr_avg:{3:.2f}".format(count_average_b, count_average_g,
                                                                                    count_average_r, (
                                                                                            count_average_b + count_average_g + count_average_r) / 3.0))

    count_average_ssim = count_ssim / len(image_input_list)
    print("ssim:{0}".format(count_average_ssim))

    # 计算CPP的值
    all_change_intensity_rate = 0
    t1 = time.time()
    for image_idx in range(len(image_input_list)):
        origin_image = Image.open(fp=image_input_list[image_idx])
        encode_image = Image.open(fp=image_encoded_list[image_idx])
        change_intensity_ratio = get_change_ratio(origin_image, encode_image)
        all_change_intensity_rate += change_intensity_ratio
    t2 = time.time()
    all_change_intensity_rate = all_change_intensity_rate / 100.
    print("t={0}, same_rate:{1:.2f}".format(t2 - t1, all_change_intensity_rate))
