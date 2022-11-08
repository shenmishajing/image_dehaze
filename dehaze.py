import argparse
import math
import os.path

import cv2
import numpy as np


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx = 0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


def dehaze(input_img, output_path, alg):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    src = cv2.imread(input_img)

    I = src.astype('float64') / 255

    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    t = TransmissionEstimate(I, A, 15)
    if alg == 'guided_filter':
        t = TransmissionRefine(src, t)
    J = Recover(I, t, A, 0.1)

    cv2.imwrite(os.path.join(output_path, 'dark_channel.png'), dark * 255)
    cv2.imwrite(os.path.join(output_path, 'transmission.png'), t * 255)
    cv2.imwrite(os.path.join(output_path, 'original.png'), src)
    cv2.imwrite(os.path.join(output_path, 'dehaze.png'), J * 255)


def parse_args():
    parser = argparse.ArgumentParser(description = 'Process some integers.')
    parser.add_argument('--input-img', type = str, default = 'image/example1.png', help = 'path for input image')
    parser.add_argument('--output-path', type = str, default = 'output', help = 'path for output image')
    parser.add_argument('--alg', type = str, default = 'dark_channel', choices = ['dark_channel', 'guided_filter'])

    return parser.parse_args()


def dehaze_all(img_path = 'image', ouput_path = 'output'):
    for name in os.listdir(img_path):
        if name.endswith('.png') or name.endswith('.jpg'):
            input_img = os.path.join(img_path, name)
            for alg in ['dark_channel', 'guided_filter']:
                dehaze(input_img, os.path.join(ouput_path, os.path.splitext(name)[0], alg), alg)


def main():
    args = parse_args()
    dehaze(args.input_img, args.output_path, args.alg)


if __name__ == '__main__':
    main()
