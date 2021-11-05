from typing import Tuple

import cv2 as cv
import numpy as np


class ImageAugmentation:

    @staticmethod
    def flip(image: np.ndarray, flip_code: int = 1) -> np.ndarray:
        """Flips image in x, y or both directions.

        Args:
            image (np.ndarray): image to flip
            flip_code (int, optional): how to flip image, 1 - horizontal,
                0 - vertical, -1 - both horizontal and vertical. Defaults to 1.

        Returns:
            np.ndarray: flipped image
        """
        image = cv.flip(image, flip_code)
        return image

    @staticmethod
    def light(gamma: float, image: np.ndarray) -> np.ndarray:
        """Adds or removed light from image.

        Args:
            gamma (float): constant defining how much light is added or
                removed from the image
            image (np.ndarray): image to add light to

        Returns:
            np.ndarray: image
        """
        inv_gamma = 1.0 / gamma
        table = np.array(
            [
                ((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)
            ]).astype("uint8")

        image = cv.LUT(image, table)
        return image

    @staticmethod
    def saturation(saturation: float, image: np.ndarray) -> np.ndarray:
        """Adds saturation to the image.

        Args:
            saturation (float): saturation value
            image (np.ndarray): image to add saturation to

        Returns:
            np.ndarray: image with saturation
        """
        v = image[:, :, 2]
        v = np.where(v <= 255 - saturation, v + saturation, 255)
        image[:, :, 2] = v
        return image

    @staticmethod
    def gaussian_blur(blur: float, image: np.ndarray) -> np.ndarray:
        """Blurs image using Gaussian.

        Args:
            blur (float): blur amount
            image (np.ndarray): image to blur

        Returns:
            np.ndarray: blurred image
        """
        image = cv.GaussianBlur(image, (5, 5), blur)
        return image

    @staticmethod
    def bilateral_blur(
        d: int,
        sigma_color: int,
        sigma_space: int,
        image: np.ndarray,
    ) -> np.ndarray:
        """Applies the bilateral filter to an image.

        Args:
            d (int): diameter of each pixel neighborhood that is used during
                filtering
            sigma_color (int): filter sigma in the color space
            sigma_space (int): filter sigma in the coordinate space
            image (np.ndarray): image to blur

        Returns:
            np.ndarray: blurred image
        """
        image = cv.bilateralFilter(image, d, sigma_color, sigma_space)
        return image

    def erode(kernel_shape: Tuple[int, int], image: np.ndarray) -> np.ndarray:
        """Erodes an image by using a specific structuring element, kernel of
        shape `kernel_shape` which consists of ones.

        Args:
            kernel_shape (Tuple[int, int]): eroding kernel shape
            image (np.ndarray): image to erode

        Returns:
            np.ndarray: eroded image
        """
        kernel = np.ones(kernel_shape, np.uint8)
        image = cv.erode(image, kernel, iterations=1)
        return image

    def dilate(kernel_shape: Tuple[int, int], image: np.ndarray) -> np.ndarray:
        """Dilates an image by using a specific structuring element, kernel of
        shape `kernel_shape` which consists of ones.

        Args:
            kernel_shape (Tuple[int, int]): dilation kernel shape
            image (np.ndarray): image to dilate

        Returns:
            np.ndarray: eroded image
        """
        kernel = np.ones(kernel_shape, np.uint8)
        image = cv.dilate(image, kernel, iterations=1)
        return image

    def sharpen(image: np.ndarray) -> np.ndarray:
        """Sharpens image using specific kernel.

        Args:
            image (np.ndarray): image to sharpen

        Returns:
            np.ndarray: sharpened image
        """
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv.filter2D(image, -1, kernel)
        return image
