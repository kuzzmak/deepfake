from typing import Tuple

import cv2 as cv
import numpy as np

from core.face import Face


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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def warp_image(interpolation: int, image: np.ndarray) -> np.ndarray:
        """Applies sligt warp effect on the image.

        Args:
            interpolation (int): interpolation type
            image (np.ndarray): image to warp

        Returns:
            np.ndarray: warped image
        """
        h, w, _ = image.shape
        cell_size = [w // (2**i) for i in range(1, 4)][np.random.randint(3)]
        cell_count = w // cell_size + 1
        grid_points = np.linspace(0, w, cell_count)
        mapx = np.broadcast_to(grid_points, (cell_count, cell_count)).copy()
        mapy = mapx.T
        mapx[1:-1, 1:-1] = mapx[1:-1, 1:-1] + \
            np.random.normal(
                size=(cell_count-2, cell_count-2), scale=0.2)*(cell_size*0.24)
        mapy[1:-1, 1:-1] = mapy[1:-1, 1:-1] + \
            np.random.normal(size=(cell_count-2, cell_count-2),
                             scale=0.2)*(cell_size*0.24)
        half_cell_size = cell_size // 2
        mapx = cv.resize(
            mapx,
            (w+cell_size,)*2,
        )[half_cell_size:-half_cell_size, half_cell_size:-half_cell_size] \
            .astype(np.float32)
        mapy = cv.resize(
            mapy,
            (w+cell_size,)*2,
        )[half_cell_size:-half_cell_size, half_cell_size:-half_cell_size] \
            .astype(np.float32)
        return cv.remap(image, mapx, mapy, interpolation)

    # @staticmethod
    # def _get_2d_grid_for_shape(shape: int) -> Tuple[np.ndarray, np.ndarray]:
    #     return np.mgrid[0:shape - 1:shape * 1j, 0:shape - 1:shape * 1j]

    # @staticmethod
    # def _get_edge_points(shape: int) -> List[Tuple[int, int]]:
    #     return [
    #         (0, 0),
    #         (0, shape - 1),
    #         (shape - 1, shape - 1),
    #         (shape - 1, 0),
    #         (shape // 2 - 1, 0),
    #         (shape // 2 - 1, shape - 1),
    #         (shape - 1, shape // 2 - 1),
    #         (0, shape // 2 - 1),
    #     ]

    # @staticmethod
    # def warp_faces_and_masks(face_A: Face, face_B: Face) -> np.ndarray:
    #     shape = face_A.aligned_image.shape[1]
    #     grid_x, grid_y = ImageAugmentation._get_2d_grid_for_shape(shape)
    #     edge_points = ImageAugmentation._get_edge_points(shape)
    #     source = face_A.aligned_landmarks
    #     destination = face_B.aligned_landmarks
    #     source = np.append(np.flip(source, axis=1), edge_points, 0)
    #     destination = np.append(np.flip(destination, axis=1), edge_points, 0)
    #     grid_z = griddata(destination, source, (grid_x, grid_y))
    #     map_x = np.array([ar[:, 1] for ar in grid_z], dtype=np.float32) \
    #         .reshape(shape, shape)
    #     map_y = np.array([ar[:, 0] for ar in grid_z], dtype=np.float32) \
    #         .reshape(shape, shape)
    #     inter = cv.INTER_LINEAR
    #     border = cv.BORDER_TRANSPARENT
    #     return (
    #         cv.remap(face_A.aligned_image, map_x, map_y, inter, border),
    #         cv.remap(face_B.aligned_image, map_x, map_y, inter, border),
    #         cv.remap(face_A.aligned_mask, map_x, map_y, inter, border),
    #         cv.remap(face_B.aligned_mask, map_x, map_y, inter, border),
    #     )
