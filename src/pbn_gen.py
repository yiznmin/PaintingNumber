import cv2
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.utils import shuffle
from shapely.geometry import Polygon, Point
import svgwrite
import json
import random

# Change me to an integer for consistent results between runs, or set to None to allow randomness in K-means
random_state = None


class PbnGen:
    def __init__(
        self, f_name, num_colors=None, min_num_colors=10, pruningThreshold=6.25e-5, fixed_palette=None, suggestion_threshold=40
    ):
        # 支援直接傳入影像矩陣或檔案路徑
        bgr_image = cv2.imread(f_name) if isinstance(f_name, str) else f_name
        # change to RGB
        rgbImage = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB) if bgr_image is not None else np.zeros((100,100,3), dtype=np.uint8)

        # Retain an original image copy for easy testing
        self.originalImage = rgbImage
        self.originalImg1d = self.get1DImg(self.originalImage)

        # Set the actual working image to a copy of the original
        self.setImage(self.originalImage.copy())

        # The minimum percentage of the image's area a color cluster can be before getting absorbed by surrounding colors
        self.pruningThreshold = pruningThreshold

        # This will contain a dict of colors and binary masks of the pruned clusters
        self.prunableClusters = None

        # Business Logic: Fixed Palette and Suggestions
        self.fixed_palette = np.array(fixed_palette) if fixed_palette is not None else None
        self.suggestion_threshold = suggestion_threshold
        self.color_suggestions = []

        self.num_colors = num_colors if num_colors else self.get_num_clusters()
        # make sure number of colors is at least minimum number
        self.num_colors = (
            self.num_colors + min_num_colors
            if self.num_colors < min_num_colors
            else self.num_colors
        )
        print(f"Quantized to {self.num_colors} colors")

    def cluster_colors(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """
        Performs K means clustering on the image to quantize it to a fixed number of colors.

        Returns:
            (palette, labels, q_img)

            palette: A (N, 3) numpy array representing the quantized colors in a float32 format.
            labels: A (H*W,) numpy array which holds the assigned labels for each pixel in the image.
            q_img: A (H, W, 3) quantized image which holds the original image quantized to the specified number of colors.
        """

        model = KMeans(
            n_clusters=self.num_colors, n_init="auto", random_state=random_state
        )
        model.fit(self.img1d)

        centers = model.cluster_centers_
        snapped_centers = []
        self.color_suggestions = []

        if self.fixed_palette is not None:
            # 將 K-Means 中心和固定色盤都轉換到 LAB 空間再計算距離
            # LAB 距離更符合人眼感知，避免 RGB 空間的非線性誤差
            def rgb_to_lab(rgb_array):
                # rgb_array shape: (N, 3)，值域 0-255
                rgb_u8 = rgb_array.astype(np.uint8).reshape(-1, 1, 3)
                lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
                return lab

            centers_lab = rgb_to_lab(centers)
            palette_lab = rgb_to_lab(self.fixed_palette)

            for i, center in enumerate(centers):
                # LAB 空間計算歐氏距離
                distances = np.linalg.norm(palette_lab - centers_lab[i], axis=1)
                min_dist_idx = np.argmin(distances)
                min_dist = float(distances[min_dist_idx])
                nearest_fixed_color = self.fixed_palette[min_dist_idx]

                if min_dist > self.suggestion_threshold:
                    self.color_suggestions.append({
                        "ideal_rgb": center.astype(int).tolist(),
                        "nearest_fixed_rgb": nearest_fixed_color.tolist(),
                        "distance": min_dist
                    })

                snapped_centers.append(nearest_fixed_color)
            self.palette = np.array(snapped_centers) / 255
        else:
            self.palette = centers / 255

        self.labels = model.labels_
        # get quantized image
        q_img = self.palette[self.labels].reshape(self.image.shape)
        return self.palette, self.labels, q_img

    def cluster_colors_(self):
        """
        An in-place clustering of colors, replaces existing image with the quantized version
        """

        colors, labels, q_img = self.cluster_colors()
        q_img = (q_img * 255).astype(np.uint8)
        # print(q_img.dtype)

        if self.color_suggestions:
            print(f"⚠️ 提醒：發現 {len(self.color_suggestions)} 個區域色差較大，建議增加色系：", self.color_suggestions)

        self.setImage(q_img.copy())

    def get_num_clusters(self):
        """
        Algorithmically gets optimal number of clusters for k means using knee method

        Returns:
            numClusters: The optimal number of clusters found by the K Knee method
        """

        max_test = 25
        num_samples = 10000
        inertias = []
        x_vals = np.arange(1, max_test)
        for i in x_vals:
            # run on sample to save time for approximation
            image_arr_sample = shuffle(
                self.img1d, random_state=0, n_samples=num_samples
            )
            kmeans = KMeans(n_clusters=i, n_init="auto", random_state=random_state)
            kmeans.fit(image_arr_sample)
            inertia = kmeans.inertia_
            inertias.append(inertia)

        # plt.plot(x_vals, inertias)
        # plt.show()

        kn = KneeLocator(
            x=x_vals,
            y=inertias,
            curve="convex",
            direction="decreasing",
        )
        return kn.knee

    def plt_cluster_pie(self):
        """
        Plots a pie chart based on the percentage of each color in the image
        """

        unique_values, counts = np.unique(self.labels, return_counts=True)
        percentages = (counts / len(self.labels)) * 100
        plt.pie(
            percentages,
            colors=np.array(self.palette),
            labels=np.arange(len(self.palette)),
        )
        plt.show()

    def resetImage(self):
        """
        Resets the existing image with the stored original image for easier testing of variants
        """

        self.setImage(self.originalImage.copy())

    def showImg(self, img=None, title="", figsize=(12, 12)):
        """
        A utility function to show the current image

        Arguments:
            title: The title for the plot
            figsize: The figure size of the image
        """

        displayImage = None
        if img is None:
            displayImage = self.getImage()
        else:
            displayImage = img

        plt.figure(figsize=figsize)
        plt.imshow(displayImage)
        plt.title(title)
        plt.show()

    def get1DImg(self, image: np.ndarray) -> np.ndarray:
        """
        Vectorizes a given image of shape (H, W, C) to shape (H*W, C)

        Arguments:
            image: The image to vectorize to 1 dimension

        Returns:
            A reshaped image vectorized over the channel axis.
        """

        H, W, C = image.shape
        return image.reshape((H * W, C))

    def setImage(self, img: np.ndarray):
        """
        Updates the currently stored image to img and updates the img1d class variable

        Arguments:
            img: The image that should replace the existing image. Will also update the 1d representation accordingly, but not clustering or other variables.
        """

        self.image = img.copy()
        self.img1d = self.get1DImg(self.image)

    def getImage(self) -> np.ndarray:
        """
        Returns a copy of the current image that can be stored or modified

        Returns:
            copy: A copy of the current self.image
        """

        return self.image.copy()

    def getImageArea(self) -> int:
        """
        Returns the image area

        Returns:
            area: H*W for an image of shape (H, W, C)
        """

        H, W, C = self.image.shape
        return H * W

    def resizeImage(
        self, image=None, scale: float = 1, dimension: tuple = None
    ) -> None:
        """
        Return a resized version of the provided image or self.image if image=None.

        Arguments:
            scale: A float > 0 that is used to scale the image up or down with a scale of 1 returning the same image.
            dimension=None: A tuple representing the manual size the image should be in the form (H, W). Overrides any given scale value.
        """

        assert scale > 0, f"Given scale {scale} must be greater than 0!"

        img = None
        if image is None:
            img = self.getImage().astype(np.uint8)
        else:
            img = image

        H, W = 0, 0

        if img.ndim == 3:
            H, W, C = img.shape
        elif img.ndim == 2:
            H, W = img.shape

        resized = None

        if dimension is not None:
            NH, NW = dimension

            # If upsampling, use INTER_NEAREST, otherwise, use INTER_AREA. We want to use INTER_NEAREST for upsampling to preserve the number of colors
            if (
                NH * NW >= H * W
            ):  # This is a crude estimate for up vs downsampling, but it works well enough
                resized = cv2.resize(img, (NW, NH), interpolation=cv2.INTER_NEAREST)
            else:
                resized = cv2.resize(img, (NW, NH), interpolation=cv2.INTER_AREA)
        else:
            NH, NW = int(H * scale), int(W * scale)

            if scale <= 1:
                resized = cv2.resize(img, (NW, NH), interpolation=cv2.INTER_AREA)
            else:
                resized = cv2.resize(img, (NW, NH), interpolation=cv2.INTER_NEAREST)

        return resized

    def resizeImage_(self, scale: float = 1, dimension: tuple = None) -> None:
        """
        Resize the stored image in place by some scale factor.

        Arguments:
            scale: A float > 0 that is used to scale the image up or down with a scale of 1 returning the same image.
            dimension=None: A tuple representing the manual size the image should be in the form (H, W). Overrides any given scale value.
        """

        resized = self.resizeImage(scale=scale, dimension=dimension)

        self.setImage(resized)

    def blurImage_(
        self,
        blurType: str,
        ksize: int = 13,
        sigma: float = 3,
        sigmaColor: float = 21,
        sigmaSpace: float = 21,
    ) -> None:
        """
        Blurs the current image in place according to the arguments of the function.
        Updates self.image and self.img1d

        Arguments:
            blurType: 'gaussian', 'median', or 'bilateral'. Determines the kind of filter to be applied
            ksize: The size of the blurring kernel to be applied
            sigma: A sigma for the gaussian kernel type
            sigmaColor: How large of a range colors should be blended, higher values means more distant colors will be blended
            sigmaSpace: How intensely pixels in the kernel are blurred
        """

        image = self.image.astype(np.uint8)
        blurred = None

        if blurType == "gaussian":
            kernel = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
            blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel)
            blurred = cv2.filter2D(image, ddepth=-1, kernel=kernel.T)
        elif blurType == "median":
            blurred = cv2.medianBlur(image, ksize=ksize)
        elif blurType == "bilateral":
            blurred = cv2.bilateralFilter(
                image, d=ksize, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
            )

        self.image = blurred
        self.img1d = self.get1DImg(self.image)

    def getUniqueColors(self, image=None) -> np.ndarray:
        """
        Gets an array of shape (N, 3) which represents all the unique colors present in self.image

        Arguments:
            image=None: If None, returns the unique colors of self.image, otherwise, performs the operations for the provided image.

        Returns:
            uniqueColors: A (N, 3) numpy array which represents the found unique colors in the provided or current image.
        """

        reshaped_image = None
        if image is None:
            # Reshape to a 2D array
            reshaped_image = self.image.reshape(-1, self.image.shape[2])
        else:
            reshaped_image = image.reshape(-1, image.shape[2])

        # Find unique color values across the channels
        uniqueColors = np.unique(reshaped_image, axis=0)

        return uniqueColors

    def getUniqueColorsMasks(self) -> dict:
        """
        Returns a dictionary with indices of each unique color and a binary numpy array representing where each unique color is
        a key and each value is a binary mask of the image representing where that color is.

        Returns:
            colorsDict: A dictionary with keys of RGB tuples and values of binary masks representing the presence of that key in the image
        """

        colorsDict = {}

        uniqueColors = self.getUniqueColors()

        for color in uniqueColors:
            colorsDict[tuple(color)] = np.repeat(
                np.all(self.image == color, axis=2)[..., np.newaxis], repeats=3, axis=2
            )

        self.colorMasks = colorsDict

        return colorsDict

    def generatePrunableClusters(self, showPlots=False):
        """
        Stores color masks in self.prunableClusters which can be pruned from the main image. The small pruned clusters can be replaced by the nearest color
        in the original image in a different function. The treshold used to determine which clusters should be removed is defined as self.pruningThreshold

        Arguments:
            showPlots=False: Whether or not to show plots of pruned clusters
        """

        colorsDict = self.getUniqueColorsMasks()

        prunableClusters = {}

        for color in colorsDict.keys():
            mask = colorsDict[color]

            # Convert color tuple to an array
            color = np.array(color, dtype=np.uint8)
            singleColorImage = color * mask

            if showPlots:
                plt.imshow(singleColorImage), plt.title(color)
                plt.show()

            # The mask seems to need to be a "binary" image but the binary values are 0 and 255 instead of 0 and 1
            (
                numLabels,
                labels,
                stats,
                centroids,
            ) = cv2.connectedComponentsWithStatsWithAlgorithm(
                (mask[..., 0] * 255).astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU
            )

            if showPlots:
                plt.imshow(labels), plt.title("Before pruning")
                plt.show()

            labelIndices = np.arange(1, numLabels)
            areas = stats[labelIndices, -1]

            imageArea = self.getImageArea()
            # Get an array representing the clusters that are too small and should be pruned
            tooSmall = imageArea * self.pruningThreshold > areas
            # Negatable is an array containing the labels that should be pruned, these are then negated in the labels mask
            negatable = labelIndices[tooSmall]
            negateMask = np.isin(labels, negatable)
            labels[negateMask] *= -1

            # Convert from labels to a mask where each pruned cluster has its unique segmented label
            labels[labels > 0] = 0
            labels = -labels

            if showPlots:
                plt.imshow(labels), plt.title("Pruned clusters")
                plt.show()

            prunableClusters[tuple(color)] = labels

            if showPlots:
                binaryLabels = (labels > 0).astype(np.uint8)
                plt.imshow(mask[..., 0] - binaryLabels), plt.title("After pruning")
                plt.show()

        self.prunableClusters = prunableClusters

    def getClusteringEffectiveness(
        self,
    ) -> "tuple[dict, dict, dict, dict, int, int, float]":
        """
        Returns stats about the effectiveness of cluster pruning by comparing raw cluster counts to theoretical pruned ones

        Returns:
            (rawStats, prunedStats, remainingStats, reductionFactors)
            rawStats: The number of clusters per color of the current image
            prunedStats: The number of clusters per color that pruning will remove
            remainingStats: How many clusters remain per color
            reductionFactory: By what percentage did pruning reduce cluster counts
            totalRawClusters: How many raw clusters there were
            totalPrunedClusters: How many clusters were pruned
            totalReduction: The percentage of total clusters pruned
        """

        rawStats, prunedStats = self._getClusterStats()

        totalRawClusters = 0
        totalPrunedClusters = 0
        remainingStats = {}
        reductionFactors = {}

        for color in rawStats.keys():
            rawCount = rawStats[color]
            prunedCount = prunedStats[color]
            totalRawClusters += rawCount
            totalPrunedClusters += prunedCount

            remainingStats[color] = rawCount - prunedCount
            reductionFactors[color] = round(
                (1 - (rawCount - prunedCount) / rawCount) * 100, 2
            )

        totalReduction = round(
            (1 - (totalRawClusters - totalPrunedClusters) / totalRawClusters) * 100, 2
        )

        return (
            rawStats,
            prunedStats,
            remainingStats,
            reductionFactors,
            totalRawClusters,
            totalPrunedClusters,
            totalReduction,
        )

    def _getClusterStats(self) -> "tuple[dict, dict]":
        """
        Gets a dictionary in the format {(R, G, B): clusterCount} where clusterCount is the number of clusters of that color.
        Also returns the number of clusters that would be pruned


        Returns:
            (rawStats, prunedStats)
            Dictionaries with the number of clusters per color in the current image, and how many will be pruned
        """

        colorsDict = self.getUniqueColorsMasks()

        rawCounts = {}
        prunedCounts = {}

        for color in colorsDict.keys():
            mask = colorsDict[color]

            # The mask seems to need to be a "binary" image but the binary values are 0 and 255 instead of 0 and 1
            (
                numLabels,
                labels,
                stats,
                centroids,
            ) = cv2.connectedComponentsWithStatsWithAlgorithm(
                (mask[..., 0] * 255).astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU
            )

            rawCounts[color] = numLabels

            labelIndices = np.arange(1, numLabels)
            areas = stats[labelIndices, -1]

            imageArea = self.getImageArea()
            # Get an array representing the clusters that are too small and should be pruned
            tooSmall = imageArea * self.pruningThreshold > areas
            tooSmallCount = np.sum(tooSmall.astype(np.uint8))
            prunedCounts[color] = tooSmallCount

        return rawCounts, prunedCounts

    def getMainSurroundingColor(self, image, mask) -> np.ndarray:
        """
        Returns the main surrounding color given a binary mask and image. The function will check the edges of the mask to determine the present colors
        and will return the most common color surrounding the mask.

        Arguments:
            image: The image to use as a reference for the surrounding colors
            mask: A binary mask which will be used to determine the cluster of pixels we want to find the common color around

        Returns:
            mostCommonColor: A (3,) numpy array which holds the RGB value of the most common color
        """

        assert image.shape[:-1] == mask.shape, "Image and mask shapes are different!"

        edgeFilter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]))

        maskEdges = cv2.filter2D(mask, ddepth=-1, kernel=edgeFilter)

        # plt.figure(figsize=(20, 20)), plt.imshow(maskEdges), plt.title('Small cluster edge'), plt.show()

        surroundingColors = image[maskEdges.astype(bool)]

        # most_common(1) returns a list with a single tuple (key, count)
        mostCommonColor = Counter(map(tuple, surroundingColors)).most_common(1)[0][0]
        return np.array(mostCommonColor, dtype=np.uint8)

    def getMainSurroundingColorVectorized(
        self, image, mask, uniqueLabels
    ) -> np.ndarray:
        """
        Returns the main surrounding colors given a labeled mask and image. The function will check the edges of the mask to determine the present colors
        and will return the most common color surrounding the mask.

        Arguments:
            image: The image to use as a reference for the surrounding colors
            mask: A 3D binary mask of shape (H, W, N) where N is the number of unique clusters excluding the background. The mask should be 1 where
                a certain unique label exists and 0 elsewhere.

        Returns:
            modeColors: A (N, 3) numpy array which holds the RGB values of the most common colors for each label
        """

        # assert image.shape[:-1] == mask.shape, 'Image and mask shapes are different!'

        edgeFilter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]), dtype=np.int32)

        modeColors = []
        for label in uniqueLabels:
            maskEdges = cv2.filter2D(
                (mask == label).astype(np.uint8), ddepth=-1, kernel=edgeFilter
            ).astype(bool)
            # plt.figure(figsize=(20, 20)), plt.imshow(maskEdges), plt.title('Small cluster edge'), plt.show()
            modeColors.append(
                Counter(map(tuple, image[maskEdges])).most_common(1)[0][0]
            )

        return np.array(modeColors, dtype=np.uint8)

    # TODO: If time allows, re-write this to merge similar intensities along strong gradients to preserve things like the whiskers in the Red Panda image
    def pruneClustersSmart(
        self,
        iterations: int = 3,
        pruneBySize=False,
        reversePruneBySize=False,
        reversePruneByIntensity=True,
        showPlots=False,
    ):
        """
        Prunes small color clusters in order of cluster size and color intensity. The pruned clusters are based on the self.pruningThreshold class variable
        and this function just determines the order that clusters are pruned to produce slightly different results.

        Arguments:
            iterations: How many times clusters are pruned by repeating this same function.
            pruneBySize=False: Whether prunable clusters should be pruned from smallest to largest. This has a large impact on performance if set to True.
                When set to False, the smart method is pretty much as fast as the simple method
            reversePruneBySize=False: By default, prunes clusters from smallest to largest. Set to True to prune by largest to smallest.
            reversePruneByIntensity=True: Whether clusters should be pruned based on color intensity in order from darkest to lightest by default.
            showPlots=False: Whether to show intermediate pruning plots for each iteration.
        """

        for i in range(iterations):
            self.generatePrunableClusters(showPlots=False)

            image = self.image.copy().astype(np.int32)
            prunableClusters = self.prunableClusters

            mergedColors = -np.ones_like(image, dtype=np.int32)

            if showPlots:
                plt.figure(figsize=(20, 20)), plt.imshow(self.image), plt.title(
                    "Before pruning"
                ), plt.show()

            colorsOrdered = sorted(
                prunableClusters.items(),
                key=lambda x: np.sum(np.array(x[0])) ** 2,
                reverse=reversePruneByIntensity,
            )

            for color, labelMask in colorsOrdered:
                color = np.array(color, dtype=np.uint8)

                uniqueLabels = np.unique(labelMask)[1:]

                # Get the labels in an order sorted by their patch size in the labelMask excluding the last element which is the background
                if pruneBySize:
                    uniqueLabels = np.array(
                        sorted(
                            uniqueLabels, key=lambda x: Counter(labelMask.flatten())[x]
                        )[:-1]
                    )
                    if reversePruneBySize:
                        uniqueLabels[::-1]

                # plt.figure(figsize=(20, 20)), plt.imshow(labelMask), plt.title('labelMask'), plt.show()

                # If no unique labels are detected, continue to the next color
                if uniqueLabels.shape[0] == 0:
                    continue

                surroundingColors = self.getMainSurroundingColorVectorized(
                    image, labelMask, uniqueLabels
                )

                # Create an index mapping for each unique label
                consistentIndexingMap = {
                    label: index for index, label in enumerate(uniqueLabels)
                }

                # Create an indexed version of labelMask for non-zero labels
                consistentLabelMask = np.zeros_like(labelMask)

                # Edit the current label mask so it has consistent integer values to quickly map them to colors
                for label, index in consistentIndexingMap.items():
                    consistentLabelMask[labelMask == label] = index

                # Apply the mapping only to non-zero labels
                image[labelMask != 0] = surroundingColors[
                    consistentLabelMask[labelMask != 0]
                ]

            if showPlots:
                plt.figure(figsize=(20, 20)), plt.imshow(mergedColors), plt.title(
                    "mergedColors"
                ), plt.show()

                mergedColorsMask = (mergedColors == -1).astype(np.uint8)
                plt.figure(figsize=(20, 20)), plt.imshow(
                    mergedColorsMask * 255
                ), plt.title("mergedColorsMask"), plt.show()

                plt.figure(figsize=(20, 20)), plt.imshow(self.image), plt.title(
                    "Before pruning"
                ), plt.show()
                # plt.figure(figsize=(20, 20)), plt.imshow(prunedImage), plt.title('After pruning'), plt.show()
                plt.figure(figsize=(20, 20)), plt.imshow(image), plt.title(
                    "After pruning"
                ), plt.show()

                plt.figure(figsize=(20, 20)), plt.imshow(
                    np.abs(self.image - image)
                ), plt.title("Diff"), plt.show()

            self.setImage(image.copy())

    def pruneClustersSimple(self, iterations: int = 3, showPlots=False, trySlow=False):
        """
        A simple cluster pruning method which iteratively prunes the smallest clusters below the self.pruningThreshold class variable.
        In most cases, this simple method produces similar results to pruneClustersSmart(), but is faster.

        Arguments:
            iterations: How many times clusters are pruned by repeating this same function. A single iteration probably isn't
                guaranteed to remove all small clusters, so this function can be run any number of times to ensure all clusters are pruned
        """

        print(f"Starting pruning... \nIteration (of {iterations}): ", end="")

        if trySlow:
            print(
                "WARNING: USING ITERATIVE self.getMainSurroundingColor()! EXPECT POOR PERFORMANCE"
            )

        for i in range(iterations):
            print(f"{i+1} ", end="")

            image = self.image.copy().astype(np.int32)
            # print('Starting generatePrunableClusters()')
            self.generatePrunableClusters(showPlots=False)
            # print('Done!')

            prunableClusters = self.prunableClusters

            if showPlots:
                plt.figure(figsize=(20, 20)), plt.imshow(self.image), plt.title(
                    "Before pruning"
                ), plt.show()

            # print('Starting pruning loop')
            for color, labelMask in prunableClusters.items():
                color = np.array(color, dtype=np.uint8)

                uniqueLabels = np.unique(labelMask)[
                    1:
                ]  # Exclude the first label which refers to the background

                # If no unique labels are detected, continue to the next color
                if uniqueLabels.shape[0] == 0:
                    continue

                if trySlow:
                    # A much slower iterative version of cluster pruning
                    surroundingColorsList = []
                    for label in uniqueLabels:
                        clusterMask = (labelMask != label).astype(np.uint8)
                        surroundingColor = self.getMainSurroundingColor(
                            image, clusterMask
                        )
                        surroundingColorsList.append(surroundingColor)

                    surroundingColors = np.array(surroundingColorsList, dtype=np.uint8)

                    for idx in range(uniqueLabels.shape[0]):
                        currentLabel = uniqueLabels[idx]
                        currentColor = surroundingColors[idx, :]
                        image[labelMask == currentLabel] = currentColor

                else:
                    # The fast vectorized version
                    surroundingColors = self.getMainSurroundingColorVectorized(
                        image, labelMask, uniqueLabels
                    )

                    # Create an index mapping for each unique label
                    consistentIndexingMap = {
                        label: index for index, label in enumerate(uniqueLabels)
                    }

                    # Create an indexed version of labelMask for non-zero labels
                    consistentLabelMask = np.zeros_like(labelMask)

                    # Edit the current label mask so it has consistent integer values to quickly map them to colors
                    for label, index in consistentIndexingMap.items():
                        consistentLabelMask[labelMask == label] = index

                    # Apply the mapping only to non-zero labels
                    image[labelMask != 0] = surroundingColors[
                        consistentLabelMask[labelMask != 0]
                    ]

            if showPlots:
                plt.figure(figsize=(20, 20)), plt.imshow(self.image), plt.title(
                    "Before pruning"
                ), plt.show()
                plt.figure(figsize=(20, 20)), plt.imshow(image), plt.title(
                    "After pruning"
                ), plt.show()

                plt.figure(figsize=(20, 20)), plt.imshow(
                    np.abs(self.image - image)
                ), plt.title("Diff"), plt.show()

            self.setImage(image.astype(np.uint8))

        print("\nDone!")

    def getBoundaryImage(
        self, image: np.ndarray = None, scale: float = 1
    ) -> np.ndarray:
        """
        Gets a boundary image between colors in a PBN template by running an edge filter on the provided image or self.image.
        Upscaling the image before passing it to this function gives better resolution.

        Arguments:
            image: An input image to get the edges of. Uses self.image if image is None
            scale: A value to scale the image by before applying the edge filter. Useful if you want higher resolution
                in the resulting boundary image for labeling regions.

        Returns:
            boundaryImage: A binary image that represents the boundaries found when applying the edge filter.
        """

        img = None
        if image is None:
            img = self.getImage().astype(np.uint8)
        else:
            img = image.copy().astype(np.uint8)

        edgeFilter = np.array(([0, 1, 0], [1, -4, 1], [0, 1, 0]))

        if scale != 1:
            img = self.resizeImage(image=img, scale=scale)

        boundaryImage = cv2.filter2D(img, ddepth=-1, kernel=edgeFilter)
        boundaryImage = np.sum(boundaryImage, axis=2)
        boundaryImage[boundaryImage > 0] = 1

        return boundaryImage

    def refine_region(self, mask: np.ndarray, extra_colors: int = 10):
        """
        對遮罩區域單獨做更細的 K-Means，結果直接覆蓋原量化圖。
        mask: 與影像同尺寸的灰階遮罩（255=選取區域，0=其餘）
        extra_colors: 這個區域額外分配的顏色數（不限制與基礎色重複）

        稀少顏色保護：K-Means 完成後，對指派誤差偏高的像素（前 15%）
        再跑一次 mini K-Means（最多 3 色），把救回的顏色合入 centers，
        避免嘴唇等小面積高彩色被多數派膚色吞掉。
        """
        img = self.getImage()
        h, w = img.shape[:2]

        # 調整遮罩尺寸
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        region_pixels_idx = np.where(mask_resized > 127)

        if len(region_pixels_idx[0]) == 0:
            print("⚠️ 遮罩區域沒有像素，跳過細化")
            return

        # 取出遮罩區域的原始像素
        region_pixels = self.originalImage[region_pixels_idx]  # (N, 3)

        if len(region_pixels) < extra_colors:
            extra_colors = max(1, len(region_pixels) // 2)

        # 對這個區域單獨跑 K-Means
        from sklearn.cluster import KMeans as _KMeans
        model = _KMeans(n_clusters=extra_colors, n_init="auto", random_state=42)
        labels = model.fit_predict(region_pixels)
        centers = model.cluster_centers_  # (extra_colors, 3)

        # ── 稀少顏色保護 ──────────────────────────────────────────────
        region_pixels_f = region_pixels.astype(np.float32)
        assigned = centers[labels]
        distances = np.linalg.norm(region_pixels_f - assigned, axis=1)
        threshold = np.percentile(distances, 85)   # 前 15% 高誤差像素
        outlier_mask = distances > threshold

        # 亮度保護：LAB L* > 78 的像素（眼白、高光）強制加入
        lab_pixels = cv2.cvtColor(
            region_pixels.astype(np.uint8).reshape(-1, 1, 3),
            cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float32)
        bright_mask = lab_pixels[:, 0] > 200
        outlier_mask = outlier_mask | bright_mask
        if bright_mask.sum() > 0:
            print(f"  → 亮度保護：{int(bright_mask.sum())} 個高亮像素（眼白/高光）")

        if outlier_mask.sum() >= 10:
            outlier_pixels = region_pixels_f[outlier_mask]
            n_rescue = min(5, max(1, int(outlier_mask.sum() // 50)))
            if len(outlier_pixels) >= n_rescue:
                rm = _KMeans(n_clusters=n_rescue, n_init="auto", random_state=42)
                rm.fit(outlier_pixels)
                centers = np.vstack([centers, rm.cluster_centers_])
                print(f"  → 稀少顏色保護：救回 {n_rescue} 色（outlier {outlier_mask.sum()} px）")
        # ──────────────────────────────────────────────────────────────

        # 如果有固定色盤，在 LAB 空間 snap
        if self.fixed_palette is not None:
            def rgb_to_lab(arr):
                u8 = arr.astype(np.uint8).reshape(-1, 1, 3)
                return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            centers_lab = rgb_to_lab(centers)
            palette_lab = rgb_to_lab(self.fixed_palette)
            snapped = []
            for c_lab in centers_lab:
                idx = np.argmin(np.linalg.norm(palette_lab - c_lab, axis=1))
                snapped.append(self.fixed_palette[idx])
            centers = np.array(snapped, dtype=np.float32)

        # 將每個遮罩像素換成最近的細化顏色
        diff = region_pixels_f[:, np.newaxis, :] - centers[np.newaxis, :, :]
        nearest = np.argmin((diff ** 2).sum(axis=2), axis=1)
        new_colors = centers[nearest].astype(np.uint8)

        # ── 微小色塊合併（可繪性過濾）────────────────────────────────
        # 找出佔比過小的顏色（低於遮罩區域總像素數的 0.5%，且絕對值 < 30 pixels）
        total_pixels = len(new_colors)
        min_pixels = max(30, int(total_pixels * 0.005))
        unique_colors, counts = np.unique(new_colors.reshape(-1, 3), axis=0, return_counts=True)
        tiny_colors = unique_colors[counts < min_pixels]

        if len(tiny_colors) > 0:
            merged = 0
            # 建立「有效顏色」集合（排除微小色）
            valid_colors = unique_colors[counts >= min_pixels].astype(np.float32)
            if len(valid_colors) > 0:
                for tc in tiny_colors:
                    tc_f = tc.astype(np.float32)
                    # 找最近的有效顏色
                    dists = np.linalg.norm(valid_colors - tc_f, axis=1)
                    nearest_valid = valid_colors[np.argmin(dists)].astype(np.uint8)
                    # 把這個微小色的所有像素換成最近有效顏色
                    match = np.all(new_colors == tc, axis=1)
                    new_colors[match] = nearest_valid
                    merged += match.sum()
                print(f"  → 微小色塊合併：{len(tiny_colors)} 色 / {merged} 像素 併入鄰近色")
        # ──────────────────────────────────────────────────────────────

        # 寫回影像
        img[region_pixels_idx] = new_colors
        self.setImage(img)
        unique = len(np.unique(new_colors.reshape(-1, 3), axis=0))
        print(f"✅ 遮罩區域細化完成，{unique} 個細化色")

    def merge_tiny_colors(self, min_pixels: int = None, min_ratio: float = 0.005,
                          exclude_mask: np.ndarray = None):
        """
        小色塊合併：把佔比過低的顏色替換成最近的顏色。
        exclude_mask: 若提供（與影像同尺寸的灰階遮罩），遮罩內外分開處理，
                      避免遮罩內救回的稀少顏色被遮罩外的統計吃掉。
        min_pixels:   絕對像素數下限（None = 自動由 min_ratio 計算）
        min_ratio:    各區域總像素佔比下限（預設 0.5%）
        """
        img = self.getImage().astype(np.uint8)
        h, w = img.shape[:2]

        def _merge_region(pixel_arr, ratio):
            """對一段像素陣列做小色合併，回傳修改後的陣列"""
            total = len(pixel_arr)
            thresh = max(30, int(total * ratio))
            unique_c, counts = np.unique(pixel_arr, axis=0, return_counts=True)
            tiny = unique_c[counts < thresh]
            valid = unique_c[counts >= thresh].astype(np.float32)
            if len(tiny) == 0 or len(valid) == 0:
                return pixel_arr, 0, 0
            merged = 0
            for tc in tiny:
                nearest = valid[np.argmin(np.linalg.norm(valid - tc.astype(np.float32), axis=1))].astype(np.uint8)
                match = np.all(pixel_arr == tc, axis=1)
                pixel_arr[match] = nearest
                merged += int(match.sum())
            return pixel_arr, len(tiny), merged

        pixels = img.reshape(-1, 3)

        if exclude_mask is not None:
            mask_r = cv2.resize(exclude_mask, (w, h), interpolation=cv2.INTER_NEAREST).reshape(-1)
            fg_idx = mask_r > 127   # 遮罩內（SAM 細化區）
            bg_idx = ~fg_idx        # 遮罩外

            fg_pixels = pixels[fg_idx].copy()
            bg_pixels = pixels[bg_idx].copy()

            fg_pixels, fg_n, fg_px = _merge_region(fg_pixels, min_ratio)
            bg_pixels, bg_n, bg_px = _merge_region(bg_pixels, min_ratio)

            pixels[fg_idx] = fg_pixels
            pixels[bg_idx] = bg_pixels

            total_n = fg_n + bg_n
            total_px = fg_px + bg_px
            region_info = f"（遮罩內 {fg_n} 色/{fg_px} px；遮罩外 {bg_n} 色/{bg_px} px）"
        else:
            if min_pixels is None:
                min_pixels = max(30, int(h * w * min_ratio))
            pixels, total_n, total_px = _merge_region(pixels, min_ratio)
            region_info = ""

        if total_n > 0:
            self.setImage(pixels.reshape(h, w, 3))
            print(f"  → 小色塊合併：{total_n} 個顏色 / {total_px} 個像素{region_info}")

    def apply_weighted_region(self, mask: np.ndarray, total_colors: int,
                               weight_ratio: float = 0.65,
                               bg_extra_blur: int = 0):
        """
        將色數預算按比例分配：選取區佔 weight_ratio，非選取區佔其餘。
        兩區分別跑 K-Means 後合回同一張圖。

        mask:           灰階遮罩（255=選取，0=非選取）
        total_colors:   總色數預算（來自難易度設定）
        weight_ratio:   選取區佔總色數比例（預設 0.65）
        bg_extra_blur:  非選取區額外模糊 ksize（0=不加，入門建議 15）
        """
        h, w = self.originalImage.shape[:2]
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        fg_idx = np.where(mask_r > 127)   # 選取區像素位置
        bg_idx = np.where(mask_r <= 127)  # 非選取區像素位置

        fg_count = len(fg_idx[0])
        bg_count = len(bg_idx[0])

        if fg_count == 0 or bg_count == 0:
            print("⚠️ 遮罩區域異常，跳過 weighted 處理")
            return

        fg_colors = max(1, round(total_colors * weight_ratio))
        bg_colors = max(1, total_colors - fg_colors)
        print(f"  選取區: {fg_colors} 色 | 非選取區: {bg_colors} 色")

        from sklearn.cluster import KMeans as _KMeans

        def snap_to_palette(centers):
            if self.fixed_palette is None:
                return centers.astype(np.uint8)
            def rgb_to_lab(arr):
                u8 = arr.astype(np.uint8).reshape(-1, 1, 3)
                return cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            c_lab = rgb_to_lab(centers)
            p_lab = rgb_to_lab(self.fixed_palette)
            snapped = []
            for c in c_lab:
                idx = np.argmin(np.linalg.norm(p_lab - c, axis=1))
                snapped.append(self.fixed_palette[idx])
            return np.array(snapped, dtype=np.uint8)

        # --- 選取區 ---
        fg_pixels = self.originalImage[fg_idx].astype(np.float32)
        fg_k = min(fg_colors, len(fg_pixels))
        model_fg = _KMeans(n_clusters=fg_k, n_init="auto", random_state=None)
        model_fg.fit(fg_pixels)
        fg_centers = snap_to_palette(model_fg.cluster_centers_)
        diff = fg_pixels[:, np.newaxis, :] - fg_centers[np.newaxis, :, :].astype(np.float32)
        fg_new = fg_centers[np.argmin((diff**2).sum(axis=2), axis=1)]

        # --- 非選取區（可選額外模糊） ---
        bg_src = self.originalImage.copy()
        if bg_extra_blur > 0:
            ksize = bg_extra_blur | 1  # 確保奇數
            bg_src = cv2.GaussianBlur(bg_src, (ksize, ksize), 0)
        bg_pixels = bg_src[bg_idx].astype(np.float32)
        bg_k = min(bg_colors, len(bg_pixels))
        model_bg = _KMeans(n_clusters=bg_k, n_init="auto", random_state=None)
        model_bg.fit(bg_pixels)
        bg_centers = snap_to_palette(model_bg.cluster_centers_)
        diff = bg_pixels[:, np.newaxis, :] - bg_centers[np.newaxis, :, :].astype(np.float32)
        bg_new = bg_centers[np.argmin((diff**2).sum(axis=2), axis=1)]

        # --- 合回影像（明確轉 uint8，確保後續 pruning 和輸出不出現 dtype 錯誤）---
        result = self.originalImage.copy()
        result[fg_idx] = fg_new
        result[bg_idx] = bg_new
        self.setImage(result.astype(np.uint8))
        print(f"✅ weighted_region 完成")

    def set_final_pbn(self, blur_ksize=21, blur_sigma_color=21, blur_sigma_space=14, prune_iterations=6):
        """
        Runs all necessary functions to get the final paint by number image.
        細緻度由外部參數控制：
          blur_ksize/sigma: 越大越模糊（色塊越大越簡單）
          prune_iterations: 越多小色塊越少（越簡單）
          pruningThreshold 在 __init__ 設定
        """
        originalDims = self.getImage().shape[:-1]
        self.blurImage_(blurType="bilateral", ksize=blur_ksize,
                        sigmaColor=blur_sigma_color, sigmaSpace=blur_sigma_space)
        self.resizeImage_(0.5)
        self.cluster_colors_()
        self.pruneClustersSimple(iterations=prune_iterations)
        self.resizeImage_(dimension=originalDims)
        # draw rectangle around image so border is recognized
        img = self.getImage()
        img = cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), 10)
        self.setImage(img)

    @staticmethod
    def _contour_to_bezier_path(contour, tension: float = 0.3) -> str:
        """
        將輪廓點轉成 SVG Catmull-Rom Bezier 路徑字串。
        tension: 0 = 直線, 0.5 = 標準 Catmull-Rom, 建議 0.2~0.4
        """
        pts = contour.squeeze()
        if len(pts.shape) == 1:
            pts = pts.reshape(1, 2)
        n = len(pts)
        if n < 3:
            # 點太少，退回直線
            d = f"M {pts[0][0]},{pts[0][1]}"
            for p in pts[1:]:
                d += f" L {p[0]},{p[1]}"
            return d + " Z"

        def p(i):
            return pts[i % n].astype(float)

        d = f"M {p(0)[0]:.2f},{p(0)[1]:.2f}"
        for i in range(n):
            p0, p1, p2, p3 = p(i - 1), p(i), p(i + 1), p(i + 2)
            cp1 = p1 + (p2 - p0) * tension
            cp2 = p2 - (p3 - p1) * tension
            d += (f" C {cp1[0]:.2f},{cp1[1]:.2f}"
                  f" {cp2[0]:.2f},{cp2[1]:.2f}"
                  f" {p2[0]:.2f},{p2[1]:.2f}")
        return d + " Z"

    @staticmethod
    def _smooth_masks_for_svg(quantized_rgb, color_masks, blur_ksize=5):
        """
        對量化圖做 median blur 後 snap 回調色盤顏色，
        讓相鄰色塊共用同一條邊界（消除鋸齒 + 消除 gap）。
        只用於 SVG 輪廓計算，不修改實際量化結果。
        """
        q_bgr = cv2.cvtColor(quantized_rgb, cv2.COLOR_RGB2BGR)
        q_blurred = cv2.medianBlur(q_bgr, blur_ksize)
        q_blurred_rgb = cv2.cvtColor(q_blurred, cv2.COLOR_BGR2RGB)

        # 調色盤顏色陣列
        palette_colors = np.array([list(c) for c in color_masks.keys()], dtype=np.float32)
        h, w = quantized_rgb.shape[:2]
        flat = q_blurred_rgb.reshape(-1, 3).astype(np.float32)

        # 每個像素 snap 回最近的調色盤顏色（RGB 歐氏距離）
        dists = np.sum((flat[:, None, :] - palette_colors[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(dists, axis=1)
        snapped = palette_colors[nearest].reshape(h, w, 3).astype(np.uint8)

        # 從 snapped 圖重建各色遮罩
        smoothed_masks = {}
        for color_key in color_masks.keys():
            color_arr = np.array(list(color_key), dtype=np.uint8)
            mask_bool = np.all(snapped == color_arr, axis=2)
            smoothed_masks[color_key] = np.stack([mask_bool] * 3, axis=2)

        return smoothed_masks

    def output_to_svg(self, svg_path: str, output_palette_path: str = None):
        h, w = self.getImage().shape[:2]
        dwg = svgwrite.Drawing(svg_path, profile="tiny", viewBox=(f"0 0 {w} {h}"))
        i = 0
        color_masks = self.getUniqueColorsMasks()

        # 對量化圖預平滑後重建遮罩 → 相鄰色塊共用邊界，消除鋸齒與 gap
        smoothed_masks = self._smooth_masks_for_svg(self.getImage(), color_masks, blur_ksize=5)

        # 單遍渲染：fill + stroke 用同一個輪廓（stroke 覆蓋接縫）
        all_contours = []
        for idx, (color, mask) in enumerate(color_masks.items()):
            smooth_mask = smoothed_masks.get(color, mask)
            mask_uint8 = np.all(smooth_mask, axis=2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
            )
            for c in contours:
                area = cv2.contourArea(c)
                if area >= 4 and len(c) >= 3:
                    all_contours.append((area, idx, color, c))

        all_contours.sort(key=lambda x: x[0], reverse=True)

        # 建立這幅畫的 1~N 專屬編號（面積最大的顏色得 #1）
        color_to_seq = {}
        for _, idx, color, _ in all_contours:
            key = tuple(int(v) for v in color)
            if key not in color_to_seq:
                color_to_seq[key] = len(color_to_seq) + 1

        # 建立 JSON 對照表
        palette_map = {}
        for idx, (color, mask) in enumerate(color_masks.items()):
            key = tuple(int(v) for v in color)
            seq_num = color_to_seq.get(key, idx + 1)

            master_id = "N/A"
            if self.fixed_palette is not None:
                match = np.where((self.fixed_palette == color).all(axis=1))[0]
                if len(match) > 0:
                    master_id = int(match[0] + 1)

            palette_map[idx] = {
                "template_id": seq_num,
                "master_id": master_id,
                "rgb": [int(c) for c in color],
                "shapes": []
            }
        palette = list(palette_map.values())

        # 單遍：fill（白底 + 25% 提示色）+ stroke（黑邊）同一 polygon
        for area, idx, color, c in all_contours:
            color_fill = "rgb({},{},{})".format(int(color[0]), int(color[1]), int(color[2]))
            points = c.squeeze().tolist()
            if len(c.squeeze().shape) == 1:
                points = [points]

            group = dwg.g(id=str(i))
            group.add(dwg.polygon(points, fill="white", stroke="black", stroke_width="1",
                                  stroke_linejoin="round", stroke_linecap="round"))
            group.add(dwg.polygon(points, fill=color_fill, stroke="none", fill_opacity="0.25"))
            key = tuple(int(v) for v in color)
            label = str(color_to_seq.get(key, idx + 1))
            text = self.add_text_label(dwg, c, label)
            group.add(text)
            dwg.add(group)

            palette[idx]["shapes"].append(str(i))
            i += 1

        dwg.save()
        print(f"{i} shapes")

        if output_palette_path:
            with open(output_palette_path, "w", encoding="utf-8") as outfile:
                json.dump(palette, outfile, ensure_ascii=False, indent=2)

        return palette

    def output_filled_image(self, output_path: str, border: bool = False):
        """
        輸出填色完成效果圖：直接輸出量化後的影像，每個像素已有正確顏色，不需重繪輪廓。
        border: 是否疊加黑色輪廓線（預設 False）。
        """
        filled_rgb = self.getImage().copy()

        if border:
            color_masks = self.getUniqueColorsMasks()
            for color, mask in color_masks.items():
                mask_uint8 = np.all(mask, axis=2).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for c in contours:
                    if cv2.contourArea(c) >= 4:
                        cv2.drawContours(filled_rgb, [c], -1, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        filled_bgr = cv2.cvtColor(filled_rgb, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", filled_bgr)
        if success:
            with open(output_path, "wb") as f:
                f.write(buf.tobytes())
            print(f"填色效果圖已儲存至: {output_path}")
        else:
            print(f"❌ 填色效果圖編碼失敗: {output_path}")

    def point_inside_contour(self, point, contour):
        """Check if a point is inside a contour."""
        return cv2.pointPolygonTest(contour, (point[0], point[1]), False) >= 0

    def sample_text_position(self, contour, num_samples=150):
        if len(contour) < 4:
            # Not enough points to form a polygon return the centroid or the first point of the contour
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                return (
                    int(moments["m10"] / moments["m00"]),
                    int(moments["m01"] / moments["m00"]),
                )
            else:
                return (0, 0)

        # Convert contour to a shapely polygon for area computation
        polygon = Polygon([pt[0] for pt in contour])
        min_x, min_y, max_x, max_y = polygon.bounds
        best_point = (0, 0)
        max_distance = -1

        for _ in range(num_samples):
            x, y = random.uniform(min_x, max_x), random.uniform(min_y, max_y)
            point = Point(x, y)

            # Check if the sampled point is within the polygon and its distance to edges
            if polygon.contains(point):
                distance = polygon.exterior.distance(point)
                if distance > max_distance:
                    max_distance = distance
                    best_point = (x, y)

        return best_point

    def add_text_label(self, dwg, contour, label):
        best_point = self.sample_text_position(contour)

        # Estimate a suitable text size
        text_size = np.clip(np.sqrt(cv2.contourArea(contour)) / 8, 4, 12)

        text = dwg.text(
            label,
            insert=best_point,
            font_size=str(text_size),
            text_anchor="middle",
        )
        return text
        # dwg.add(text)
