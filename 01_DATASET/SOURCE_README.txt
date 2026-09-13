This archive contains A Time Series Dataset of NIR Spectra and RGB and NIR-HSI Images of the Barley Germination Process by Engstrøm et al. (2025)

A description of this archive follows.

README.txt
- This file.

LICENSE.txt
- This dataset is licensed under CC BY-NC 4.0

annotations.csv
- A csv file with annotations for every single barley kernel.
- It contains four columns: petri_dish, grain_id, germination_day, variety.
- petri_dish and grain_id form a unique tuple identifier for a grain.
- germination_day is the day on which the specific grain germinated. If -1, the grain
  did not germinate at all during the experiment.
- variety is one of four grain varieties, indicating which variety the specific grain
  belongs to.

mono12p_read_write.py
- A Python file containing two functions to process the mono12p file format. mono12p is
  a way to pack two 12-bit values sequentially in three 8-bit bytes. The hyperspectral
  images are mono12p.
- mono12p_to_uint16() converts an array from mono12p to uint16.
- uint16_to_mono12p() converts an array from uint16 to mono12p.

image_crops_and_spectra/
- contains RGB/, HSI/, and HSI_abs_mean_spectra/
- RGB/ and HSI/ contain subdirectories for each day, which in turn contain
  subdirectories for each petri_dish. Each of these subdirectories contains files for
  grain_id n. If an image is present, it is guaranteed to contain a barley kernel.
  Additionally, if an image of a specific kernel is present for any day, it is present
  for all days and always contains the kernel. Let n be an arbitrary kernel. Then,
  - RGB/
    contains n_binary_mask.npy and n_binary_mask.png contain NumPy and png files of
    the binary mask segmenting barley kernel from the background. n_masked_rgb_img.png and
    n_rgb_img.png contain masked and unmasked RGB images of the barley kernel n.
  - HSI/
    contains n_binary_mask.npy, n_binary_mask.png with the binary mask for kernel n. Also
    contains n_masked_hsi_img_grayscale.png and n_hsi_img_grayscale.png, which are masked
    and unmasked grayscale versions of the hyperspectral image. Finally, contains
    n_hsi_img_mono12p.npy, which is the hyperspectral image of kernel n stored in mono12p.
  - HSI_abs_mean_spectra/
    contains subdirectories for each day. Each of these contain
    - mean_spectra.npy
      A NumPy array with NIR spectra of each barley kernel. The spectra are computed
      by taking the negative logarithm (log10) of each pixel in the hyperspectral image
      inside the binary mask (to get pseudo absorbance from reflectance) and then taking
      the average of the spatial dimensions inside the mask to get a spectrum. Finally,
      the first and last ten wavelength channels were discarded as they contained much
      noise per the hyperspectral camera's whitepaper. The result is a 204-wavelength
      NIR spectrum for each barley kernel, which corresponds to uniformly distributed
      channels from approximately 950-1650 nm.
    - mean_spectra_order.csv
      The order of spectra in mean_spectra.npy. It contains two columns: petri_dish and
      grain_id, which together form the unique identifier for a given barley kernel.
    

full_images/
- Directories pre_moisture/ and day_1 to day_5. Which in turn contain
  - petri_dish_n
    where n is a number identifying the petri_dish. These, in turn, contain 
    - HSI/ and RGB/.
      RGB/ contains rgb_img.png, which is the full RGB image of the petri dish. It also
      contains 
      - grid_coordinates.npy, which is a (6, 6, 2) array of coordinates of the grid
        intersections. Let x index the first axis of the array and y the second axis.
        Then, the grain with grain_id x * 5 + y is surrounded by four of these
        coordinates. These four coordinates are given by indexing the array at indices
        (x,y), (x, y + 1), (x + 1, y + 1), and (x + 1, y). Note that all 36
        coordinates are always contained in this array but provide no guarantee that
        grain kernels are actually present inside the corresponding grid cells.
      HSI/ contains
      - hsi_img_grayscale.png
        which is a grayscale visualization of the hyperspectral image.
      - hsi_img_pseudo_rgb.png
        which is a pseudo rgb visualization of the hyperspectral image.
      - hsi_img_mono12p.npy
        which is the full hyperspectral image stored in the mono12p format.
      - grid_coordinates.npy
        which contains the (6, 6, 2) grid coordinate array for the hyperspectral image.
        The RGB image equivalent is described above.
- HSI_grid_coordinates.json
  contains the coordinates of the four-point polygon surrounding each barley kernel on
  each hyperspectral image. For grain kernel 11 from petri dish 20 on day 3 the json
  file can be loaded as a dictionary and indexed with dict["3"]["20"]["11"] to yield a
  dictionary with eight keys: x0, x1, x2, x3, y1, y2, y3, y4, which define the
  coordinates of the polygon surrounding grain kernel 11 on the HSI image of petri dish
  20 taken on day 3. The polygon coordinates are to be constructed as
  (x0, y0), (x1, y1), (x2, y2), (x3, y3). Polygons for other kernels on other days in
  other Petri dishes are extracted similarly. Note that this file contains coordinates
  for any grid intersection and does not guarantee that a grain kernel is present
  inside a grid cell surrounded by the corresponding coordinates.
- RGB_grid_coordinates.json
  Same as above, but for the RGB images.