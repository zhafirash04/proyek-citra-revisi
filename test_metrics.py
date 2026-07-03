import unittest
import numpy as np
from filter_comparison import (
    calculate_mse,
    calculate_psnr,
    add_salt_pepper_noise,
    apply_median_filter,
    apply_gaussian_filter
)

class TestMetrics(unittest.TestCase):

    def test_calculate_mse_identical_images(self):
        """Test MSE calculation with identical images."""
        img1 = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        img2 = np.array([[100, 150], [200, 250]], dtype=np.uint8)

        mse = calculate_mse(img1, img2)
        self.assertEqual(mse, 0.0, f"MSE untuk gambar yang identik harus 0, didapat: {mse}")

    def test_calculate_mse_different_images(self):
        """Test MSE calculation with different images."""
        img1 = np.array([[100, 100], [100, 100]], dtype=np.uint8)
        img2 = np.array([[110, 110], [110, 110]], dtype=np.uint8)

        mse = calculate_mse(img1, img2)
        self.assertEqual(mse, 100.0, f"MSE yang diharapkan 100.0, didapat: {mse}")

    def test_calculate_mse_overflow_handling(self):
        """Test MSE calculation overflow handling using uint8 values."""
        img1 = np.array([[0]], dtype=np.uint8)
        img2 = np.array([[255]], dtype=np.uint8)

        mse = calculate_mse(img1, img2)
        self.assertEqual(mse, 65025.0, f"MSE overflow tidak ditangani dengan benar, didapat: {mse}")

    def test_calculate_psnr_zero_mse(self):
        """Test PSNR calculation with MSE = 0 (identical images)."""
        psnr = calculate_psnr(0)
        self.assertEqual(psnr, float('inf'), f"PSNR untuk MSE=0 harus infinity, didapat: {psnr}")

    def test_calculate_psnr_normal(self):
        """Test PSNR calculation with normal MSE."""
        mse = 100.0
        expected_psnr = 10 * np.log10((255 ** 2) / mse)
        psnr = calculate_psnr(mse)
        self.assertTrue(np.isclose(psnr, expected_psnr), f"PSNR yang diharapkan {expected_psnr}, didapat: {psnr}")

    def test_add_salt_pepper_noise_density(self):
        """Test if the noise density added is mathematically accurate."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        density = 0.2
        noisy_img = add_salt_pepper_noise(img, density)

        # Hitung jumlah piksel yang berubah (menjadi 0 atau 255)
        changed_pixels = np.sum((noisy_img == 0) | (noisy_img == 255))
        expected_pixels = int(10000 * density)

        # Toleransi 1 piksel karena pembagian bilangan bulat salt & pepper
        self.assertAlmostEqual(changed_pixels, expected_pixels, delta=2)

    def test_apply_median_filter_odd_kernel(self):
        """Test if median filter handles kernel sizes properly."""
        img = np.array([
            [10, 10, 10, 10, 10],
            [10, 255, 10, 10, 10],  # Single salt noise pixel
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10]
        ], dtype=np.uint8)

        filtered_img = apply_median_filter(img, 3)
        # Piksel noise 255 harus diganti oleh median (10)
        self.assertEqual(filtered_img[1, 1], 10, "Median filter 3x3 gagal mereduksi single salt noise")

    def test_apply_gaussian_filter_smoothing(self):
        """Test if gaussian filter smooths values with custom sigma."""
        img = np.array([
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0]
        ], dtype=np.uint8)

        filtered_img = apply_gaussian_filter(img, 3, 1.0)
        # Nilai tengah 255 harus menyebar/berkurang karena smoothing
        self.assertTrue(filtered_img[1, 1] < 255, "Gaussian filter gagal melakukan smoothing")

    # ── Tests untuk citra berwarna (3-channel RGB) ──

    def test_color_salt_pepper_noise(self):
        """Test noise injection on a 3-channel color image."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        density = 0.2
        noisy = add_salt_pepper_noise(img, density)

        self.assertEqual(noisy.shape, (100, 100, 3), "Noisy color image harus tetap 3-channel")
        # Setiap channel harus punya piksel noise
        for c in range(3):
            changed = np.sum((noisy[:, :, c] == 0) | (noisy[:, :, c] == 255))
            self.assertGreater(changed, 0, f"Channel {c} harus memiliki piksel noise")

    def test_color_median_filter(self):
        """Test median filter on a 3-channel color image."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 100
        img[5, 5, :] = 255  # Salt noise di tengah
        filtered = apply_median_filter(img, 3)

        self.assertEqual(filtered.shape, (10, 10, 3), "Filtered color image harus tetap 3-channel")
        # Noise impulsif harus tereduksi oleh median filter
        for c in range(3):
            self.assertLess(filtered[5, 5, c], 255, f"Median filter gagal mereduksi noise pada channel {c}")

    def test_color_gaussian_filter(self):
        """Test gaussian filter on a 3-channel color image."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[5, 5, :] = 255
        filtered = apply_gaussian_filter(img, 3, 1.0)

        self.assertEqual(filtered.shape, (10, 10, 3), "Filtered color image harus tetap 3-channel")
        self.assertTrue(filtered[5, 5, 0] < 255, "Gaussian filter gagal melakukan smoothing pada citra berwarna")

    def test_color_mse_identical(self):
        """Test MSE calculation with identical color images."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        mse = calculate_mse(img, img)
        self.assertEqual(mse, 0.0, "MSE untuk citra berwarna identik harus 0")

    def test_color_mse_different(self):
        """Test MSE calculation with different color images."""
        img1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        img2 = np.ones((10, 10, 3), dtype=np.uint8) * 110
        mse = calculate_mse(img1, img2)
        self.assertEqual(mse, 100.0, f"MSE citra berwarna yang diharapkan 100.0, didapat: {mse}")

    def test_color_full_pipeline(self):
        """Test full pipeline: noise → filter → MSE/PSNR on a color image."""
        np.random.seed(42)
        # Use a uniform-value image so salt-and-pepper noise is clearly outlier
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        noisy = add_salt_pepper_noise(img, 0.2)
        filtered = apply_median_filter(noisy, 3)
        mse = calculate_mse(img, filtered)
        psnr = calculate_psnr(mse)

        self.assertGreater(psnr, 0, "PSNR harus positif")
        # Filter median seharusnya memperbaiki citra (PSNR meningkat)
        noisy_mse = calculate_mse(img, noisy)
        self.assertLess(mse, noisy_mse, "MSE setelah median filter harus lebih kecil daripada MSE noisy")

if __name__ == '__main__':
    unittest.main()
