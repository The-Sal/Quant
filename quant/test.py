import unittest
from quant import OpenClose, covariance


class TestQuant(unittest.TestCase):
    def test_OC(self):
        p = OpenClose(1, 2)
        self.assertEqual(p.returns, 1)

    def test_covariance(self):
        r_t = [
            2.3,
            1.8,
            2.1,
            1.9,
            2.4
        ]

        self.assertEqual(covariance(r_t, 1), -0.029999999999999985)

    def test_prices_to_covariance(self):
        data = [
            [222.78, 221.41],
            [223.66, 222.3],
            [223.83, 219.79],
            [222.64, 219.38],
            [229.98, 228.48],
            [228.26, 228.03],
            [237.87, 234.43],
            [233.28, 232.472],
            [234.4, 229.72],
            [236.85, 233],
            [242.7, 240.05],
            [242.21, 241.35],
            [245, 243.2],
            [243.36, 241.89],
            [243.85, 241.8201],
            [250.42, 249.43],
            [252.2, 250.75],
            [255.59, 253.06],
            [259.02, 257.63],
        ]
        oc = list(map(lambda x: OpenClose(x[0], x[1]), data))
        r_t = list(map(lambda x: x.log_returns, oc))
        self.assertEqual(covariance(r_t, 1), 1.018133060358903e-06)


