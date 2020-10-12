import unittest
import torch
from ..fastai_audio.utils import one_hot_tensor, one_hot_decode


class TestOneHotEncode(unittest.TestCase):
    def test_single_one_in_column(self):
        n_classes = 32
        signal = torch.randint(0, n_classes, (64,))
        signal_oh = one_hot_tensor(signal, n_classes)
        # that signal has no more than one value per each column
        self.assertTrue(all(signal_oh.sum(dim=0) == 1))

    def test_only_ones_and_zeros(self):
        n_classes = 32
        signal = torch.randint(0, n_classes, (64,))
        signal_oh = one_hot_tensor(signal, n_classes)
        # all different from 0 equal 1
        self.assertTrue(torch.all(signal_oh[signal_oh != 0] == 1))

    def test_identity_matrix(self):
        n_classes = 10
        signal = torch.arange(n_classes)
        signal_oh = one_hot_tensor(signal, n_classes)
        self.assertTrue(torch.equal(signal_oh, torch.eye(n_classes)))

    def test_known_one_hot(self):
        oh_tensor = torch.tensor([[1, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0]], dtype=torch.float)
        signal = torch.tensor([0, 1, 0, 3, 2])
        signal_oh = one_hot_tensor(signal, 4)
        self.assertTrue(torch.equal(signal_oh, oh_tensor))

    def test_random_one_hot(self):
        n_classes = 32
        signal = torch.randint(0, n_classes, (64,))
        signal_oh = one_hot_tensor(signal, n_classes)
        guessed_signal = signal_oh.argmax(0)
        self.assertTrue(torch.equal(signal, guessed_signal))


class TestOneHotDecode(unittest.TestCase):
    def test_decode_identity(self):
        n = 5
        identity = torch.eye(n)
        source = torch.arange(n)
        decoded = one_hot_decode(identity)
        self.assertTrue(torch.equal(source, decoded))

    def test_decode_known(self):
        oh_tensor = torch.tensor([[1, 0, 1, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 1],
                                  [0, 0, 0, 1, 0]], dtype=torch.float)
        decoded = one_hot_decode(oh_tensor)
        source = torch.tensor([0, 1, 0, 3, 2])
        self.assertTrue(torch.equal(source, decoded))

    def test_decode_random_int(self):
        n_classes = 32
        signal = torch.randint(0, n_classes, (64,))
        signal_oh = one_hot_tensor(signal, n_classes)
        guessed_signal = one_hot_decode(signal_oh)
        self.assertTrue(torch.equal(signal, guessed_signal))

    def test_decode_wrong(self):
        wrong = torch.arange(10).repeat(10, 1)
        decoded = one_hot_decode(wrong)
        self.assertTrue(torch.all(decoded == 0))

    def test_decode_random_float(self):
        size = 23
        oh_tensor = torch.rand((10, 23))
        guess_signal = oh_tensor.argmax(0)
        decoded = one_hot_decode(oh_tensor)
        self.assertEqual(size, len(decoded))
        self.assertTrue(torch.equal(guess_signal, decoded))


if __name__ == '__main__':
    unittest.main()
