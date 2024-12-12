import unittest
from cognitum.dataset import DatasetManager
from cognitum.types import DataItem
from cognitum.exceptions import DatasetError

class TestDatasetManager(unittest.TestCase):

    def setUp(self):
        self.valid_data = [
            ("id1", "text1"),
            ("id2", "text2"),
            ("id3", "text3")
        ]
        self.invalid_data = [
            ("id1", "text1"),
            ("id1", "text2")  # Duplicate id
        ]

    def test_initialization_with_valid_data(self):
        manager = DatasetManager(self.valid_data)
        self.assertEqual(len(manager), 3)

    def test_initialization_with_invalid_data(self):
        with self.assertRaises(DatasetError):
            DatasetManager(self.invalid_data)

    def test_hash(self):
        manager = DatasetManager(self.valid_data)
        hash1 = manager.hash()
        hash2 = manager.hash()
        self.assertEqual(hash1, hash2)

    def test_sample(self):
        manager = DatasetManager(self.valid_data)
        sample = manager.sample(2, seed=42)
        self.assertEqual(len(sample), 2)
        # Check if the sample is consistent with the seed
        self.assertEqual(sample, [("id3", "text3"), ("id1", "text1")])

    def test_sample_too_large(self):
        manager = DatasetManager(self.valid_data)
        with self.assertRaises(DatasetError):
            manager.sample(10)

    def test_len(self):
        manager = DatasetManager(self.valid_data)
        self.assertEqual(len(manager), 3)

    def test_getitem(self):
        manager = DatasetManager(self.valid_data)
        self.assertEqual(manager[0], ("id1", "text1"))
        self.assertEqual(manager[1], ("id2", "text2"))

if __name__ == '__main__':
    unittest.main()
