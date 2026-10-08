import unittest

from placement_model import (
    load_data,
    train_model,
    predict_placement
)


class TestPlacementModel(unittest.TestCase):

    def test_dataset_columns(self):
        """Verify required dataset columns."""

        df = load_data()

        self.assertIn(
            "cgpa",
            df.columns
        )

        self.assertIn(
            "placement_exam_marks",
            df.columns
        )

        self.assertIn(
            "placed",
            df.columns
        )

    def test_dataset_row_count(self):
        """Verify the dataset contains 1000 records."""

        df = load_data()

        self.assertEqual(
            len(df),
            1000
        )

    def test_target_values(self):
        """Verify placed contains only 0 and 1."""

        df = load_data()

        values = set(
            df["placed"].unique()
        )

        self.assertTrue(
            values.issubset({0, 1})
        )

    def test_model_training(self):
        """Verify that the ML model trains."""

        model, accuracy = train_model()

        self.assertIsNotNone(model)

        self.assertGreaterEqual(
            accuracy,
            0.50
        )

        self.assertLessEqual(
            accuracy,
            1.00
        )

    def test_prediction_output(self):
        """Verify prediction returns 0 or 1."""

        prediction = predict_placement(
            7.5,
            60
        )

        self.assertIn(
            prediction,
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
