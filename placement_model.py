import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_FILE = "placement_data.csv"


def load_data():
    """Load and validate the placement dataset."""

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "cgpa",
        "placement_exam_marks",
        "placed"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


def train_model():
    """Train the student placement prediction model."""

    df = load_data()

    X = df[
        [
            "cgpa",
            "placement_exam_marks"
        ]
    ]

    y = df["placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy


def predict_placement(cgpa, placement_exam_marks):
    """Predict whether a student will be placed."""

    model, _ = train_model()

    input_data = pd.DataFrame(
        [
            {
                "cgpa": cgpa,
                "placement_exam_marks": placement_exam_marks
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    return int(prediction)


if __name__ == "__main__":

    model, accuracy = train_model()

    print("=" * 50)
    print("STUDENT PLACEMENT PREDICTION")
    print("=" * 50)

    print(f"Model Accuracy: {accuracy:.4f}")

    cgpa = 7.5
    placement_exam_marks = 60

    prediction = predict_placement(
        cgpa,
        placement_exam_marks
    )

    print(f"CGPA: {cgpa}")
    print(
        f"Placement Exam Marks: "
        f"{placement_exam_marks}"
    )

    if prediction == 1:
        print("Prediction: PLACED")
    else:
        print("Prediction: NOT PLACED")

    print("=" * 50)
