"""Synthetic interface illustration, authored for this release; not dataset content."""


def examples():
    return [
        {
            "title": "Synthetic example / 合成示例",
            "state": {"left_box": "red", "right_box": "blue"},
            "questions": {
                "box": {
                    "type": "choice",
                    "instructions": "Which box is red?",
                    "criteria": {"left": "The left box", "right": "The right box"},
                }
            },
        }
    ]
