from implicit.als import AlternatingLeastSquares


def build_als_model(
    factors: int,
    regularization: float,
    iterations: int,
    random_state: int
):
    return AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state
    )