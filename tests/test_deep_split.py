from fiber.classify.deep_split import (
    DEEP_EPOCHS,
    DEEP_EVAL_SEEDS,
    DEEP_MAX_CABLES,
    DEEP_PER_EVENT,
    DEEP_PUBLISHED_EVAL,
    train_seed,
)


def test_deep_holdout_does_not_overlap_training_seeds():
    assert DEEP_EPOCHS > 6
    assert DEEP_PER_EVENT > 32
    assert DEEP_EVAL_SEEDS == tuple(range(900_000, 900_004))
    assert DEEP_PUBLISHED_EVAL == tuple(range(800_000, 800_004))
    trained = [train_seed(epoch, index) for epoch in range(1, DEEP_EPOCHS + 1) for index in range(DEEP_MAX_CABLES)]
    assert set(trained).isdisjoint(DEEP_EVAL_SEEDS)
    assert set(trained).isdisjoint(DEEP_PUBLISHED_EVAL)
    assert set(DEEP_EVAL_SEEDS).isdisjoint(DEEP_PUBLISHED_EVAL)
    assert train_seed(1, 0) == 1000
