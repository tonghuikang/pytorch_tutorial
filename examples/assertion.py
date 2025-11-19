"""
Contains a function to assert equality for loss values.
"""


def assert_loss(
    epoch: int,
    loss: float,
    epoch_to_expected_loss: dict[int, float],
    skip_to_print_loss: bool = False,
) -> None:
    """
    Assert that the loss at a given epoch matches the expected value.

    Args:
        epoch: Current training epoch
        loss: Current loss value
        epoch_to_expected_loss: Dictionary mapping epoch numbers to expected loss values
        skip_to_print_loss: If True, print the loss value instead of asserting.
                           Useful for collecting new expected loss values.
    """
    if epoch in epoch_to_expected_loss:
        if skip_to_print_loss:
            print(f"    {epoch}: {loss:.4f},")
            return
        assert abs(loss - epoch_to_expected_loss[epoch]) < 0.001, (
            f"Epoch {epoch}: expected {epoch_to_expected_loss[epoch]}, got {loss}"
        )
