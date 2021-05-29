def split_nbr(number):
    """
    Split the given number into the multiplication of two numbers evenly weighted.
    :param number: integer
        A number to split into multiplication of two numbers.
    :return tuple
        Tuple of two numbers.
    """

    # if we don't pass to the other greater number `number + 1`
    not_it = False
    if number in [0, 1, 2]:
        return 1, 2, not_it  # one row, two columns

    # use the above number if fails to find two
    # number which has multiplication equal to `number`

    while True:
        # Search for a divider less than its middle
        i = int(number / 2) - 1
        while i > 1:
            if number % i == 0:
                return i, int(number / i), not_it
            i -= 1
        number += 1
        not_it = True
