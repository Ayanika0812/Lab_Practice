class PowerCalculator:
    def __init__(self, x: float, n: int):
        self.x = x
        self.n = n

    def pow(self) -> float:
        if self.n == 0:
            return 1
        elif self.n < 0:
            return 1 / self._power(self.x, -self.n)
        else:
            return self._power(self.x, self.n)

    def _power(self, base: float, exponent: int) -> float:
        result = 1
        while exponent > 0:
            if exponent % 2 == 1:  # If exponent is odd
                result *= base
            base *= base
            exponent //= 2
        return result

class StringProcessor:
    def __init__(self):
        self.user_string = ""

    def get_String(self):
        self.user_string = input("Enter a string: ")

    def print_String(self):
        print(self.user_string.upper())

# Example usage
if __name__ == "__main__":
    # PowerCalculator example
    x = 2.0
    n = 10
    calculator = PowerCalculator(x, n)
    print(calculator.pow())  # Output: 1024.0

    x = 2.0
    n = -3
    calculator = PowerCalculator(x, n)
    print(calculator.pow())  # Output: 0.125

    # StringProcessor example
    processor = StringProcessor()
    processor.get_String()
    processor.print_String()
