from typing import List, Tuple, Optional

class PairFinder:
    def __init__(self, numbers: List[int], target: int):
        self.numbers = numbers
        self.target = target

    def find_pair(self) -> Optional[Tuple[int, int]]:
        # Dictionary to store numbers and their indices
        seen = {}
        for index, number in enumerate(self.numbers):
            complement = self.target - number
            if complement in seen:
                # Return indices as 1-based instead of 0-based if required
                return (seen[complement] + 1, index + 1)
            seen[number] = index
        return None

# Example usage
if __name__ == "__main__":
    numbers = [10, 20, 10, 40, 50, 60, 70]
    target = 50
    finder = PairFinder(numbers, target)
    result = finder.find_pair()
    if result:
        print(result)  # Output: 3, 4
    else:
        print("No pair found.")