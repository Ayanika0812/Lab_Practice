from typing import List

class SubsetGenerator:
    def __init__(self, input_set: List[int]):
        self.input_set = input_set

    def get_subsets(self) -> List[List[int]]:
        # Start with an empty subset
        result = [[]]
        for num in self.input_set:
            # For each number, add it to all existing subsets to form new subsets
            result += [current + [num] for current in result]
        return result

# Example usage
if __name__ == "__main__":
    input_set = [4, 5, 6]
    generator = SubsetGenerator(input_set)
    subsets = generator.get_subsets()
    print(subsets)


