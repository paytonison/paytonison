#!/usr/bin/env python3

def sugma_ligma(limit: int = 100) -> None:
    """
    Print numbers from 1‒limit replacing:
     - multiples of 3 with "sugma"
     - multiples of 5 with "ligma"
     - multiples of both with "sugma-ligma"
    """
    for num in range(1, limit + 1):
        match (num % 3 == 0, num % 5 == 0):
            case (True, True):
                print("sugma-ligma")
            case (True, False):
                print("sugma")
            case (False, True):
                print("ligma")
            case _:
                print(num)

if __name__ == "__main__":
    sugma_ligma()
