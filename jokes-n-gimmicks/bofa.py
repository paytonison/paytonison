def sugma_ligma(limit: int = 100) -> None:
    for num in range(1, limit + 1):
        match (num % 3 == 0, num % 5 == 0):
            case (True, True):
                print("bofa")
            case (True, False):
                print("sugma")
            case (False, True):
                print("ligma")
            case _:
                print(num)

if __name__ == "__main__":
    sugma_ligma()
