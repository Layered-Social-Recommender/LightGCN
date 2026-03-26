import sys
import traceback
import lightgcn_movielens

if __name__ == "__main__":
    with open("output_fixed.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        sys.stderr = f
        try:
            lightgcn_movielens.main()
        except Exception as e:
            traceback.print_exc(file=f)
