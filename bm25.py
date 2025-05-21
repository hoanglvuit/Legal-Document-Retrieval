from rank_bm25 import BM25Okapi,BM25L,BM25Plus
import argparse

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Use BM25 for retrieval') 
    parser.add_argument("--model", type=str, default="bm25o", choices=["bm25o", "bm25l", "bm25plus"])  
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75) 
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=0.5) 
    parser.add_argument("--input_dirr")

args = parser.parse_args() 

# load data


