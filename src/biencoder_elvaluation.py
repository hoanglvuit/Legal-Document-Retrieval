from sentence_transformers import SentenceTransformer
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Evaluation for Bi Encoder model") 
    parser.add_argument("--model_path", type=str, default="../saved_model/BiEncoder/model1/best")