import argparse
import json
import torch
import math
from pplm_classification_head import ClassificationHead
from run_pplm_discrim_train import Discriminator


def load_model(pretrained_model,sentence,discrim_weights,discrim_meta, device='cpu',cached=False):

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights

    classifier = ClassificationHead(
        class_size=meta["class_size"],
        embed_size=meta["embed_size"]
    ).to('cpu').eval()

    classifier.load_state_dict(
        torch.load(discrim_weights, map_location=device))
    classifier.eval()

    model = Discriminator(pretrained_model=pretrained_model, classifier_head=classifier, cached_mode=cached, device=device)
    model.eval()

    classes = [c for i, c in enumerate(meta["class_vocab"])]
    predict(sentence, model, classes)

def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument('--sentence', type=str, default='I am so happy!',
                        help='Sentence to input')
    parser.add_argument('--discrim_weights', type=str, default='emotion.pt',
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default='emotion.json',
                        help='Meta information for the generic discriminator')

    args = parser.parse_args()

    load_model(**(vars(args)))
