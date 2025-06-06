import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, logging


warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()

softmax = nn.Softmax(dim=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLMConfig:
    MULTILINGUAL_MODELS = ['google-bert/bert-base-multilingual-cased', 'FacebookAI/xlm-roberta-large']
    GENERAL_MODELS = ['google-bert/bert-base-cased', 'distilbert/distilbert-base-cased']
    DOMAIN_SPECIFIC_MODELS = ['allenai/scibert_scivocab_uncased']
    MODELS_LIST = MULTILINGUAL_MODELS + GENERAL_MODELS + DOMAIN_SPECIFIC_MODELS
    TOP_K = 10
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results/MLM_results")


def load_book_content(book_path: Path) -> dict:
    """Load JSON book content from a file path."""
    with open(book_path, 'r') as f:
        return json.load(f)


def extract_prompts_and_labels(book_content: dict) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Extract paragraph-level and sentence-level prompts and labels."""
    paragraph_prompts, paragraph_labels = [], []
    sentence_prompts, sentence_labels = [], []

    for concept in book_content:
        para_level = book_content[concept].get("paragraph_level", [])
        for entry in para_level:
            if not entry.get('ignore', False):
                paragraph_prompts.append(entry['prompt'])
                paragraph_labels.append(entry['label'])

        sent_level = book_content[concept].get("sentence_level", [])
        for entry in sent_level:
            if not entry.get('ignore', False):
                sentence_prompts.append(entry['prompt'])
                sentence_labels.append(entry['label'])

    return paragraph_prompts, paragraph_labels, sentence_prompts, sentence_labels


def get_masked_token_replacement(tokenizer: AutoTokenizer) -> str:
    """Return the masked token used by the tokenizer."""
    return tokenizer.mask_token


def truncate_token_ids(token_ids: torch.Tensor, max_length: int) -> torch.Tensor:
    """Truncate token ids tensor to the model max length."""
    if token_ids.size(1) > max_length:
        return token_ids[:, :max_length]
    return token_ids


def calculate_top_k_scores(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    labels: List[str],
    top_k: int,
    device: torch.device
) -> Tuple[List[List[str]], Dict[str, float]]:
    """
    Given a list of prompts and labels, compute top-k predictions and
    calculate accuracy scores for top 1, 5, and 10 predictions.
    """
    results_predictions = []
    top_k_score = {'1': 0, '5': 0, '10': 0}

    entities = list(set(labels))
    ent_ids_lists = [tokenizer.encode(ent, add_special_tokens=False) for ent in entities]
    entity_lengths = [len(ent_ids) for ent_ids in ent_ids_lists]
    unique_entity_lengths = list(set(entity_lengths))

    softmax_fn = nn.Softmax(dim=0)

    calculated_prompts_counter = 0

    for idx, prompt in enumerate(prompts):
        try:
            if prompt.count(tokenizer.mask_token) != 1:
                # Skip prompts with more than one masked token for this approach
                continue

            calculated_prompts_counter += 1
            correct_answer = labels[idx]

            # Duplicate prompt for each unique entity length with corresponding mask tokens
            masked_prompts = [
                prompt.replace(tokenizer.mask_token, tokenizer.mask_token * length).replace('][', '] [')
                for length in unique_entity_lengths
            ]

            probabilities_per_length = {}

            for masked_prompt, ent_length in zip(masked_prompts, unique_entity_lengths):
                token_ids = tokenizer.encode(masked_prompt, return_tensors='pt')
                token_ids = truncate_token_ids(token_ids, tokenizer.model_max_length).to(device)

                masked_positions = (token_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

                with torch.no_grad():
                    outputs = model(token_ids)
                logits = outputs.logits.squeeze(0)

                probs_per_token = []
                for pos in masked_positions:
                    token_logits = logits[pos]
                    probs = softmax_fn(token_logits)
                    probs_per_token.append(probs)
                probabilities_per_length[ent_length] = probs_per_token

            entity_scores = []
            for ent_ids in ent_ids_lists:
                log_probs = []
                for i, token_id in enumerate(ent_ids):
                    prob = probabilities_per_length[len(ent_ids)][i][token_id]
                    log_probs.append(torch.log(prob))
                entity_scores.append(torch.stack(log_probs).mean())

            topk_indices = torch.topk(torch.tensor(entity_scores), top_k).indices.tolist()
            top_k_entities = [entities[i] for i in topk_indices]

            results_predictions.append(top_k_entities)

            # Check if correct answer or its lowercase variant is in the top-k
            def in_top_k(k):
                return (correct_answer in top_k_entities[:k]) or (correct_answer.lower() in top_k_entities[:k])

            if in_top_k(1):
                top_k_score['1'] += 1
            if in_top_k(5):
                top_k_score['5'] += 1
            if in_top_k(10):
                top_k_score['10'] += 1

            if idx % 200 == 0:
                print(f"Processed {idx}/{len(prompts)} prompts.")

        except (IndexError, RuntimeError) as e:
            # Prompt is too long -- happens rarely
            continue

    # Normalize scores by number of valid prompts
    for k in top_k_score:
        if calculated_prompts_counter > 0:
            top_k_score[k] /= calculated_prompts_counter
        else:
            top_k_score[k] = 0.0

    return results_predictions, top_k_score


def run_evaluation():
    results = {}
    models_predictions = {}

    for language_dir in MLMConfig.DATA_DIR.iterdir():
        if not language_dir.is_dir():
            continue
        language = language_dir.name
        print(f"Processing language: {language}")

        book_paths = list(language_dir.iterdir())
        for book_idx, book_path in enumerate(book_paths, 1):
            book_name = book_path.name
            print(f"\nProcessing book ({book_idx}/{len(book_paths)}): {book_name}")

            book_content = load_book_content(book_path)
            paragraph_prompts, paragraph_labels, sentence_prompts, sentence_labels = extract_prompts_and_labels(book_content)

            for model_idx, model_name in enumerate(MLMConfig.MODELS_LIST, 1):
                print(f"\nEvaluating model ({model_idx}/{len(MLMConfig.MODELS_LIST)}): {model_name}")
                model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True)

                # Replace masked tokens
                sentence_level_prompts = [p.replace('[MASK]', get_masked_token_replacement(tokenizer)) for p in sentence_prompts]
                paragraph_level_prompts = [p.replace('[MASK]', get_masked_token_replacement(tokenizer)) for p in paragraph_prompts]

                # Sentence-level predictions
                print("Running sentence-level predictions...")
                sent_preds, sent_scores = calculate_top_k_scores(
                    model, tokenizer, sentence_level_prompts, sentence_labels, MLMConfig.TOP_K, device
                )
                key_sent = f"{language}_{model_name}_{book_name}_sentence_level"
                models_predictions[key_sent] = sent_preds
                results[key_sent] = sent_scores
                print(f"Sentence-level top-1 accuracy: {sent_scores['1']:.4f}, top-5: {sent_scores['5']:.4f}, top-10: {sent_scores['10']:.4f}")

                # Paragraph-level predictions
                print("Running paragraph-level predictions...")
                para_preds, para_scores = calculate_top_k_scores(
                    model, tokenizer, paragraph_level_prompts, paragraph_labels, MLMConfig.TOP_K, device
                )
                key_para = f"{language}_{model_name}_{book_name}_paragraph_level"
                models_predictions[key_para] = para_preds
                results[key_para] = para_scores
                print(f"Paragraph-level top-1 accuracy: {para_scores['1']:.4f}, top-5: {para_scores['5']:.4f}, top-10: {para_scores['10']:.4f}")

                # Save results progressively
                MLMConfig.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                with open(MLMConfig.RESULTS_DIR / f"{language}_results.json", "w") as f:
                    json.dump(results, f, indent=2)

                with open(MLMConfig.RESULTS_DIR / f"{language}_models_predictions.json", "w") as f:
                    json.dump(models_predictions, f, indent=2)


if __name__ == "__main__":
    run_evaluation()
