import numpy as np
import spacy

import torch

model_ru = spacy.load('ru_core_news_lg')
case_to_integer = {
  'Nom': 1,  # Nominative
  'Acc': 2,  # Accusative
  'Gen': 3,  # Genitive
  'Ins': 4,  # Instrumental
  'Dat': 5,  # Dative
  'Loc': 6,  # Locative
  'DNE': 0  # Does Not Exist or Not Applicable
}
def encode_and_adjust_length(cases, desired_length):
  """
  Encodes a list of grammatical cases to integers, applying padding or trimming to match the desired length.

  :param cases: List of grammatical cases.
  :param desired_length: The desired length of the output list.
  :param case_to_integer_map: A dictionary mapping grammatical cases to integers.
  :return: List of integers with applied padding or trimming.
  """

  # Map the cases to integers
  encoded_cases = [case_to_integer.get(case, 0) for case in cases]

  # Trim the list if it's longer than the desired length
  if len(encoded_cases) > desired_length:
    encoded_cases = encoded_cases[:desired_length]

  # Pad the list with -1 if it's shorter than the desired length
  elif len(encoded_cases) < desired_length:
    padding = [-1] * (desired_length - len(encoded_cases))
    encoded_cases.extend(padding)

  return encoded_cases
def case_tagger(sent):
  tagged_sent = []
  doc = model_ru(sent)
  for token in doc:
    label_list = str(token.morph).split(sep="|")
    case_tag = "DNE"
    for label in label_list:
      if label.split(sep='=')[0] == "Case":
        case_tag = label.split(sep='=')[1]
    tagged_sent.append(case_tag)
  return encode_and_adjust_length(tagged_sent, 20)


def append_list_to_tensor(tensor, numbers_list):
  # Move tensor to CPU if it's on GPU
  if tensor.is_cuda:
    tensor = tensor.cpu()
  numbers_tensor = torch.tensor([numbers_list], dtype=tensor.dtype)
  return torch.cat((tensor, numbers_tensor), dim=1)


