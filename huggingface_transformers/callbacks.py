import numpy as np

from transformers import EvalPrediction
from transformers import TrainerCallback


class EvalCallback(TrainerCallback):
  def on_evaluate(self, args, state, control, model, eval_dataloader=None, **kwargs):
    eval(eval_dataloader, model)


def evaluation_metric(eval_prediction):
  predictions = eval_prediction.predictions
  target = eval_prediction.label_ids
  pred, trg = predictions, target
  sse, mse = model_performance(pred, trg, print_output=True)
  return {"sse": str(sse), "mse": str(mse)}

# How we print the model performance
def model_performance(output, target, print_output=False):
    """
    Returns SSE and MSE per batch (printing the MSE and the RMSE)
    """

    sq_error = (output - target)**2

    sse = np.sum(sq_error)
    mse = np.mean(sq_error)
    rmse = np.sqrt(mse)

    if print_output:
        print(f'| MSE: {mse:.2f} | RMSE: {rmse:.2f} |')

    return sse, mse