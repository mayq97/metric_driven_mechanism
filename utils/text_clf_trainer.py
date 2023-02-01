from torch import nn
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
import torch
from typing import Any,  Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers.utils import logging
from sklearn.metrics import f1_score
import torch


def collate_to_max_length(dict_batch):
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """

    key_list = ['input_ids', 'token_type_ids',
                'attention_mask', 'position_ids']
    batch = [[example[k] for k in key_list] for example in dict_batch]

    max_length = max(x[0].shape[0] for x in batch)

    output = {}

    for item in dict_batch:
        for feat_name, feat_values in item.items():
            if not feat_name == "labels":
                len_feat = len(feat_values)
                feat_values = torch.cat(
                    (feat_values, torch.LongTensor([0]*(max_length-len_feat))), 0)
            temp = output.get(feat_name, [])
            temp.append(feat_values)
            output[feat_name] = temp
    key_list = ['input_ids', 'token_type_ids',
                'attention_mask', 'position_ids']
    for k in key_list:
        v = output[k]
        output[k] = torch.stack(v).to("cuda")
    output["labels"] = torch.LongTensor(output["labels"]).to("cuda")
    return output


logger = logging.get_logger(__name__)


class TextClfTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(
                **inputs
            )

        return (outputs.loss, outputs.logits, inputs["labels"])

    # def evaluate(
    #     self,
    #     eval_dataset,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> Dict[str, float]:

    #     self._memory_tracker.start()

    #     eval_dataloader = self.get_eval_dataloader(eval_dataset)
    #     start_time = time.time()

    #     eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
    #     output = self.predict_loop(
    #         eval_dataloader,
    #         description="Evaluation",
    #         prediction_loss_only=True if self.compute_metrics is None else None,
    #     )

    #     self.log(output.metrics)

    #     self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

    #     self._memory_tracker.stop_and_update_metrics(output.metrics)

    #     return output.metrics

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> PredictionOutput:
        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        self.callback_handler.eval_dataloader = dataloader

        label_ids = []
        preds = []
        metrics = {}
        for step, inputs in enumerate(dataloader):
            loss, pred_logits, real_labels = self.prediction_step(
                model, inputs, prediction_loss_only)
            pred_labels = torch.argmax(pred_logits, dim=1)
            preds.extend(pred_labels.cpu().numpy().tolist())
            label_ids.extend(real_labels.cpu().numpy().tolist())

        # print(classification_report(y_true = label_ids, y_pred = preds,labels = ["non_mechan","mechan"],digits = 4,zero_division=0))
        # clf_res = classification_report(y_true = label_ids, y_pred = preds,labels = ["non_mechan","mechan"])
        metrics["eval_macro_f1"] = f1_score(
            y_true=label_ids, y_pred=preds, average="macro")
        metrics["eval_f1"] = f1_score(
            y_true=label_ids, y_pred=preds, average="binary")
        print(metrics)
        # if description == "Evaluation":
        return EvalLoopOutput(predictions=preds, label_ids=label_ids, metrics=metrics, num_samples=num_examples)
        # return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset
        train_sampler = self._get_train_sampler()
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=collate_to_max_length,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=collate_to_max_length,
            num_workers=self.args.dataloader_num_workers,
        )
