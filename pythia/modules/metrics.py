# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from pythia.common.registry import registry
    from pythia.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_attributes:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""
import collections

import torch

from pythia.common.registry import registry


class Metrics:
    """Internally used by Pythia, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (List[ConfigNode]): List of ConfigNodes where each ConfigNode
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, list):
            metric_list = [metric_list]

        self.writer = registry.get("writer")
        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        for metric in metric_list:
            params = {}
            if isinstance(metric, collections.abc.Mapping):
                if not hasattr(metric, "type"):
                    raise ValueError(
                        "Metric {} needs to have 'type' attribute".format(metric)
                    )
                metric = metric.type
                params = getattr(metric, "params", {})
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )

            metric_cls = registry.get_metric_class(metric)
            if metric_cls is None:
                raise ValueError(
                    "No metric named {} registered to registry".format(metric)
                )
            metrics[metric] = metric_cls(**params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}
        if not hasattr(sample_list, "targets"):
            return values

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            for metric_name, metric_object in self.metrics.items():
                key = "{}/{}/{}".format(dataset_type, dataset_name, metric_name)
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values


class BaseMetric:
    """Base class to be inherited by all metrics registered to Pythia. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError(
            "'calculate' must be implemented in the child class"
        )

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value


@registry.register_metric("accuracy")
class Accuracy(BaseMetric):
    """Metric for calculating accuracy.

    **Key:** ``accuracy``
    """

    def __init__(self):
        super().__init__("accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        output = model_output["scores"]
        expected = sample_list["targets"]

        assert output.dim() <= 2, ("Output from model shouldn't have "
                                   " more than dim 2 for accuracy")
        assert expected.dim() <= 2, ("Expected target shouldn't have "
                                     "more than dim 2 for accuracy")

        if output.dim() == 2:
            output = torch.max(output, 1)[1]

        # If more than 1
        if expected.dim() == 2:
            expected = torch.max(expected, 1)[1]

        correct = (expected == output.squeeze()).sum().float()
        total = len(expected)

        value = correct / total
        return value


@registry.register_metric("caption_bleu4")
class CaptionBleu4Metric(BaseMetric):
    """Metric for calculating caption accuracy using BLEU4 Score.

    **Key:** ``caption_bleu4``
    """

    def __init__(self):
        import nltk.translate.bleu_score as bleu_score
        self._bleu_score = bleu_score
        super().__init__("caption_bleu4")
        self.caption_processor = registry.get("coco_caption_processor")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: bleu4 score.

        """
        # Create reference and hypotheses captions.
        references = []
        hypotheses = []

        # References
        targets = sample_list.answers
        for j, p in enumerate(targets):
            img_captions = [
                self.caption_processor(c)["tokens"] for c in targets[j].tolist()
            ]
            references.append(img_captions)

        # Hypotheses
        scores = torch.max(model_output["scores"], dim=-1)[1]
        scores = scores.tolist()
        predictions = []
        for j, p in enumerate(scores):
            caption = self.caption_processor(scores[j])["tokens"]
            predictions.append(caption)
        hypotheses.extend(predictions)

        assert len(references) == len(hypotheses)

        bleu4 = self._bleu_score.corpus_bleu(references, hypotheses)

        return targets.new_tensor(bleu4, dtype=torch.float)


@registry.register_metric("vqa_accuracy")
class VQAAccuracy(BaseMetric):
    """
    Calculate VQAAccuracy. Find more information here_

    **Key**: ``vqa_accuracy``.

    .. _here: https://visualqa.org/evaluation.html
    """

    def __init__(self):
        super().__init__("vqa_accuracy")

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate vqa accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: VQA Accuracy

        """
        output = model_output["scores"]
        expected = sample_list["targets"]

        output = self._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected.new_zeros(*expected.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = one_hots * expected
        accuracy = torch.sum(scores) / expected.size(0)

        return accuracy


class RecallAtK(BaseMetric):
    def __init__(self, name="recall@k"):
        super().__init__(name)

    def score_to_ranks(self, scores):
        # sort in descending order - largest score gets highest rank
        sorted_ranks, ranked_idx = scores.sort(1, descending=True)

        # convert from ranked_idx to ranks
        ranks = ranked_idx.clone().fill_(0)
        for i in range(ranked_idx.size(0)):
            for j in range(100):
                ranks[i][ranked_idx[i][j]] = j
        ranks += 1
        return ranks

    def get_gt_ranks(self, ranks, ans_ind):
        _, ans_ind = ans_ind.max(dim=1)
        ans_ind = ans_ind.view(-1)
        gt_ranks = torch.LongTensor(ans_ind.size(0))

        for i in range(ans_ind.size(0)):
            gt_ranks[i] = int(ranks[i, ans_ind[i].long()])
        return gt_ranks

    def get_ranks(self, sample_list, model_output, *args, **kwargs):
        output = model_output["scores"]
        expected = sample_list["targets"]

        ranks = self.score_to_ranks(output)
        gt_ranks = self.get_gt_ranks(ranks, expected)

        ranks = self.process_ranks(gt_ranks)
        return ranks.float()

    def calculate(self, sample_list, model_output, k, *args, **kwargs):
        ranks = self.get_ranks(sample_list, model_output)
        recall = float(torch.sum(torch.le(ranks, k))) / ranks.size(0)
        return recall


@registry.register_metric("r@1")
class RecallAt1(RecallAtK):
    """
    Calculate Recall@1 which specifies how many time the chosen candidate
    was rank 1.

    **Key**: ``r@1``.
    """

    def __init__(self):
        super().__init__("r@1")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@1

        """
        return self.calculate(sample_list, model_output, k=1)


@registry.register_metric("r@5")
class RecallAt5(RecallAtK):
    """
    Calculate Recall@5 which specifies how many time the chosen candidate
    was among first 5 rank.

    **Key**: ``r@5``.
    """

    def __init__(self):
        super().__init__("r@5")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@5 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@5

        """
        return self.calculate(sample_list, model_output, k=5)


@registry.register_metric("r@10")
class RecallAt10(RecallAtK):
    """
    Calculate Recall@10 which specifies how many time the chosen candidate
    was among first 10 ranks.

    **Key**: ``r@10``.
    """

    def __init__(self):
        super().__init__("r@10")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall@10 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Recall@10

        """
        return self.calculate(sample_list, model_output, k=10)


@registry.register_metric("mean_r")
class MeanRank(RecallAtK):
    """
    Calculate MeanRank which specifies what was the average rank of the chosen
    candidate.

    **Key**: ``mean_r``.
    """

    def __init__(self):
        super().__init__("mean_r")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: mean rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks)


@registry.register_metric("mean_rr")
class MeanReciprocalRank(RecallAtK):
    """
    Calculate reciprocal of mean rank..

    **Key**: ``mean_rr``.
    """

    def __init__(self):
        super().__init__("mean_rr")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Mean Reciprocal Rank and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: Mean Reciprocal Rank

        """
        ranks = self.get_ranks(sample_list, model_output)
        return torch.mean(ranks.reciprocal())


@registry.register_metric("textvqa_accuracy")
class TextVQAAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("textvqa_accuracy")
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.TextVQAAccuracyEvaluator()
        self.gt_key = 'gt_answers_enc'

    def calculate(self, sample_list, model_output, *args, **kwargs):
        answer_processor = registry.get(
            sample_list.dataset_name + "_answer_processor"
        )

        batch_size = sample_list.context_tokens_enc.size(0)
        pred_answers = model_output["scores"].argmax(dim=-1)
        context_tokens_enc = sample_list.context_tokens_enc.cpu().numpy()
        gt_answers_enc = sample_list[self.gt_key].cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        from pythia.utils.objects_to_byte_tensor import dec_bytes2obj
        from pythia.utils.text_utils import word_tokenize
        for idx in range(batch_size):
            context_tokens = dec_bytes2obj(context_tokens_enc[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            gt_answers = dec_bytes2obj(gt_answers_enc[idx])
            predictions.append({
                "pred_answer": pred_answer,
                "gt_answers": gt_answers,
            })

        accuracy = self.evaluator.eval_pred_list(predictions)
        accuracy = torch.tensor(accuracy).cuda()

        return accuracy


@registry.register_metric("stvqa_anls")
class STVQAANLS(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "stvqa_anls"
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.STVQAANLSEvaluator()


@registry.register_metric("stvqa_accuracy")
class STVQAAccuracy(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "stvqa_accuracy"
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.STVQAAccuracyEvaluator()


@registry.register_metric("ocrvqa_accuracy")
class OCRVQAAccuracy(STVQAAccuracy):
    def __init__(self):
        super().__init__()
        # same as STVQAAccuracy except for the name
        self.name = "ocrvqa_accuracy"


@registry.register_metric("textcaps_bleu4")
class TextCapsBleu4(TextVQAAccuracy):
    def __init__(self):
        super().__init__()
        self.name = "textcaps_bleu4"
        self.gt_key = 'ref_strs'
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.TextCapsBleu4Evaluator()


@registry.register_metric("maskpred_accuracy")
class PreTrainMLMAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("maskpred_accuracy")
        self.contrapred_accuracy = PreTrainContraAccuracy()
        self.rpp_accuracy = PreTrainRPPAccuracy()

    def calculate(self, sample_list, model_output, *args, **kwargs):
        ## (B,T,token), (B,T)
        batch_size, tokenlen, vocab_size = model_output["textcls_scores"].shape
        scores = model_output["textcls_scores"].view(batch_size*tokenlen,-1)
        targets = sample_list["cmb_text_mask_label"].view(batch_size*tokenlen)
        loss_mask = (targets!=-1).float()
        pollute_mask = sample_list["tag_pollute"].float()
        pollute_mask = (1-pollute_mask).repeat(1,tokenlen).view(batch_size*tokenlen)   ## 1 is polluted, not compute MLM
        loss_mask = (loss_mask * pollute_mask).bool()
        if loss_mask.float().sum()==0.:
            accuracy = torch.zeros(1).cuda()
        else:
            scores, targets = scores[loss_mask], targets[loss_mask]
            accuracy = torch.sum(scores.argmax(1)==targets).float()/max(targets.shape[0],1)
        return (accuracy + self.contrapred_accuracy.calculate(sample_list, model_output) + self.rpp_accuracy.calculate(sample_list, model_output))/3.

@registry.register_metric("contrapred_accuracy")
class PreTrainContraAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("contrapred_accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = model_output["pollutecls_scores"]
        if "tag_pollute" not in sample_list:
            targets = sample_list["ocrtag_pollute"].float()
        else:
            targets = sample_list["tag_pollute"].float()
        scores, targets = (torch.sigmoid(scores)>0.5).float(), (targets>0.5).float()
        accuracy = (scores==targets).float().sum()/max(scores.shape[0],1)
        return accuracy

@registry.register_metric("sourcepred_accuracy")
class SourceAccuracy(BaseMetric):
    """
    Measure accuracy for predicting source of answer.
    """
    def __init__(self):
        super().__init__("sourcepred_accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        scores = model_output["src"]
        targets = sample_list["tag_source"].float()
        scores, targets = (torch.sigmoid(scores)>0.5).float(), (targets>0.5).float()
        accuracy = (scores==targets).float().sum()/max(scores.shape[0],1)
        return accuracy

@registry.register_metric("regionpred_loss")
class RegionPredLoss(BaseMetric):
    """
    Measure loss for region prediction task.
    """
    def __init__(self):
        super().__init__("regionpred_loss")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        diff = torch.absolute(model_output['region'] - torch.squeeze(sample_list['reg_box']))
        return torch.mean(diff)

@registry.register_metric("positionpred_accuracy")
class PreTrainRPPAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("positionpred_accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        ## binary
        scores = model_output["overlapcls_scores"].squeeze(-1)
        targets = sample_list["overlap"].float()
        mask = targets!=-1
        scores, targets = (scores[mask]>0.5).float(), (targets[mask]>0.5).float()
        accuracy = (scores==targets).float().sum()/max(scores.shape[0],1)
        # # cls
        # scores = model_output["overlapcls_scores"].view(-1,model_output["overlapcls_scores"].shape[-1])
        # targets = sample_list["overlap"].float().view(-1)
        # mask = targets!=-1
        # accuracy = (torch.argmax(scores,dim=1)[mask]==targets[mask]).float().sum()/max(targets[mask].shape[0],1)
        return accuracy