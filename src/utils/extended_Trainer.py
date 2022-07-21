from naslib.defaults.trainer import Trainer
from naslib.utils import utils
from naslib.utils.logging import log_every_n_seconds, log_first_n

from typing import Callable
import numpy as np
import torch
import time
import logging

logger = logging.getLogger(__name__)

class ExtendedTrainer(Trainer):
    """
    Adding extra logging information for configurations to the trainer. logging sampled architecture
    """

    def __init__(self, *args, **kwargs):
        self.configs = {"configs": []}
        super(ExtendedTrainer, self).__init__(*args, **kwargs)

    def search(self, resume_from="", summary_writer=None, after_epoch: Callable[[int], None]=None, report_incumbent=True):
        """
                Start the architecture search.

                Generates a json file with training statistics.

                Args:
                    resume_from (str): Checkpoint file to resume from. If not given then
                        train from scratch.
                """
        logger.info("Start training")

        np.random.seed(self.config.search.seed)
        torch.manual_seed(self.config.search.seed)

        self.optimizer.before_training()
        checkpoint_freq = self.config.search.checkpoint_freq
        if self.optimizer.using_step_function:
            self.scheduler = self.build_search_scheduler(
                self.optimizer.op_optimizer, self.config
            )

            start_epoch = self._setup_checkpointers(
                resume_from, period=checkpoint_freq, scheduler=self.scheduler
            )
        else:
            start_epoch = self._setup_checkpointers(resume_from, period=checkpoint_freq)

        if self.optimizer.using_step_function:
            self.train_queue, self.valid_queue, _ = self.build_search_dataloaders(
                self.config
            )

        for e in range(start_epoch, self.epochs):

            start_time = time.time()
            self.optimizer.new_epoch(e)

            if self.optimizer.using_step_function:
                for step, data_train in enumerate(self.train_queue):

                    data_train = (
                        data_train[0].to(self.device),
                        data_train[1].to(self.device, non_blocking=True),
                    )
                    data_val = next(iter(self.valid_queue))

                    data_val = (
                        data_val[0].to(self.device),
                        data_val[1].to(self.device, non_blocking=True),
                    )

                    stats = self.optimizer.step(data_train, data_val)
                    logits_train, logits_val, train_loss, val_loss = stats

                    self._store_accuracies(logits_train, data_train[1], "train")
                    self._store_accuracies(logits_val, data_val[1], "val")

                    log_every_n_seconds(
                        logging.INFO,
                        "Epoch {}-{}, Train loss: {:.5f}, validation loss: {:.5f}, learning rate: {}".format(
                            e, step, train_loss, val_loss, self.scheduler.get_last_lr()
                        ),
                        n=5,
                    )

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=3,
                        )

                    self.train_loss.update(float(train_loss.detach().cpu()))
                    self.val_loss.update(float(val_loss.detach().cpu()))

                self.scheduler.step()

                end_time = time.time()

                self.errors_dict.train_acc.append(self.train_top1.avg)
                self.errors_dict.train_loss.append(self.train_loss.avg)
                self.errors_dict.valid_acc.append(self.val_top1.avg)
                self.errors_dict.valid_loss.append(self.val_loss.avg)
                self.errors_dict.runtime.append(end_time - start_time)
            else:
                end_time = time.time()
                # TODO: nasbench101 does not have train_loss, valid_loss, test_loss implemented, so this is a quick fix for now
                # train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss = self.optimizer.train_statistics()
                (
                    train_acc,
                    valid_acc,
                    test_acc,
                    train_time,
                ) = self.optimizer.train_statistics(report_incumbent)
                train_loss, valid_loss, test_loss = -1, -1, -1

                self.errors_dict.train_acc.append(train_acc)
                self.errors_dict.train_loss.append(train_loss)
                self.errors_dict.valid_acc.append(valid_acc)
                self.errors_dict.valid_loss.append(valid_loss)
                self.errors_dict.test_acc.append(test_acc)
                self.errors_dict.test_loss.append(test_loss)
                self.errors_dict.runtime.append(end_time - start_time)
                self.errors_dict.train_time.append(train_time)
                self.train_top1.avg = train_acc
                self.val_top1.avg = valid_acc

            # get sampled architecture from the optimizer
            if report_incumbent:
                arch_config = self.optimizer.get_final_architecture()
            else:
                try:
                    arch_config = self.optimizer.sampled_archs[-1].arch
                except AttributeError:
                    try:
                        arch_config = self.optimizer.population[-1].arch
                    except AttributeError:
                        arch_config = self.optimizer.train_data[-1].arch

            # convert sampled architecture into op_indices string
            op_indices = "".join([str(i) for i in arch_config.get_op_indices()])
            self.configs["configs"].append(op_indices)

            self.periodic_checkpointer.step(e)

            anytime_results = self.optimizer.test_statistics()
            if anytime_results:
                # record anytime performance
                self.errors_dict.arch_eval.append(anytime_results)
                log_every_n_seconds(
                    logging.INFO,
                    "Epoch {}, Anytime results: {}".format(e, anytime_results),
                    n=5,
                )

            self.errors_dict.update(self.configs)
            self._log_to_json()

            self._log_and_reset_accuracies(e, summary_writer)

            if after_epoch is not None:
                after_epoch(e)

        self.optimizer.after_training()

        if summary_writer is not None:
            summary_writer.close()

        logger.info("Training finished")