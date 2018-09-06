def infer_mask_from_batch_data(batch_data):
    """
    Create binary mask for all non-empty timesteps
    :param batch_data: BatchSize x SequenceLen x Features
    :return: BatchSize x SequenceLen
    """
    return batch_data.abs().sum(-1) > 0


def infer_lengths_from_mask(mask):
    """
    Get array of lengths from binary mask
    :param mask: BatchSize x SequenceLen
    :return: BatchSize
    """
    return mask.long().sum(1)
