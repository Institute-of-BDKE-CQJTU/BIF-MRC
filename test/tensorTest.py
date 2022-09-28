import torch

def gather_positions(input_tensor, positions):
    """
    :param input_tensor: shape [batch_size, seq_length, dim]
    :param positions: shape [batch_size, num_positions]
    :return: [batch_size, num_positions, dim]
    """
    _, _, dim = input_tensor.size()
    index = positions.unsqueeze(-1).repeat(1, 1, dim).permute(1, 0, 2)  # [batch_size, num_positions, dim]
    print(index.size())
    print(index)
    gathered_output = torch.gather(input_tensor, dim=1, index=index)  # [batch_size, num_positions, dim]
    print(gathered_output.size())
    return gathered_output

A = torch.rand((20, 512, 768))
B = torch.randint(0, 511, [20])

print(B)
gather_positions(A, B)