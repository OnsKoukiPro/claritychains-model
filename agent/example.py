from sequencer import get_sequence, plot_sequence
if __name__ == "__main__":
    outputs = get_sequence("data/test.bvh", verbose=True)
    plot_sequence(outputs)
    a = 1