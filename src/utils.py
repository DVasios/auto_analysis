import matplotlib.pyplot as plt

# Utils
def has_missing_data(df_descr):
    for f, d in df_descr['features'].items():

        if (d['eda']['missing_data']['percentage'] > 0.0):
            return True
    return False

def plot_convergence_random(accuracies):

    x_labels = [i for i in range(len(accuracies))]
    y_labels = [1-i for i in accuracies]

    # Create the line plot
    plt.plot(x_labels, y_labels, label='Optimization Accuracies', color='blue', marker='o')

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Line Plot Example')

    # Show legend (for the label)
    plt.legend()

    # Show grid (optional)
    plt.grid(True)

    # Display the plot
    plt.show()