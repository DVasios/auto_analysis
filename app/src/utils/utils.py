import matplotlib.pyplot as plt

# Utils
def has_missing_data(df_descr):
    for f, d in df_descr['features'].items():

        if (d['eda']['missing_data']['percentage'] > 0.0):
            print('yes')
            print(f)

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

#
def results(opt_type, res):

    if (opt_type == 'bayesian'):

        print(f"---- Auto Analysis Report ----")
        # Best Accuracy
        print('\n\n\n')
        print(f"Best Accuracy: {1 - res[0][0]}\n")

        # Suggested Pipeline
        print(f"Suggested Pipeline:")

        pip = res[1]['x']
        # Outliers
        remove_out = 'True' if pip[2] == 'True' else 'False'
        print(f"Remove Outliers: {remove_out}")


